#include "rys.h"
#include "hgp.h"       // Auto path picks between OS, HGP and Rys per quartet.
#include "os.h"        // Auto path OS branch (high-L corner).
#include "rys_roots.h"
#include "screening.h" // Shared HGP-based Schwarz table for the Auto path.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include <tuple>

// ─── Scratch buffer dimensions ────────────────────────────────────────────────
//
// VRR_DIM = 2*MAX_L+1 = 13  (per-axis, matches os.cpp convention)
// The compact 1D VRR tables (Ix/Iy/Iz, [VRR_DIM]^2) and the 3D CD-HRR slice
// (V0_CD/W, [VRR_DIM]^3) are small stack arrays kept at the fixed bound. The
// large 6D accumulator is sized per quartet in RysScratch below (was a fixed
// [VRR_DIM]^6 = 38.5 MB/thread).

static constexpr int VRR_DIM = 2 * MAX_L + 1; // 13; per-axis bound for the compact 1D/3D stack helpers

// Accumulated Rys 6D intermediate: sum_{roots} w_r * Ix[ax][cx] * Iy[ay][cy] * Iz[az][cz].
//
// The 6D accumulator is the only large Rys scratch buffer. A fixed
// [VRR_DIM]^6 = [13]^6 array would be 38.5 MB/thread (sized off the global
// MAX_L=6), almost all of it unused: under Auto only L_AB+L_CD<=1 quartets
// reach Rys, and even explicit `engine rys` tops out at F in practice. Instead
// size it per quartet, mirroring the HGP/OS EriScratch pattern: a thread-local
// std::vector grown only when the active (lAB*+1)x(lCD*+1) dimensions change,
// reused across quartets. Under Auto the dimension is constant so it resizes
// once per thread then pure-reuses. Index layout matches the old C array,
// [ax][ay][az][cx][cy][cz] row-major, so the recurrence reads/writes are
// unchanged bit-for-bit.
namespace
{
    struct RysScratch
    {
        int ax_dim = 0, ay_dim = 0, az_dim = 0;
        int cx_dim = 0, cy_dim = 0, cz_dim = 0;
        std::size_t ax_stride = 0, ay_stride = 0, az_stride = 0;
        std::size_t cx_stride = 0, cy_stride = 0, cz_stride = 0;
        std::size_t size = 0;
        std::vector<double> buf;
        double *data = nullptr;

        // Size the 6D accumulator for one quartet. Dimensions are the total
        // bra/ket angular momentum per axis plus one (indices run 0..lAB*,
        // 0..lCD*). HRR reads one cell beyond the active index along the swept
        // axis, which stays within [0, lAB*] / [0, lCD*], so these dims suffice.
        void resize_for_quartet(
            int lABx, int lABy, int lABz,
            int lCDx, int lCDy, int lCDz)
        {
            ax_dim = lABx + 1;
            ay_dim = lABy + 1;
            az_dim = lABz + 1;
            cx_dim = lCDx + 1;
            cy_dim = lCDy + 1;
            cz_dim = lCDz + 1;
            cz_stride = 1;
            cy_stride = static_cast<std::size_t>(cz_dim) * cz_stride;
            cx_stride = static_cast<std::size_t>(cy_dim) * cy_stride;
            az_stride = static_cast<std::size_t>(cx_dim) * cx_stride;
            ay_stride = static_cast<std::size_t>(az_dim) * az_stride;
            ax_stride = static_cast<std::size_t>(ay_dim) * ay_stride;
            const std::size_t needed =
                static_cast<std::size_t>(ax_dim) * ay_dim * az_dim *
                cx_dim * cy_dim * cz_dim;
            if (buf.size() != needed)
                buf.resize(needed);
            size = needed;
            data = buf.data();
        }

        std::size_t index(int ax, int ay, int az, int cx, int cy, int cz) const noexcept
        {
            return static_cast<std::size_t>(ax) * ax_stride +
                   static_cast<std::size_t>(ay) * ay_stride +
                   static_cast<std::size_t>(az) * az_stride +
                   static_cast<std::size_t>(cx) * cx_stride +
                   static_cast<std::size_t>(cy) * cy_stride +
                   static_cast<std::size_t>(cz) * cz_stride;
        }

        double &at(int ax, int ay, int az, int cx, int cy, int cz) noexcept
        {
            return data[index(ax, ay, az, cx, cy, cz)];
        }
    };

    thread_local RysScratch g_rys_scratch;
} // namespace

// ─── Schwarz screening table ──────────────────────────────────────────────────
//
// Q(i,j) = sqrt((ij|ij)) — identical to os.cpp's _compute_schwarz_table.
// Declared here; uses _rys_contracted_eri internally once rys.cpp is complete.
// For now, forward-declare and implement after the ERI functions.

static Eigen::MatrixXd _rys_schwarz_table(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    std::size_t nbasis,
    const std::vector<HartreeFock::SignedAOSymOp> *sym_ops);

namespace
{
    using SymOps = std::vector<HartreeFock::SignedAOSymOp>;

    struct PairOrbitElem
    {
        std::size_t i = 0;
        std::size_t j = 0;
        int sign = 1;
    };

    struct QuartetOrbitElem
    {
        std::size_t i = 0;
        std::size_t j = 0;
        std::size_t k = 0;
        std::size_t l = 0;
        int sign = 1;
    };

    struct ScreenedKernelData
    {
        double rho = 0.0;
        double prefactor_scale = 1.0;
        double boys_scale = 1.0;
    };

    static ScreenedKernelData screened_kernel_data(
        double rho,
        HartreeFock::ERIKernel kernel,
        double omega) noexcept
    {
        // Range-separated kernels are implemented by replacing the Coulomb rho
        // parameter, prefactor, and Boys-function argument with their screened
        // equivalents. Keeping that map in one helper lets OS and Rys share the
        // same physics-level interpretation.
        if (kernel == HartreeFock::ERIKernel::Coulomb)
            return ScreenedKernelData{.rho = rho, .prefactor_scale = 1.0, .boys_scale = 1.0};

        if (omega <= 0.0)
        {
            return kernel == HartreeFock::ERIKernel::LongRange
                       ? ScreenedKernelData{.rho = 0.0, .prefactor_scale = 0.0, .boys_scale = 0.0}
                       : ScreenedKernelData{.rho = rho, .prefactor_scale = 1.0, .boys_scale = 1.0};
        }

        const double omega2 = omega * omega;
        const double lambda = omega2 / (omega2 + rho);
        return ScreenedKernelData{
            .rho = lambda * rho,
            .prefactor_scale = std::sqrt(lambda),
            .boys_scale = lambda};
    }

    static bool use_symmetry_ops(const SymOps *sym_ops)
    {
        return sym_ops != nullptr && sym_ops->size() > 1;
    }

    // ─── Shell-quartet iteration support (H-10 step A3, Auto path) ───────────
    //
    // Reconstruct the shell grouping from the per-AO shell_pairs list so the
    // Auto _compute_2e_auto build can iterate at shell-quartet granularity.
    // build_shellpairs emits every diagonal pair (i,i), so each AO i appears as
    // some pair's A side with A._index == i; runs of AOs sharing the same Shell*
    // (contiguous by construction) form the groups. Mirrors the OS/HGP
    // shell_groups_from_pairs; engine-local because the helpers live in this
    // translation unit's anonymous namespace.
    struct RysShellGroup
    {
        std::size_t first_ao = 0;     // _index of component 0
        std::size_t n_components = 0; // (L+1)(L+2)/2
    };

    static std::vector<RysShellGroup> shell_groups_from_pairs(
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        std::size_t nbasis,
        std::vector<const HartreeFock::ContractedView *> &ao_views)
    {
        ao_views.assign(nbasis, nullptr);
        for (const auto &sp : shell_pairs)
        {
            if (sp.A._index < nbasis)
                ao_views[sp.A._index] = &sp.A;
            if (sp.B._index < nbasis)
                ao_views[sp.B._index] = &sp.B;
        }

        std::vector<RysShellGroup> groups;
        const HartreeFock::Shell *current = nullptr;
        for (std::size_t ao = 0; ao < nbasis; ++ao)
        {
            const HartreeFock::ContractedView *view = ao_views[ao];
            const HartreeFock::Shell *shell = view ? view->_shell : nullptr;
            if (groups.empty() || shell != current)
            {
                groups.push_back(RysShellGroup{ao, 1});
                current = shell;
            }
            else
            {
                ++groups.back().n_components;
            }
        }
        return groups;
    }

    static void canonicalize_pair(std::size_t &i, std::size_t &j)
    {
        if (i > j)
            std::swap(i, j);
    }

    static void canonicalize_quartet(
        std::size_t &i, std::size_t &j,
        std::size_t &k, std::size_t &l)
    {
        canonicalize_pair(i, j);
        canonicalize_pair(k, l);
        if (std::tie(i, j) > std::tie(k, l))
        {
            std::swap(i, k);
            std::swap(j, l);
        }
    }

    static bool append_pair_orbit(
        std::vector<PairOrbitElem> &orbit,
        std::size_t i, std::size_t j, int sign)
    {
        for (const auto &elem : orbit)
        {
            if (elem.i == i && elem.j == j)
                return elem.sign == sign;
        }
        orbit.push_back({i, j, sign});
        return true;
    }

    static bool append_quartet_orbit(
        std::vector<QuartetOrbitElem> &orbit,
        std::size_t i, std::size_t j,
        std::size_t k, std::size_t l,
        int sign)
    {
        for (const auto &elem : orbit)
        {
            if (elem.i == i && elem.j == j &&
                elem.k == k && elem.l == l)
                return elem.sign == sign;
        }
        orbit.push_back({i, j, k, l, sign});
        return true;
    }

    static std::pair<std::vector<PairOrbitElem>, bool> build_pair_orbit(
        std::size_t i, std::size_t j, const SymOps &sym_ops)
    {
        // Symmetry replication is shared with the OS engine: identify the
        // canonical pair once, then reuse it across the whole symmetry orbit.
        std::vector<PairOrbitElem> orbit;
        orbit.reserve(sym_ops.size());

        for (const auto &op : sym_ops)
        {
            std::size_t ii = static_cast<std::size_t>(op.ao_map[i]);
            std::size_t jj = static_cast<std::size_t>(op.ao_map[j]);
            const int sign =
                static_cast<int>(op.ao_sign[i]) *
                static_cast<int>(op.ao_sign[j]);
            canonicalize_pair(ii, jj);
            if (!append_pair_orbit(orbit, ii, jj, sign))
                return {orbit, true};
        }

        std::sort(
            orbit.begin(), orbit.end(),
            [](const PairOrbitElem &a, const PairOrbitElem &b)
            {
                return std::tie(a.i, a.j) < std::tie(b.i, b.j);
            });
        return {orbit, false};
    }

    static std::pair<std::vector<QuartetOrbitElem>, bool> build_quartet_orbit(
        std::size_t i, std::size_t j,
        std::size_t k, std::size_t l,
        const SymOps &sym_ops)
    {
        // A sign conflict inside the orbit means the quartet vanishes by
        // symmetry, so the caller can skip the expensive recurrence entirely.
        std::vector<QuartetOrbitElem> orbit;
        orbit.reserve(sym_ops.size());

        for (const auto &op : sym_ops)
        {
            std::size_t ii = static_cast<std::size_t>(op.ao_map[i]);
            std::size_t jj = static_cast<std::size_t>(op.ao_map[j]);
            std::size_t kk = static_cast<std::size_t>(op.ao_map[k]);
            std::size_t ll = static_cast<std::size_t>(op.ao_map[l]);
            const int sign =
                static_cast<int>(op.ao_sign[i]) *
                static_cast<int>(op.ao_sign[j]) *
                static_cast<int>(op.ao_sign[k]) *
                static_cast<int>(op.ao_sign[l]);
            canonicalize_quartet(ii, jj, kk, ll);
            if (!append_quartet_orbit(orbit, ii, jj, kk, ll, sign))
                return {orbit, true};
        }

        std::sort(
            orbit.begin(), orbit.end(),
            [](const QuartetOrbitElem &a, const QuartetOrbitElem &b)
            {
                return std::tie(a.i, a.j, a.k, a.l) <
                       std::tie(b.i, b.j, b.k, b.l);
            });
        return {orbit, false};
    }

    static void write_eri_permutations(
        std::vector<double> &eri,
        std::size_t nb, std::size_t nb2, std::size_t nb3,
        std::size_t i, std::size_t j,
        std::size_t k, std::size_t l,
        double val)
    {
        auto write_slot = [&](std::size_t idx)
        {
#ifdef USE_OPENMP
#pragma omp atomic write
#endif
            eri[idx] = val;
        };

        write_slot(i * nb3 + j * nb2 + k * nb + l);
        write_slot(j * nb3 + i * nb2 + k * nb + l);
        write_slot(i * nb3 + j * nb2 + l * nb + k);
        write_slot(j * nb3 + i * nb2 + l * nb + k);
        write_slot(k * nb3 + l * nb2 + i * nb + j);
        write_slot(l * nb3 + k * nb2 + i * nb + j);
        write_slot(k * nb3 + l * nb2 + j * nb + i);
        write_slot(l * nb3 + k * nb2 + j * nb + i);
    }

} // namespace

// Auto-dispatch engine selection: three-way OS / HGP / Rys, keyed on the
// quartet's angular-momentum bucket (L_AB, L_CD).
//
// The region table below is the calibration result emitted by
// scripts/fit_auto_dispatch.py (the `region_table` / generated C++ in
// docs/auto_dispatch_fit.json), derived as "pick the engine with the lowest
// cross-case median per-quartet time in that bucket" over the benchmark in
// tests/auto_dispatch_benchmark.cpp. It is a dense lookup, not a hand-fitted
// inequality, because the OS/HGP/Rys boundaries are irregular and move when the
// engines are optimized — keeping the table verbatim from the fitter avoids
// drift. After re-benchmarking and re-running the fitter, regenerate this table
// from docs/auto_dispatch_fit.json's `rule_in_code`.
//
// Current shape (median of 9 runs, post HGP-A4 and OS-A4 HRR hoists):
//   - HGP wins the low/mid-L bulk (incl. the L_AB+L_CD<=1 corner).
//   - OS wins a high-L corner where HGP's per-quartet HRR overhead overtakes
//     its primitive-loop savings.
//   - Rys wins only the extreme tail (7,8) and (8,8).
static constexpr int kAutoMaxL = 8;
static constexpr HartreeFock::IntegralMethod
    kAutoEngine[kAutoMaxL + 1][kAutoMaxL + 1] = {
        // L_CD =  0    1    2    3    4    5    6    7    8
        /*L_AB=0*/ {HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople},
        /*L_AB=1*/ {HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople},
        /*L_AB=2*/ {HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople},
        /*L_AB=3*/ {HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::ObaraSaika},
        /*L_AB=4*/ {HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::ObaraSaika, HartreeFock::IntegralMethod::ObaraSaika},
        /*L_AB=5*/ {HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::ObaraSaika, HartreeFock::IntegralMethod::ObaraSaika},
        /*L_AB=6*/ {HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::ObaraSaika, HartreeFock::IntegralMethod::ObaraSaika, HartreeFock::IntegralMethod::ObaraSaika},
        /*L_AB=7*/ {HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::ObaraSaika, HartreeFock::IntegralMethod::ObaraSaika, HartreeFock::IntegralMethod::RysQuadrature},
        /*L_AB=8*/ {HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::HeadGordonPople, HartreeFock::IntegralMethod::ObaraSaika, HartreeFock::IntegralMethod::ObaraSaika, HartreeFock::IntegralMethod::ObaraSaika, HartreeFock::IntegralMethod::RysQuadrature},
};

// Look up the calibrated engine for a quartet bucket. L beyond the benchmarked
// reach (kAutoMaxL) is clamped to the table edge; the table edges already
// capture the asymptotic high-L preference.
static inline HartreeFock::IntegralMethod _auto_engine(int L_AB, int L_CD) noexcept
{
    if (L_AB < 0)
        L_AB = 0;
    if (L_CD < 0)
        L_CD = 0;
    if (L_AB > kAutoMaxL)
        L_AB = kAutoMaxL;
    if (L_CD > kAutoMaxL)
        L_CD = kAutoMaxL;
    return kAutoEngine[L_AB][L_CD];
}

static double _auto_contracted_eri(
    const HartreeFock::ShellPair &spAB,
    const HartreeFock::ShellPair &spCD,
    int lAx, int lAy, int lAz,
    int lBx, int lBy, int lBz,
    int lCx, int lCy, int lCz,
    int lDx, int lDy, int lDz,
    HartreeFock::ERIKernel kernel,
    double omega) noexcept
{
    const int L_AB = spAB.A._cartesian[0] + spAB.A._cartesian[1] + spAB.A._cartesian[2] +
                     spAB.B._cartesian[0] + spAB.B._cartesian[1] + spAB.B._cartesian[2];
    const int L_CD = spCD.A._cartesian[0] + spCD.A._cartesian[1] + spCD.A._cartesian[2] +
                     spCD.B._cartesian[0] + spCD.B._cartesian[1] + spCD.B._cartesian[2];

    switch (_auto_engine(L_AB, L_CD))
    {
    case HartreeFock::IntegralMethod::RysQuadrature:
        return HartreeFock::RysQuad::_rys_contracted_eri(
            spAB, spCD, lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz, kernel, omega);
    case HartreeFock::IntegralMethod::ObaraSaika:
        return HartreeFock::ObaraSaika::_contracted_eri_elem(
            spAB, spCD, lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz, kernel, omega);
    default:
        return HartreeFock::HeadGordonPople::_contracted_eri_elem(
            spAB, spCD, lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz, kernel, omega);
    }
}

// ─── 1D Rys VRR ───────────────────────────────────────────────────────────────
//
// Fills vrr[a][c] for a in [0, lAB], c in [0, lCD] using the recurrences:
//
//   [0,0]     = 1.0                                       (seed)
//   [a+1, 0]  = C00 * [a,0]  + a*B10 * [a-1,0]
//   [a, c+1]  = D00 * [a,c]  + c*B01 * [a,c-1]  + a*B00 * [a-1,c]
//
// where for Rys root u = t_r^2:
//   C00 = PA_q + u*WP_q,  D00 = QC_q + u*WQ_q
//   B00 = u/(2*delta),  B10 = 1/(2*zeta) - u/(2*delta),  B01 = 1/(2*eta) - u/(2*delta)

static void _rys_vrr_1d(
    double vrr[VRR_DIM][VRR_DIM],
    const int lAB, const int lCD,
    const double C00, const double D00,
    const double B00, const double B10, const double B01) noexcept
{
    vrr[0][0] = 1.0;

    // Build A-axis (c=0)
    for (int a = 1; a <= lAB; ++a)
    {
        vrr[a][0] = C00 * vrr[a - 1][0];
        if (a >= 2)
            vrr[a][0] += (a - 1) * B10 * vrr[a - 2][0];
    }

    // Build C-axis and mixed (increment c for each a)
    for (int c = 1; c <= lCD; ++c)
    {
        vrr[0][c] = D00 * vrr[0][c - 1];
        if (c >= 2)
            vrr[0][c] += (c - 1) * B01 * vrr[0][c - 2];

        for (int a = 1; a <= lAB; ++a)
        {
            vrr[a][c] = D00 * vrr[a][c - 1] + a * B00 * vrr[a - 1][c - 1];
            if (c >= 2)
                vrr[a][c] += (c - 1) * B01 * vrr[a][c - 2];
        }
    }
}

// ─── HRR (reused from OS logic) ───────────────────────────────────────────────
//
// AB-HRR: transfer angular momentum from A to B using displacement AB.
// Operates in-place on the 6D W[ax][ay][az][cx][cy][cz] buffer.
// After the sweep, W[lAx][lAy][lAz][cx][cy][cz] holds (lA,lB | cx,cy,cz, 0).
//
// Transfer rule: [a, b+1 | c, d] = [a+1, b | c, d] + AB_q * [a, b | c, d]
//
// We apply the sweep independently for x, then y, then z, mirroring os.cpp's
// _eri_hrr_ab function.

static void _rys_hrr_ab(
    RysScratch &W,
    const int lAx, const int lAy, const int lAz,
    const int lBx, const int lBy, const int lBz,
    const int lCDx, const int lCDy, const int lCDz,
    const double ABx, const double ABy, const double ABz) noexcept
{
    for (int kz = 0; kz < lBz; ++kz)
        for (int ax = 0; ax <= lAx + lBx; ++ax)
            for (int ay = 0; ay <= lAy + lBy; ++ay)
                for (int az = 0; az <= lAz + lBz - kz - 1; ++az)
                    for (int cx = 0; cx <= lCDx; ++cx)
                        for (int cy = 0; cy <= lCDy; ++cy)
                            for (int cz = 0; cz <= lCDz; ++cz)
                                W.at(ax, ay, az, cx, cy, cz) =
                                    W.at(ax, ay, az + 1, cx, cy, cz) + ABz * W.at(ax, ay, az, cx, cy, cz);

    for (int ky = 0; ky < lBy; ++ky)
        for (int ax = 0; ax <= lAx + lBx; ++ax)
            for (int ay = 0; ay <= lAy + lBy - ky - 1; ++ay)
                for (int az = 0; az <= lAz; ++az)
                    for (int cx = 0; cx <= lCDx; ++cx)
                        for (int cy = 0; cy <= lCDy; ++cy)
                            for (int cz = 0; cz <= lCDz; ++cz)
                                W.at(ax, ay, az, cx, cy, cz) =
                                    W.at(ax, ay + 1, az, cx, cy, cz) + ABy * W.at(ax, ay, az, cx, cy, cz);

    for (int kx = 0; kx < lBx; ++kx)
        for (int ax = 0; ax <= lAx + lBx - kx - 1; ++ax)
            for (int ay = 0; ay <= lAy; ++ay)
                for (int az = 0; az <= lAz; ++az)
                    for (int cx = 0; cx <= lCDx; ++cx)
                        for (int cy = 0; cy <= lCDy; ++cy)
                            for (int cz = 0; cz <= lCDz; ++cz)
                                W.at(ax, ay, az, cx, cy, cz) =
                                    W.at(ax + 1, ay, az, cx, cy, cz) + ABx * W.at(ax, ay, az, cx, cy, cz);
}

// CD-HRR: transfer C→D using a 3D slice V0[cx][cy][cz].
// Mirrors _nuclear_hrr in os.cpp exactly.

static double _rys_hrr_cd(
    double V0[VRR_DIM][VRR_DIM][VRR_DIM],
    const int lCx, const int lCy, const int lCz,
    const int lDx, const int lDy, const int lDz,
    const double CDx, const double CDy, const double CDz) noexcept
{
    // Working copy
    double W[VRR_DIM][VRR_DIM][VRR_DIM];
    for (int cx = 0; cx <= lCx + lDx; ++cx)
        for (int cy = 0; cy <= lCy + lDy; ++cy)
            for (int cz = 0; cz <= lCz + lDz; ++cz)
                W[cx][cy][cz] = V0[cx][cy][cz];

    for (int kx = 0; kx < lDx; ++kx)
        for (int cx = 0; cx <= lCx + lDx - kx - 1; ++cx)
            for (int cy = 0; cy <= lCy + lDy; ++cy)
                for (int cz = 0; cz <= lCz + lDz; ++cz)
                    W[cx][cy][cz] = W[cx + 1][cy][cz] + CDx * W[cx][cy][cz];

    for (int ky = 0; ky < lDy; ++ky)
        for (int cy = 0; cy <= lCy + lDy - ky - 1; ++cy)
            for (int cz = 0; cz <= lCz + lDz; ++cz)
                W[lCx][cy][cz] = W[lCx][cy + 1][cz] + CDy * W[lCx][cy][cz];

    for (int kz = 0; kz < lDz; ++kz)
        for (int cz = 0; cz <= lCz + lDz - kz - 1; ++cz)
            W[lCx][lCy][cz] = W[lCx][lCy][cz + 1] + CDz * W[lCx][lCy][cz];

    return W[lCx][lCy][lCz];
}

// ─── Primitive ERI: B-0 seam (Phase B) ──────────────────────────────────────
//
// `_rys_eri_primitive` is split into three statics so the Rys shell-quartet
// hoist (Phase B) can later build the per-primitive-pair geometry once and the
// 6D `sum` once per shell quartet, then read each component's HRR out of it.
// In B-0 the three are simply called in sequence by `_rys_eri_primitive`, so
// the result is bitwise-identical to the previous single-function form (same
// arithmetic, same order).
//
//   _rys_eri_prep        — per-primitive-pair geometry (kernel-screened),
//                          independent of the Cartesian component split
//   _rys_eri_build_sum   — roots + per-root 1D VRR + 6D outer-product
//                          accumulation + prefactor scale, at a given box
//   _rys_eri_hrr_to_eri  — AB-HRR then CD-HRR readout to the scalar ERI

namespace
{
    struct RysPrimGeom
    {
        double PAx, PAy, PAz;
        double QCx, QCy, QCz;
        double WPx, WPy, WPz;
        double WQx, WQy, WQz;
        double T;
        double prefac;
        double inv_delta;
        double inv_zetaAB; // ppAB.inv_zeta
        double inv_zetaCD; // ppCD.inv_zeta
        double rho_over_zeta;
        double rho_over_eta;
        double b00_scale; // 1 for Coulomb, screen.boys_scale otherwise
    };

    // Per-primitive-pair derived quantities. Independent of how the bra/ket
    // angular momentum splits between A/B and C/D, so it is computed once per
    // primitive pair (component-independent).
    static RysPrimGeom _rys_eri_prep(
        const HartreeFock::PrimitivePair &ppAB,
        const HartreeFock::PrimitivePair &ppCD,
        HartreeFock::ERIKernel kernel,
        double omega) noexcept
    {
        const double zeta = ppAB.zeta;
        const double eta = ppCD.zeta;
        const double delta = zeta + eta;
        const double inv_delta = 1.0 / delta;
        const double rho = zeta * eta * inv_delta;
        const ScreenedKernelData screen = screened_kernel_data(rho, kernel, omega);
        const double effective_rho = screen.rho;

        const double Px = ppAB.center[0], Py = ppAB.center[1], Pz = ppAB.center[2];
        const double Qx = ppCD.center[0], Qy = ppCD.center[1], Qz = ppCD.center[2];

        const double PQx = Px - Qx, PQy = Py - Qy, PQz = Pz - Qz;

        const double Wx = (zeta * Px + eta * Qx) * inv_delta;
        const double Wy = (zeta * Py + eta * Qy) * inv_delta;
        const double Wz = (zeta * Pz + eta * Qz) * inv_delta;

        const double wpwq_scale =
            (kernel == HartreeFock::ERIKernel::Coulomb) ? 1.0 : screen.boys_scale;
        const double b00_scale =
            (kernel == HartreeFock::ERIKernel::Coulomb) ? 1.0 : screen.boys_scale;

        return RysPrimGeom{
            .PAx = ppAB.pA[0], .PAy = ppAB.pA[1], .PAz = ppAB.pA[2],
            .QCx = ppCD.pA[0], .QCy = ppCD.pA[1], .QCz = ppCD.pA[2],
            .WPx = (Wx - Px) * wpwq_scale,
            .WPy = (Wy - Py) * wpwq_scale,
            .WPz = (Wz - Pz) * wpwq_scale,
            .WQx = (Wx - Qx) * wpwq_scale,
            .WQy = (Wy - Qy) * wpwq_scale,
            .WQz = (Wz - Qz) * wpwq_scale,
            .T = screen.boys_scale * rho * (PQx * PQx + PQy * PQy + PQz * PQz),
            .prefac = ppAB.prefactor * ppCD.prefactor * 2.0 *
                      std::sqrt(rho / std::numbers::pi) * screen.prefactor_scale,
            .inv_delta = inv_delta,
            .inv_zetaAB = ppAB.inv_zeta,
            .inv_zetaCD = ppCD.inv_zeta,
            .rho_over_zeta = effective_rho * ppAB.inv_zeta,
            .rho_over_eta = effective_rho * ppCD.inv_zeta,
            .b00_scale = b00_scale,
        };
    }

    // Roots + per-root 1D VRR + 6D outer-product accumulation + prefactor scale,
    // at the box (lABx..lCDz) using `n_roots` Rys roots. `sum` is sized and
    // zero-filled here.
    //
    // The root count is an explicit parameter, NOT derived from the box: the box
    // is per-axis (lABx..lCDz) but the Rys quadrature degree depends on the
    // quartet's *total* L = (L_A+L_B)+(L_C+L_D), capped per axis. Deriving n from
    // the summed per-axis box would over-count badly (a g max-box sums to L=48 ->
    // n=25, past RYS_MAX_ROOTS=11), so the caller passes the correct n. Per-
    // component, n = L_comp/2+1; the hoist passes the quartet n_max (Gauss
    // over-integration makes n_max >= n_comp exact for every component).
    static void _rys_eri_build_sum(
        const RysPrimGeom &g,
        const int lABx, const int lABy, const int lABz,
        const int lCDx, const int lCDy, const int lCDz,
        const int n_roots,
        RysScratch &sum) noexcept
    {
        const int n = n_roots;
        double t2[HartreeFock::Rys::RYS_MAX_ROOTS];
        double w[HartreeFock::Rys::RYS_MAX_ROOTS];
        HartreeFock::Rys::rys_roots_weights(n, g.T, t2, w);

        sum.resize_for_quartet(lABx, lABy, lABz, lCDx, lCDy, lCDz);
        std::fill(sum.buf.begin(), sum.buf.end(), 0.0);

        for (int r = 0; r < n; ++r)
        {
            const double u = t2[r];
            const double wr = w[r];

            const double B00 = 0.5 * g.inv_delta * u * g.b00_scale;
            const double B10 = 0.5 * g.inv_zetaAB * (1.0 - g.rho_over_zeta * u);
            const double B01 = 0.5 * g.inv_zetaCD * (1.0 - g.rho_over_eta * u);

            double Ix[VRR_DIM][VRR_DIM];
            double Iy[VRR_DIM][VRR_DIM];
            double Iz[VRR_DIM][VRR_DIM];

            _rys_vrr_1d(Ix, lABx, lCDx, g.PAx + u * g.WPx, g.QCx + u * g.WQx, B00, B10, B01);
            _rys_vrr_1d(Iy, lABy, lCDy, g.PAy + u * g.WPy, g.QCy + u * g.WQy, B00, B10, B01);
            _rys_vrr_1d(Iz, lABz, lCDz, g.PAz + u * g.WPz, g.QCz + u * g.WQz, B00, B10, B01);

            for (int ax = 0; ax <= lABx; ++ax)
                for (int ay = 0; ay <= lABy; ++ay)
                    for (int az = 0; az <= lABz; ++az)
                        for (int cx = 0; cx <= lCDx; ++cx)
                            for (int cy = 0; cy <= lCDy; ++cy)
                                for (int cz = 0; cz <= lCDz; ++cz)
                                    sum.at(ax, ay, az, cx, cy, cz) +=
                                        wr * Ix[ax][cx] * Iy[ay][cy] * Iz[az][cz];
        }

        for (int ax = 0; ax <= lABx; ++ax)
            for (int ay = 0; ay <= lABy; ++ay)
                for (int az = 0; az <= lABz; ++az)
                    for (int cx = 0; cx <= lCDx; ++cx)
                        for (int cy = 0; cy <= lCDy; ++cy)
                            for (int cz = 0; cz <= lCDz; ++cz)
                                sum.at(ax, ay, az, cx, cy, cz) *= g.prefac;
    }

    // AB-HRR (in place on `sum`) then CD-HRR readout to the scalar ERI for the
    // component (lA*,lB*,lC*,lD*). `sum` must already hold the contracted
    // (built) block at a box that covers this component (lAB* = lA*+lB*, etc.).
    static double _rys_eri_hrr_to_eri(
        RysScratch &sum,
        const int lAx, const int lAy, const int lAz,
        const int lBx, const int lBy, const int lBz,
        const int lCx, const int lCy, const int lCz,
        const int lDx, const int lDy, const int lDz,
        const double ABx, const double ABy, const double ABz,
        const double CDx, const double CDy, const double CDz) noexcept
    {
        const int lCDx = lCx + lDx, lCDy = lCy + lDy, lCDz = lCz + lDz;

        _rys_hrr_ab(sum,
                    lAx, lAy, lAz, lBx, lBy, lBz,
                    lCDx, lCDy, lCDz,
                    ABx, ABy, ABz);

        double V0_CD[VRR_DIM][VRR_DIM][VRR_DIM];
        for (int cx = 0; cx <= lCDx; ++cx)
            for (int cy = 0; cy <= lCDy; ++cy)
                for (int cz = 0; cz <= lCDz; ++cz)
                    V0_CD[cx][cy][cz] = sum.at(lAx, lAy, lAz, cx, cy, cz);

        return _rys_hrr_cd(V0_CD, lCx, lCy, lCz, lDx, lDy, lDz, CDx, CDy, CDz);
    }
} // namespace

double HartreeFock::RysQuad::_rys_eri_primitive(
    const HartreeFock::PrimitivePair &ppAB,
    const HartreeFock::PrimitivePair &ppCD,
    const int lAx, const int lAy, const int lAz,
    const int lBx, const int lBy, const int lBz,
    const int lCx, const int lCy, const int lCz,
    const int lDx, const int lDy, const int lDz,
    const double ABx, const double ABy, const double ABz,
    const double CDx, const double CDy, const double CDz,
    HartreeFock::ERIKernel kernel,
    double omega) noexcept
{
    const int lABx = lAx + lBx, lABy = lAy + lBy, lABz = lAz + lBz;
    const int lCDx = lCx + lDx, lCDy = lCy + lDy, lCDz = lCz + lDz;
    const int n = (lABx + lABy + lABz + lCDx + lCDy + lCDz) / 2 + 1;

    const RysPrimGeom geom = _rys_eri_prep(ppAB, ppCD, kernel, omega);

    RysScratch &sum = g_rys_scratch;
    _rys_eri_build_sum(geom, lABx, lABy, lABz, lCDx, lCDy, lCDz, n, sum);

    return _rys_eri_hrr_to_eri(sum,
                               lAx, lAy, lAz, lBx, lBy, lBz,
                               lCx, lCy, lCz, lDx, lDy, lDz,
                               ABx, ABy, ABz, CDx, CDy, CDz);
}

// ─── Contracted ERI ───────────────────────────────────────────────────────────

double HartreeFock::RysQuad::_rys_contracted_eri(
    const HartreeFock::ShellPair &spAB,
    const HartreeFock::ShellPair &spCD,
    const int lAx, const int lAy, const int lAz,
    const int lBx, const int lBy, const int lBz,
    const int lCx, const int lCy, const int lCz,
    const int lDx, const int lDy, const int lDz,
    HartreeFock::ERIKernel kernel,
    double omega) noexcept
{
    const double ABx = spAB.R[0], ABy = spAB.R[1], ABz = spAB.R[2];
    const double CDx = spCD.R[0], CDy = spCD.R[1], CDz = spCD.R[2];

    double eri = 0.0;
    for (const auto &ppAB : spAB.primitive_pairs)
        for (const auto &ppCD : spCD.primitive_pairs)
        {
            const double full =
                _rys_eri_primitive(ppAB, ppCD, lAx, lAy, lAz, lBx, lBy, lBz,
                                   lCx, lCy, lCz, lDx, lDy, lDz,
                                   ABx, ABy, ABz, CDx, CDy, CDz,
                                   HartreeFock::ERIKernel::Coulomb, 0.0);

            double value = full;
            if (kernel != HartreeFock::ERIKernel::Coulomb)
            {
                const double long_range =
                    _rys_eri_primitive(ppAB, ppCD, lAx, lAy, lAz, lBx, lBy, lBz,
                                       lCx, lCy, lCz, lDx, lDy, lDz,
                                       ABx, ABy, ABz, CDx, CDy, CDz,
                                       HartreeFock::ERIKernel::LongRange, omega);
                value = (kernel == HartreeFock::ERIKernel::LongRange) ? long_range : (full - long_range);
            }

            eri += ppAB.coeff_product * ppCD.coeff_product * value;
        }
    return eri;
}

// ─── Test hook (Phase B / B-1): box-size invariance of the 6D sum ────────────
//
// Fills `out` with the full per-primitive-pair 6D Rys `sum` buffer (row-major
// in RysScratch's stride convention: index = ((((ax*ay_dim+ay)*az_dim+az)*
// cx_dim+cx)*cy_dim+cy)*cz_dim+cz, with each *_dim = box+1), at a caller-given
// box and root count, BEFORE HRR. Filling the whole buffer once lets the caller
// read every coordinate from one build instead of rebuilding per cell.
//
// The root count is explicit (not box-derived) because it is the quartet's
// quadrature degree n = (L_AB_total + L_CD_total)/2 + 1 — the summed per-axis
// box would over-count (g max-box sums to L=48 -> n=25, past RYS_MAX_ROOTS). B-1
// builds twice per quartet/component: the max box with n_max and the component
// box with n_comp; box-invariance + Gauss over-integration require the two to
// agree at every component coordinate. ShortRange = Coulomb − LongRange,
// matching production. No production path calls this; it exists for
// tests/rys_box_invariance.cpp.
void HartreeFock::RysQuad::_build_sum_native_test(
    const HartreeFock::PrimitivePair &ppAB,
    const HartreeFock::PrimitivePair &ppCD,
    int lABx, int lABy, int lABz,
    int lCDx, int lCDy, int lCDz,
    int n_roots,
    HartreeFock::ERIKernel kernel,
    double omega,
    std::vector<double> &out) noexcept
{
    RysScratch &sum = g_rys_scratch;

    if (kernel == HartreeFock::ERIKernel::ShortRange)
    {
        // Build Coulomb, snapshot, then LongRange, and combine cell-wise
        // (matching production's full − long_range).
        const RysPrimGeom gc = _rys_eri_prep(ppAB, ppCD, HartreeFock::ERIKernel::Coulomb, 0.0);
        _rys_eri_build_sum(gc, lABx, lABy, lABz, lCDx, lCDy, lCDz, n_roots, sum);
        std::vector<double> coulomb(sum.buf.begin(), sum.buf.begin() + sum.size);

        const RysPrimGeom gl = _rys_eri_prep(ppAB, ppCD, HartreeFock::ERIKernel::LongRange, omega);
        _rys_eri_build_sum(gl, lABx, lABy, lABz, lCDx, lCDy, lCDz, n_roots, sum);

        out.assign(sum.size, 0.0);
        for (std::size_t i = 0; i < sum.size; ++i)
            out[i] = coulomb[i] - sum.buf[i];
        return;
    }

    const RysPrimGeom geom = _rys_eri_prep(ppAB, ppCD, kernel, omega);
    _rys_eri_build_sum(geom, lABx, lABy, lABz, lCDx, lCDy, lCDz, n_roots, sum);
    out.assign(sum.buf.begin(), sum.buf.begin() + sum.size);
}

// ─── Schwarz screening (mirrors os.cpp _compute_schwarz_table) ────────────────

static Eigen::MatrixXd _rys_schwarz_table(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const std::size_t nbasis,
    const std::vector<HartreeFock::SignedAOSymOp> *sym_ops)
{
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nbasis, nbasis);
    const bool use_sym = use_symmetry_ops(sym_ops);

    for (const auto &sp : shell_pairs)
    {
        const std::size_t i = sp.A._index;
        const std::size_t j = sp.B._index;
        std::vector<PairOrbitElem> orbit;

        if (use_sym)
        {
            auto [orb, forced_zero] = build_pair_orbit(i, j, *sym_ops);
            orbit = std::move(orb);
            // Q(i,j) = sqrt((ij|ij)) is a diagonal two-electron bound, so the
            // pair phase cancels between bra and ket. A pair that looks odd for
            // one-electron matrices is still valid here and must not be
            // screened out.
            (void)forced_zero;
            if (orbit.front().i != i || orbit.front().j != j)
                continue;
        }

        const int lAx = sp.A._cartesian[0], lAy = sp.A._cartesian[1], lAz = sp.A._cartesian[2];
        const int lBx = sp.B._cartesian[0], lBy = sp.B._cartesian[1], lBz = sp.B._cartesian[2];

        const double val = HartreeFock::RysQuad::_rys_contracted_eri(
            sp, sp, lAx, lAy, lAz, lBx, lBy, lBz, lAx, lAy, lAz, lBx, lBy, lBz);

        const double q = std::sqrt(std::abs(val));
        if (!use_sym)
        {
            Q(i, j) = Q(j, i) = q;
            continue;
        }

        for (const auto &elem : orbit)
        {
            Q(elem.i, elem.j) = q;
            Q(elem.j, elem.i) = q;
        }
    }
    return Q;
}

std::vector<double> HartreeFock::RysQuad::_compute_2e(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const std::size_t nbasis,
    const HartreeFock::ERIKernel kernel,
    const double omega,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp> *sym_ops)
{
    const std::size_t nb = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;
    const bool use_sym = use_symmetry_ops(sym_ops);

    const Eigen::MatrixXd Q = _rys_schwarz_table(shell_pairs, nb, sym_ops);
    std::vector<double> eri(nb * nb * nb * nb, 0.0);

    const std::size_t npairs = shell_pairs.size();

    // Flatten the upper-triangle (p,q) iteration space for load balance; see
    // ObaraSaika::_compute_2e for the rationale and the closed-form t->(p,q)
    // inversion. The scatter is store-only, so the tensor is independent of
    // visitation order and this stays bitwise-identical to the serial-row form.
    const std::size_t ntri = npairs * (npairs + 1) / 2;
    const auto tri_base = [npairs](std::size_t r) -> std::size_t
    { return r * npairs - r * (r - 1) / 2; };

#pragma omp parallel for schedule(dynamic, 64)
    for (std::size_t t = 0; t < ntri; ++t)
    {
        long long pp = static_cast<long long>(std::floor(
            (static_cast<double>(2 * npairs + 1) -
             std::sqrt(static_cast<double>(2 * npairs + 1) *
                           static_cast<double>(2 * npairs + 1) -
                       8.0 * static_cast<double>(t))) /
            2.0));
        if (pp < 0)
            pp = 0;
        while (pp > 0 && tri_base(static_cast<std::size_t>(pp)) > t)
            --pp;
        while (static_cast<std::size_t>(pp) + 1 < npairs &&
               tri_base(static_cast<std::size_t>(pp) + 1) <= t)
            ++pp;
        const std::size_t p = static_cast<std::size_t>(pp);
        const std::size_t q = p + (t - tri_base(p));

        const auto &spAB = shell_pairs[p];
        const std::size_t i = spAB.A._index;
        const std::size_t j = spAB.B._index;
        const int lAx = spAB.A._cartesian[0], lAy = spAB.A._cartesian[1], lAz = spAB.A._cartesian[2];
        const int lBx = spAB.B._cartesian[0], lBy = spAB.B._cartesian[1], lBz = spAB.B._cartesian[2];

        const auto &spCD = shell_pairs[q];
        const std::size_t k = spCD.A._index;
        const std::size_t l = spCD.B._index;
        std::vector<QuartetOrbitElem> orbit;
        if (Q(i, j) * Q(k, l) < tol_eri)
            continue;

        if (use_sym)
        {
            auto [orb, forced_zero] =
                build_quartet_orbit(i, j, k, l, *sym_ops);
            if (forced_zero)
                continue;
            orbit = std::move(orb);
            if (orbit.front().i != i || orbit.front().j != j ||
                orbit.front().k != k || orbit.front().l != l)
                continue;
        }

        const int lCx = spCD.A._cartesian[0], lCy = spCD.A._cartesian[1], lCz = spCD.A._cartesian[2];
        const int lDx = spCD.B._cartesian[0], lDy = spCD.B._cartesian[1], lDz = spCD.B._cartesian[2];
        const double val = HartreeFock::RysQuad::_rys_contracted_eri(
            spAB, spCD,
            lAx, lAy, lAz, lBx, lBy, lBz,
            lCx, lCy, lCz, lDx, lDy, lDz,
            kernel, omega);

        if (!use_sym)
        {
            write_eri_permutations(eri, nb, nb2, nb3, i, j, k, l, val);
            continue;
        }

        for (const auto &elem : orbit)
        {
            write_eri_permutations(
                eri, nb, nb2, nb3,
                elem.i, elem.j, elem.k, elem.l,
                static_cast<double>(elem.sign) * val);
        }
    }

    return eri;
}

std::vector<double> HartreeFock::RysQuad::_compute_2e_auto(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const std::size_t nbasis,
    const HartreeFock::ERIKernel kernel,
    const double omega,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp> *sym_ops)
{
    const std::size_t nb = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;
    const bool use_sym = use_symmetry_ops(sym_ops);

    // Use the shared HGP Schwarz table instead of _rys_schwarz_table:
    // profiling showed the Rys variant was 4-5x slower and accounted for
    // the entire Auto-vs-HGP Fock-build gap. The Schwarz bound is
    // engine-independent at the value level, so this is a pure speed swap.
    const std::vector<double> Q = HartreeFock::Screening::schwarz_table_hgp(
        shell_pairs, nb, sym_ops);
    std::vector<double> eri(nb * nb * nb * nb, 0.0);

    // H-10 step A3/A4-3: iterate the default Auto path at *shell-quartet*
    // granularity instead of per Cartesian-AO. Reconstruct the shell grouping
    // from the per-AO shell_pairs list and form the upper triangle of *shell*
    // pairs. Per-component Schwarz screening, symmetry orbit, and the 8-fold
    // store-only scatter run in the inner component loops unchanged.
    //
    // The Rys/HGP choice keys on shell-level total L, so it is constant across a
    // shell quartet's components. A4-3 exploits this: HGP-chosen quartets route
    // through the hoisted block (one shared (a0|c0) contraction per quartet +
    // cheap per-component HRR readout) instead of re-running the per-primitive
    // contraction per component. Rys-chosen quartets (L_AB+L_CD<=1, <=3
    // components) stay on the per-component _auto_contracted_eri path — Rys has
    // no VRR/HRR to hoist. The hoisted block is computed lazily on the first
    // surviving component, so a fully screened HGP quartet pays nothing.
    //
    // Not bitwise vs the per-AO build for HGP-chosen d-shell quartets: the hoist
    // applies _component_norm after HRR while the per-component path folds it in
    // before contraction, so they round differently at the last FP bit (~1e-15).
    // Gated by planck-os-block-kernel's hoisted arm and the Auto-vs-OS check in
    // planck-compute-2e (1e-12). Store-only scatter keeps the tensor independent
    // of visitation order. The per-component (k,l) >=_lex (i,j) check reproduces
    // the old flat-pair q >= p ordering exactly.
    std::vector<const HartreeFock::ContractedView *> ao_views;
    const std::vector<RysShellGroup> groups =
        shell_groups_from_pairs(shell_pairs, nb, ao_views);
    const std::size_t ngroups = groups.size();

    struct GroupPair
    {
        std::size_t a;
        std::size_t b;
    };
    std::vector<GroupPair> group_pairs;
    group_pairs.reserve(ngroups * (ngroups + 1) / 2);
    for (std::size_t sa = 0; sa < ngroups; ++sa)
        for (std::size_t sb = sa; sb < ngroups; ++sb)
            group_pairs.push_back({sa, sb});

    const std::size_t ngp = group_pairs.size();

#pragma omp parallel for schedule(dynamic, 8)
    for (std::size_t bra = 0; bra < ngp; ++bra)
    {
        const RysShellGroup &gA = groups[group_pairs[bra].a];
        const RysShellGroup &gB = groups[group_pairs[bra].b];

        // Iterate every ket shell pair; the per-component bra-ket canonical
        // check ((k,l) >=_lex (i,j)) is the exact filter, so we must not prune
        // ket shell pairs by their flat index.
        for (std::size_t ket = 0; ket < ngp; ++ket)
        {
            const RysShellGroup &gC = groups[group_pairs[ket].a];
            const RysShellGroup &gD = groups[group_pairs[ket].b];

            // The engine choice is constant across the quartet's components
            // (it keys on shell-level total L), so decide once here off the
            // component-0 views — cheaply, without building ShellPairs. The
            // shell's total L is the component-0 cartesian sum (component 0 of an
            // L-shell carries all L on one axis), so this matches _auto_engine
            // exactly. Only HGP-chosen quartets use the hoisted block fast path
            // (one shared contraction); OS- and Rys-chosen quartets fall to the
            // per-component path, which re-dispatches through _auto_contracted_eri.
            const auto total_L = [](const HartreeFock::ContractedView &v)
            {
                return v._cartesian[0] + v._cartesian[1] + v._cartesian[2];
            };
            const int L_AB = total_L(*ao_views[gA.first_ao]) + total_L(*ao_views[gB.first_ao]);
            const int L_CD = total_L(*ao_views[gC.first_ao]) + total_L(*ao_views[gD.first_ao]);
            const bool quartet_uses_hgp =
                _auto_engine(L_AB, L_CD) == HartreeFock::IntegralMethod::HeadGordonPople;

            const std::size_t nCq = gC.n_components;
            const std::size_t nDq = gD.n_components;
            const std::size_t nCDq = nCq * nDq;
            std::vector<double> hgp_block; // filled lazily on first survivor
            bool hgp_block_ready = false;
            auto ensure_hgp_block = [&]()
            {
                if (hgp_block_ready)
                    return;
                hgp_block.assign(
                    gA.n_components * gB.n_components * nCDq, 0.0);
                const HartreeFock::ContractedView *const *vA =
                    ao_views.data() + gA.first_ao;
                const HartreeFock::ContractedView *const *vB =
                    ao_views.data() + gB.first_ao;
                const HartreeFock::ContractedView *const *vC =
                    ao_views.data() + gC.first_ao;
                const HartreeFock::ContractedView *const *vD =
                    ao_views.data() + gD.first_ao;
                HartreeFock::HeadGordonPople::_contracted_eri_block_hoisted_views(
                    vA, gA.n_components, vB, gB.n_components,
                    vC, nCq, vD, nDq, kernel, omega, hgp_block.data());
                hgp_block_ready = true;
            };

            for (std::size_t ca = 0; ca < gA.n_components; ++ca)
            {
                const HartreeFock::ContractedView &cvA = *ao_views[gA.first_ao + ca];
                const std::size_t i = cvA._index;
                const int lAx = cvA._cartesian[0], lAy = cvA._cartesian[1], lAz = cvA._cartesian[2];

                for (std::size_t cb = 0; cb < gB.n_components; ++cb)
                {
                    const HartreeFock::ContractedView &cvB = *ao_views[gB.first_ao + cb];
                    const std::size_t j = cvB._index;
                    if (j < i) // bra upper triangle: j >= i
                        continue;
                    const int lBx = cvB._cartesian[0], lBy = cvB._cartesian[1], lBz = cvB._cartesian[2];

                    for (std::size_t cc = 0; cc < gC.n_components; ++cc)
                    {
                        const HartreeFock::ContractedView &cvC = *ao_views[gC.first_ao + cc];
                        const std::size_t k = cvC._index;
                        const int lCx = cvC._cartesian[0], lCy = cvC._cartesian[1], lCz = cvC._cartesian[2];

                        for (std::size_t cd = 0; cd < gD.n_components; ++cd)
                        {
                            const HartreeFock::ContractedView &cvD = *ao_views[gD.first_ao + cd];
                            const std::size_t l = cvD._index;
                            if (l < k) // ket upper triangle: l >= k
                                continue;
                            // bra-ket canonical: (k,l) >=_lex (i,j)
                            if (k < i || (k == i && l < j))
                                continue;

                            const int lDx = cvD._cartesian[0], lDy = cvD._cartesian[1], lDz = cvD._cartesian[2];

                            if (Q[i * nb + j] * Q[k * nb + l] < tol_eri)
                                continue;

                            std::vector<QuartetOrbitElem> orbit;
                            if (use_sym)
                            {
                                auto [orb, forced_zero] =
                                    build_quartet_orbit(i, j, k, l, *sym_ops);
                                if (forced_zero)
                                    continue;
                                orbit = std::move(orb);
                                if (orbit.front().i != i || orbit.front().j != j ||
                                    orbit.front().k != k || orbit.front().l != l)
                                    continue;
                            }

                            double val;
                            if (quartet_uses_hgp)
                            {
                                ensure_hgp_block();
                                val = hgp_block[((ca * gB.n_components + cb) * nCq + cc) * nDq + cd];
                            }
                            else
                            {
                                const HartreeFock::ShellPair spAB(cvA, cvB);
                                const HartreeFock::ShellPair spCD(cvC, cvD);
                                val = _auto_contracted_eri(
                                    spAB, spCD,
                                    lAx, lAy, lAz, lBx, lBy, lBz,
                                    lCx, lCy, lCz, lDx, lDy, lDz,
                                    kernel, omega);
                            }

                            if (!use_sym)
                            {
                                write_eri_permutations(eri, nb, nb2, nb3, i, j, k, l, val);
                                continue;
                            }

                            for (const auto &elem : orbit)
                                write_eri_permutations(
                                    eri, nb, nb2, nb3,
                                    elem.i, elem.j, elem.k, elem.l,
                                    static_cast<double>(elem.sign) * val);
                        }
                    }
                }
            }
        }
    }

    return eri;
}

// ─── Public: RHF 2e Fock (direct SCF) ────────────────────────────────────────

Eigen::MatrixXd HartreeFock::RysQuad::_compute_2e_fock(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const Eigen::MatrixXd &density,
    const std::size_t nbasis,
    const HartreeFock::ERIKernel kernel,
    const double omega,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp> *sym_ops)
{
    const std::size_t nb = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;

    std::vector<double> eri = _compute_2e(shell_pairs, nbasis, kernel, omega, tol_eri, sym_ops);

    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nb, nb);

#pragma omp parallel for schedule(static)
    for (std::size_t mu = 0; mu < nb; ++mu)
        for (std::size_t nu = 0; nu < nb; ++nu)
            for (std::size_t lam = 0; lam < nb; ++lam)
                for (std::size_t sig = 0; sig < nb; ++sig)
                    G(mu, nu) += density(lam, sig) *
                                 (eri[mu * nb3 + nu * nb2 + lam * nb + sig] - 0.5 * eri[mu * nb3 + lam * nb2 + nu * nb + sig]);

    return G;
}

// ─── Public: UHF 2e Fock (direct SCF) ────────────────────────────────────────

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
HartreeFock::RysQuad::_compute_2e_fock_uhf(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const Eigen::MatrixXd &Pa,
    const Eigen::MatrixXd &Pb,
    const std::size_t nbasis,
    const HartreeFock::ERIKernel kernel,
    const double omega,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp> *sym_ops)
{
    const std::size_t nb = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;

    const Eigen::MatrixXd Pt = Pa + Pb;
    std::vector<double> eri = _compute_2e(shell_pairs, nbasis, kernel, omega, tol_eri, sym_ops);

    Eigen::MatrixXd Ga = Eigen::MatrixXd::Zero(nb, nb);
    Eigen::MatrixXd Gb = Eigen::MatrixXd::Zero(nb, nb);

#pragma omp parallel for schedule(static)
    for (std::size_t mu = 0; mu < nb; ++mu)
        for (std::size_t nu = 0; nu < nb; ++nu)
            for (std::size_t lam = 0; lam < nb; ++lam)
                for (std::size_t sig = 0; sig < nb; ++sig)
                {
                    const double coulomb = eri[mu * nb3 + nu * nb2 + lam * nb + sig];
                    const double exch = eri[mu * nb3 + lam * nb2 + nu * nb + sig];
                    Ga(mu, nu) += Pt(lam, sig) * coulomb - Pa(lam, sig) * exch;
                    Gb(mu, nu) += Pt(lam, sig) * coulomb - Pb(lam, sig) * exch;
                }

    return {Ga, Gb};
}

// ─── Auto-dispatch variants ───────────────────────────────────────────────────
//
// Per-quartet three-way engine selection (see `_auto_engine` above). OS, HGP
// and Rys are all reachable from the auto path. The Fock-auto entries reuse
// `_compute_2e_auto` so they pick up the same per-quartet dispatch.

Eigen::MatrixXd HartreeFock::RysQuad::_compute_2e_fock_auto(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const Eigen::MatrixXd &density,
    const std::size_t nbasis,
    const HartreeFock::ERIKernel kernel,
    const double omega,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp> *sym_ops)
{
    const std::size_t nb = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;
    std::vector<double> eri = _compute_2e_auto(shell_pairs, nbasis, kernel, omega, tol_eri, sym_ops);
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nb, nb);

#pragma omp parallel for schedule(static)
    for (std::size_t mu = 0; mu < nb; ++mu)
        for (std::size_t nu = 0; nu < nb; ++nu)
            for (std::size_t lam = 0; lam < nb; ++lam)
                for (std::size_t sig = 0; sig < nb; ++sig)
                    G(mu, nu) += density(lam, sig) *
                                 (eri[mu * nb3 + nu * nb2 + lam * nb + sig] - 0.5 * eri[mu * nb3 + lam * nb2 + nu * nb + sig]);
    return G;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
HartreeFock::RysQuad::_compute_2e_fock_uhf_auto(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const Eigen::MatrixXd &Pa,
    const Eigen::MatrixXd &Pb,
    const std::size_t nbasis,
    const HartreeFock::ERIKernel kernel,
    const double omega,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp> *sym_ops)
{
    const std::size_t nb = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;
    const Eigen::MatrixXd Pt = Pa + Pb;
    std::vector<double> eri = _compute_2e_auto(shell_pairs, nbasis, kernel, omega, tol_eri, sym_ops);
    Eigen::MatrixXd Ga = Eigen::MatrixXd::Zero(nb, nb);
    Eigen::MatrixXd Gb = Eigen::MatrixXd::Zero(nb, nb);

#pragma omp parallel for schedule(static)
    for (std::size_t mu = 0; mu < nb; ++mu)
        for (std::size_t nu = 0; nu < nb; ++nu)
            for (std::size_t lam = 0; lam < nb; ++lam)
                for (std::size_t sig = 0; sig < nb; ++sig)
                {
                    const double coulomb = eri[mu * nb3 + nu * nb2 + lam * nb + sig];
                    const double exch = eri[mu * nb3 + lam * nb2 + nu * nb + sig];
                    Ga(mu, nu) += Pt(lam, sig) * coulomb - Pa(lam, sig) * exch;
                    Gb(mu, nu) += Pt(lam, sig) * coulomb - Pb(lam, sig) * exch;
                }
    return {Ga, Gb};
}
