// Where does the direct Fock build spend its ERI time, by angular-momentum class?
//
// C0 of docs/ERI_LOW_L_CLOSED_FORM_SCOPE.md. Before writing any closed-form
// low-L kernel we need the share of real Fock-build time each (L_AB, L_CD)
// class takes, because that share -- not the per-quartet speedup -- caps what
// a specialized kernel can buy (Amdahl).
//
// This drives the PRODUCTION loop, HartreeFock::Integrals::fused_fock_build,
// exactly as the default Auto engine does (Schwarz table, canonical filter,
// per-component ShellPair construction, accumulation), and swaps in a callable
// that times the HGP per-quartet kernel and buckets it by class. HGP is what
// kAutoEngine routes every low bucket to, so the numbers are the Auto path's.
//
// Two builds per fixture:
//   unwrapped -- the plain HGP callable; its wall time is the honest Fock cost.
//   wrapped   -- the timing callable; gives the per-class split. The two
//                steady_clock reads per quartet inflate it, so class shares are
//                reported as fractions of the wrapped KERNEL time, and the
//                kernel's share of the real build is estimated by subtracting
//                the measured timer overhead.
//
// Run single-threaded (OMP_NUM_THREADS=1): the per-class tallies are per-thread
// and summed, but a shared-core timing split is not what the projection wants.
//
//   planck-eri-class-share                 # the four C0 fixtures
//   planck-eri-class-share --profile N     # N unwrapped builds of 4 H2O/6-31G*,
//                                          # for `sample` to attribute
//                                          # kernel vs loop time

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "base/basis.h"
#include "base/types.h"
#include "basis/basis.h"
#include "integrals/fused_fock.h"
#include "integrals/hgp.h"
#include "integrals/screening.h"
#include "integrals/shellpair.h"
#include "require_boys_library.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace
{
    using Clock = std::chrono::steady_clock;

    constexpr int kMaxL = 12; // per pair (L_A + L_B); cc-pVTZ f+f = 6

    double seconds_since(Clock::time_point t0)
    {
        return std::chrono::duration<double>(Clock::now() - t0).count();
    }

    struct Fixture
    {
        int nwater;
        const char *basis;
    };

    // Water chain along z, 3 A spacing -- the same geometry as
    // tests/benchmarks/scale_bench.py::water_chain, so numbers are comparable.
    HartreeFock::Calculator make_water_chain(int nwater, const std::string &basis_name)
    {
        HartreeFock::Calculator calc;
        HartreeFock::Molecule &mol = calc._molecule;
        mol.natoms = 3 * nwater;
        mol.charge = 0;
        mol.multiplicity = 1;
        mol.atomic_numbers.resize(mol.natoms);
        mol.atomic_masses.resize(mol.natoms);
        mol.coordinates.resize(mol.natoms, 3);
        for (int w = 0; w < nwater; ++w)
        {
            const double z = 3.0 * w;
            mol.atomic_numbers.segment(3 * w, 3) << 8, 1, 1;
            mol.atomic_masses.segment(3 * w, 3) << 16.0, 1.0, 1.0;
            mol.coordinates.row(3 * w) << 0.000, 0.000, z;
            mol.coordinates.row(3 * w + 1) << 0.757, 0.586, z;
            mol.coordinates.row(3 * w + 2) << -0.757, 0.586, z;
        }
        calc._basis._basis = HartreeFock::BasisType::Cartesian;
        calc.prepare_coordinates();
        mol.set_standard_from_bohr(mol._coordinates);

        const std::filesystem::path gbs = std::filesystem::path(get_basis_path()) / basis_name;
        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(gbs.string(), mol, calc._basis._basis);
        if (!basis_res)
        {
            std::fprintf(stderr, "read_gbs_basis failed for %s: %s\n", basis_name.c_str(), basis_res.error().c_str());
            std::exit(1);
        }
        calc._shells = std::move(*basis_res);
        return calc;
    }

    int pair_L(const HartreeFock::ShellPair &sp) noexcept
    {
        return sp.A._cartesian[0] + sp.A._cartesian[1] + sp.A._cartesian[2] +
               sp.B._cartesian[0] + sp.B._cartesian[1] + sp.B._cartesian[2];
    }

    double hgp_elem(const HartreeFock::ShellPair &spAB, const HartreeFock::ShellPair &spCD,
                    int lAx, int lAy, int lAz, int lBx, int lBy, int lBz,
                    int lCx, int lCy, int lCz, int lDx, int lDy, int lDz,
                    HartreeFock::ERIKernel k, double w) noexcept
    {
        return HartreeFock::HeadGordonPople::_contracted_eri_elem(
            spAB, spCD, lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz, k, w);
    }

    // Per-thread tallies indexed [L_lo][L_hi], summed after the build.
    struct ClassTally
    {
        std::array<std::array<std::size_t, kMaxL + 1>, kMaxL + 1> count{};
        std::array<std::array<double, kMaxL + 1>, kMaxL + 1> seconds{};
    };

    struct BuildSetup
    {
        HartreeFock::Calculator calc;
        std::vector<HartreeFock::ShellPair> pairs;
        Eigen::MatrixXd Q;
        Eigen::MatrixXd P;
        std::size_t nb = 0;
    };

    BuildSetup prepare(const Fixture &f)
    {
        BuildSetup s{make_water_chain(f.nwater, f.basis), {}, {}, {}, 0};
        s.pairs = build_shellpairs(s.calc._shells);
        s.nb = s.calc._shells.nbasis();
        const std::vector<double> flat = HartreeFock::Screening::schwarz_table_hgp(s.pairs, s.nb, nullptr);
        s.Q.resize(s.nb, s.nb);
        for (std::size_t i = 0; i < s.nb; ++i)
            for (std::size_t j = 0; j < s.nb; ++j)
                s.Q(i, j) = flat[i * s.nb + j];
        // ERI cost does not depend on the density; any symmetric matrix will do.
        s.P = Eigen::MatrixXd::Random(s.nb, s.nb);
        s.P = (0.5 * (s.P + s.P.transpose())).eval();
        return s;
    }

    // Bitwise fingerprint of the last build's G (FNV-1a over its bytes), so a
    // loop change that must not move a single bit can be checked here.
    std::uint64_t g_fingerprint = 0;
    Eigen::MatrixXd g_last;

    // closed_form_low_l routes total-L <= 1 quartets through low_l_eri.h,
    // as the Auto engine does.
    template <typename Elem>
    double timed_build(const BuildSetup &s, Elem &&elem, bool closed_form_low_l = false)
    {
        Eigen::MatrixXd G, Ga, Gb;
        HartreeFock::Integrals::FusedFockDensities dens;
        dens.P = &s.P;
        const auto t0 = Clock::now();
        HartreeFock::Integrals::fused_fock_build(
            s.pairs, s.nb, s.Q, HartreeFock::ERIKernel::Coulomb, 0.0, /*tol_eri=*/1e-10,
            /*spin_resolved=*/false, dens, G, Ga, Gb, elem, nullptr,
            HartreeFock::Integrals::FusedTerm::Combined, closed_form_low_l);
        const double dt = seconds_since(t0);
        std::uint64_t h = 1469598103934665603ULL;
        const auto *bytes = reinterpret_cast<const unsigned char *>(G.data());
        for (std::size_t b = 0; b < sizeof(double) * static_cast<std::size_t>(G.size()); ++b)
            h = (h ^ bytes[b]) * 1099511628211ULL;
        g_fingerprint = h;
        g_last = std::move(G);
        return dt;
    }

    // Cost of the two steady_clock reads the wrapper adds per quartet.
    double timer_pair_seconds()
    {
        constexpr int n = 1 << 20;
        volatile double sink = 0.0;
        const auto t0 = Clock::now();
        for (int i = 0; i < n; ++i)
        {
            const auto a = Clock::now();
            sink = sink + std::chrono::duration<double>(Clock::now() - a).count();
        }
        return seconds_since(t0) / n;
    }

    void run_fixture(const Fixture &f, double timer_pair)
    {
        const BuildSetup s = prepare(f);

        timed_build(s, hgp_elem); // warm the HGP scratch and the caches
        const double t_unwrapped = timed_build(s, hgp_elem);
        const std::uint64_t fingerprint = g_fingerprint;
        const Eigen::MatrixXd G_hgp = g_last;
        timed_build(s, hgp_elem, true);
        const double t_low = timed_build(s, hgp_elem, true);
        const double low_dG = (g_last - G_hgp).cwiseAbs().maxCoeff();

#ifdef USE_OPENMP
        const int nthreads = omp_get_max_threads();
#else
        const int nthreads = 1;
#endif
        std::vector<ClassTally> tallies(static_cast<std::size_t>(nthreads));
        auto timing_elem = [&](const HartreeFock::ShellPair &spAB, const HartreeFock::ShellPair &spCD,
                               int lAx, int lAy, int lAz, int lBx, int lBy, int lBz,
                               int lCx, int lCy, int lCz, int lDx, int lDy, int lDz,
                               HartreeFock::ERIKernel k, double w) noexcept
        {
            const auto t0 = Clock::now();
            const double v = hgp_elem(spAB, spCD, lAx, lAy, lAz, lBx, lBy, lBz,
                                      lCx, lCy, lCz, lDx, lDy, lDz, k, w);
            const double dt = seconds_since(t0);
#ifdef USE_OPENMP
            ClassTally &t = tallies[static_cast<std::size_t>(omp_get_thread_num())];
#else
            ClassTally &t = tallies[0];
#endif
            const int la = std::min(pair_L(spAB), kMaxL), lc = std::min(pair_L(spCD), kMaxL);
            const int lo = std::min(la, lc), hi = std::max(la, lc);
            ++t.count[lo][hi];
            t.seconds[lo][hi] += dt;
            return v;
        };
        const double t_wrapped = timed_build(s, timing_elem);

        ClassTally sum;
        for (const ClassTally &t : tallies)
            for (int a = 0; a <= kMaxL; ++a)
                for (int b = 0; b <= kMaxL; ++b)
                {
                    sum.count[a][b] += t.count[a][b];
                    sum.seconds[a][b] += t.seconds[a][b];
                }

        std::size_t n_total = 0;
        double kernel_wrapped = 0.0;
        for (int a = 0; a <= kMaxL; ++a)
            for (int b = 0; b <= kMaxL; ++b)
            {
                n_total += sum.count[a][b];
                kernel_wrapped += sum.seconds[a][b];
            }

        // One timer pair lands inside each measured interval's edges; the
        // kernel time net of it, as a share of the UNWRAPPED build, is the
        // fraction of the real Fock build that is ERI evaluation.
        const double kernel_net = std::max(0.0, kernel_wrapped - n_total * 0.5 * timer_pair);
        const double kernel_share = std::min(1.0, kernel_net / t_unwrapped);

        std::printf("\n%d H2O / %s: G fingerprint %016llx\n", f.nwater, f.basis,
                    static_cast<unsigned long long>(fingerprint));
        std::printf("  closed-form L_tot<=1: %.3f s vs %.3f s (%.3fx), max|dG| %.1e\n",
                    t_low, t_unwrapped, t_unwrapped / t_low, low_dG);
        std::printf("\n%d H2O / %s: nb=%zu, %zu evaluated component quartets\n",
                    f.nwater, f.basis, s.nb, n_total);
        std::printf("  Fock build %.3f s (unwrapped), %.3f s (wrapped); HGP kernel ~%.0f%% of the build\n",
                    t_unwrapped, t_wrapped, 100.0 * kernel_share);
        std::printf("  %-10s %10s %8s %8s %8s\n", "(Llo,Lhi)", "quartets", "%count", "%kernel", "%build");

        std::array<double, 2 * kMaxL + 1> share_by_Ltot{};
        for (int a = 0; a <= kMaxL; ++a)
            for (int b = a; b <= kMaxL; ++b)
            {
                if (sum.count[a][b] == 0)
                    continue;
                const double frac = sum.seconds[a][b] / kernel_wrapped;
                share_by_Ltot[static_cast<std::size_t>(a + b)] += frac;
                std::printf("  (%d,%d)      %10zu %8.1f %8.1f %8.1f\n", a, b, sum.count[a][b],
                            100.0 * sum.count[a][b] / n_total, 100.0 * frac,
                            100.0 * frac * kernel_share);
            }

        // The C2-C4 projections: share of the BUILD each coverage level owns,
        // the build speedup if that share runs 5x faster, and the ceiling.
        std::printf("  coverage    %%build   build@5x   ceiling\n");
        double cum = 0.0;
        for (int Lmax = 0; Lmax <= 2 * kMaxL; ++Lmax)
        {
            cum += share_by_Ltot[static_cast<std::size_t>(Lmax)] * kernel_share;
            if (Lmax != 1 && Lmax != 2 && Lmax != 4)
                continue;
            std::printf("  L_tot<=%d   %6.1f     %5.2fx    %5.2fx\n", Lmax, 100.0 * cum,
                        1.0 / (1.0 - cum + cum / 5.0), 1.0 / (1.0 - cum));
        }
    }
} // namespace

int main(int argc, char **argv)
{
    if (!require_boys_library())
        return 1;

    if (argc == 3 && std::string(argv[1]) == "--profile")
    {
        const int reps = std::atoi(argv[2]);
        const BuildSetup s = prepare({4, "6-31g*"});
        double total = 0.0;
        for (int r = 0; r < reps; ++r)
            total += timed_build(s, hgp_elem);
        std::printf("--profile: %d unwrapped builds of 4 H2O / 6-31g*, %.3f s each\n", reps, total / reps);
        return 0;
    }

    const double timer_pair = timer_pair_seconds();
    std::printf("steady_clock pair: %.1f ns per quartet\n", 1e9 * timer_pair);

    for (const Fixture &f : {Fixture{1, "6-31g*"}, Fixture{4, "6-31g*"},
                             Fixture{1, "cc-pVDZ"}, Fixture{8, "6-31g"}})
        run_fixture(f, timer_pair);
    return 0;
}
