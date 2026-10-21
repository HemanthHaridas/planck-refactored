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
#include <map>
#include <utility>
#include <span>
#include <numbers>
#include <cmath>
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
#include "symmetry/integral_symmetry.h"
#include "base/types.h"
#include "basis/basis.h"
#include "integrals/fused_fock.h"
#include "integrals/hgp.h"
#include "integrals/hgp_block_eri.h"
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
        int nwater; // 0 = naphthalene with its D2h integral symmetry (sym_ops)
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

    // Naphthalene, planar in xy with the long axis on x: exactly D2h by
    // construction, so update_integral_symmetry finds all 8 sign-flip
    // operations. The symmetric fixture for the fused loop's sym_ops path.
    HartreeFock::Calculator make_naphthalene(const std::string &basis_name)
    {
        HartreeFock::Calculator calc;
        HartreeFock::Molecule &mol = calc._molecule;
        const double C[10][2] = {{0.0, 0.716}, {0.0, -0.716}, {1.240, 1.403}, {-1.240, 1.403},
                                 {1.240, -1.403}, {-1.240, -1.403}, {2.432, 0.703}, {-2.432, 0.703},
                                 {2.432, -0.703}, {-2.432, -0.703}};
        const double H[8][2] = {{1.240, 2.490}, {-1.240, 2.490}, {1.240, -2.490}, {-1.240, -2.490},
                                {3.380, 1.243}, {-3.380, 1.243}, {3.380, -1.243}, {-3.380, -1.243}};
        mol.natoms = 18;
        mol.charge = 0;
        mol.multiplicity = 1;
        mol.atomic_numbers.resize(18);
        mol.atomic_masses.resize(18);
        mol.coordinates.resize(18, 3);
        for (int a = 0; a < 10; ++a)
        {
            mol.atomic_numbers[a] = 6;
            mol.atomic_masses[a] = 12.0;
            mol.coordinates.row(a) << C[a][0], C[a][1], 0.0;
        }
        for (int a = 0; a < 8; ++a)
        {
            mol.atomic_numbers[10 + a] = 1;
            mol.atomic_masses[10 + a] = 1.0;
            mol.coordinates.row(10 + a) << H[a][0], H[a][1], 0.0;
        }
        calc._basis._basis = HartreeFock::BasisType::Cartesian;
        calc.prepare_coordinates();
        mol.set_standard_from_bohr(mol._coordinates);
        mol._symmetry = true;
        mol._point_group = "D2h";

        const std::filesystem::path gbs = std::filesystem::path(get_basis_path()) / basis_name;
        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(gbs.string(), mol, calc._basis._basis);
        if (!basis_res)
        {
            std::fprintf(stderr, "read_gbs_basis failed for %s: %s\n", basis_name.c_str(), basis_res.error().c_str());
            std::exit(1);
        }
        calc._shells = std::move(*basis_res);
        HartreeFock::Symmetry::update_integral_symmetry(calc);
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
        bool sym = false; // pass calc._integral_symmetry_ops to the build
    };

    BuildSetup prepare(const Fixture &f)
    {
        BuildSetup s{f.nwater > 0 ? make_water_chain(f.nwater, f.basis) : make_naphthalene(f.basis),
                     {}, {}, {}, 0, f.nwater == 0};
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
        // The symmetric fixture's density must carry the molecule's symmetry,
        // or the build falls back from the skeleton path (S1) to replication.
        if (s.sym)
            HartreeFock::Integrals::symmetrize_over_ops(s.P, s.calc._integral_symmetry_ops);
        return s;
    }

    // Bitwise fingerprint of the last build's G (FNV-1a over its bytes), so a
    // loop change that must not move a single bit can be checked here.
    std::uint64_t g_fingerprint = 0;
    Eigen::MatrixXd g_last;

    // `block` is the block kernel (block_eri.h); the Auto engine uses the
    // closed forms (low_l_eri.h) at total L <= 3.
    template <typename Elem, typename Block>
    double timed_build_block(const BuildSetup &s, Elem &&elem, Block block)
    {
        Eigen::MatrixXd G, Ga, Gb;
        HartreeFock::Integrals::FusedFockDensities dens;
        dens.P = &s.P;
        const auto t0 = Clock::now();
        HartreeFock::Integrals::fused_fock_build(
            s.pairs, s.nb, s.Q, HartreeFock::ERIKernel::Coulomb, 0.0, /*tol_eri=*/1e-10,
            /*spin_resolved=*/false, dens, G, Ga, Gb, elem,
            s.sym ? &s.calc._integral_symmetry_ops : nullptr,
            HartreeFock::Integrals::FusedTerm::Combined, block);
        const double dt = seconds_since(t0);
        std::uint64_t h = 1469598103934665603ULL;
        const auto *bytes = reinterpret_cast<const unsigned char *>(G.data());
        for (std::size_t b = 0; b < sizeof(double) * static_cast<std::size_t>(G.size()); ++b)
            h = (h ^ bytes[b]) * 1099511628211ULL;
        g_fingerprint = h;
        g_last = std::move(G);
        return dt;
    }

    // The Auto engine's block kernels (rys.cpp auto_block_kernel): closed forms
    // at total L <= 3, HGP block kernel at 4-12.
    auto production_kernel()
    {
        using namespace HartreeFock::Integrals;
        return ChainBlockKernel<ClosedFormBlockKernel, HgpBlockKernel>{ClosedFormBlockKernel{3}, HgpBlockKernel{4, 12}};
    }

    // closed_form_max_l routes total-L <= max quartets through the closed forms.
    template <typename Elem>
    double timed_build(const BuildSetup &s, Elem &&elem, int closed_form_max_l = 0)
    {
        return timed_build_block(s, elem, HartreeFock::Integrals::ClosedFormBlockKernel{closed_form_max_l});
    }

    // D1: the HGP block kernel (hgp_block_eri.h) one total-L class at a time,
    // on top of the closed forms, against the closed forms alone. A class wins
    // when its build is faster; max|dG| is against the per-component HGP build.
    void report_hgp_block_by_class(const BuildSetup &s, const Eigen::MatrixXd &G_hgp, double t_cf3)
    {
        using HartreeFock::Integrals::ChainBlockKernel;
        using HartreeFock::Integrals::ClosedFormBlockKernel;
        using HartreeFock::Integrals::HgpBlockKernel;
        int max_shell_l = 0;
        for (const auto &bf : s.calc._shells._basis_functions)
            max_shell_l = std::max(max_shell_l, bf._cartesian[0] + bf._cartesian[1] + bf._cartesian[2]);
        std::printf("  HGP block kernel on top of the closed forms (L_tot<=3 build %.3f s):\n", t_cf3);
        for (int lt = 4; lt <= 4 * max_shell_l; ++lt)
        {
            const ChainBlockKernel<ClosedFormBlockKernel, HgpBlockKernel> k{ClosedFormBlockKernel{3}, HgpBlockKernel{lt, lt}};
            timed_build_block(s, hgp_elem, k);
            const double t = timed_build_block(s, hgp_elem, k);
            std::printf("    L_tot=%2d: %.3f s, saves %+.3f s, max|dG| %.1e\n",
                        lt, t, t_cf3 - t, (g_last - G_hgp).cwiseAbs().maxCoeff());
        }
        const ChainBlockKernel<ClosedFormBlockKernel, HgpBlockKernel> all{ClosedFormBlockKernel{3}, HgpBlockKernel{4}};
        timed_build_block(s, hgp_elem, all);
        const double t = timed_build_block(s, hgp_elem, all);
        std::printf("    all L_tot>=4: %.3f s (%.2fx over the closed forms alone), max|dG| %.1e\n",
                    t, t_cf3 / t, (g_last - G_hgp).cwiseAbs().maxCoeff());
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

    // D2a (docs/ERI_LOW_L_CLOSED_FORM_SCOPE.md): shared-exponent families.
    // A family is a set of shells on one atom whose exponent sets are contained
    // in the largest one's (cc general contractions within L; Pople SP shells
    // across L unless `within_l`). A family quartet runs ONE primitive pass over
    // the union of its members' primitive pairs, building the HGP VRR table at
    // the family's maximum L, and accumulates every member quartet from it with
    // that member's own weight; each member then runs its own HRR (finish()).
    // Timed against the production kernels (closed forms + HGP block kernel) on
    // the same member blocks. No Schwarz or canonical filtering on either side;
    // a family pair with p == q includes both orders of a member quartet on
    // both sides alike.
    // The closed-form raises of low_l_shell_eri for ONE primitive quartet, from
    // coefficient-free th[m] = base0 F_m, written (not accumulated) into out.
    // NR raises fixed at compile time; ACC accumulates into out instead of writing it.
    template <int NR, bool ACC>
    void cfr(const HartreeFock::PrimitivePair &ab, const HartreeFock::PrimitivePair &cd,
             double rho, double delta, const double (&PQ)[3], const double *th,
             const int (&c)[3], double *out)
    {
        const auto put = [&](int k, double v) { if constexpr (ACC) out[k] += v; else out[k] = v; };
        if constexpr (NR == 0)
        {
            put(0, th[0]);
            return;
        }
        const auto d_of = [&](int k) -> const Eigen::Vector3d &
        { return k == 0 ? ab.pA : k == 1 ? ab.pB : k == 2 ? cd.pA : cd.pB; };
        const auto w_scale = [&](int k) { return k < 2 ? -cd.zeta / delta : ab.zeta / delta; };
        const int c1 = c[0], c2 = c[1], c3 = c[2];
        (void)c2; (void)c3;
        const Eigen::Vector3d &d1 = d_of(c1);
        const double s1 = w_scale(c1);
        if constexpr (NR == 1)
        {
            for (int i = 0; i < 3; ++i)
                put(i, d1[i] * th[0] + PQ[i] * (th[1] * s1));
            return;
        }
        const Eigen::Vector3d &d2 = d_of(c2);
        const double s2 = w_scale(c2);
        if constexpr (NR == 2)
        {
            double R0[3], R1[3];
            for (int i = 0; i < 3; ++i)
            {
                const double w1 = s1 * PQ[i];
                R0[i] = d1[i] * th[0] + w1 * th[1];
                R1[i] = d1[i] * th[1] + w1 * th[2];
            }
            const double kappa = c2 < 2 ? (th[0] - (rho / ab.zeta) * th[1]) * (0.5 / ab.zeta)
                                 : c1 >= 2 ? (th[0] - (rho / cd.zeta) * th[1]) * (0.5 / cd.zeta)
                                           : th[1] * (0.5 / delta);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    put(3 * i + j, d2[j] * R0[i] + s2 * PQ[j] * R1[i] + (i == j ? kappa : 0.0));
            return;
        }
        if constexpr (NR == 3)
        {
        const Eigen::Vector3d &d3 = d_of(c3);
        const double s3 = w_scale(c3);
        const auto cpl = [&](int y, int x, const double *X, int m)
        {
            if (y < 2 && x < 2)
                return (X[m] - (rho / ab.zeta) * X[m + 1]) * (0.5 / ab.zeta);
            if (y >= 2 && x >= 2)
                return (X[m] - (rho / cd.zeta) * X[m + 1]) * (0.5 / cd.zeta);
            return X[m + 1] * (0.5 / delta);
        };
        double A[3][3], A2[3][2], B[3][3][2];
        for (int i = 0; i < 3; ++i)
        {
            for (int m = 0; m < 3; ++m)
                A[i][m] = d1[i] * th[m] + s1 * PQ[i] * th[m + 1];
            for (int m = 0; m < 2; ++m)
                A2[i][m] = d2[i] * th[m] + s2 * PQ[i] * th[m + 1];
        }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int m = 0; m < 2; ++m)
                    B[i][j][m] = d2[j] * A[i][m] + s2 * PQ[j] * A[i][m + 1] + (i == j ? cpl(c1, c2, th, m) : 0.0);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                {
                    double v = d3[k] * B[i][j][0] + s3 * PQ[k] * B[i][j][1];
                    if (k == i)
                        v += cpl(c1, c3, A2[j], 0);
                    if (k == j)
                        v += cpl(c2, c3, A[i], 0);
                    put(9 * i + 3 * j + k, v);
                }
        }
    }

    void report_families(const BuildSetup &s, bool within_l)
    {
        using namespace HartreeFock;
        using namespace HartreeFock::Integrals;
        const std::size_t nb = s.nb;
        std::vector<const ShellPair *> pair_of(nb * nb, nullptr);
        std::vector<const ContractedView *> views(nb, nullptr);
        for (const auto &sp : s.pairs)
        {
            pair_of[sp.A._index * nb + sp.B._index] = &sp;
            views[sp.A._index] = &sp.A;
            views[sp.B._index] = &sp.B;
        }
        struct Group { std::size_t first, n; int L; const Shell *sh; };
        std::vector<Group> groups;
        for (std::size_t ao = 0; ao < nb; ++ao)
        {
            const auto &v = *views[ao];
            const int L = v._cartesian[0] + v._cartesian[1] + v._cartesian[2];
            if (ao == 0 || v._shell != views[ao - 1]->_shell)
                groups.push_back({ao, 1, L, v._shell});
            else
                ++groups.back().n;
        }
        std::vector<double> norm_ratio(nb, 1.0);
        for (const Group &g : groups)
            for (std::size_t c = 0; c < g.n; ++c)
                norm_ratio[g.first + c] = views[g.first + c]->_component_norm / views[g.first]->_component_norm;

        // Families: per atom, largest exponent set first; a shell joins the
        // first family whose representative contains all its exponents.
        const auto contains = [](const Shell &big, const Shell &small)
        {
            for (int i = 0; i < small._primitives.size(); ++i)
            {
                bool hit = false;
                for (int j = 0; j < big._primitives.size(); ++j)
                    hit = hit || big._primitives[j] == small._primitives[i];
                if (!hit)
                    return false;
            }
            return true;
        };
        std::vector<std::size_t> order(groups.size());
        for (std::size_t i = 0; i < order.size(); ++i)
            order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b)
                         { const Shell &x = *groups[a].sh, &y = *groups[b].sh;
                           return x._atom_index != y._atom_index ? x._atom_index < y._atom_index
                                                                 : x._primitives.size() > y._primitives.size(); });
        std::vector<std::vector<std::size_t>> fam; // group indices; fam[k][0] is the representative
        for (std::size_t g : order)
        {
            bool placed = false;
            for (auto &F : fam)
            {
                const Group &r = groups[F[0]];
                if (r.sh->_atom_index == groups[g].sh->_atom_index && (!within_l || r.L == groups[g].L) &&
                    contains(*r.sh, *groups[g].sh))
                {
                    F.push_back(g);
                    placed = true;
                    break;
                }
            }
            if (!placed)
                fam.push_back({g});
        }

        // Family pairs: members are the canonical shell pairs (first AO of A <=
        // first AO of B) with one shell in each family. The union primitive pair
        // is keyed by (exponent on the FA shell, exponent on the FB shell).
        struct Member { const ShellPair *sp; std::size_t ga, gb; int LA, LB; std::vector<double> w; };
        struct FamPair { std::vector<const PrimitivePair *> u; std::vector<Member> m; int Lmax = 0; };
        std::vector<FamPair> fps;
        for (std::size_t p = 0; p < fam.size(); ++p)
            for (std::size_t q = p; q < fam.size(); ++q)
            {
                FamPair fp;
                std::vector<std::pair<double, double>> keys;
                const bool same_atom = groups[fam[p][0]].sh->_atom_index == groups[fam[q][0]].sh->_atom_index;
                for (std::size_t ga : fam[p])
                    for (std::size_t gb : fam[q])
                    {
                        if (p == q && gb < ga)
                            continue;
                        const bool flip = groups[gb].first < groups[ga].first;
                        if (flip && !same_atom)
                        {
                            std::fprintf(stderr, "family probe: cross-atom member pair against family order\n");
                            std::exit(1);
                        }
                        const std::size_t a = flip ? gb : ga, b = flip ? ga : gb;
                        const ShellPair *sp = pair_of[groups[a].first * nb + groups[b].first];
                        Member mem{sp, a, b, groups[a].L, groups[b].L, {}};
                        std::vector<std::pair<std::size_t, double>> wl;
                        for (const auto &pp : sp->primitive_pairs)
                        {
                            const std::pair<double, double> key = flip ? std::pair{pp.beta, pp.alpha} : std::pair{pp.alpha, pp.beta};
                            std::size_t u = 0;
                            while (u < keys.size() && keys[u] != key)
                                ++u;
                            if (u == keys.size())
                            {
                                keys.push_back(key);
                                fp.u.push_back(&pp);
                            }
                            wl.push_back({u, pp.coeff_product});
                        }
                        mem.w.assign(keys.size() + 64, 0.0); // resized below once the union is final
                        for (auto [u, w] : wl)
                            mem.w[u] = w;
                        fp.Lmax = std::max(fp.Lmax, mem.LA + mem.LB);
                        fp.m.push_back(std::move(mem));
                    }
                for (Member &mem : fp.m)
                    mem.w.resize(fp.u.size(), 0.0);
                fps.push_back(std::move(fp));
            }

        const auto request = [&](const Member &x, const Member &y)
        {
            BlockEriRequest r;
            r.ab = x.sp;
            r.cd = y.sp;
            r.norm_ratio = norm_ratio.data();
            const std::size_t g4[4] = {x.ga, x.gb, y.ga, y.gb};
            for (int k = 0; k < 4; ++k)
            {
                r.views[k] = &views[groups[g4[k]].first];
                r.n[k] = groups[g4[k]].n;
                r.L[k] = groups[g4[k]].L;
            }
            return r;
        };

        // Per family quartet, member quartets split by their own total L:
        //   low  (<= 3): (a) closed forms per member  vs (c) one closed-form family
        //                pass: T, prefactor and Boys once per union primitive
        //                quartet, raises once per distinct (L_A,L_B,L_C,L_D)
        //                layout, each member adding w_k w_l x them;
        //   high (>= 4): (a) HGP block kernel per member vs (b) one HGP VRR pass
        //                at the high members' maximum L, each member accumulating
        //                its own [e0|f0] and running its own HRR (finish()).
        using hgp_block::n_upto;
        const auto &T = hgp_block::table();
        auto prod = production_kernel();
        HgpBlockKernel fin;
        std::vector<double> V, F(4 * MAX_L + 1);
        std::vector<std::vector<double>> accs;
        std::vector<std::array<double, 27>> cf_acc;
        // B0: per-family-quartet bookkeeping, reused across quartets (no allocation per quartet).
        struct Lay { int nr; int c[3]; std::array<int, 4> L4; };
        std::vector<Lay> lays;
        std::vector<int> lay;
        double ta[2] = {0, 0}, tf[2] = {0, 0}, dmax[2] = {0, 0}; // [0] low, [1] high
        std::size_t nfq = 0, nmem_q[2] = {0, 0}, pa_passes[2] = {0, 0}, pf_passes[2] = {0, 0};
        volatile double sink = 0.0;
        const double two_over_sqrt_pi = 2.0 / std::sqrt(std::numbers::pi);
        const auto check = [&](const BlockEriRequest &r, auto &&got, double &dm)
        {
            prod.prepare(r);
            for (std::size_t ca = 0; ca < r.n[0]; ++ca)
                for (std::size_t cb = 0; cb < r.n[1]; ++cb)
                    for (std::size_t cc = 0; cc < r.n[2]; ++cc)
                        for (std::size_t cd2 = 0; cd2 < r.n[3]; ++cd2)
                        {
                            const double x = prod.value(ca, cb, cc, cd2), y = got(ca, cb, cc, cd2);
                            dm = std::max(dm, std::abs(x - y) / std::max(1.0, std::abs(x)));
                        }
        };
        for (std::size_t P = 0; P < fps.size(); ++P)
            for (std::size_t Qi = P; Qi < fps.size(); ++Qi)
            {
                const FamPair &fa = fps[P], &fc = fps[Qi];
                const std::size_t na = fa.m.size(), nc = fc.m.size(), nm = na * nc;
                ++nfq;
                const auto ltot = [&](std::size_t i, std::size_t j) { return fa.m[i].LA + fa.m[i].LB + fc.m[j].LA + fc.m[j].LB; };
                int LABh = -1, LCDh = -1, Mlow = -1;
                for (std::size_t i = 0; i < na; ++i)
                    for (std::size_t j = 0; j < nc; ++j)
                        if (ltot(i, j) >= 4)
                            LABh = std::max(LABh, fa.m[i].LA + fa.m[i].LB), LCDh = std::max(LCDh, fc.m[j].LA + fc.m[j].LB);
                        else
                            Mlow = std::max(Mlow, ltot(i, j));
                // (a), split low / high
                for (int h = 0; h < 2; ++h)
                {
                    const auto t0 = Clock::now();
                    for (std::size_t i = 0; i < na; ++i)
                        for (std::size_t j = 0; j < nc; ++j)
                            if ((ltot(i, j) >= 4) == (h == 1))
                            {
                                prod.prepare(request(fa.m[i], fc.m[j]));
                                sink = sink + prod.value(0, 0, 0, 0);
                                pa_passes[h] += fa.m[i].sp->primitive_pairs.size() * fc.m[j].sp->primitive_pairs.size();
                                ++nmem_q[h];
                            }
                    ta[h] += seconds_since(t0);
                }
                // (c) low members, closed-form family pass (B0: production-quality).
                // Layouts precomputed per family quartet with their members; a
                // single-member layout folds its weight into theta and accumulates
                // directly; a shared layout raises once and adds per member;
                // scalar boys(0, T) when every low member is (ss|ss).
                if (Mlow >= 0)
                {
                    const auto t0 = Clock::now();
                    cf_acc.resize(nm);
                    lay.assign(nm, -1);
                    lays.clear();
                    for (std::size_t i = 0; i < na; ++i)
                        for (std::size_t j = 0; j < nc; ++j)
                        {
                            if (ltot(i, j) >= 4)
                                continue;
                            cf_acc[i * nc + j].fill(0.0);
                            const std::array<int, 4> L4 = {fa.m[i].LA, fa.m[i].LB, fc.m[j].LA, fc.m[j].LB};
                            std::size_t id = 0;
                            while (id < lays.size() && lays[id].L4 != L4)
                                ++id;
                            if (id == lays.size())
                            {
                                Lay l{0, {0, 0, 0}, L4};
                                for (int s2 = 0; s2 < 4; ++s2)
                                    for (int u = 0; u < L4[s2]; ++u)
                                        l.c[l.nr++] = s2;
                                lays.push_back(l);
                            }
                            lay[i * nc + j] = static_cast<int>(id);
                        }
                    const auto raise = [&](int nr, bool acc, const PrimitivePair &ab, const PrimitivePair &cd, double rho,
                                           double delta, const double (&PQ)[3], const double *th, const int (&c)[3], double *out)
                    {
                        switch (nr * 2 + (acc ? 1 : 0))
                        {
                        case 0: cfr<0, false>(ab, cd, rho, delta, PQ, th, c, out); break;
                        case 1: cfr<0, true>(ab, cd, rho, delta, PQ, th, c, out); break;
                        case 2: cfr<1, false>(ab, cd, rho, delta, PQ, th, c, out); break;
                        case 3: cfr<1, true>(ab, cd, rho, delta, PQ, th, c, out); break;
                        case 4: cfr<2, false>(ab, cd, rho, delta, PQ, th, c, out); break;
                        case 5: cfr<2, true>(ab, cd, rho, delta, PQ, th, c, out); break;
                        case 6: cfr<3, false>(ab, cd, rho, delta, PQ, th, c, out); break;
                        default: cfr<3, true>(ab, cd, rho, delta, PQ, th, c, out); break;
                        }
                    };
                    std::array<double, 27> tmp;
                    std::array<std::array<double, 27>, 64> raised_l;
                    for (std::size_t u = 0; u < fa.u.size(); ++u)
                        for (std::size_t w2 = 0; w2 < fc.u.size(); ++w2)
                        {
                            const PrimitivePair &ab = *fa.u[u], &cd = *fc.u[w2];
                            ++pf_passes[0];
                            const double delta = ab.zeta + cd.zeta, rho = ab.zeta * cd.zeta / delta;
                            const double PQ[3] = {ab.center[0] - cd.center[0], ab.center[1] - cd.center[1], ab.center[2] - cd.center[2]};
                            const double Tb = rho * (PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2]);
                            const double base0 = ab.prefactor * cd.prefactor * two_over_sqrt_pi * std::sqrt(rho);
                            double th[4];
                            if (Mlow == 0)
                                th[0] = base0 * HartreeFock::Lookup::boys(0, Tb);
                            else
                            {
                                HartreeFock::Lookup::boys_vec(Tb, std::span<double>(th, Mlow + 1));
                                for (int m = 0; m <= Mlow; ++m)
                                    th[m] *= base0;
                            }
                            // Rows first: a bra member whose weight is zero at this primitive
                            // pair skips its whole row (cc's small members contribute at one
                            // exponent of nine). Iterating layout member lists instead
                            // measured 1.25x slower on cc-pVDZ, 1.65x on cc-pVTZ.
                            {
                                std::uint64_t done = 0;
                                for (std::size_t i = 0; i < na; ++i)
                                {
                                    const double wa = fa.m[i].w[u];
                                    if (wa == 0.0)
                                        continue;
                                    for (std::size_t j = 0; j < nc; ++j)
                                    {
                                        const int id = lay[i * nc + j];
                                        if (id < 0)
                                            continue;
                                        const double wc = fc.m[j].w[w2];
                                        if (wc == 0.0)
                                            continue;
                                        const Lay &l = lays[id];
                                        const double ww = wa * wc;
                                        const int nout = l.nr == 3 ? 27 : l.nr == 2 ? 9 : l.nr == 1 ? 3 : 1;
                                        if (!(done >> id & 1))
                                        {
                                            raise(l.nr, false, ab, cd, rho, delta, PQ, th, l.c, raised_l[id].data());
                                            done |= std::uint64_t{1} << id;
                                        }
                                        double *o = cf_acc[i * nc + j].data();
                                        for (int k = 0; k < nout; ++k)
                                            o[k] += ww * raised_l[id][k];
                                    }
                                }
                            }
                        }
                    tf[0] += seconds_since(t0);
                    for (std::size_t i = 0; i < na; ++i)
                        for (std::size_t j = 0; j < nc; ++j)
                        {
                            if (ltot(i, j) >= 4)
                                continue;
                            const BlockEriRequest r = request(fa.m[i], fc.m[j]);
                            const int nr = lays[lay[i * nc + j]].nr;
                            check(r, [&](std::size_t ca, std::size_t cb, std::size_t cc, std::size_t cd2)
                                  {
                                      const std::size_t comp[4] = {ca, cb, cc, cd2};
                                      int axes[3] = {0, 0, 0}, n_ax = 0;
                                      double nrm = 1.0;
                                      for (int s2 = 0; s2 < 4; ++s2)
                                      {
                                          const ContractedView &vv = *r.views[s2][comp[s2]];
                                          nrm *= norm_ratio[vv._index];
                                          for (int ax = 0; ax < 3 && n_ax < nr; ++ax)
                                              for (int uu = 0; uu < vv._cartesian[ax]; ++uu)
                                                  axes[n_ax++] = ax;
                                      }
                                      const int slot = nr == 3 ? 9 * axes[0] + 3 * axes[1] + axes[2]
                                                       : nr == 2 ? 3 * axes[0] + axes[1] : axes[0];
                                      return cf_acc[i * nc + j][slot] * nrm;
                                  }, dmax[0]);
                        }
                }
                // (b) high members, HGP family pass
                if (LABh >= 0)
                {
                    const auto t0 = Clock::now();
                    const int LAB = LABh, LCD = LCDh, M = LAB + LCD;
                    const int Ne = n_upto(LAB), Nf = n_upto(LCD), Md = M + 1;
                    V.resize(static_cast<std::size_t>(Ne) * Nf * Md);
                    const auto v = [&](int e, int f) { return &V[(static_cast<std::size_t>(e) * Nf + f) * Md]; };
                    accs.resize(std::max(accs.size(), nm));
                    for (std::size_t i = 0; i < na; ++i)
                        for (std::size_t j = 0; j < nc; ++j)
                            if (ltot(i, j) >= 4)
                                accs[i * nc + j].assign(
                                    static_cast<std::size_t>(n_upto(fa.m[i].LA + fa.m[i].LB)) * n_upto(fc.m[j].LA + fc.m[j].LB), 0.0);
                    for (std::size_t u = 0; u < fa.u.size(); ++u)
                        for (std::size_t w2 = 0; w2 < fc.u.size(); ++w2)
                        {
                            const PrimitivePair &ab = *fa.u[u], &cd = *fc.u[w2];
                            ++pf_passes[1];
                            const double zeta = ab.zeta, eta = cd.zeta, delta = zeta + eta, rho = zeta * eta / delta;
                            double WP[3], WQ[3], PQ2 = 0.0;
                            for (int x = 0; x < 3; ++x)
                            {
                                const double W = (zeta * ab.center[x] + eta * cd.center[x]) / delta;
                                WP[x] = W - ab.center[x];
                                WQ[x] = W - cd.center[x];
                                const double pq = ab.center[x] - cd.center[x];
                                PQ2 += pq * pq;
                            }
                            const double prefac = ab.prefactor * cd.prefactor * two_over_sqrt_pi * std::sqrt(rho);
                            HartreeFock::Lookup::boys_vec(rho * PQ2, std::span<double>(F.data(), M + 1));
                            for (int m = 0; m <= M; ++m)
                                v(0, 0)[m] = prefac * F[m];
                            const double i2z = 0.5 / zeta, i2e = 0.5 / eta, i2d = 0.5 / delta, rz = rho / zeta, re = rho / eta;
                            for (int e = 1; e < Ne; ++e)
                            {
                                const auto &c = T[e];
                                const int i = c.axis, em = c.down[i], mt = M - c.L;
                                const double *a = v(em, 0);
                                double *o = v(e, 0);
                                for (int m = 0; m <= mt; ++m)
                                    o[m] = ab.pA[i] * a[m] + WP[i] * a[m + 1];
                                if (c.e[i] > 1)
                                {
                                    const double *bb = v(T[em].down[i], 0);
                                    const double sc = (c.e[i] - 1) * i2z;
                                    for (int m = 0; m <= mt; ++m)
                                        o[m] += sc * (bb[m] - rz * bb[m + 1]);
                                }
                            }
                            for (int f = 1; f < Nf; ++f)
                            {
                                const auto &cf = T[f];
                                const int j = cf.axis, fm = cf.down[j];
                                const int fmm = cf.e[j] > 1 ? T[fm].down[j] : -1;
                                const double sf = (cf.e[j] - 1) * i2e;
                                for (int e = 0; e < Ne; ++e)
                                {
                                    const auto &ce = T[e];
                                    const int mt = M - ce.L - cf.L;
                                    const double *a = v(e, fm);
                                    double *o = v(e, f);
                                    for (int m = 0; m <= mt; ++m)
                                        o[m] = cd.pA[j] * a[m] + WQ[j] * a[m + 1];
                                    if (fmm >= 0)
                                    {
                                        const double *bb = v(e, fmm);
                                        for (int m = 0; m <= mt; ++m)
                                            o[m] += sf * (bb[m] - re * bb[m + 1]);
                                    }
                                    if (ce.e[j] > 0)
                                    {
                                        const double *bb = v(ce.down[j], fm);
                                        const double sc = ce.e[j] * i2d;
                                        for (int m = 0; m <= mt; ++m)
                                            o[m] += sc * bb[m + 1];
                                    }
                                }
                            }
                            for (std::size_t i = 0; i < na; ++i)
                            {
                                const double wa = fa.m[i].w[u];
                                if (wa == 0.0)
                                    continue;
                                const int e0 = n_upto(fa.m[i].LA - 1), ne = n_upto(fa.m[i].LA + fa.m[i].LB);
                                for (std::size_t j = 0; j < nc; ++j)
                                {
                                    const double wc = fc.m[j].w[w2];
                                    if (wc == 0.0 || ltot(i, j) < 4)
                                        continue;
                                    const double ww = wa * wc;
                                    const int f0 = n_upto(fc.m[j].LA - 1), nf = n_upto(fc.m[j].LA + fc.m[j].LB);
                                    double *acc = accs[i * nc + j].data();
                                    for (int e = e0; e < ne; ++e)
                                        for (int f = f0; f < nf; ++f)
                                            acc[static_cast<std::size_t>(e) * nf + f] += ww * v(e, f)[0];
                                }
                            }
                        }
                    for (std::size_t i = 0; i < na; ++i)
                        for (std::size_t j = 0; j < nc; ++j)
                            if (ltot(i, j) >= 4)
                            {
                                const BlockEriRequest r = request(fa.m[i], fc.m[j]);
                                fin.acc_buffer() = accs[i * nc + j];
                                fin.finish(r);
                                sink = sink + fin.value(0, 0, 0, 0);
                            }
                    tf[1] += seconds_since(t0);
                    for (std::size_t i = 0; i < na; ++i)
                        for (std::size_t j = 0; j < nc; ++j)
                            if (ltot(i, j) >= 4)
                            {
                                const BlockEriRequest r = request(fa.m[i], fc.m[j]);
                                fin.acc_buffer() = accs[i * nc + j];
                                fin.finish(r);
                                check(r, [&](std::size_t ca, std::size_t cb, std::size_t cc, std::size_t cd2)
                                      { return fin.value(ca, cb, cc, cd2); }, dmax[1]);
                            }
                }
            }
        std::size_t nmem = 0;
        for (const auto &F2 : fam)
            nmem += F2.size();
        std::printf("  families (%s): %zu families over %zu shells, %zu family quartets\n",
                    within_l ? "within L" : "across L", fam.size(), nmem, nfq);
        const char *nm2[2] = {"L_tot <= 3 (closed-form family)", "L_tot >= 4 (HGP family)    "};
        for (int h = 0; h < 2; ++h)
            std::printf("    %s: %zu member blocks, passes %.2fx fewer, members %.3f s, family %.3f s (%.2fx), max rel diff %.1e\n",
                        nm2[h], nmem_q[h], double(pa_passes[h]) / std::max<double>(1, pf_passes[h]), ta[h], tf[h], ta[h] / tf[h], dmax[h]);
        std::printf("    total: members %.3f s, families %.3f s (%.2fx)\n", ta[0] + ta[1], tf[0] + tf[1],
                    (ta[0] + ta[1]) / (tf[0] + tf[1]));
    }

    // 5b S0 (docs/ERI_LOW_L_CLOSED_FORM_SCOPE.md): if the fused build collected
    // its surviving blocks per chunk of bra group pairs and pooled the
    // primitive quartets of same-class blocks into B = 4 lanes, how full would
    // the lanes be? Replays the fused loop's enumeration exactly (group pairs,
    // L1 start, block Schwarz, at least one component surviving the canonical
    // and component Schwarz filters). No symmetry. Weighted by primitive
    // quartets, the kernel's unit of work.
    void report_census(const BuildSetup &s)
    {
        using namespace HartreeFock;
        using namespace HartreeFock::Integrals;
        constexpr double tol = 1e-10;
        constexpr int B = 4;
        const std::size_t nb = s.nb;
        std::vector<const ContractedView *> ao_views;
        const auto groups = fused_shell_groups(s.pairs, nb, ao_views);
        const std::size_t ng = groups.size();
        std::vector<int> gL(ng);
        for (std::size_t g = 0; g < ng; ++g)
        {
            const auto &c = ao_views[groups[g].first_ao]->_cartesian;
            gL[g] = c[0] + c[1] + c[2];
        }
        std::vector<const ShellPair *> pair_of(nb * nb, nullptr);
        for (const auto &sp : s.pairs)
            pair_of[sp.A._index * nb + sp.B._index] = &sp;
        std::vector<std::array<std::size_t, 2>> gp;
        for (std::size_t a = 0; a < ng; ++a)
            for (std::size_t b = a; b < ng; ++b)
                gp.push_back({a, b});
        const std::size_t ngp = gp.size();
        std::vector<std::size_t> first_pair_of(ng, 0);
        for (std::size_t p = ngp; p-- > 0;)
            first_pair_of[gp[p][0]] = p;
        std::vector<double> qmax(ngp, 0.0);
        for (std::size_t p = 0; p < ngp; ++p)
            for (std::size_t ca = 0; ca < groups[gp[p][0]].n_components; ++ca)
                for (std::size_t cb = 0; cb < groups[gp[p][1]].n_components; ++cb)
                    qmax[p] = std::max(qmax[p], s.Q(groups[gp[p][0]].first_ao + ca, groups[gp[p][1]].first_ao + cb));

        struct Blk { std::size_t bra; int cls; std::size_t K; bool live; };
        std::vector<Blk> blks;
        const auto t0 = Clock::now();
        for (std::size_t bra = 0; bra < ngp; ++bra)
            for (std::size_t ket = first_pair_of[gp[bra][0]]; ket < ngp; ++ket)
            {
                if (qmax[bra] * qmax[ket] < tol)
                    continue;
                const auto &gA = groups[gp[bra][0]], &gB = groups[gp[bra][1]];
                const auto &gC = groups[gp[ket][0]], &gD = groups[gp[ket][1]];
                bool live = false;
                for (std::size_t ca = 0; ca < gA.n_components && !live; ++ca)
                    for (std::size_t cb = 0; cb < gB.n_components && !live; ++cb)
                    {
                        const std::size_t i = gA.first_ao + ca, j = gB.first_ao + cb;
                        if (j < i)
                            continue;
                        for (std::size_t cc = 0; cc < gC.n_components && !live; ++cc)
                            for (std::size_t cd = 0; cd < gD.n_components && !live; ++cd)
                            {
                                const std::size_t k = gC.first_ao + cc, l = gD.first_ao + cd;
                                if (l < k || k < i || (k == i && l < j))
                                    continue;
                                live = s.Q(i, j) * s.Q(k, l) >= tol;
                            }
                    }
                const int L4[4] = {gL[gp[bra][0]], gL[gp[bra][1]], gL[gp[ket][0]], gL[gp[ket][1]]};
                const int cls = ((L4[0] * 8 + L4[1]) * 8 + L4[2]) * 8 + L4[3];
                const std::size_t K = pair_of[gA.first_ao * nb + gB.first_ao]->primitive_pairs.size() *
                                      pair_of[gC.first_ao * nb + gD.first_ao]->primitive_pairs.size();
                blks.push_back({bra, cls, K, live});
            }
        const double t_enum = seconds_since(t0);
        const double t_build = timed_build_block(s, hgp_elem, production_kernel());
        std::size_t n_live = 0, K_dead = 0, K_all = 0;
        for (const Blk &b : blks)
        {
            n_live += b.live;
            K_all += b.K;
            K_dead += b.live ? 0 : b.K;
        }
        std::printf("  census: %zu blocks pass block filters, %zu with a surviving component (%.1f%% of their primitive quartets dead);\n"
                    "          enumeration %.1f ms against a %.1f ms production build\n",
                    blks.size(), n_live, 100.0 * K_dead / std::max<std::size_t>(1, K_all), 1e3 * t_enum, 1e3 * t_build);
        for (int high = 1; high >= 0; --high)
        {
            // within-block (step 2b's rule): lanes when K >= 3, else scalar
            std::size_t useful = 0, wb_lane = 0, wb_slots = 0;
            for (const Blk &b : blks)
            {
                const int lt = (b.cls >> 9) + ((b.cls >> 6) & 7) + ((b.cls >> 3) & 7) + (b.cls & 7);
                if (!b.live || (lt >= 4) != (high == 1))
                    continue;
                useful += b.K;
                if (b.K >= 3)
                    wb_lane += b.K, wb_slots += B * ((b.K + B - 1) / B);
            }
            std::printf("  %s: %.3g live primitive quartets; within-block lanes carry %.1f%% of them (padding %.1f%%)\n",
                        high ? "HGP classes (L_tot >= 4)" : "closed-form classes (L_tot <= 3)",
                        double(useful), 100.0 * wb_lane / std::max<std::size_t>(1, useful),
                        100.0 * (wb_slots - wb_lane) / std::max<std::size_t>(1, wb_slots));
            for (std::size_t chunk : {std::size_t{1}, std::size_t{8}, std::size_t{64}, ngp})
            {
                // pooled: per (chunk, class), all live primitive quartets in lanes
                std::map<std::pair<std::size_t, int>, std::array<std::size_t, 2>> pool; // K sum, blocks
                for (const Blk &b : blks)
                {
                    const int lt = (b.cls >> 9) + ((b.cls >> 6) & 7) + ((b.cls >> 3) & 7) + (b.cls & 7);
                    if (!b.live || (lt >= 4) != (high == 1))
                        continue;
                    auto &e = pool[{b.bra / chunk, b.cls}];
                    e[0] += b.K;
                    ++e[1];
                }
                std::size_t slots = 0;
                std::vector<std::size_t> nblk;
                for (const auto &[key, e] : pool)
                {
                    slots += B * ((e[0] + B - 1) / B);
                    nblk.push_back(e[1]);
                }
                std::sort(nblk.begin(), nblk.end());
                std::printf("    pooled per %s: padding %.2f%%, %zu (chunk, class) batches, median %zu blocks per batch\n",
                            chunk == ngp ? "whole build   " : (std::to_string(chunk) + " bra pairs").c_str(),
                            100.0 * (slots - useful) / std::max<std::size_t>(1, slots), pool.size(),
                            nblk.empty() ? std::size_t{0} : nblk[nblk.size() / 2]);
            }
        }
    }

    void run_fixture(const Fixture &f, double timer_pair)
    {
        BuildSetup s = prepare(f);

        timed_build(s, hgp_elem); // warm the HGP scratch and the caches
        const double t_unwrapped = timed_build(s, hgp_elem);
        const std::uint64_t fingerprint = g_fingerprint;
        const Eigen::MatrixXd G_hgp = g_last;
        double t_low[4] = {t_unwrapped, 0.0, 0.0, 0.0}, low_dG[4] = {0.0, 0.0, 0.0, 0.0};
        std::uint64_t low_fp[4] = {fingerprint, 0, 0, 0};
        for (int max_l = 1; max_l <= 3; ++max_l)
        {
            timed_build(s, hgp_elem, max_l);
            t_low[max_l] = timed_build(s, hgp_elem, max_l);
            low_dG[max_l] = (g_last - G_hgp).cwiseAbs().maxCoeff();
            low_fp[max_l] = g_fingerprint;
        }

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

        if (s.sym)
        {
            s.sym = false;
            timed_build_block(s, hgp_elem, production_kernel());
            const double t_nosym = timed_build_block(s, hgp_elem, production_kernel());
            const Eigen::MatrixXd G_nosym = g_last;
            s.sym = true;
            timed_build_block(s, hgp_elem, production_kernel());
            const double t_sym = timed_build_block(s, hgp_elem, production_kernel());
            std::printf("\nnaphthalene / %s: %zu integral-symmetry ops; production build (Auto block kernels) %.3f s with sym_ops, %.3f s without, max|dG| %.1e\n",
                        f.basis, s.calc._integral_symmetry_ops.size(), t_sym, t_nosym,
                        (g_last - G_nosym).cwiseAbs().maxCoeff());
        }
        std::printf("\n%d H2O / %s: G fingerprint %016llx\n", f.nwater, f.basis,
                    static_cast<unsigned long long>(fingerprint));
        for (int max_l = 1; max_l <= 3; ++max_l)
            std::printf("  closed-form L_tot<=%d: %.3f s vs %.3f s (%.3fx), max|dG| %.1e, fingerprint %016llx\n",
                        max_l, t_low[max_l], t_unwrapped, t_unwrapped / t_low[max_l], low_dG[max_l],
                        static_cast<unsigned long long>(low_fp[max_l]));
        report_hgp_block_by_class(s, G_hgp, t_low[3]);
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

    // --profile N [nwater basis]: repeat the production-shaped build (the Auto
    // engine's block kernels: closed forms at L <= 3, HGP block kernel at 4-12)
    // so `sample` can attribute it. nwater 0 is naphthalene.
    if ((argc == 3 || argc == 5) && std::string(argv[1]) == "--profile")
    {
        const int reps = std::atoi(argv[2]);
        const int nwater = argc == 5 ? std::atoi(argv[3]) : 4;
        const char *basis = argc == 5 ? argv[4] : "6-31g*";
        const BuildSetup s = prepare({nwater, basis});
        double total = 0.0;
        for (int r = 0; r < reps; ++r)
            total += timed_build_block(s, hgp_elem, production_kernel());
        std::printf("--profile: %d builds of %d H2O / %s, %.3f s each\n", reps, nwater, basis, total / reps);
        return 0;
    }

    // --census nwater basis: 5b S0, lane fill if same-class blocks were pooled.
    if (argc == 4 && std::string(argv[1]) == "--census")
    {
        const BuildSetup s = prepare({std::atoi(argv[2]), argv[3]});
        report_census(s);
        return 0;
    }

    // --family nwater basis [within]: the D2a shared-exponent family probe.
    if ((argc == 4 || argc == 5) && std::string(argv[1]) == "--family")
    {
        const BuildSetup s = prepare({std::atoi(argv[2]), argv[3]});
        const bool within = argc == 5 && std::string(argv[4]) == "within";
        report_families(s, within); // warm-up
        report_families(s, within);
        return 0;
    }

    const double timer_pair = timer_pair_seconds();
    std::printf("steady_clock pair: %.1f ns per quartet\n", 1e9 * timer_pair);

    for (const Fixture &f : {Fixture{1, "6-31g*"}, Fixture{4, "6-31g*"},
                             Fixture{1, "cc-pVDZ"}, Fixture{8, "6-31g"}, Fixture{0, "6-31g"},
                             Fixture{4, "cc-pVDZ"}, Fixture{1, "cc-pVTZ"}})
        run_fixture(f, timer_pair);
    return 0;
}
