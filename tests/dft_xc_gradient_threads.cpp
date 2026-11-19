// The KS XC nuclear gradient must be bitwise identical at any OpenMP thread
// count (docs/DFT_XC_GRADIENT_PERFORMANCE.md, X3): RKS and UKS, LDA and
// GGA, on a C1 water cation with more grid blocks than accumulation bins.
// Also (X3b): the O(natoms^2) Becke owner-weight derivative against the
// O(natoms^3) form it replaced, kept below as the oracle.
// Also (S2b, docs/DFT_DEFAULT_GRID_SCOPE.md): the KS XC matrix
// (assemble_xc_matrix) is bitwise identical at any thread count, and equals
// a per-point oracle (the pre-S2b algorithm, written out below).
// Also (S7): the screened AO blocks give the same density, XC matrix and XC
// gradient as dense arrays holding the same AOs, are bitwise at any thread
// count, and the default screening tolerance moves nothing that matters
// (separated water-cation dimer, so most blocks drop columns).
#include "dft/dft_gradient.h"
#include "dft/ks_matrix.h"
#include "base/basis.h"
#include "basis/basis.h"
#include "integrals/base.h"
#include "integrals/shellpair.h"
#include "require_boys_library.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <omp.h>

namespace
{
    using namespace DFT;
    using M = Eigen::MatrixXd;
    bool ok = true;

    template <class T> T require(std::expected<T, std::string> r)
    {
        if (!r)
            throw std::runtime_error(r.error());
        return std::move(*r);
    }
    XC::Functional functional(const char *name, XC::Spin spin)
    {
        return require(XC::Functional::create(require(XC::functional_id(name)), spin));
    }

    template <class AO>
    M xc_gradient(const HartreeFock::Calculator &calc, const MolecularGrid &grid, const AO &ao,
                  const M &pa, const M &pb, bool uks, const char *xname, const char *cname)
    {
        const auto spin = uks ? XC::Spin::Polarized : XC::Spin::Unpolarized;
        const auto x = functional(xname, spin), c = functional(cname, spin);
        if (uks)
        {
            const auto xc = require(evaluate_xc_on_grid(grid, ao, pa, pb, x, c));
            return require(Gradient::compute_xc_nuclear_gradient_uks(calc._molecule, calc._shells, grid, ao, xc, pa, pb));
        }
        const M p = pa + pb;
        const auto xc = require(evaluate_xc_on_grid(grid, ao, p, x, c));
        return require(Gradient::compute_xc_nuclear_gradient_rks(calc._molecule, calc._shells, grid, ao, xc, p));
    }

    template <class AO>
    XCGridEvaluation xc_eval(const MolecularGrid &grid, const AO &ao, const M &pa, const M &pb,
                             bool uks, const char *xname, const char *cname)
    {
        const auto spin = uks ? XC::Spin::Polarized : XC::Spin::Unpolarized;
        const auto x = functional(xname, spin), c = functional(cname, spin);
        return uks ? require(evaluate_xc_on_grid(grid, ao, pa, pb, x, c))
                   : require(evaluate_xc_on_grid(grid, ao, M(pa + pb), x, c));
    }

    // Pre-S2b assemble_xc_matrix, written out: per point,
    // V += w vrho phi phi^T + w (phi g^T + g phi^T), g = sum_k c_k dphi/dk.
    std::pair<M, M> xc_matrix_oracle(const MolecularGrid &grid, const AOGridEvaluation &ao, const XCGridEvaluation &xc)
    {
        const Eigen::Index nb = ao.nbasis();
        const bool pol = xc.density.polarized, gga = xc.vsigma.cols() > 0;
        M va = M::Zero(nb, nb), vb = M::Zero(nb, nb);
        for (Eigen::Index p = 0; p < grid.points.rows(); ++p)
        {
            const double w = grid.points(p, 3);
            if (w == 0.0)
                continue;
            const Eigen::VectorXd phi = ao.values.row(p).transpose();
            const auto add = [&](M &v, double vrho, const Eigen::Vector3d &c)
            {
                const Eigen::VectorXd g = c.x() * ao.grad_x.row(p).transpose() + c.y() * ao.grad_y.row(p).transpose() +
                                          c.z() * ao.grad_z.row(p).transpose();
                v += w * vrho * phi * phi.transpose() + w * (phi * g.transpose() + g * phi.transpose());
            };
            const auto grad = [&](const DensityChannelOnGrid &d)
            { return Eigen::Vector3d(d.grad_x(p), d.grad_y(p), d.grad_z(p)); };
            if (!pol)
            {
                add(va, xc.vrho(p, 0), gga ? Eigen::Vector3d(2.0 * xc.vsigma(p, 0) * grad(xc.density.total))
                                           : Eigen::Vector3d::Zero());
                continue;
            }
            Eigen::Vector3d ca = Eigen::Vector3d::Zero(), cb = Eigen::Vector3d::Zero();
            if (gga)
            {
                const Eigen::Vector3d ga = grad(xc.density.alpha), gb = grad(xc.density.beta);
                ca = 2.0 * xc.vsigma(p, 0) * ga + xc.vsigma(p, 1) * gb;
                cb = xc.vsigma(p, 1) * ga + 2.0 * xc.vsigma(p, 2) * gb;
            }
            add(va, xc.vrho(p, 0), ca);
            add(vb, xc.vrho(p, 1), cb);
        }
        return {va, pol ? vb : va};
    }

    // The pre-X3b becke_partition_owner_derivatives, verbatim: oracle only.
    std::expected<Eigen::MatrixXd, std::string> becke_oracle(
        const MolecularGrid &grid,
        const HartreeFock::Molecule &mol,
        Eigen::Index ip)
    {
        // Differentiate the owner atom's Becke partition weight with
        // respect to every nuclear Cartesian. This is the core moving-grid
        // correction needed for analytic DFT gradients.
        const Eigen::Index natoms = static_cast<Eigen::Index>(mol.natoms);
        if (grid.owner.size() != grid.points.rows())
            return std::unexpected("molecular grid owner array does not match point count");
        if (grid.atomic_weights.size() != grid.points.rows() ||
            grid.partition_weights.size() != grid.points.rows())
        {
            return std::unexpected("molecular grid partition metadata does not match point count");
        }

        const int owner = grid.owner(ip);
        if (owner < 0 || owner >= natoms)
            return std::unexpected("molecular grid owner index is out of range");

        const Eigen::Vector3d point = grid.points.row(ip).head<3>().transpose();
        const Eigen::MatrixXd &coordinates = mol._standard;
        const Eigen::VectorXi &atomic_numbers = mol.atomic_numbers;

        std::vector<double> products(static_cast<std::size_t>(natoms), 1.0);
        // d(product_k)/dR (natoms x 3) for every k, side by side in one buffer.
        Eigen::MatrixXd dproducts = Eigen::MatrixXd::Zero(natoms, 3 * natoms);
        const auto dprod = [&dproducts](Eigen::Index k)
        { return dproducts.middleCols(3 * k, 3); };

        for (Eigen::Index i = 0; i < natoms; ++i)
        {
            const Eigen::Vector3d ri = coordinates.row(i).transpose();
            for (Eigen::Index j = i + 1; j < natoms; ++j)
            {
                const Eigen::Vector3d rj = coordinates.row(j).transpose();
                const Eigen::Vector3d rij = ri - rj;
                const double Rij = rij.norm();
                if (Rij < 1e-14)
                    return std::unexpected("Becke partition derivative encountered coincident atoms");

                const Eigen::Vector3d eij = rij / Rij;
                const Eigen::Vector3d dpi = point - ri;
                const Eigen::Vector3d dpj = point - rj;
                const double di = dpi.norm();
                const double dj = dpj.norm();
                if (di < 1e-14 || dj < 1e-14)
                    return std::unexpected("Becke partition derivative encountered a grid point on a nucleus");

                const Eigen::Vector3d ui = dpi / di;
                const Eigen::Vector3d uj = dpj / dj;
                const double mu_raw = (di - dj) / Rij;
                const double aij = DFT::detail::treutler_becke_adjustment(
                    atomic_numbers(i),
                    atomic_numbers(j));
                const double mu_adjusted = mu_raw + aij * (1.0 - mu_raw * mu_raw);
                const double mu_clamped = std::clamp(mu_adjusted, -1.0, 1.0);
                const double sij = DFT::detail::becke_switch(mu_clamped);
                const double ds_dmu =
                    (std::abs(mu_adjusted - mu_clamped) < 1e-14)
                        ? DFT::detail::becke_switch_derivative(mu_clamped)
                        : 0.0;

                const double old_pi = products[static_cast<std::size_t>(i)];
                const double old_pj = products[static_cast<std::size_t>(j)];
                products[static_cast<std::size_t>(i)] = old_pi * sij;
                products[static_cast<std::size_t>(j)] = old_pj * (1.0 - sij);
                dprod(i) *= sij;
                dprod(j) *= (1.0 - sij);

                // d s_ij / dR is nonzero only in the rows of the owner, i and j.
                const Eigen::Index rows[3] = {owner, i, j};
                for (int r = 0; r < 3; ++r)
                {
                    const Eigen::Index atom = rows[r];
                    if (r > 0 && atom == owner)
                        continue;
                    const double delta_owner = (atom == owner) ? 1.0 : 0.0;
                    const double delta_i = (atom == i) ? 1.0 : 0.0;
                    const double delta_j = (atom == j) ? 1.0 : 0.0;

                    const Eigen::Vector3d ddi = (delta_owner - delta_i) * ui;
                    const Eigen::Vector3d ddj = (delta_owner - delta_j) * uj;
                    const Eigen::Vector3d dR = (delta_i - delta_j) * eij;
                    const Eigen::Vector3d dmu_raw =
                        ((ddi - ddj) * Rij - (di - dj) * dR) / (Rij * Rij);
                    const Eigen::Vector3d dmu_adjusted =
                        (1.0 - 2.0 * aij * mu_raw) * dmu_raw;
                    const Eigen::RowVector3d ds = (ds_dmu * dmu_adjusted).transpose();
                    dprod(i).row(atom) += old_pi * ds;
                    dprod(j).row(atom) -= old_pj * ds;
                }
            }
        }

        double sum = 0.0;
        Eigen::MatrixXd dsum = Eigen::MatrixXd::Zero(natoms, 3);
        for (Eigen::Index atom = 0; atom < natoms; ++atom)
        {
            sum += products[static_cast<std::size_t>(atom)];
            dsum += dprod(atom);
        }
        if (sum <= 0.0)
            return std::unexpected("Becke partition derivative: partition denominator underflowed");

        const double owner_product = products[static_cast<std::size_t>(owner)];
        Eigen::MatrixXd derivatives = Eigen::MatrixXd::Zero(natoms, 3);
        for (Eigen::Index atom = 0; atom < natoms; ++atom)
        {
            for (int q = 0; q < 3; ++q)
            {
                derivatives(atom, q) =
                    (dprod(owner)(atom, q) * sum -
                     owner_product * dsum(atom, q)) /
                    (sum * sum);
            }
        }

        return derivatives;
    }

    // Max |new - old| over every point of a C1 7-atom fixture, relative to max |old|.
    bool becke_matches_oracle()
    {
        HartreeFock::Calculator calc;
        auto &m = calc._molecule;
        m.natoms = 7;
        m.atomic_numbers.resize(7);
        m.atomic_numbers << 6, 8, 1, 1, 1, 1, 7;
        m.atomic_masses = m.atomic_numbers.cast<double>() * 2.0;
        m.coordinates.resize(7, 3);
        m.coordinates << 0.00, 0.00, 0.00, 1.41, 0.12, -0.05, -0.38, 1.02, 0.11, -0.41, -0.53, 0.88,
            -0.29, -0.47, -0.93, 1.72, -0.71, 0.33, 2.9, 1.1, 0.6;
        calc._basis._basis = HartreeFock::BasisType::Cartesian;
        calc.prepare_coordinates();
        m.set_standard_from_bohr(m._coordinates);
        const auto grid = require(MakeMolecularGrid(m, GridLevel::Coarse));
        double max_ref = 0.0, max_diff = 0.0;
        for (Eigen::Index ip = 0; ip < grid.points.rows(); ++ip)
        {
            const M ref = require(becke_oracle(grid, m, ip));
            const M got = require(Gradient::becke_partition_owner_derivatives(grid, m, ip));
            max_ref = std::max(max_ref, ref.cwiseAbs().maxCoeff());
            max_diff = std::max(max_diff, (got - ref).cwiseAbs().maxCoeff());
        }
        std::cout << "Becke derivative vs oracle: " << grid.points.rows() << " points, max |diff| " << max_diff
                  << ", max |ref| " << max_ref << '\n';
        return max_ref > 1e-3 && max_diff <= 1e-13 * max_ref;
    }
    // A C1 doublet in 6-31G* on a `normal` grid, with core-Hamiltonian
    // densities (n_alpha = n_beta + 1 occupied orbitals).
    struct Fixture
    {
        HartreeFock::Calculator calc;
        MolecularGrid grid;
        M pa, pb;
    };
    Fixture make_fixture(const std::vector<int> &z, const M &xyz, int charge, int n_beta)
    {
        Fixture f;
        auto &m = f.calc._molecule;
        m.natoms = z.size();
        m.charge = charge;
        m.multiplicity = 2;
        m.atomic_numbers = Eigen::Map<const Eigen::VectorXi>(z.data(), z.size());
        m.atomic_masses.resize(z.size());
        for (std::size_t i = 0; i < z.size(); ++i)
            m.atomic_masses(i) = z[i] == 8 ? 16. : 1.;
        m.coordinates = xyz;
        f.calc._basis._basis = HartreeFock::BasisType::Cartesian;
        f.calc.prepare_coordinates();
        m.set_standard_from_bohr(m._coordinates);
        f.calc._shells = require(HartreeFock::BasisFunctions::read_gbs_basis(
            (std::filesystem::path(get_basis_path()) / "6-31g*").string(), m, f.calc._basis._basis));
        f.grid = require(MakeMolecularGrid(m, GridLevel::Normal));

        const auto pairs = build_shellpairs(f.calc._shells);
        const int n = f.calc._shells.nbasis();
        const auto engine = HartreeFock::IntegralMethod::ObaraSaika;
        const auto [s, t] = _compute_1e(pairs, n, engine);
        Eigen::SelfAdjointEigenSolver<M> se(s);
        const M orth = se.operatorInverseSqrt();
        Eigen::SelfAdjointEigenSolver<M> he(orth * (t + _compute_nuclear_attraction(pairs, n, m, engine)) * orth);
        const M c = orth * he.eigenvectors();
        f.pa = c.leftCols(n_beta + 1) * c.leftCols(n_beta + 1).transpose();
        f.pb = c.leftCols(n_beta) * c.leftCols(n_beta).transpose();
        return f;
    }
} // namespace

int main()
{
    if (!require_boys_library())
        return 1;
    // C1 water cation, 5 alpha / 4 beta.
    M xyz(3, 3);
    xyz << .12, -.07, .11, .91, .23, -.43, -.34, .86, .27;
    const Fixture fx = make_fixture({8, 1, 1}, xyz, 1, 4);
    const auto &calc = fx.calc;
    const auto &grid = fx.grid;
    const auto &pa = fx.pa;
    const auto &pb = fx.pb;
    const auto ao = require(evaluate_ao_basis_on_grid(calc._shells, grid));
    if (grid.points.rows() / 128 <= 64)
    {
        std::cerr << "FAIL fixture has too few blocks to fill the bins\n";
        return 1;
    }

    struct Case
    {
        bool uks;
        const char *x, *c, *label;
    };
    const Case cases[] = {{false, "lda_x", "lda_c_vwn", "RKS LDA"},
                          {false, "gga_x_pbe", "gga_c_pbe", "RKS GGA"},
                          {true, "lda_x", "lda_c_vwn", "UKS LDA"},
                          {true, "gga_x_pbe", "gga_c_pbe", "UKS GGA"}};

    if (!becke_matches_oracle())
    {
        ok = false;
        std::cerr << "FAIL Becke owner derivative differs from the O(natoms^3) oracle\n";
    }

    const int saved = omp_get_max_threads();
    for (const auto &k : cases)
    {
        omp_set_num_threads(1);
        const M ref = xc_gradient(calc, grid, ao, pa, pb, k.uks, k.x, k.c);
        if (!ref.allFinite() || ref.norm() < 1e-6)
        {
            ok = false;
            std::cerr << "FAIL " << k.label << ": degenerate reference gradient\n";
        }
        for (int threads : {2, 3, 4, 8})
        {
            omp_set_num_threads(threads);
            const M g = xc_gradient(calc, grid, ao, pa, pb, k.uks, k.x, k.c);
            if (g != ref)
            {
                ok = false;
                std::cerr << "FAIL " << k.label << " at " << threads << " threads: max |diff| "
                          << (g - ref).cwiseAbs().maxCoeff() << '\n';
            }
        }

        // KS XC matrix: thread identity and the per-point oracle.
        omp_set_num_threads(1);
        const auto xc = xc_eval(grid, ao, pa, pb, k.uks, k.x, k.c);
        const auto v1 = require(assemble_xc_matrix(grid, ao, xc));
        const auto [oa, ob] = xc_matrix_oracle(grid, ao, xc);
        const double scale = std::max(oa.cwiseAbs().maxCoeff(), ob.cwiseAbs().maxCoeff());
        const double dev = std::max((v1.alpha - oa).cwiseAbs().maxCoeff(), (v1.beta - ob).cwiseAbs().maxCoeff());
        std::cout << k.label << " XC matrix vs per-point oracle: max |diff| " << dev << ", max |V| " << scale << '\n';
        if (!(scale > 1e-3 && dev <= 1e-12 * scale) || v1.alpha != v1.alpha.transpose())
        {
            ok = false;
            std::cerr << "FAIL " << k.label << ": XC matrix differs from the oracle or is not symmetric\n";
        }
        for (int threads : {2, 3, 4, 8})
        {
            omp_set_num_threads(threads);
            const auto v = require(assemble_xc_matrix(grid, ao, xc));
            if (v.alpha != v1.alpha || v.beta != v1.beta)
            {
                ok = false;
                std::cerr << "FAIL " << k.label << " XC matrix at " << threads << " threads: max |diff| "
                          << std::max((v.alpha - v1.alpha).cwiseAbs().maxCoeff(),
                                      (v.beta - v1.beta).cwiseAbs().maxCoeff()) << '\n';
            }
        }
    }

    // S7. A loose tolerance (physically meaningless) drops columns from most
    // blocks, so the gather and scatter paths run; the dense arrays rebuilt
    // from the same blocks hold the same AOs, so only summation order may
    // differ. Not for the GGA gradient: the dense path evaluates the AO
    // Hessian of every AO, including those a block dropped, so it keeps a
    // term the screened path drops; the default-tolerance check covers it.
    // Fixture: the cation next to a neutral water 6 A away (10 alpha / 9
    // beta), so most blocks lie on one monomer and drop the other's AOs.
    M xyz2(6, 3);
    xyz2 << xyz, xyz.rowwise() + Eigen::RowVector3d(6.0, 0.4, -0.3);
    const Fixture f2 = make_fixture({8, 1, 1, 8, 1, 1}, xyz2, 1, 9);
    const auto S = require(evaluate_screened_ao_basis_on_grid(f2.calc._shells, f2.grid, 1e-4));
    const auto D = to_dense(S);
    const auto S0 = require(evaluate_screened_ao_basis_on_grid(f2.calc._shells, f2.grid));
    const auto ao2 = require(evaluate_ao_basis_on_grid(f2.calc._shells, f2.grid));
    const double kept = static_cast<double>(S.stored_entries()) / (static_cast<double>(S.npoints()) * S.nbasis());
    std::cout << "screened fixture keeps " << kept << " of the AO entries at 1e-4\n";
    if (!(kept < 0.8))
    {
        ok = false;
        std::cerr << "FAIL screened fixture drops too few columns to test anything\n";
    }
    const auto rel = [](const M &a, const M &b) { return (a - b).cwiseAbs().maxCoeff() / std::max(1.0, b.cwiseAbs().maxCoeff()); };
    for (const auto &k : cases)
    {
        omp_set_num_threads(1);
        const M gs = xc_gradient(f2.calc, f2.grid, S, f2.pa, f2.pb, k.uks, k.x, k.c);
        const M gd = xc_gradient(f2.calc, f2.grid, D, f2.pa, f2.pb, k.uks, k.x, k.c);
        const auto xs = xc_eval(f2.grid, S, f2.pa, f2.pb, k.uks, k.x, k.c);
        const auto xd = xc_eval(f2.grid, D, f2.pa, f2.pb, k.uks, k.x, k.c);
        const auto vs = require(assemble_xc_matrix(f2.grid, S, xs));
        const auto vd = require(assemble_xc_matrix(f2.grid, D, xd));
        const double dg = rel(gs, gd), dv = std::max(rel(vs.alpha, vd.alpha), rel(vs.beta, vd.beta));
        const double de = std::abs(xs.total_energy - xd.total_energy);
        std::cout << k.label << " screened vs dense, same AOs: gradient " << dg << ", XC matrix " << dv
                  << ", E_xc " << de << '\n';
        const bool gga = std::string(k.x).starts_with("gga");
        if (!((gga || dg <= 1e-12) && dv <= 1e-12 && de <= 1e-12))
        {
            ok = false;
            std::cerr << "FAIL " << k.label << ": screened blocks differ from the dense arrays they hold\n";
        }
        for (int threads : {2, 3, 4, 8})
        {
            omp_set_num_threads(threads);
            const M g = xc_gradient(f2.calc, f2.grid, S, f2.pa, f2.pb, k.uks, k.x, k.c);
            const auto v = require(assemble_xc_matrix(f2.grid, S, xs));
            if (g != gs || v.alpha != vs.alpha || v.beta != vs.beta)
            {
                ok = false;
                std::cerr << "FAIL " << k.label << " screened at " << threads << " threads\n";
            }
        }

        // Default tolerance against the unscreened dense arrays (also the
        // GGA term the same-AO check above cannot see).
        omp_set_num_threads(1);
        const double d0 = rel(xc_gradient(f2.calc, f2.grid, S0, f2.pa, f2.pb, k.uks, k.x, k.c),
                              xc_gradient(f2.calc, f2.grid, ao2, f2.pa, f2.pb, k.uks, k.x, k.c));
        std::cout << k.label << " default screening vs unscreened: gradient " << d0 << '\n';
        if (!(d0 <= 1e-10))
        {
            ok = false;
            std::cerr << "FAIL " << k.label << ": default screening moves the gradient by " << d0 << '\n';
        }
    }

    omp_set_num_threads(saved);
    std::cout << (ok ? "XC gradient and matrix thread identity: PASS\n" : "XC gradient and matrix thread identity: FAIL\n");
    return ok ? 0 : 1;
}
