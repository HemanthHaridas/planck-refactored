// The KS XC nuclear gradient must be bitwise identical at any OpenMP thread
// count (docs/DFT_XC_GRADIENT_PERFORMANCE.md, X3): RKS and UKS, LDA and
// GGA, on a C1 water cation with more grid blocks than accumulation bins.
// Also (X3b): the O(natoms^2) Becke owner-weight derivative against the
// O(natoms^3) form it replaced, kept below as the oracle.
// Also (S2b, docs/DFT_DEFAULT_GRID_SCOPE.md): the KS XC matrix
// (assemble_xc_matrix) is bitwise identical at any thread count, and equals
// a per-point oracle (the pre-S2b algorithm, written out below).
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

    M xc_gradient(const HartreeFock::Calculator &calc, const MolecularGrid &grid, const AOGridEvaluation &ao,
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

    XCGridEvaluation xc_eval(const MolecularGrid &grid, const AOGridEvaluation &ao, const M &pa, const M &pb,
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
} // namespace

int main()
{
    if (!require_boys_library())
        return 1;
    HartreeFock::Calculator calc;
    auto &m = calc._molecule;
    m.natoms = 3;
    m.charge = 1;
    m.multiplicity = 2;
    m.atomic_numbers.resize(3);
    m.atomic_numbers << 8, 1, 1;
    m.atomic_masses.resize(3);
    m.atomic_masses << 16., 1., 1.;
    m.coordinates.resize(3, 3);
    m.coordinates << .12, -.07, .11, .91, .23, -.43, -.34, .86, .27;
    calc._basis._basis = HartreeFock::BasisType::Cartesian;
    calc.prepare_coordinates();
    m.set_standard_from_bohr(m._coordinates);
    calc._shells = require(HartreeFock::BasisFunctions::read_gbs_basis(
        (std::filesystem::path(get_basis_path()) / "6-31g*").string(), m, calc._basis._basis));

    const auto grid = require(MakeMolecularGrid(m, GridLevel::Normal));
    const auto ao = require(evaluate_ao_basis_on_grid(calc._shells, grid));
    if (grid.points.rows() / 128 <= 64)
    {
        std::cerr << "FAIL fixture has too few blocks to fill the bins\n";
        return 1;
    }

    // Core-Hamiltonian orbitals, 5 alpha / 4 beta.
    const auto pairs = build_shellpairs(calc._shells);
    const int n = ao.nbasis();
    const auto engine = HartreeFock::IntegralMethod::ObaraSaika;
    const auto [s, t] = _compute_1e(pairs, n, engine);
    Eigen::SelfAdjointEigenSolver<M> se(s);
    const M orth = se.operatorInverseSqrt();
    Eigen::SelfAdjointEigenSolver<M> he(orth * (t + _compute_nuclear_attraction(pairs, n, m, engine)) * orth);
    const M c = orth * he.eigenvectors();
    const M pa = c.leftCols(5) * c.leftCols(5).transpose();
    const M pb = c.leftCols(4) * c.leftCols(4).transpose();

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
    omp_set_num_threads(saved);
    std::cout << (ok ? "XC gradient and matrix thread identity: PASS\n" : "XC gradient and matrix thread identity: FAIL\n");
    return ok ? 0 : 1;
}
