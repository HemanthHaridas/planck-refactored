#include "hessian.h"

#include <cmath>
#include <expected>
#include <format>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "base/tables.h"
#include "base/types.h"
#include "basis/basis.h"
#include "gradient/gradient.h"
#include "integrals/base.h"
#include "integrals/shellpair.h"
#include "io/logging.h"
#include "lookup/elements.h"
#include "scf/scf.h"
#include "symmetry/integral_symmetry.h"
#include "symmetry/vibrational_symmetry.h"

// ─── Physical constants ────────────────────────────────────────────────────────
//
// HESSIAN_TO_WAVENUMBER: sqrt(Eₕ / (amu · a₀²)) → cm⁻¹
//   = sqrt(Eₕ/amu) / a₀ / (100 · 2π·c)
//   Numerical value matches the Python planck reference (5140.487).
//
// CM_INV_TO_HARTREE: 1 cm⁻¹ in Hartree = hc / (Eₕ · 100 m)
static constexpr double HESSIAN_TO_WAVENUMBER = 5140.487;
static constexpr double CM_INV_TO_HARTREE = 4.5563352527e-6;

// ─── Single-point SCF + gradient helper ──────────────────────────────────────
//
// Mirrors _run_sp_gradient in geomopt.cpp.
// Rebuilds basis, 1e integrals and SCF for the geometry currently stored in
// calc._molecule._standard; then computes the analytic gradient.
// Returns the gradient as a natoms×3 matrix (atom-major).

static std::expected<Eigen::MatrixXd, std::string> _run_sp_gradient_freq_hf(HartreeFock::Calculator &calc)
{
    // Sync coordinate frames
    calc._molecule._coordinates = calc._molecule._standard;
    calc._molecule.coordinates = calc._molecule._standard / ANGSTROM_TO_BOHR;
    calc._molecule.set_standard_from_bohr(calc._molecule._standard);

    // Rebuild basis
    const std::string gbs_path =
        calc._basis._basis_path + "/" + calc._basis._basis_name;
    auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(
        gbs_path, calc._molecule, calc._basis._basis);
    if (!basis_res)
        return std::unexpected("Hessian basis rebuild failed: " + basis_res.error());
    calc._shells = std::move(*basis_res);

    // Reset SCF state (no SAO blocking during finite-difference steps)
    calc._info._scf = HartreeFock::DataSCF(
        calc._scf._scf != HartreeFock::SCFType::RHF);
    calc._info._scf.initialize(calc._shells.nbasis());
    calc._scf.set_scf_mode_auto(calc._shells.nbasis());
    calc._info._is_converged = false;
    calc._use_sao_blocking = false;

    calc._compute_nuclear_repulsion();

    auto shell_pairs = build_shellpairs(calc._shells);
    HartreeFock::Symmetry::update_integral_symmetry(calc);

    auto [S, T] = _compute_1e(shell_pairs, calc._shells.nbasis(),
                              calc._integral._engine,
                              calc._use_integral_symmetry ? &calc._integral_symmetry_ops : nullptr);
    auto V = _compute_nuclear_attraction(shell_pairs, calc._shells.nbasis(),
                                         calc._molecule, calc._integral._engine,
                                         calc._use_integral_symmetry ? &calc._integral_symmetry_ops : nullptr);
    calc._overlap = S;
    calc._hcore = T + V;

    std::expected<void, std::string> scf_res;
    if (calc._scf._scf == HartreeFock::SCFType::UHF)
        scf_res = HartreeFock::SCF::run_uhf(calc, shell_pairs);
    else if (calc._scf._scf == HartreeFock::SCFType::ROHF)
        scf_res = HartreeFock::SCF::run_rohf(calc, shell_pairs);
    else
        scf_res = HartreeFock::SCF::run_rhf(calc, shell_pairs);

    if (!scf_res)
        return std::unexpected("Hessian SCF failed: " + scf_res.error());

    Eigen::MatrixXd grad;
    if (calc._scf._scf == HartreeFock::SCFType::UHF)
    {
        auto grad_res = HartreeFock::Gradient::compute_uhf_gradient(calc, shell_pairs);
        if (!grad_res)
            return std::unexpected("Hessian UHF gradient failed: " + grad_res.error());
        grad = std::move(*grad_res);
    }
    else if (calc._scf._scf == HartreeFock::SCFType::ROHF)
    {
        return std::unexpected("Hessian ROHF gradient is not implemented");
    }
    else
    {
        auto grad_res = HartreeFock::Gradient::compute_rhf_gradient(calc, shell_pairs);
        if (!grad_res)
            return std::unexpected("Hessian RHF gradient failed: " + grad_res.error());
        grad = std::move(*grad_res);
    }

    calc._gradient = grad;
    return grad;
}

// ─── Eckart translation+rotation basis ───────────────────────────────────────
//
// Builds an orthonormal basis for the translation+rotation subspace
// (6 vectors for non-linear, 5 for linear) following the Eckart conditions.
// Uses SVD to discard any near-zero singular values that arise for linear
// molecules (where one rotation is degenerate).
//
// Returns Q (3N × n_tr) with orthonormal columns.

static Eigen::MatrixXd _eckart_basis(const HartreeFock::Calculator &calc)
{
    const std::size_t N = calc._molecule.natoms;
    const int n3 = static_cast<int>(3 * N);

    // Masses and sqrt-masses
    Eigen::VectorXd m(N), sq(N);
    for (std::size_t a = 0; a < N; ++a)
    {
        m[a] = calc._molecule.atomic_masses[a];
        sq[a] = std::sqrt(m[a]);
    }

    // Centre of mass (Bohr)
    const double M_tot = m.sum();
    Eigen::Vector3d r_cm = Eigen::Vector3d::Zero();
    for (std::size_t a = 0; a < N; ++a)
        r_cm += m[a] * calc._molecule._standard.row(a).transpose();
    r_cm /= M_tot;

    // Build 6 T+R vectors
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n3, 6);
    for (std::size_t a = 0; a < N; ++a)
    {
        const int i = static_cast<int>(a) * 3;
        const double s = sq[a];
        Eigen::Vector3d d =
            calc._molecule._standard.row(a).transpose() - r_cm;

        // Translations: T_α[3a+α] = sqrt(m_a)
        D(i + 0, 0) = s;
        D(i + 1, 1) = s;
        D(i + 2, 2) = s;

        // Rotations R_0 = e_x × d, R_1 = e_y × d, R_2 = e_z × d
        // (each weighted by sqrt(m_a))
        //   e_x × d = ( 0,  dz, -dy)
        //   e_y × d = (-dz,  0,  dx)
        //   e_z × d = ( dy,-dx,   0)
        D(i + 0, 3) = 0.0;
        D(i + 1, 3) = -s * d[2];
        D(i + 2, 3) = s * d[1];
        D(i + 0, 4) = s * d[2];
        D(i + 1, 4) = 0.0;
        D(i + 2, 4) = -s * d[0];
        D(i + 0, 5) = -s * d[1];
        D(i + 1, 5) = s * d[0];
        D(i + 2, 5) = 0.0;
    }

    // Orthonormalise via thin SVD; discard near-zero singular values
    Eigen::BDCSVD<Eigen::MatrixXd> svd(D,
                                       Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd &sv = svd.singularValues();
    const double tol = 1e-10 * sv[0];
    int n_tr = 0;
    for (int k = 0; k < sv.size(); ++k)
        if (sv[k] > tol)
            ++n_tr;

    return svd.matrixU().leftCols(n_tr); // (3N, n_tr)
}

// ─── Vibrational analysis ─────────────────────────────────────────────────────

void HartreeFock::Freq::vibrational_analysis(
    HessianResult &result,
    const HartreeFock::Calculator &calc)
{
    const std::size_t N = calc._molecule.natoms;
    const int n3 = static_cast<int>(3 * N);

    // Mass vector (amu), one entry per Cartesian DOF
    Eigen::VectorXd mass_vec(n3);
    for (std::size_t a = 0; a < N; ++a)
    {
        const double m = calc._molecule.atomic_masses[a];
        mass_vec[static_cast<int>(a) * 3 + 0] = m;
        mass_vec[static_cast<int>(a) * 3 + 1] = m;
        mass_vec[static_cast<int>(a) * 3 + 2] = m;
    }

    // Mass-weighted Hessian: H_mw[i,j] = H[i,j] / sqrt(m_i * m_j)
    Eigen::MatrixXd H_mw =
        result.hessian.array() /
        (mass_vec * mass_vec.transpose()).array().sqrt();

    // Build Eckart T+R basis Q (3N × n_tr)
    const Eigen::MatrixXd Q_tr = _eckart_basis(calc);
    const int n_tr = static_cast<int>(Q_tr.cols());
    const int n_vib = n3 - n_tr;

    result.n_vib = n_vib;
    result.is_linear = (n_tr == 5);

    // Project T+R modes out of the mass-weighted Hessian.
    // P = I - Q Q^T is the projector onto the vibrational subspace.
    // Its rank-n_vib column space is spanned by the leading n_vib left
    // singular vectors of P.
    const Eigen::MatrixXd P =
        Eigen::MatrixXd::Identity(n3, n3) - Q_tr * Q_tr.transpose();

    Eigen::BDCSVD<Eigen::MatrixXd> svd_p(P,
                                         Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::MatrixXd Q_vib =
        svd_p.matrixU().leftCols(n_vib); // (3N, n_vib)

    // Reduced mass-weighted Hessian in vibrational subspace
    const Eigen::MatrixXd H_vib = Q_vib.transpose() * H_mw * Q_vib;

    // Diagonalise (symmetric)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H_vib);
    const Eigen::VectorXd &eigenvalues = es.eigenvalues();   // ascending
    const Eigen::MatrixXd evecs = Q_vib * es.eigenvectors(); // (3N, n_vib)

    // Convert eigenvalues to cm⁻¹ (imaginary → negative)
    result.frequencies.resize(n_vib);
    for (int i = 0; i < n_vib; ++i)
    {
        const double ev = eigenvalues[i];
        result.frequencies[i] = (ev >= 0.0)
                                    ? HESSIAN_TO_WAVENUMBER * std::sqrt(ev)
                                    : -HESSIAN_TO_WAVENUMBER * std::sqrt(-ev);
    }

    // Un-mass-weight eigenvectors and normalise columns in Cartesian space
    const Eigen::VectorXd M_sqrt = mass_vec.array().sqrt();
    Eigen::MatrixXd L_cart = evecs.array().colwise() / M_sqrt.array();
    for (int c = 0; c < n_vib; ++c)
    {
        const double norm = L_cart.col(c).norm();
        if (norm > 1e-12)
            L_cart.col(c) /= norm;
    }
    result.normal_modes = std::move(L_cart);
    auto mode_symmetry = HartreeFock::Symmetry::assign_vibrational_symmetry(
        calc, result.normal_modes);
    if (!mode_symmetry)
    {
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Warning,
            "Vibrational Symmetry :",
            "Unavailable: " + mode_symmetry.error());
        result.mode_symmetry.clear();
    }
    else
    {
        result.mode_symmetry = std::move(*mode_symmetry);
    }

    // Zero-point energy (sum over real modes only)
    result.zpe = 0.0;
    result.n_imaginary = 0;
    for (int i = 0; i < n_vib; ++i)
    {
        if (result.frequencies[i] > 0.0)
            result.zpe += 0.5 * result.frequencies[i] * CM_INV_TO_HARTREE;
        else
            ++result.n_imaginary;
    }
}

// ─── Compute semi-numerical Hessian ──────────────────────────────────────────

std::expected<HartreeFock::Freq::HessianResult, std::string>
HartreeFock::Freq::compute_hessian(HartreeFock::Calculator &calc)
{
    return compute_hessian(calc, _run_sp_gradient_freq_hf);
}

std::expected<HartreeFock::Freq::HessianResult, std::string>
HartreeFock::Freq::compute_hessian(
    HartreeFock::Calculator &calc,
    const GradientMatrixRunner &gradient_runner)
{
    const std::size_t N = calc._molecule.natoms;
    const int n3 = static_cast<int>(3 * N);
    const double h = calc._hessian_step;

    HessianResult result;
    result.hessian = Eigen::MatrixXd::Zero(n3, n3);

    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Hessian :",
                                 std::format("Semi-numerical (central differences, h = {:.4f} Bohr, {} evaluations)",
                                             h, 2 * n3));

    for (int j = 0; j < n3; ++j)
    {
        const std::size_t atom = static_cast<std::size_t>(j / 3);
        const int dir = j % 3;

        const double x_orig = calc._molecule._standard(atom, dir);

        // Forward displacement
        calc._molecule._standard(atom, dir) = x_orig + h;
        auto g_fwd_res = gradient_runner(calc);
        if (!g_fwd_res)
            return std::unexpected(g_fwd_res.error());
        Eigen::MatrixXd g_fwd = std::move(*g_fwd_res);

        // Backward displacement
        calc._molecule._standard(atom, dir) = x_orig - h;
        auto g_bck_res = gradient_runner(calc);
        if (!g_bck_res)
            return std::unexpected(g_bck_res.error());
        Eigen::MatrixXd g_bck = std::move(*g_bck_res);

        // Restore
        calc._molecule._standard(atom, dir) = x_orig;

        // Central difference: H[:,j] = (g_fwd - g_bck) / (2h)
        // Both g_fwd and g_bck are natoms×3; flatten atom-major
        for (std::size_t a = 0; a < N; ++a)
            for (int k = 0; k < 3; ++k)
                result.hessian(static_cast<int>(a) * 3 + k, j) =
                    (g_fwd(static_cast<int>(a), k) - g_bck(static_cast<int>(a), k)) / (2.0 * h);

        if ((j + 1) % std::max(1, n3 / 5) == 0 || j + 1 == n3)
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Hessian :",
                                         std::format("  {}/{} displacements done", j + 1, n3));
    }

    // Symmetrise to remove numerical noise
    result.hessian = 0.5 * (result.hessian + result.hessian.transpose());

    // Restore the calculator's SCF state to the undisplaced geometry so
    // the energy / density printed after the hessian block is consistent.
    calc._molecule._coordinates = calc._molecule._standard;
    calc._molecule.coordinates = calc._molecule._standard / ANGSTROM_TO_BOHR;
    calc._molecule.set_standard_from_bohr(calc._molecule._standard);

    // Run vibrational analysis
    vibrational_analysis(result, calc);

    return result;
}
