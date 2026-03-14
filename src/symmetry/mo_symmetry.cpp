#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "mo_symmetry.h"
#include "wrapper.h"
#include "external/libmsym/install/include/libmsym/msym.h"

// ─── Internal helpers ─────────────────────────────────────────────────────────

namespace
{

// Factorial lookup (0! … 12!)
static const int FACT[13] = {1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800,479001600};

static int multinomial(int n, int k0, int k1, int k2)
{
    if (k0 < 0 || k1 < 0 || k2 < 0 || k0+k1+k2 != n) return 0;
    return FACT[n] / (FACT[k0] * FACT[k1] * FACT[k2]);
}

static double ipow(double x, int n)
{
    if (n == 0) return 1.0;
    double r = 1.0;
    for (int i = 0; i < n; ++i) r *= x;
    return r;
}

// ── Compute the 3×3 Cartesian matrix for a symmetry operation ─────────────────
//
// Conventions (libmsym):
//   - Proper rotation   : rotate by 2π·power/order about axis v
//   - Reflection        : reflect through the plane with normal v
//   - Improper rotation : S_n^k = σ_h · C_n^k  (σ_h = plane ⊥ axis v)
//   - Inversion         : -I
static Eigen::Matrix3d sop_to_matrix(const msym_symmetry_operation_t& sop)
{
    using M3d = Eigen::Matrix3d;
    using V3d = Eigen::Vector3d;

    // Use integer constants matching the enum (IDENTITY=0, PROPER=1, IMPROPER=2, REFLECTION=3, INVERSION=4)
    switch (static_cast<int>(sop.type))
    {
        case 0: // IDENTITY
            return M3d::Identity();

        case 4: // INVERSION
            return -M3d::Identity();

        case 3: // REFLECTION
        {
            V3d n(sop.v[0], sop.v[1], sop.v[2]);
            n.normalize();
            return M3d::Identity() - 2.0 * n * n.transpose();
        }

        case 1: // PROPER_ROTATION
        {
            V3d v(sop.v[0], sop.v[1], sop.v[2]);
            v.normalize();
            const double angle = 2.0 * M_PI * sop.power / sop.order;
            const double c = std::cos(angle), s = std::sin(angle);
            M3d K;
            K <<      0, -v.z(),  v.y(),
                 v.z(),       0, -v.x(),
                -v.y(),  v.x(),       0;
            return c * M3d::Identity() + (1.0 - c) * v * v.transpose() + s * K;
        }

        case 2: // IMPROPER_ROTATION: S_n^k = σ_h ∘ C_n^k (σ_h reflects through plane ⊥ v)
        {
            V3d v(sop.v[0], sop.v[1], sop.v[2]);
            v.normalize();
            const double angle = 2.0 * M_PI * sop.power / sop.order;
            const double c = std::cos(angle), s = std::sin(angle);
            M3d K;
            K <<      0, -v.z(),  v.y(),
                 v.z(),       0, -v.x(),
                -v.y(),  v.x(),       0;
            const M3d Cn      = c * M3d::Identity() + (1.0 - c) * v * v.transpose() + s * K;
            const M3d sigma_h = M3d::Identity() - 2.0 * v * v.transpose();
            return sigma_h * Cn;
        }

        default:
            return M3d::Identity();
    }
}

// ── Coefficient of vx^ax vy^ay vz^az in (M⁻¹v)_x^lx · (M⁻¹v)_y^ly · (M⁻¹v)_z^lz ──
//
// Since M is orthogonal: M⁻¹ = Mᵀ, so (Mᵀv)_q = Σ_r M[r][q] v_r.
// This gives the AO angular-momentum transformation coefficient when the
// source Cartesian function (lx,ly,lz) is acted on by M.
static double angular_coeff(const Eigen::Matrix3d& M,
                             int lx, int ly, int lz,
                             int ax, int ay, int az)
{
    if (lx + ly + lz != ax + ay + az) return 0.0;
    if (lx + ly + lz == 0)            return 1.0;

    // cx[r] = M(r,0), cy[r] = M(r,1), cz[r] = M(r,2)
    const double cx[3] = {M(0,0), M(1,0), M(2,0)};
    const double cy[3] = {M(0,1), M(1,1), M(2,1)};
    const double cz[3] = {M(0,2), M(1,2), M(2,2)};

    double result = 0.0;

    for (int i0 = 0; i0 <= lx; ++i0)
    for (int i1 = 0; i1 <= lx-i0; ++i1)
    {
        const int i2 = lx - i0 - i1;
        const double cx_term = multinomial(lx, i0, i1, i2)
                              * ipow(cx[0], i0) * ipow(cx[1], i1) * ipow(cx[2], i2);

        for (int j0 = 0; j0 <= ly; ++j0)
        for (int j1 = 0; j1 <= ly-j0; ++j1)
        {
            const int j2 = ly - j0 - j1;
            // k is fully determined by the constraint i+j+k = (ax,ay,az)
            const int k0 = ax - i0 - j0;
            const int k1 = ay - i1 - j1;
            const int k2 = az - i2 - j2;
            if (k0 < 0 || k1 < 0 || k2 < 0 || k0+k1+k2 != lz) continue;

            const double cy_term = multinomial(ly, j0, j1, j2)
                                 * ipow(cy[0], j0) * ipow(cy[1], j1) * ipow(cy[2], j2);
            const double cz_term = multinomial(lz, k0, k1, k2)
                                 * ipow(cz[0], k0) * ipow(cz[1], k1) * ipow(cz[2], k2);

            result += cx_term * cy_term * cz_term;
        }
    }
    return result;
}

// ── Atom permutation: perm[a] = b if M maps atom a's position to atom b ──────
//
// Positions are taken from molecule.standard (Angstrom, symmetrized frame).
// tol: distance threshold in Angstrom.
static std::vector<int> build_permutation(const Eigen::Matrix3d& M,
                                          const HartreeFock::Molecule& mol,
                                          double tol = 0.25)
{
    const int N = static_cast<int>(mol.natoms);
    std::vector<int> perm(N, -1);

    for (int a = 0; a < N; ++a)
    {
        const Eigen::Vector3d Mpa = M * mol.standard.row(a).transpose();
        for (int b = 0; b < N; ++b)
        {
            if (mol.atomic_numbers[a] != mol.atomic_numbers[b]) continue;
            if ((Mpa - mol.standard.row(b).transpose()).norm() < tol)
            {
                perm[a] = b;
                break;
            }
        }
        if (perm[a] == -1)
            throw std::runtime_error(
                "MO symmetry: atom " + std::to_string(a) +
                " has no permutation image under this operation");
    }
    return perm;
}

// ── nb × nb AO representation matrix for symmetry op (M, perm) ───────────────
//
// D_R[ν][μ] = transform coeff of Cartesian function μ into ν, accounting for
// the component-norm ratio between the two functions.
static Eigen::MatrixXd build_ao_transform(const Eigen::Matrix3d& M,
                                           const std::vector<int>& perm,
                                           const HartreeFock::Basis& basis)
{
    const std::size_t nb = basis.nbasis();
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(nb, nb);

    const auto& bfs = basis._basis_functions;

    for (std::size_t mu = 0; mu < nb; ++mu)
    {
        const auto& cv_mu   = bfs[mu];
        const int   atom_a  = static_cast<int>(cv_mu._shell->_atom_index);
        const int   atom_b  = perm[atom_a];
        const int   lx = cv_mu._cartesian[0];
        const int   ly = cv_mu._cartesian[1];
        const int   lz = cv_mu._cartesian[2];
        const int   l  = lx + ly + lz;
        const double norm_mu = cv_mu._component_norm;

        for (std::size_t nu = 0; nu < nb; ++nu)
        {
            const auto& cv_nu = bfs[nu];
            if (static_cast<int>(cv_nu._shell->_atom_index) != atom_b) continue;

            const int ax = cv_nu._cartesian[0];
            const int ay = cv_nu._cartesian[1];
            const int az = cv_nu._cartesian[2];
            if (ax + ay + az != l) continue;

            const double c = angular_coeff(M, lx, ly, lz, ax, ay, az);
            if (std::abs(c) < 1e-14) continue;

            D(nu, mu) = c * (norm_mu / cv_nu._component_norm);
        }
    }
    return D;
}

// ── Identify Λ and assign irrep labels for a linear molecule ─────────────────
//
// After msymAlignAxes the molecular axis is z.  A rotation R_θ about z gives:
//
//   Λ | χ(π/2) | χ(π/3)
//   ──┼────────┼───────
//   0 |  1.0   |  1.0    → Σ
//   1 |  0.0   |  0.5    → Π
//   2 | -1.0   | -0.5    → Δ
//   3 |  0.0   | -1.0    → Φ
//   4 |  1.0   | -0.5    → Γ
//
// For Σ: sign of χ_i(σv) distinguishes + (symmetric) from − (antisymmetric).
// For D∞h: sign of χ_i(i) distinguishes g from u.
static void assign_for_linear(
    const Eigen::MatrixXd&     C,
    const Eigen::MatrixXd&     SD_pi2,   // S · D_{R_{π/2}}
    const Eigen::MatrixXd&     SD_pi3,   // S · D_{R_{π/3}}
    const Eigen::MatrixXd&     SD_sv,    // S · D_{σv(xz)}
    const Eigen::MatrixXd*     SD_inv,   // S · D_{i}  (nullptr for C∞v)
    std::vector<std::string>&  labels)
{
    struct LamMatch { double c1, c2; int lam; };
    static const LamMatch LAMBDA_TABLE[] = {
        { 1.0,  1.0, 0 },   // Σ
        { 0.0,  0.5, 1 },   // Π
        {-1.0, -0.5, 2 },   // Δ
        { 0.0, -1.0, 3 },   // Φ
        { 1.0, -0.5, 4 },   // Γ
    };

    const int nb = static_cast<int>(C.cols());
    labels.resize(nb);

    for (int i = 0; i < nb; ++i)
    {
        const Eigen::VectorXd ci = C.col(i);

        const double chi_pi2 = ci.dot(SD_pi2 * ci);
        const double chi_pi3 = ci.dot(SD_pi3 * ci);

        // Nearest-neighbour match in (χ(π/2), χ(π/3)) space
        int    best_lam  = 0;
        double best_dist = 1e10;
        for (const auto& m : LAMBDA_TABLE)
        {
            const double d = (chi_pi2 - m.c1)*(chi_pi2 - m.c1)
                           + (chi_pi3 - m.c2)*(chi_pi3 - m.c2);
            if (d < best_dist) { best_dist = d; best_lam = m.lam; }
        }

        std::string label;
        if (best_lam == 0)
        {
            const double chi_sv = ci.dot(SD_sv * ci);
            label = (chi_sv >= 0.0) ? "SIG+" : "SIG-";
        }
        else if (best_lam == 1) label = "PI";
        else if (best_lam == 2) label = "DEL";
        else if (best_lam == 3) label = "PHI";
        else                    label = "LAM" + std::to_string(best_lam);

        if (SD_inv != nullptr)
        {
            const double chi_inv = ci.dot((*SD_inv) * ci);
            label += (chi_inv >= 0.0) ? "g" : "u";
        }

        labels[i] = label;
    }
}

// ── Full linear-molecule MO symmetry driver ───────────────────────────────────
//
// Builds the five operator matrices (two rotations, σv, optionally i) and
// calls assign_for_linear for alpha (and beta for UHF).
static void assign_mo_symmetry_linear(HartreeFock::Calculator& calc)
{
    const HartreeFock::Molecule& mol = calc._molecule;
    const HartreeFock::Basis&    bs  = calc._shells;
    const Eigen::MatrixXd&       S   = calc._overlap;

    // D∞h contains both 'D' and 'inf'; C∞v contains only 'C'.
    const bool is_Dinfh = (mol._point_group.find('D') != std::string::npos);

    // After msymAlignAxes the molecular axis is z.
    // All atoms lie on (0,0,z_a) → rotation about z leaves each atom fixed,
    // σv through xz-plane leaves each atom fixed,
    // inversion maps (0,0,z_a) → (0,0,−z_a) (finds the paired image atom).

    auto build_SD = [&](const Eigen::Matrix3d& M) -> Eigen::MatrixXd {
        const std::vector<int> perm = build_permutation(M, mol);
        const Eigen::MatrixXd  D    = build_ao_transform(M, perm, bs);
        return S * D;
    };

    // Rotation about z by angle θ
    auto rot_z = [](double theta) -> Eigen::Matrix3d {
        const double c = std::cos(theta), s = std::sin(theta);
        Eigen::Matrix3d R;
        R << c, -s, 0.0,
             s,  c, 0.0,
           0.0, 0.0, 1.0;
        return R;
    };

    const Eigen::MatrixXd SD_pi2 = build_SD(rot_z(M_PI / 2.0));
    const Eigen::MatrixXd SD_pi3 = build_SD(rot_z(M_PI / 3.0));

    // σv through xz-plane: flip y → diag(1, -1, 1)
    Eigen::Matrix3d R_sv = Eigen::Matrix3d::Identity();
    R_sv(1, 1) = -1.0;
    const Eigen::MatrixXd SD_sv = build_SD(R_sv);

    // Inversion for D∞h
    std::unique_ptr<Eigen::MatrixXd> SD_inv_ptr;
    if (is_Dinfh)
        SD_inv_ptr = std::make_unique<Eigen::MatrixXd>(
            build_SD(-Eigen::Matrix3d::Identity()));

    auto classify = [&](const Eigen::MatrixXd& C, std::vector<std::string>& lbl) {
        assign_for_linear(C, SD_pi2, SD_pi3, SD_sv, SD_inv_ptr.get(), lbl);
    };

    classify(calc._info._scf.alpha.mo_coefficients,
             calc._info._scf.alpha.mo_symmetry);

    if (calc._info._scf.is_uhf &&
        calc._info._scf.beta.mo_coefficients.cols() > 0)
    {
        classify(calc._info._scf.beta.mo_coefficients,
                 calc._info._scf.beta.mo_symmetry);
    }
}

// ── Assign irreps to MO columns using the character table ────────────────────
static void assign_for_channel(
    const Eigen::MatrixXd&        C,          // nbasis × nmo
    const Eigen::MatrixXd&        S,          // overlap matrix
    const std::vector<Eigen::MatrixXd>& SD,   // S · D_Rc for each class c
    const msym_character_table_t* ct,
    std::vector<std::string>&     labels)
{
    const int nb  = static_cast<int>(C.cols());
    const int nc  = ct->d;   // number of classes = number of irreps

    // Group order h = Σ_c classc[c]
    int h = 0;
    for (int c = 0; c < nc; ++c) h += ct->classc[c];

    const double* table = static_cast<const double*>(ct->table);

    labels.resize(nb);

    for (int i = 0; i < nb; ++i)
    {
        const Eigen::VectorXd ci = C.col(i);

        // Character of MO i under representative op of class c:
        //   χ_i(R_c) = cᵢᵀ (S D_{R_c}) cᵢ
        std::vector<double> chars(nc);
        for (int c = 0; c < nc; ++c)
            chars[c] = ci.dot(SD[c] * ci);

        // Projection weight for irrep Γ:
        //   w_Γ = (dΓ / h) Σ_c n_c χ^Γ_c χ_i(R_c)
        int    best_irrep = 0;
        double best_w    = -1.0;

        for (int gamma = 0; gamma < nc; ++gamma)
        {
            double w = 0.0;
            for (int c = 0; c < nc; ++c)
                w += ct->classc[c] * table[gamma * nc + c] * chars[c];
            w *= ct->s[gamma].d / static_cast<double>(h);
            w = std::abs(w);

            if (w > best_w)
            {
                best_w    = w;
                best_irrep = gamma;
            }
        }
        labels[i] = ct->s[best_irrep].name;
    }
}

} // anonymous namespace

// ─── Public entry point ───────────────────────────────────────────────────────

void HartreeFock::Symmetry::assign_mo_symmetry(HartreeFock::Calculator& calculator)
{
    if (!calculator._molecule._symmetry)   return;
    if (!calculator._info._is_converged)   return;

    // Linear molecules (C∞v / D∞h) use a dedicated handler.
    const std::string& pg = calculator._molecule._point_group;
    if (pg.find("inf") != std::string::npos)
    {
        assign_mo_symmetry_linear(calculator);
        return;
    }

    // ── Rebuild msym context on the symmetrized coordinates ──────────────────
    // molecule.standard is in Angstrom (symmetrized, pre-alignment frame).
    // Basis function centers are consistent with this frame (via _standard = Bohr).
    HartreeFock::Symmetry::SymmetryContext ctx;
    HartreeFock::Symmetry::SymmetryElements atoms(calculator._molecule.natoms);

    for (std::size_t i = 0; i < calculator._molecule.natoms; ++i)
    {
        atoms.data()[i].m    = calculator._molecule.atomic_masses[i];
        atoms.data()[i].n    = calculator._molecule.atomic_numbers[i];
        atoms.data()[i].v[0] = calculator._molecule.standard(i, 0);
        atoms.data()[i].v[1] = calculator._molecule.standard(i, 1);
        atoms.data()[i].v[2] = calculator._molecule.standard(i, 2);
    }

    if (MSYM_SUCCESS != msymSetElements(ctx.get(), atoms.size(), atoms.data()))
        throw std::runtime_error("assign_mo_symmetry: msymSetElements failed");

    // Re-detect symmetry (do NOT align axes — keep same frame as basis centers)
    if (MSYM_SUCCESS != msymFindSymmetry(ctx.get()))
        throw std::runtime_error("assign_mo_symmetry: msymFindSymmetry failed");

    // ── Obtain character table ────────────────────────────────────────────────
    const msym_character_table_t* ct = nullptr;
    if (MSYM_SUCCESS != msymGetCharacterTable(ctx.get(), &ct) || ct == nullptr)
        throw std::runtime_error("assign_mo_symmetry: msymGetCharacterTable failed");

    const int nc = ct->d;   // number of conjugacy classes (= number of irreps for finite groups)

    // ── For each conjugacy class: build 3×3 matrix, atom permutation, AO transform ──
    std::vector<Eigen::MatrixXd> SD(nc);
    for (int c = 0; c < nc; ++c)
    {
        const Eigen::Matrix3d M    = sop_to_matrix(*ct->sops[c]);
        const std::vector<int> perm = build_permutation(M, calculator._molecule);
        const Eigen::MatrixXd D     = build_ao_transform(M, perm, calculator._shells);
        SD[c] = calculator._overlap * D;
    }

    // ── Assign irreps ─────────────────────────────────────────────────────────
    const Eigen::MatrixXd& S = calculator._overlap;
    (void)S;

    assign_for_channel(
        calculator._info._scf.alpha.mo_coefficients,
        calculator._overlap,
        SD, ct,
        calculator._info._scf.alpha.mo_symmetry);

    if (calculator._info._scf.is_uhf &&
        calculator._info._scf.beta.mo_coefficients.cols() > 0)
    {
        assign_for_channel(
            calculator._info._scf.beta.mo_coefficients,
            calculator._overlap,
            SD, ct,
            calculator._info._scf.beta.mo_symmetry);
    }
}
