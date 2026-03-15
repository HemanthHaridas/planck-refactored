#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "mo_symmetry.h"
#include "wrapper.h"
#include "external/libmsym/install/include/libmsym/msym.h"
#include "io/logging.h"

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
//
// Shell-level correspondence: when atom a has multiple contracted shells of the
// same angular momentum l (e.g. O 1s and O 2s are both l=0), the k-th shell of
// type l at atom a maps to the k-th shell of type l at atom b = perm[a].  This
// prevents mixing of O 1s and O 2s under operations that fix atom a.
static Eigen::MatrixXd build_ao_transform(const Eigen::Matrix3d& M,
                                           const std::vector<int>& perm,
                                           const HartreeFock::Basis& basis)
{
    const std::size_t nb = basis.nbasis();
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(nb, nb);

    const auto& bfs    = basis._basis_functions;
    const auto& shells = basis._shells;

    // Precompute: for each (atom, l), the ordered list of shell pointers.
    // Used to find the k-th shell of angular type l at a given atom.
    std::map<std::pair<int,int>, std::vector<const HartreeFock::Shell*>> atom_l_shells;
    for (const auto& sh : shells)
    {
        const int atm = static_cast<int>(sh._atom_index);
        const int  l  = static_cast<int>(sh._shell);
        atom_l_shells[{atm, l}].push_back(&sh);
    }

    for (std::size_t mu = 0; mu < nb; ++mu)
    {
        const auto& cv_mu  = bfs[mu];
        const int   atom_a = static_cast<int>(cv_mu._shell->_atom_index);
        const int   atom_b = perm[atom_a];
        const int   lx     = cv_mu._cartesian[0];
        const int   ly     = cv_mu._cartesian[1];
        const int   lz     = cv_mu._cartesian[2];
        const int   l      = lx + ly + lz;
        const double norm_mu = cv_mu._component_norm;

        // Identify which shell (within the set of l-type shells at atom_a) μ belongs to.
        const auto& src_list = atom_l_shells.at({atom_a, l});
        int shell_k = -1;
        for (int k = 0; k < static_cast<int>(src_list.size()); ++k)
        {
            if (src_list[k] == cv_mu._shell) { shell_k = k; break; }
        }

        // Corresponding target shell: k-th l-type shell at atom_b.
        const auto& tgt_list = atom_l_shells.at({atom_b, l});
        const HartreeFock::Shell* tgt_shell = tgt_list[shell_k];

        for (std::size_t nu = 0; nu < nb; ++nu)
        {
            const auto& cv_nu = bfs[nu];
            if (cv_nu._shell != tgt_shell) continue;   // must come from the corresponding shell

            const int ax = cv_nu._cartesian[0];
            const int ay = cv_nu._cartesian[1];
            const int az = cv_nu._cartesian[2];

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

// ── Strip redundant E-type subscript "1" when no higher E exists ─────────────
//
// libmsym names the doubly-degenerate irrep "E1" (and "E1g"/"E1u") for groups
// like D3d, C3v, D3h, where there is only one E type.  The standard Mulliken
// convention omits the subscript number in that case ("Eg", "Eu", "E").
// Groups like D6h, D4h have both E1 and E2, so the digit must be kept.
//
// Rule: if the character table contains no irrep whose name starts with "E"
// followed by a digit ≥ 2, strip the leading "1" from every "E1..." label.
static void normalize_e_labels(const msym_character_table_t* ct,
                                std::vector<std::string>&     labels)
{
    const int nc = ct->d;
    for (int g = 0; g < nc; ++g)
    {
        const std::string name = ct->s[g].name;
        // "E2...", "E3..." → multiple E types present → keep numbering
        if (name.size() >= 2 && name[0] == 'E' && name[1] >= '2' && name[1] <= '9')
            return;
    }
    // Only "E1" (or plain "E") → strip the "1" subscript
    for (auto& lbl : labels)
        if (lbl.size() >= 2 && lbl[0] == 'E' && lbl[1] == '1')
            lbl = "E" + lbl.substr(2);
}

// ── Fix B1/B2 convention to match standard chemistry (σv = xz plane) ─────────
//
// libmsym labels B1 as the irrep symmetric under whichever σv plane it
// encounters first (typically the molecular plane).  The standard chemistry
// convention always uses the xz plane as σv, so B1 is symmetric under xz and
// B2 is symmetric under yz.  When the two conventions disagree we detect it by
// checking: if libmsym's B2 has character +1 under the xz reflection (normal
// ≈ y-hat), B1 and B2 are swapped → rename them in the label list.
static void fix_b1b2_convention(const msym_character_table_t* ct,
                                 std::vector<std::string>&     labels)
{
    const int    nc    = ct->d;
    const double* table = static_cast<const double*>(ct->table);

    // Find B1 and B2 irrep row indices (exact-name match for C2v labels).
    int b1_idx = -1, b2_idx = -1;
    for (int g = 0; g < nc; ++g)
    {
        const std::string name = ct->s[g].name;
        if (name == "B1") b1_idx = g;
        if (name == "B2") b2_idx = g;
    }
    if (b1_idx < 0 || b2_idx < 0) return;   // group has no separate B1/B2

    // Find the conjugacy class whose representative is a reflection in the xz
    // plane (normal ≈ ±y-hat).
    for (int c = 0; c < nc; ++c)
    {
        const msym_symmetry_operation_t* sop = ct->sops[c];
        if (static_cast<int>(sop->type) != 3) continue;   // not a reflection

        Eigen::Vector3d n(sop->v[0], sop->v[1], sop->v[2]);
        n.normalize();
        if (std::abs(std::abs(n[1]) - 1.0) > 0.1) continue;   // not y-hat

        // This is the xz-plane reflection.  Standard convention: B1 should be
        // +1 here.  If libmsym's B2 is +1 → they are swapped → correct.
        if (table[b2_idx * nc + c] > 0.5)
        {
            for (auto& lbl : labels)
            {
                if      (lbl == "B1") lbl = "B2";
                else if (lbl == "B2") lbl = "B1";
            }
        }
        break;
    }
}

// ── Per-shell Cartesian→Spherical pseudoinverse T⁺  [n_sph × n_cart] ─────────
//
// Cartesian order follows _cartesian_shell_order(L): lx descending, then ly.
// Spherical order: m = −L, −L+1, …, 0, …, +L  (libmsym convention).
//
// Libmsym convention (from msym.h comment):
//   pz = m=0,  px = m=+1,  py = m=-1
//   dz2 = m=0, dxz = m=+1, dyz = m=-1, dx2y2 = m=+2, dxy = m=-2
//
// T [n_cart × n_sph]:  column m = coefficients of φ_sph_m in the
//   component-norm-weighted Cartesian basis.
// T⁺ = (TᵀT)⁻¹ Tᵀ  [n_sph × n_cart]:  maps Cartesian AO coefficients
//   to spherical AO coefficients (discards the r²-contamination subspace for L≥2).
static Eigen::MatrixXd cart_to_sph_block(int L)
{
    if (L == 0)
        return Eigen::MatrixXd::Identity(1, 1);

    if (L == 1)
    {
        // Cartesian order: px(1,0,0)=col0, py(0,1,0)=col1, pz(0,0,1)=col2
        // Spherical order: m=-1=py, m=0=pz, m=+1=px
        // T is a 3×3 permutation → T⁺ = Tᵀ
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(3, 3);
        T(0, 1) = 1.0;   // m=-1 ← py
        T(1, 2) = 1.0;   // m= 0 ← pz
        T(2, 0) = 1.0;   // m=+1 ← px
        return T;
    }

    if (L == 2)
    {
        // Cartesian order: xx=0, xy=1, xz=2, yy=3, yz=4, zz=5
        // Spherical order: m=-2=dxy, m=-1=dyz, m=0=dz2, m=+1=dxz, m=+2=dx2y2
        //
        // T (6×5):
        //   col m=-2: T[xy,m=-2] = 1
        //   col m=-1: T[yz,m=-1] = 1
        //   col m= 0: T[xx,m=0]=-1/2, T[yy,m=0]=-1/2, T[zz,m=0]=1
        //   col m=+1: T[xz,m=+1] = 1
        //   col m=+2: T[xx,m=+2]=√3/2, T[yy,m=+2]=-√3/2
        //
        // TᵀT = diag(1, 1, 3/2, 1, 3/2)  →  (TᵀT)⁻¹ = diag(1,1,2/3,1,2/3)
        // T⁺ (5×6) = (TᵀT)⁻¹ Tᵀ:
        //   row m=-2:  [0,    1,  0,    0,    0,   0   ]
        //   row m=-1:  [0,    0,  0,    0,    1,   0   ]
        //   row m= 0:  [-1/3, 0,  0,   -1/3,  0,  2/3 ]
        //   row m=+1:  [0,    0,  1,    0,    0,   0   ]
        //   row m=+2:  [1/√3, 0,  0,  -1/√3,  0,   0  ]
        const double s3 = std::sqrt(3.0);
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(5, 6);
        //            xx      xy  xz    yy      yz  zz
        T(0, 1) = 1.0;                                        // m=-2: dxy
        T(1, 4) = 1.0;                                        // m=-1: dyz
        T(2, 0) = -1.0/3.0; T(2, 3) = -1.0/3.0; T(2, 5) = 2.0/3.0; // m=0: dz2
        T(3, 2) = 1.0;                                        // m=+1: dxz
        T(4, 0) = 1.0/s3;   T(4, 3) = -1.0/s3;              // m=+2: dx2y2
        return T;
    }

    if (L == 3)
    {
        // n_cart=10, n_sph=7
        // Cartesian: xxx=0 xxy=1 xxz=2 xyy=3 xyz=4 xzz=5 yyy=6 yyz=7 yzz=8 zzz=9
        // Spherical: m=-3..+3 → cols 0..6
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(10, 7);
        T(1,0)= 3; T(6,0)=-1;                        // m=-3: y(3x²-y²)
        T(4,1)= 1;                                    // m=-2: xyz
        T(8,2)= 4; T(1,2)=-1; T(6,2)=-1;            // m=-1: y(4z²-x²-y²)
        T(9,3)= 2; T(2,3)=-3; T(7,3)=-3;            // m= 0: z(2z²-3x²-3y²)
        T(5,4)= 4; T(0,4)=-1; T(3,4)=-1;            // m=+1: x(4z²-x²-y²)
        T(2,5)= 1; T(7,5)=-1;                        // m=+2: z(x²-y²)
        T(0,6)= 1; T(3,6)=-3;                        // m=+3: x(x²-3y²)
        return T.completeOrthogonalDecomposition().pseudoInverse();
    }

    if (L == 4)
    {
        // n_cart=15, n_sph=9
        // Cartesian: x⁴=0 x³y=1 x³z=2 x²y²=3 x²yz=4 x²z²=5 xy³=6 xy²z=7 xyz²=8
        //            xz³=9 y⁴=10 y³z=11 y²z²=12 yz³=13 z⁴=14
        // Spherical: m=-4..+4 → cols 0..8
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(15, 9);
        T(1,0)= 1; T(6,0)=-1;                                   // m=-4: xy(x²-y²)
        T(4,1)= 3; T(11,1)=-1;                                  // m=-3: yz(3x²-y²)
        T(8,2)= 6; T(1,2)=-1; T(6,2)=-1;                       // m=-2: xy(6z²-x²-y²)
        T(13,3)=4; T(4,3)=-3; T(11,3)=-3;                      // m=-1: yz(4z²-3x²-3y²)
        T(0,4)= 3; T(3,4)= 6; T(5,4)=-24;                      // m= 0: 3x⁴+6x²y²-24x²z²
        T(10,4)=3; T(12,4)=-24; T(14,4)= 8;                    //       +3y⁴-24y²z²+8z⁴
        T(9,5)= 4; T(2,5)=-3; T(7,5)=-3;                       // m=+1: xz(4z²-3x²-3y²)
        T(0,6)=-1; T(10,6)=1; T(5,6)= 6; T(12,6)=-6;          // m=+2: (x²-y²)(6z²-x²-y²)
        T(2,7)= 1; T(7,7)=-3;                                   // m=+3: xz(x²-3y²)
        T(0,8)= 1; T(3,8)=-6; T(10,8)=1;                       // m=+4: x⁴-6x²y²+y⁴
        return T.completeOrthogonalDecomposition().pseudoInverse();
    }

    if (L == 5)
    {
        // n_cart=21, n_sph=11
        // Cartesian: x⁵=0 x⁴y=1 x⁴z=2 x³y²=3 x³yz=4 x³z²=5 x²y³=6 x²y²z=7 x²yz²=8 x²z³=9
        //            xy⁴=10 xy³z=11 xy²z²=12 xyz³=13 xz⁴=14
        //            y⁵=15 y⁴z=16 y³z²=17 y²z³=18 yz⁴=19 z⁵=20
        // Spherical: m=-5..+5 → cols 0..10
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(21, 11);
        T(1,0)= 5; T(6,0)=-10; T(15,0)=1;                      // m=-5: y(5x⁴-10x²y²+y⁴)
        T(4,1)= 4; T(11,1)=-4;                                  // m=-4: 4xyz(x²-y²)
        T(8,2)=24; T(1,2)=-3; T(6,2)=-2;                       // m=-3: y(3x²-y²)(8z²-x²-y²)
        T(17,2)=-8; T(15,2)=1;
        T(13,3)=2; T(4,3)=-1; T(11,3)=-1;                      // m=-2: xyz(2z²-x²-y²)
        T(1,4)= 1; T(6,4)= 2; T(15,4)=1;                       // m=-1: y(x⁴+2x²y²+y⁴+8z⁴-12x²z²-12y²z²)
        T(19,4)=8; T(8,4)=-12; T(17,4)=-12;
        T(20,5)=8; T(9,5)=-40; T(18,5)=-40;                    // m= 0: z(8z⁴-40x²z²-40y²z²+15x⁴+30x²y²+15y⁴)
        T(2,5)=15; T(7,5)= 30; T(16,5)=15;
        T(0,6)= 1; T(3,6)= 2; T(10,6)= 1;                      // m=+1: x(x⁴+2x²y²+y⁴+8z⁴-12x³z²-12xy²z²)
        T(14,6)=8; T(5,6)=-12; T(12,6)=-12;
        T(9,7)= 2; T(2,7)=-1; T(16,7)=1; T(18,7)=-2;          // m=+2: z(x²-y²)(2z²-x²-y²)
        T(0,8)=-1; T(3,8)= 2; T(10,8)=3;                       // m=+3: x(x²-3y²)(8z²-x²-y²)
        T(5,8)= 8; T(12,8)=-24;
        T(2,9)= 1; T(7,9)=-6; T(16,9)=1;                       // m=+4: z(x⁴-6x²y²+y⁴)
        T(0,10)=1; T(3,10)=-10; T(10,10)=5;                    // m=+5: x(x⁴-10x²y²+5y⁴)
        return T.completeOrthogonalDecomposition().pseudoInverse();
    }

    throw std::runtime_error(
        "assign_mo_symmetry: Cartesian→Spherical transform not implemented for L=" +
        std::to_string(L) + " (max supported: L=5)");
}

// ── Does a point group have only 1D real irreps? ─────────────────────────────
//
// Groups with all-1D irreps in the real character table (no E, T, … types):
//   Ci, Cs             — always
//   Cn, Cnh, Cnv,      — only for n ≤ 2
//   Dn, Dnh, Dnd, Sn   — only for n ≤ 2
//
// Note: S6, C3, C3v, … are Abelian groups but have 2D real irreps ("E").
// We require all-1D so every MO gets a unique, unambiguous irrep label.
static bool is_all_1d_irreps(msym_point_group_type_t t, int n)
{
    switch (static_cast<int>(t))
    {
        case 2:  // Ci  — always 1D
        case 3:  // Cs  — always 1D
            return true;
        case 4:  // Cn
        case 5:  // Cnh
        case 6:  // Cnv
        case 7:  // Dn
        case 8:  // Dnh
        case 9:  // Dnd
        case 10: // Sn
            return n <= 2;
        default:
            return false;
    }
}

} // anonymous namespace

// ─── SAO basis construction ───────────────────────────────────────────────────
//
// Builds the unitary transform U whose columns are symmetry-adapted orbitals
// (SAOs), grouped by irreducible representation.  The SAOs are orthonormal by
// construction (modified Gram-Schmidt), so U^T S U = I and the Fock matrix
// transformed to the SAO basis is block-diagonal.
//
// Algorithm (for a non-linear Abelian group):
//   1. Rebuild libmsym context (same frame as assign_mo_symmetry — no axis align).
//   2. Select largest Abelian subgroup if the full group is non-Abelian.
//   3. Get the character table.  For Abelian groups, ct->d = h = group order,
//      and ct->sops[c] is the unique operation of conjugacy class c.
//   4. Build D_R[c] = AO representation matrix for each group operation c.
//   5. For each irrep g: P_Γ = (1/h) Σ_c χ_Γ(c) D_R[c].
//      Apply P_Γ to every AO unit vector; keep linearly independent images via
//      modified Gram-Schmidt (threshold 1e-8).
//   6. Verify SAO count == nbasis.
//   7. Assemble U, fill metadata (block_sizes, block_offsets, irrep_names).
//   8. Apply the same B1/B2 and E-label normalisations as assign_mo_symmetry.

HartreeFock::Symmetry::SAOBasis HartreeFock::Symmetry::build_sao_basis(HartreeFock::Calculator& calculator)
{
    SAOBasis result;   // valid = false by default

    if (!calculator._molecule._symmetry)
        return result;

    const std::string& pg = calculator._molecule._point_group;

    // Skip linear molecules — SAO blocking not implemented for infinite groups.
    if (pg.find("inf") != std::string::npos)
        return result;

    // Skip C1 — one trivial block, no benefit.
    if (pg == "C1")
        return result;

    const std::size_t nb = calculator._shells.nbasis();

    // ── Rebuild libmsym context (NO axis alignment) ───────────────────────────
    HartreeFock::Symmetry::SymmetryContext  ctx;
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
        throw std::runtime_error("build_sao_basis: msymSetElements failed");

    if (MSYM_SUCCESS != msymFindSymmetry(ctx.get()))
        throw std::runtime_error("build_sao_basis: msymFindSymmetry failed");

    // ── Select largest Abelian subgroup if needed ─────────────────────────────
    {
        msym_point_group_type_t pg_type;
        int pg_n = 0;
        if (MSYM_SUCCESS != msymGetPointGroupType(ctx.get(), &pg_type, &pg_n))
            throw std::runtime_error("build_sao_basis: msymGetPointGroupType failed");

        if (!is_all_1d_irreps(pg_type, pg_n))
        {
            int nsg = 0;
            const msym_subgroup_t* sgs = nullptr;
            if (MSYM_SUCCESS != msymGetSubgroups(ctx.get(), &nsg, &sgs))
                throw std::runtime_error("build_sao_basis: msymGetSubgroups failed");

            const msym_subgroup_t* best = nullptr;
            int best_order = 0;
            for (int k = 0; k < nsg; ++k)
            {
                if (is_all_1d_irreps(sgs[k].type, sgs[k].n) && sgs[k].order > best_order)
                {
                    best_order = sgs[k].order;
                    best       = &sgs[k];
                }
            }
            if (best != nullptr)
            {
                if (MSYM_SUCCESS != msymSelectSubgroup(ctx.get(), best))
                    throw std::runtime_error("build_sao_basis: msymSelectSubgroup failed");
            }
        }
    }

    // ── Register minimal basis functions to initialise the character table ────
    // msymGetCharacterTable requires msymSetBasisFunctions to have been called.
    // We register one s-type spherical harmonic per atom (the minimal valid set).
    // The character table is a group property and independent of this choice.
    {
        int nelems = 0;
        msym_element_t* melems = nullptr;
        if (MSYM_SUCCESS != msymGetElements(ctx.get(), &nelems, &melems))
            throw std::runtime_error("build_sao_basis: msymGetElements failed");

        std::vector<msym_basis_function_t> bfs(nelems);
        std::memset(bfs.data(), 0, nelems * sizeof(msym_basis_function_t));
        for (int i = 0; i < nelems; ++i)
        {
            bfs[i].element  = &melems[i];
            bfs[i].f.rsh.n  = 1;
            bfs[i].f.rsh.l  = 0;
            bfs[i].f.rsh.m  = 0;
            // type = 0 = MSYM_BASIS_TYPE_REAL_SPHERICAL_HARMONIC (already zeroed)
        }
        if (MSYM_SUCCESS != msymSetBasisFunctions(ctx.get(), nelems, bfs.data()))
            throw std::runtime_error("build_sao_basis: msymSetBasisFunctions failed");
    }

    // ── Get character table ───────────────────────────────────────────────────
    const msym_character_table_t* ct = nullptr;
    if (MSYM_SUCCESS != msymGetCharacterTable(ctx.get(), &ct) || ct == nullptr)
        throw std::runtime_error("build_sao_basis: msymGetCharacterTable failed");

    // For an Abelian group the number of irreps == group order h.
    const int h        = ct->d;
    const int n_irreps = h;
    const double* ctable = static_cast<const double*>(ct->table);

    // ── Overlap matrix S (needed for S-metric orthogonalisation) ─────────────
    const Eigen::MatrixXd& S = calculator._overlap;

    // ── Build AO representation matrices D_R for each group operation ─────────
    // ct->sops[c] = representative of conjugacy class c.  For Abelian groups
    // each class has exactly one element, so this gives all h operations.
    std::vector<Eigen::MatrixXd> D_ops(h);
    for (int c = 0; c < h; ++c)
    {
        const Eigen::Matrix3d  M    = sop_to_matrix(*ct->sops[c]);
        const std::vector<int> perm = build_permutation(M, calculator._molecule);
        D_ops[c] = build_ao_transform(M, perm, calculator._shells);
    }

    // ── Project and S-orthonormalise SAOs for each irrep ──────────────────────
    // Use the physical (overlap) inner product <u,v>_S = u^T S v so that the
    // resulting transform U satisfies U^T S U = I.  This is required for the
    // blocked SCF diagonalisation which assumes no X step per block.
    constexpr double NORM_THRESH = 1e-8;

    std::vector<Eigen::MatrixXd> sao_blocks(n_irreps);
    std::vector<int>             block_sizes(n_irreps, 0);

    for (int g = 0; g < n_irreps; ++g)
    {
        // Projection operator: P_Γ = (1/h) Σ_c χ_Γ(R_c) D_R[c]
        Eigen::MatrixXd P = Eigen::MatrixXd::Zero(nb, nb);
        for (int c = 0; c < h; ++c)
            P += ctable[g * h + c] * D_ops[c];
        P /= static_cast<double>(h);

        // Apply P to each AO unit vector e_μ = μ-th column of the identity.
        // Collect the non-zero images and build an S-orthonormal basis for the
        // irrep-Γ subspace via modified Gram-Schmidt with the S-inner product.
        Eigen::MatrixXd accepted(nb, nb);
        int n_accepted = 0;

        for (std::size_t mu = 0; mu < nb; ++mu)
        {
            // v = P * e_mu = mu-th column of P
            Eigen::VectorXd v = P.col(static_cast<Eigen::Index>(mu));

            // S-orthogonalise against already-accepted SAOs:
            //   remove component along each accepted_k in the S-metric
            for (int k = 0; k < n_accepted; ++k)
            {
                const double proj = accepted.col(k).dot(S * v);
                v -= proj * accepted.col(k);
            }

            // S-norm: ||v||_S = sqrt(v^T S v)
            const double nrm = std::sqrt(v.dot(S * v));
            if (nrm > NORM_THRESH)
            {
                v /= nrm;   // now v^T S v = 1
                accepted.col(n_accepted++) = v;
            }
        }

        sao_blocks[g] = accepted.leftCols(n_accepted);
        block_sizes[g] = n_accepted;
    }

    // ── Verify completeness ───────────────────────────────────────────────────
    int total = 0;
    for (int g = 0; g < n_irreps; ++g) total += block_sizes[g];
    if (total != static_cast<int>(nb))
        throw std::runtime_error(
            "build_sao_basis: SAO count mismatch: got " + std::to_string(total) +
            ", expected " + std::to_string(nb));

    // ── Assemble U, block offsets, irrep_names ────────────────────────────────
    result.transform       = Eigen::MatrixXd(nb, nb);
    result.sao_irrep_index .resize(nb);
    result.block_sizes     = block_sizes;
    result.block_offsets   .resize(n_irreps);
    result.irrep_names     .resize(n_irreps);

    int col = 0;
    for (int g = 0; g < n_irreps; ++g)
    {
        result.block_offsets[g] = col;
        result.irrep_names[g]   = ct->s[g].name;   // raw Mulliken label from libmsym

        for (int k = 0; k < block_sizes[g]; ++k)
        {
            result.transform.col(col)   = sao_blocks[g].col(k);
            result.sao_irrep_index[col] = g;
            ++col;
        }
    }

    // ── Apply E-label normalisation (strip "E1" → "E" when no E2/E3 exist) ────
    {
        bool has_e2_or_higher = false;
        for (const auto& nm : result.irrep_names)
            if (nm.size() >= 2 && nm[0] == 'E' && nm[1] >= '2' && nm[1] <= '9')
                { has_e2_or_higher = true; break; }

        if (!has_e2_or_higher)
            for (auto& nm : result.irrep_names)
                if (nm.size() >= 2 && nm[0] == 'E' && nm[1] == '1')
                    nm = "E" + nm.substr(2);
    }

    // ── Apply B1/B2 convention fix ────────────────────────────────────────────
    // Standard chemistry: B1 is symmetric under the xz-plane reflection (y-hat normal).
    // If libmsym labels B2 as +1 under the xz reflection, swap the name strings.
    // Only the names are swapped; the SAO columns (and thus block ordering) remain
    // unchanged — this is intentional since the SAOs themselves are correct.
    {
        int b1_idx = -1, b2_idx = -1;
        for (int g = 0; g < n_irreps; ++g)
        {
            if (result.irrep_names[g] == "B1") b1_idx = g;
            if (result.irrep_names[g] == "B2") b2_idx = g;
        }
        if (b1_idx >= 0 && b2_idx >= 0)
        {
            for (int c = 0; c < h; ++c)
            {
                const msym_symmetry_operation_t* sop = ct->sops[c];
                if (static_cast<int>(sop->type) != 3) continue;   // not a reflection
                Eigen::Vector3d n(sop->v[0], sop->v[1], sop->v[2]);
                n.normalize();
                if (std::abs(std::abs(n[1]) - 1.0) > 0.1) continue;   // not y-hat

                // xz reflection found.  Standard B1 must be +1 here.
                if (ctable[b2_idx * h + c] > 0.5)
                    std::swap(result.irrep_names[b1_idx], result.irrep_names[b2_idx]);
                break;
            }
        }
    }

    result.valid = true;
    return result;
}

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

    // ── Select largest Abelian subgroup if the full group is non-Abelian ─────
    //
    // Standard quantum chemistry uses Abelian groups so every MO gets a unique
    // 1D irrep label.  For non-Abelian groups (D3d, Td, Oh, …) we find and
    // activate the largest Abelian subgroup before registering basis functions.
    {
        msym_point_group_type_t pg_type;
        int pg_n = 0;
        if (MSYM_SUCCESS != msymGetPointGroupType(ctx.get(), &pg_type, &pg_n))
            throw std::runtime_error("assign_mo_symmetry: msymGetPointGroupType failed");

        if (!is_all_1d_irreps(pg_type, pg_n))
        {
            int nsg = 0;
            const msym_subgroup_t* sgs = nullptr;
            if (MSYM_SUCCESS != msymGetSubgroups(ctx.get(), &nsg, &sgs))
                throw std::runtime_error("assign_mo_symmetry: msymGetSubgroups failed");

            const msym_subgroup_t* best = nullptr;
            int best_order = 0;
            for (int k = 0; k < nsg; ++k)
            {
                if (is_all_1d_irreps(sgs[k].type, sgs[k].n) && sgs[k].order > best_order)
                {
                    best_order = sgs[k].order;
                    best       = &sgs[k];
                }
            }

            if (best != nullptr)
            {
                // Capture name BEFORE SelectSubgroup may invalidate the pointer
                const std::string sg_name = best->name;

                if (MSYM_SUCCESS != msymSelectSubgroup(ctx.get(), best))
                    throw std::runtime_error("assign_mo_symmetry: msymSelectSubgroup failed");

                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "MO Symmetry :",
                    "Using Abelian subgroup " + sg_name +
                    " of " + calculator._molecule._point_group + " for MO labels");
            }
        }
        else
        {
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "MO Symmetry :",
                "Using point group " + calculator._molecule._point_group +
                " for MO labels");
        }
    }

    // Get the internal element array (needed to set element pointers on BFs)
    int nelems = 0;
    msym_element_t* melems = nullptr;
    if (MSYM_SUCCESS != msymGetElements(ctx.get(), &nelems, &melems))
        throw std::runtime_error("assign_mo_symmetry: msymGetElements failed");

    // ── Build Cart→Sph transform T⁺ and libmsym basis function array ─────────
    //
    // T⁺ is block-diagonal; each shell s contributes a (2Ls+1) × n_cart_s block.
    // The libmsym basis functions are ordered shell-by-shell, m = −L … +L.
    // The 'id' field of each BF stores its index in our ordering, so that after
    // msymGetBasisFunctions we can map internal → our ordering for wf reindexing.

    const auto& shells      = calculator._shells._shells;
    const int   n_cart_total = static_cast<int>(calculator._shells.nbasis());

    int n_sph_total = 0;
    for (const auto& sh : shells)
        n_sph_total += 2 * static_cast<int>(sh._shell) + 1;

    // Block-diagonal T⁺  [n_sph_total × n_cart_total]
    Eigen::MatrixXd T_cs = Eigen::MatrixXd::Zero(n_sph_total, n_cart_total);

    std::vector<msym_basis_function_t> bfs(n_sph_total);
    std::memset(bfs.data(), 0, n_sph_total * sizeof(msym_basis_function_t));

    // Track shell count per (atom, L) to give each contracted shell a unique n
    std::map<std::pair<int,int>, int> shell_n_counter;

    int sph_row  = 0;
    int cart_col = 0;
    int bf_idx   = 0;

    for (std::size_t si = 0; si < shells.size(); ++si)
    {
        const auto& sh     = shells[si];
        const int   L      = static_cast<int>(sh._shell);
        const int   n_cart = (L + 1) * (L + 2) / 2;
        const int   n_sph  = 2 * L + 1;
        const int   atom   = static_cast<int>(sh._atom_index);

        // Unique principal quantum number per (atom, L) — required when the same
        // atom has multiple contracted shells of the same angular momentum.
        const int shell_n = ++shell_n_counter[{atom, L}];

        // Place the per-shell T⁺ block
        T_cs.block(sph_row, cart_col, n_sph, n_cart) = cart_to_sph_block(L);

        // Fill libmsym BFs for this shell (m: −L … +L)
        for (int m = -L; m <= L; ++m, ++bf_idx)
        {
            // type field = 0 = MSYM_BASIS_TYPE_REAL_SPHERICAL_HARMONIC (already zeroed by memset)
            bfs[bf_idx].id      = reinterpret_cast<void*>(static_cast<std::intptr_t>(bf_idx));
            bfs[bf_idx].element = &melems[atom];
            bfs[bf_idx].f.rsh.n = L + shell_n;  // n >= l+1 required by libmsym
            bfs[bf_idx].f.rsh.l = L;
            bfs[bf_idx].f.rsh.m = m;
        }

        sph_row  += n_sph;
        cart_col += n_cart;
    }

    // ── Register basis functions ──────────────────────────────────────────────
    if (MSYM_SUCCESS != msymSetBasisFunctions(ctx.get(), n_sph_total, bfs.data()))
        throw std::runtime_error("assign_mo_symmetry: msymSetBasisFunctions failed");

    // ── Obtain character table ────────────────────────────────────────────────
    const msym_character_table_t* ct = nullptr;
    if (MSYM_SUCCESS != msymGetCharacterTable(ctx.get(), &ct) || ct == nullptr)
        throw std::runtime_error("assign_mo_symmetry: msymGetCharacterTable failed");
    const int n_species = ct->d;

    // ── Build reindex map: our_bf_idx → internal_bf_idx ──────────────────────
    // After msymSetBasisFunctions the context may reorder basis functions
    // internally.  We tagged each BF with its original index in the 'id' field.
    int             mbfsl = 0;
    msym_basis_function_t* mbfs = nullptr;
    if (MSYM_SUCCESS != msymGetBasisFunctions(ctx.get(), &mbfsl, &mbfs))
        throw std::runtime_error("assign_mo_symmetry: msymGetBasisFunctions failed");

    // to_internal[our_idx] = internal_idx
    std::vector<int> to_internal(n_sph_total);
    for (int j = 0; j < mbfsl; ++j)
    {
        const int our_idx = static_cast<int>(
            reinterpret_cast<std::intptr_t>(mbfs[j].id));
        to_internal[our_idx] = j;
    }

    // ── Classify MOs ─────────────────────────────────────────────────────────
    //
    // For each MO column c_i (Cartesian AO coefficients):
    //   1. Transform to spherical basis: d_i = T⁺ c_i
    //   2. Reindex to internal libmsym ordering
    //   3. Call msymSymmetrySpeciesComponents → component weights per species
    //   4. Label = species with largest weight
    auto classify = [&](const Eigen::MatrixXd& C, std::vector<std::string>& labels)
    {
        const int n_mo = static_cast<int>(C.cols());

        // C_sph[n_sph_total × n_mo]
        const Eigen::MatrixXd C_sph = T_cs * C;

        labels.resize(n_mo);
        std::vector<double> wf(mbfsl, 0.0);
        std::vector<double> comp(n_species, 0.0);

        for (int i = 0; i < n_mo; ++i)
        {
            // Reorder coefficients to internal libmsym basis ordering
            for (int k = 0; k < n_sph_total; ++k)
                wf[to_internal[k]] = C_sph(k, i);

            if (MSYM_SUCCESS != msymSymmetrySpeciesComponents(
                    ctx.get(), mbfsl, wf.data(), n_species, comp.data()))
                throw std::runtime_error(
                    "assign_mo_symmetry: msymSymmetrySpeciesComponents failed for MO " +
                    std::to_string(i));

            const int best = static_cast<int>(
                std::max_element(comp.begin(), comp.end()) - comp.begin());
            labels[i] = ct->s[best].name;
        }
    };

    classify(calculator._info._scf.alpha.mo_coefficients,
             calculator._info._scf.alpha.mo_symmetry);
    normalize_e_labels(ct, calculator._info._scf.alpha.mo_symmetry);
    fix_b1b2_convention(ct, calculator._info._scf.alpha.mo_symmetry);

    if (calculator._info._scf.is_uhf &&
        calculator._info._scf.beta.mo_coefficients.cols() > 0)
    {
        classify(calculator._info._scf.beta.mo_coefficients,
                 calculator._info._scf.beta.mo_symmetry);
        normalize_e_labels(ct, calculator._info._scf.beta.mo_symmetry);
        fix_b1b2_convention(ct, calculator._info._scf.beta.mo_symmetry);
    }
}
