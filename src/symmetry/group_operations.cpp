#include "group_operations.h"

#include <cmath>
#include <cstring>
#include <map>
#include <numbers>
#include <utility>

#include "external/libmsym/install/include/libmsym/msym.h"
#include "wrapper.h"

// The 3×3-operation, angular-coefficient, nuclear-permutation, and AO-transform
// helpers below are the SAME math as src/symmetry/mo_symmetry.cpp. They are
// duplicated here (rather than shared yet) so this module can be validated as a
// standalone unit before mo_symmetry.cpp is refactored to depend on it; once the
// O_R are trusted, mo_symmetry.cpp's anonymous-namespace copies can be deleted in
// favor of these. Keeping them byte-identical preserves the validation that
// build_sao_basis already provides for this math.

namespace
{
    static const int FACT[13] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600};

    static int multinomial(int n, int k0, int k1, int k2)
    {
        if (k0 < 0 || k1 < 0 || k2 < 0 || k0 + k1 + k2 != n)
            return 0;
        return FACT[n] / (FACT[k0] * FACT[k1] * FACT[k2]);
    }

    static double ipow(double x, int n)
    {
        if (n == 0)
            return 1.0;
        double r = 1.0;
        for (int i = 0; i < n; ++i)
            r *= x;
        return r;
    }

    // 3×3 Cartesian matrix for a libmsym symmetry operation.
    //   IDENTITY=0, PROPER=1, IMPROPER=2, REFLECTION=3, INVERSION=4
    static Eigen::Matrix3d sop_to_matrix(const msym_symmetry_operation_t &sop)
    {
        using M3d = Eigen::Matrix3d;
        using V3d = Eigen::Vector3d;

        switch (static_cast<int>(sop.type))
        {
        case 0:
            return M3d::Identity();
        case 4:
            return -M3d::Identity();
        case 3:
        {
            V3d n(sop.v[0], sop.v[1], sop.v[2]);
            n.normalize();
            return M3d::Identity() - 2.0 * n * n.transpose();
        }
        case 1:
        {
            V3d v(sop.v[0], sop.v[1], sop.v[2]);
            v.normalize();
            const double angle = 2.0 * std::numbers::pi * sop.power / sop.order;
            const double c = std::cos(angle), s = std::sin(angle);
            M3d K;
            K << 0, -v.z(), v.y(),
                v.z(), 0, -v.x(),
                -v.y(), v.x(), 0;
            return c * M3d::Identity() + (1.0 - c) * v * v.transpose() + s * K;
        }
        case 2:
        {
            V3d v(sop.v[0], sop.v[1], sop.v[2]);
            v.normalize();
            const double angle = 2.0 * std::numbers::pi * sop.power / sop.order;
            const double c = std::cos(angle), s = std::sin(angle);
            M3d K;
            K << 0, -v.z(), v.y(),
                v.z(), 0, -v.x(),
                -v.y(), v.x(), 0;
            const M3d Cn = c * M3d::Identity() + (1.0 - c) * v * v.transpose() + s * K;
            const M3d sigma_h = M3d::Identity() - 2.0 * v * v.transpose();
            return sigma_h * Cn;
        }
        default:
            return M3d::Identity();
        }
    }

    // Coefficient of the source Cartesian monomial (lx,ly,lz) projected onto the
    // target monomial (ax,ay,az) under the orthogonal operation M (M⁻¹ = Mᵀ).
    static double angular_coeff(const Eigen::Matrix3d &M,
                               int lx, int ly, int lz,
                               int ax, int ay, int az)
    {
        if (lx + ly + lz != ax + ay + az)
            return 0.0;
        if (lx + ly + lz == 0)
            return 1.0;

        const double cx[3] = {M(0, 0), M(1, 0), M(2, 0)};
        const double cy[3] = {M(0, 1), M(1, 1), M(2, 1)};
        const double cz[3] = {M(0, 2), M(1, 2), M(2, 2)};

        double result = 0.0;
        for (int i0 = 0; i0 <= lx; ++i0)
            for (int i1 = 0; i1 <= lx - i0; ++i1)
            {
                const int i2 = lx - i0 - i1;
                const double cx_term = multinomial(lx, i0, i1, i2) * ipow(cx[0], i0) * ipow(cx[1], i1) * ipow(cx[2], i2);

                for (int j0 = 0; j0 <= ly; ++j0)
                    for (int j1 = 0; j1 <= ly - j0; ++j1)
                    {
                        const int j2 = ly - j0 - j1;
                        const int k0 = ax - i0 - j0;
                        const int k1 = ay - i1 - j1;
                        const int k2 = az - i2 - j2;
                        if (k0 < 0 || k1 < 0 || k2 < 0 || k0 + k1 + k2 != lz)
                            continue;

                        const double cy_term = multinomial(ly, j0, j1, j2) * ipow(cy[0], j0) * ipow(cy[1], j1) * ipow(cy[2], j2);
                        const double cz_term = multinomial(lz, k0, k1, k2) * ipow(cz[0], k0) * ipow(cz[1], k1) * ipow(cz[2], k2);
                        result += cx_term * cy_term * cz_term;
                    }
            }
        return result;
    }

    // perm[a] = b if M maps atom a's nuclear position onto atom b's.
    static std::expected<std::vector<int>, std::string> build_permutation(
        const Eigen::Matrix3d &M, const HartreeFock::Molecule &mol, double tol = 0.25)
    {
        const int N = static_cast<int>(mol.natoms);
        std::vector<int> perm(N, -1);

        for (int a = 0; a < N; ++a)
        {
            const Eigen::Vector3d Mpa = M * mol.standard.row(a).transpose();
            for (int b = 0; b < N; ++b)
            {
                if (mol.atomic_numbers[a] != mol.atomic_numbers[b])
                    continue;
                if ((Mpa - mol.standard.row(b).transpose()).norm() < tol)
                {
                    perm[a] = b;
                    break;
                }
            }
            if (perm[a] == -1)
                return std::unexpected(
                    "build_group_operations: atom " + std::to_string(a) +
                    " has no permutation image under this operation");
        }
        return perm;
    }

    // nb×nb dense AO representation matrix for the operation (M, perm).
    // Same shell-correspondence rule as build_ao_transform: the k-th shell of
    // angular type l at atom a maps to the k-th shell of type l at atom perm[a].
    static Eigen::MatrixXd build_ao_transform(const Eigen::Matrix3d &M,
                                             const std::vector<int> &perm,
                                             const HartreeFock::Basis &basis)
    {
        const std::size_t nb = basis.nbasis();
        Eigen::MatrixXd D = Eigen::MatrixXd::Zero(nb, nb);

        const auto &bfs = basis._basis_functions;
        const auto &shells = basis._shells;

        std::map<std::pair<int, int>, std::vector<const HartreeFock::Shell *>> atom_l_shells;
        for (const auto &sh : shells)
        {
            const int atm = static_cast<int>(sh._atom_index);
            const int l = static_cast<int>(sh._shell);
            atom_l_shells[{atm, l}].push_back(&sh);
        }

        for (std::size_t mu = 0; mu < nb; ++mu)
        {
            const auto &cv_mu = bfs[mu];
            const int atom_a = static_cast<int>(cv_mu._shell->_atom_index);
            const int atom_b = perm[atom_a];
            const int lx = cv_mu._cartesian[0];
            const int ly = cv_mu._cartesian[1];
            const int lz = cv_mu._cartesian[2];
            const int l = lx + ly + lz;
            const double norm_mu = cv_mu._component_norm;

            const auto &src_list = atom_l_shells.at({atom_a, l});
            int shell_k = -1;
            for (int k = 0; k < static_cast<int>(src_list.size()); ++k)
                if (src_list[k] == cv_mu._shell)
                {
                    shell_k = k;
                    break;
                }

            const auto &tgt_list = atom_l_shells.at({atom_b, l});
            const HartreeFock::Shell *tgt_shell = tgt_list[shell_k];

            for (std::size_t nu = 0; nu < nb; ++nu)
            {
                const auto &cv_nu = bfs[nu];
                if (cv_nu._shell != tgt_shell)
                    continue;

                const int ax = cv_nu._cartesian[0];
                const int ay = cv_nu._cartesian[1];
                const int az = cv_nu._cartesian[2];

                const double c = angular_coeff(M, lx, ly, lz, ax, ay, az);
                if (std::abs(c) < 1e-14)
                    continue;

                D(nu, mu) = c * (norm_mu / cv_nu._component_norm);
            }
        }
        return D;
    }

    // Shell→shell permutation induced by the nuclear permutation `perm`: the k-th
    // shell of angular type L at atom a maps to the k-th shell of type L at atom
    // perm[a]. Same correspondence rule build_ao_transform uses, but tracked at the
    // shell level for the petite-list representative test. Indexed by position in
    // basis._shells. Returns std::nullopt-style error if any shell has no image
    // (group/basis inconsistency — should not happen for a symmetric molecule).
    static std::expected<std::vector<int>, std::string> build_shell_permutation(
        const std::vector<int> &perm, const HartreeFock::Basis &basis)
    {
        const auto &shells = basis._shells;
        const int nsh = static_cast<int>(shells.size());

        // (atom, L) -> ordered list of shell indices at that atom of that type.
        std::map<std::pair<int, int>, std::vector<int>> atom_l_shells;
        for (int s = 0; s < nsh; ++s)
        {
            const int atm = static_cast<int>(shells[s]._atom_index);
            const int l = static_cast<int>(shells[s]._shell);
            atom_l_shells[{atm, l}].push_back(s);
        }

        std::vector<int> shell_perm(nsh, -1);
        for (int s = 0; s < nsh; ++s)
        {
            const int atom_a = static_cast<int>(shells[s]._atom_index);
            const int atom_b = perm[atom_a];
            const int l = static_cast<int>(shells[s]._shell);

            const auto &src_list = atom_l_shells.at({atom_a, l});
            int k = -1;
            for (int idx = 0; idx < static_cast<int>(src_list.size()); ++idx)
                if (src_list[idx] == s)
                {
                    k = idx;
                    break;
                }

            const auto tgt_it = atom_l_shells.find({atom_b, l});
            if (k < 0 || tgt_it == atom_l_shells.end() ||
                k >= static_cast<int>(tgt_it->second.size()))
                return std::unexpected(
                    "build_group_operations: shell " + std::to_string(s) +
                    " has no image under this operation");
            shell_perm[s] = tgt_it->second[k];
        }
        return shell_perm;
    }

    // Human-readable label for a libmsym operation (diagnostics only).
    static std::string sop_label(const msym_symmetry_operation_t &sop)
    {
        switch (static_cast<int>(sop.type))
        {
        case 0:
            return "E";
        case 4:
            return "i";
        case 3:
            return "sigma";
        case 1:
            return "C" + std::to_string(sop.order) + "^" + std::to_string(sop.power);
        case 2:
            return "S" + std::to_string(sop.order) + "^" + std::to_string(sop.power);
        default:
            return "?";
        }
    }
} // namespace

std::expected<HartreeFock::Symmetry::GroupOperations, std::string>
HartreeFock::Symmetry::build_group_operations(HartreeFock::Calculator &calculator)
{
    GroupOperations result; // valid = false by default

    if (!calculator._molecule._symmetry)
        return result;

    const std::string &pg = calculator._molecule._point_group;
    if (pg == "C1" || pg.find("inf") != std::string::npos)
        return result;

    // ── Rebuild libmsym context on the symmetrized frame (NO axis alignment, NO
    //    subgroup selection — we want the FULL group's operations) ──────────────
    auto ctx_result = HartreeFock::Symmetry::SymmetryContext::create();
    if (!ctx_result)
        return std::unexpected("build_group_operations: " + ctx_result.error());
    HartreeFock::Symmetry::SymmetryContext ctx = std::move(*ctx_result);
    HartreeFock::Symmetry::SymmetryElements atoms(calculator._molecule.natoms);

    for (std::size_t i = 0; i < calculator._molecule.natoms; ++i)
    {
        atoms.data()[i].m = calculator._molecule.atomic_masses[i];
        atoms.data()[i].n = calculator._molecule.atomic_numbers[i];
        atoms.data()[i].v[0] = calculator._molecule.standard(i, 0);
        atoms.data()[i].v[1] = calculator._molecule.standard(i, 1);
        atoms.data()[i].v[2] = calculator._molecule.standard(i, 2);
    }

    if (MSYM_SUCCESS != msymSetElements(ctx.get(), atoms.size(), atoms.data()))
        return std::unexpected("build_group_operations: msymSetElements failed");
    if (MSYM_SUCCESS != msymFindSymmetry(ctx.get()))
        return std::unexpected("build_group_operations: msymFindSymmetry failed");

    // Pull the full list of symmetry operations directly (not via the character
    // table, which for Abelian groups gives one rep per class — here every class is
    // size 1, but for non-Abelian groups we want EVERY operation, so use the raw
    // operation list).
    int sopsl = 0;
    const msym_symmetry_operation_t *sops = nullptr;
    if (MSYM_SUCCESS != msymGetSymmetryOperations(ctx.get(), &sopsl, &sops) || sops == nullptr)
        return std::unexpected("build_group_operations: msymGetSymmetryOperations failed");
    if (sopsl <= 0)
        return result;

    result.operations.reserve(static_cast<std::size_t>(sopsl));
    for (int c = 0; c < sopsl; ++c)
    {
        const Eigen::Matrix3d M = sop_to_matrix(sops[c]);
        auto perm = build_permutation(M, calculator._molecule);
        if (!perm)
            return std::unexpected(perm.error());

        auto shell_perm = build_shell_permutation(*perm, calculator._shells);
        if (!shell_perm)
            return std::unexpected(shell_perm.error());

        GroupOperation op;
        op.label = sop_label(sops[c]);
        op.matrix = build_ao_transform(M, *perm, calculator._shells);
        op.shell_perm = std::move(*shell_perm);
        result.operations.push_back(std::move(op));
    }

    result.point_group = pg;
    result.order = static_cast<int>(result.operations.size());
    result.valid = true;
    return result;
}
