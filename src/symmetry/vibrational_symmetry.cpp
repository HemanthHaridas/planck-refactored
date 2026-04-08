#include "vibrational_symmetry.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "wrapper.h"

namespace
{
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
            const double angle = 2.0 * M_PI * sop.power / sop.order;
            const double c = std::cos(angle);
            const double s = std::sin(angle);
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
            const double angle = 2.0 * M_PI * sop.power / sop.order;
            const double c = std::cos(angle);
            const double s = std::sin(angle);
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

    static bool is_all_1d_irreps(msym_point_group_type_t t, int n)
    {
        switch (static_cast<int>(t))
        {
        case 2:
        case 3:
            return true;
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
            return n <= 2;
        case 9:
            return n <= 1;
        case 10:
            return false;
        default:
            return false;
        }
    }

    static void normalize_e_labels(const msym_character_table_t *ct,
                                   std::vector<std::string> &labels)
    {
        const int nc = ct->d;
        for (int g = 0; g < nc; ++g)
        {
            const std::string name = ct->s[g].name;
            if (name.size() >= 2 && name[0] == 'E' && name[1] >= '2' && name[1] <= '9')
                return;
        }
        for (auto &lbl : labels)
            if (lbl.size() >= 2 && lbl[0] == 'E' && lbl[1] == '1')
                lbl = "E" + lbl.substr(2);
    }

    static void fix_b1b2_convention(const msym_character_table_t *ct,
                                    std::vector<std::string> &labels)
    {
        const int nc = ct->d;
        const double *table = static_cast<const double *>(ct->table);

        int b1_idx = -1;
        int b2_idx = -1;
        for (int g = 0; g < nc; ++g)
        {
            const std::string name = ct->s[g].name;
            if (name == "B1")
                b1_idx = g;
            if (name == "B2")
                b2_idx = g;
        }
        if (b1_idx < 0 || b2_idx < 0)
            return;

        for (int c = 0; c < nc; ++c)
        {
            const msym_symmetry_operation_t *sop = ct->sops[c];
            if (static_cast<int>(sop->type) != 3)
                continue;

            Eigen::Vector3d n(sop->v[0], sop->v[1], sop->v[2]);
            n.normalize();
            if (std::abs(std::abs(n[1]) - 1.0) > 0.1)
                continue;

            if (table[b2_idx * nc + c] > 0.5)
            {
                for (auto &lbl : labels)
                {
                    if (lbl == "B1")
                        lbl = "B2";
                    else if (lbl == "B2")
                        lbl = "B1";
                }
            }
            break;
        }
    }

    static int find_mapped_atom(const HartreeFock::Molecule &mol,
                                const Eigen::Vector3d &transformed,
                                int atomic_number,
                                std::vector<bool> &used)
    {
        int best = -1;
        double best_dist2 = std::numeric_limits<double>::max();
        for (std::size_t b = 0; b < mol.natoms; ++b)
        {
            if (used[b])
                continue;
            if (static_cast<int>(mol.atomic_numbers[b]) != atomic_number)
                continue;

            const Eigen::Vector3d rb = mol.standard.row(static_cast<int>(b)).transpose();
            const double dist2 = (rb - transformed).squaredNorm();
            if (dist2 < best_dist2)
            {
                best_dist2 = dist2;
                best = static_cast<int>(b);
            }
        }

        if (best >= 0 && best_dist2 < 1.0e-8)
        {
            used[static_cast<std::size_t>(best)] = true;
            return best;
        }
        return -1;
    }

    static Eigen::MatrixXd build_cartesian_representation(
        const HartreeFock::Molecule &mol,
        const msym_symmetry_operation_t &sop)
    {
        const int n3 = static_cast<int>(3 * mol.natoms);
        Eigen::MatrixXd rep = Eigen::MatrixXd::Zero(n3, n3);
        const Eigen::Matrix3d M = sop_to_matrix(sop);
        std::vector<bool> used(mol.natoms, false);

        for (std::size_t a = 0; a < mol.natoms; ++a)
        {
            const Eigen::Vector3d ra = mol.standard.row(static_cast<int>(a)).transpose();
            const Eigen::Vector3d transformed = M * ra;
            const int b = find_mapped_atom(
                mol,
                transformed,
                static_cast<int>(mol.atomic_numbers[a]),
                used);
            if (b < 0)
                throw std::runtime_error("vibrational symmetry: failed to map transformed atom");

            rep.block<3, 3>(3 * b, 3 * static_cast<int>(a)) = M;
        }

        return rep;
    }
} // namespace

std::vector<std::string> HartreeFock::Symmetry::assign_vibrational_symmetry(
    const HartreeFock::Calculator &calc,
    const Eigen::MatrixXd &normal_modes)
{
    if (!calc._molecule._symmetry)
        return {};
    if (normal_modes.cols() == 0)
        return {};

    const std::string &pg = calc._molecule._point_group;
    if (pg.find("inf") != std::string::npos)
        return {};

    HartreeFock::Symmetry::SymmetryContext ctx;
    HartreeFock::Symmetry::SymmetryElements atoms(calc._molecule.natoms);

    for (std::size_t i = 0; i < calc._molecule.natoms; ++i)
    {
        atoms.data()[i].m = calc._molecule.atomic_masses[i];
        atoms.data()[i].n = calc._molecule.atomic_numbers[i];
        atoms.data()[i].v[0] = calc._molecule.standard(i, 0);
        atoms.data()[i].v[1] = calc._molecule.standard(i, 1);
        atoms.data()[i].v[2] = calc._molecule.standard(i, 2);
    }

    if (MSYM_SUCCESS != msymSetElements(ctx.get(), atoms.size(), atoms.data()))
        return {};
    if (MSYM_SUCCESS != msymFindSymmetry(ctx.get()))
        return {};

    msym_point_group_type_t pg_type;
    int pg_n = 0;
    if (MSYM_SUCCESS != msymGetPointGroupType(ctx.get(), &pg_type, &pg_n))
        return {};

    if (!is_all_1d_irreps(pg_type, pg_n))
    {
        int nsg = 0;
        const msym_subgroup_t *sgs = nullptr;
        if (MSYM_SUCCESS != msymGetSubgroups(ctx.get(), &nsg, &sgs))
            return {};

        const msym_subgroup_t *best = nullptr;
        int best_order = 0;
        for (int k = 0; k < nsg; ++k)
        {
            if (is_all_1d_irreps(sgs[k].type, sgs[k].n) && sgs[k].order > best_order)
            {
                best_order = sgs[k].order;
                best = &sgs[k];
            }
        }

        if (best == nullptr)
            return {};
        if (MSYM_SUCCESS != msymSelectSubgroup(ctx.get(), best))
            return {};
    }

    const msym_character_table_t *ct = nullptr;
    if (MSYM_SUCCESS != msymGetCharacterTable(ctx.get(), &ct))
        return {};

    const int nc = ct->d;
    if (nc <= 0)
        return {};

    std::vector<Eigen::MatrixXd> reps;
    reps.reserve(static_cast<std::size_t>(nc));
    for (int c = 0; c < nc; ++c)
    {
        reps.push_back(build_cartesian_representation(calc._molecule, *ct->sops[c]));
    }

    std::vector<std::string> labels(static_cast<std::size_t>(normal_modes.cols()));
    std::vector<std::string> irrep_names(static_cast<std::size_t>(nc));
    for (int g = 0; g < nc; ++g)
        irrep_names[static_cast<std::size_t>(g)] = ct->s[g].name;
    normalize_e_labels(ct, irrep_names);
    fix_b1b2_convention(ct, irrep_names);

    const double *table = static_cast<const double *>(ct->table);
    for (int mode = 0; mode < normal_modes.cols(); ++mode)
    {
        const Eigen::VectorXd v = normal_modes.col(mode).normalized();
        int best_irrep = -1;
        double best_err = std::numeric_limits<double>::max();

        for (int g = 0; g < nc; ++g)
        {
            double err = 0.0;
            for (int c = 0; c < nc; ++c)
            {
                const double observed = v.dot(reps[static_cast<std::size_t>(c)] * v);
                const double expected = table[g * nc + c];
                const double diff = observed - expected;
                err += diff * diff;
            }
            if (err < best_err)
            {
                best_err = err;
                best_irrep = g;
            }
        }

        labels[static_cast<std::size_t>(mode)] =
            (best_irrep >= 0 && best_err < 0.25 * nc)
                ? irrep_names[static_cast<std::size_t>(best_irrep)]
                : "?";
    }

    return labels;
}
