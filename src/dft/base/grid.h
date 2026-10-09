#ifndef DFT_GRID_H
#define DFT_GRID_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <map>
#include <iterator>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "angular.h"
#include "base/types.h"
#include "radial.h"

namespace DFT
{

    // ORCA-inspired XC grid quality presets.
    //
    // The angular scheme and IntAcc values follow the ORCA 6 manual tables for
    // AngularGrid / XC IntAcc. The radial rule here still uses the existing
    // Treutler-Ahlrichs M4 mapping from radial.h, and the five-region pruning
    // cutoffs are heuristic but ORCA-like.
    enum class GridLevel
    {
        Coarse,
        Normal,
        Fine,
        UltraFine
    };

    struct GridPreset
    {
        GridLevel level;
        int angular_scheme;
        double int_acc;
        int radial_row_factor;
        bool reduce_light_atoms;
        // PySCF grid level (gen_grid.Grids.level, 0-9) whose RAD_GRIDS row gives
        // the radial point count. Coarse/Normal/Fine/UltraFine map to 1/3/5/7,
        // putting Normal on PySCF's own default (level 3).
        int pyscf_level;
    };

    // Named schedules are immutable and versioned.  The candidates below are
    // G3 measurement controls; LegacyOrcaIndexV1 remains the production
    // default.  G43V4QualifiedV1 is a separately qualified opt-in policy;
    // it is deliberately not a default-policy migration.  In particular,
    // PySCFNWChem213 is a literal PySCF 2.13 `nwchem_prune` candidate, not a
    // claim of force convergence.
    enum class GridPruningPolicy
    {
        LegacyOrcaIndexV1,
        LegacyOrcaNoLightReductionV1,
        LegacyOrcaUnprunedMaxV1,
        PySCFNWChem213,
        G43V4QualifiedV1
    };

    // The request is the small, user-facing part of grid selection.  It is
    // intentionally independent of a molecule: resolution must not inspect a
    // changing density or geometry, otherwise a numerical displacement could
    // silently use a different quadrature rule.
    struct GridRequest
    {
        GridLevel level = GridLevel::Normal;
        // Zero retains the selected policy's shell schedule.  A supported
        // positive Lebedev size is an unpruned, uniform angular override.
        int angular_points = 0;
        // Zero uses the element/quality radial table.  A positive value is a
        // uniform per-atom shell count; it is deliberately independent of
        // density and geometry so displaced calculations retain one rule.
        int radial_points = 0;
        GridPruningPolicy pruning_policy = GridPruningPolicy::LegacyOrcaIndexV1;
    };

    // An immutable description of every geometry-independent quadrature
    // choice.  MolecularGrid below owns only the geometry-dependent points and
    // weights; consumers retain this specification to prove that SCF, response
    // and displaced calculations used the same rule.
    struct ResolvedGridSpec
    {
        GridRequest request;
        GridPreset preset;
        GridPruningPolicy pruning_policy = GridPruningPolicy::LegacyOrcaIndexV1;
        int radial_override = 0; // Reserved for G2; zero means preset count.
        bool reduce_light_atoms = true;
        std::string radial_rule = "treutler-ahlrichs-m4-xi-pyscf-2.13";
        std::string partition_rule = "original-becke-treutler-bragg";
        std::string policy_name = "legacy-orca-index-v1";

        [[nodiscard]] std::string fingerprint() const
        {
            return policy_name + ";v=1;level=" + std::to_string(static_cast<int>(request.level)) +
                   ";pyscf_level=" + std::to_string(preset.pyscf_level) +
                   ";angular_override=" + std::to_string(request.angular_points) +
                   ";radial_override=" + std::to_string(radial_override) +
                   ";light_atoms=" + (reduce_light_atoms ? "reduced" : "full") +
                   ";radial=" + radial_rule + ";partition=" + partition_rule;
        }

        // All current fields are resolver-owned identifiers or integers, so no
        // escaping is needed.  Keep this compact record separate from runtime
        // MolecularGrid arrays; it is suitable for an audit ledger or cache
        // key report without serializing geometry-dependent data.
        [[nodiscard]] std::string json() const
        {
            return "{\"policy\":\"" + policy_name + "\",\"version\":1,\"level\":" +
                   std::to_string(static_cast<int>(request.level)) +
                   ",\"pyscf_level\":" + std::to_string(preset.pyscf_level) +
                   ",\"angular_override\":" + std::to_string(request.angular_points) +
                   ",\"radial_override\":" + std::to_string(radial_override) +
                   ",\"light_atoms\":\"" + (reduce_light_atoms ? "reduced" : "full") +
                   "\",\"radial\":\"" + radial_rule + "\",\"partition\":\"" +
                   partition_rule + "\"}";
        }
    };

    struct MolecularGrid
    {
        Eigen::MatrixXd points;            // N x 4 -> x, y, z, weight  (Bohr / quadrature weight)
        Eigen::VectorXi owner;             // Generating atom index for each point
        Eigen::VectorXd atomic_weights;    // Unpartitioned atomic-grid weights
        Eigen::VectorXd partition_weights; // Becke partition weight of owner atom
    };

    namespace detail
    {

        inline std::expected<double, std::string> xc_intacc_for_scheme(int angular_scheme)
        {
            switch (angular_scheme)
            {
            case 1:
                return 4.004;
            case 2:
                return 4.004;
            case 3:
                return 4.159;
            case 4:
                return 4.388;
            case 5:
                return 4.629;
            case 6:
                return 4.959;
            case 7:
                return 4.959;
            default:
                return std::unexpected(
                    "xc_intacc_for_scheme: unsupported angular scheme " +
                    std::to_string(angular_scheme));
            }
        }

        inline std::expected<std::array<int, 5>, std::string> angular_shells_for_scheme(int angular_scheme)
        {
            switch (angular_scheme)
            {
            case 1:
                return std::array<int, 5>{14, 26, 50, 50, 26};
            case 2:
                return std::array<int, 5>{14, 26, 50, 110, 50};
            case 3:
                return std::array<int, 5>{26, 50, 110, 194, 110};
            case 4:
                return std::array<int, 5>{26, 110, 194, 302, 194};
            case 5:
                return std::array<int, 5>{26, 194, 302, 434, 302};
            case 6:
                return std::array<int, 5>{50, 302, 434, 590, 434};
            case 7:
                return std::array<int, 5>{110, 434, 590, 770, 590};
            default:
                return std::unexpected(
                    "angular_shells_for_scheme: unsupported angular scheme " +
                    std::to_string(angular_scheme));
            }
        }

        inline std::expected<int, std::string> periodic_row(int Z)
        {
            if (Z <= 0)
                return std::unexpected(
                    "periodic_row: atomic number must be positive, got " + std::to_string(Z));
            if (Z <= 2)
                return 1;
            if (Z <= 10)
                return 2;
            if (Z <= 18)
                return 3;
            if (Z <= 36)
                return 4;
            if (Z <= 54)
                return 5;
            if (Z <= 86)
                return 6;
            return 7;
        }

        inline int effective_angular_scheme(int Z, const GridPreset &preset)
        {
            int scheme = preset.angular_scheme;
            if (preset.reduce_light_atoms && Z <= 2 && scheme > 1)
                --scheme;
            return scheme;
        }

        inline std::expected<int, std::string> radial_point_count(int Z, const GridPreset &preset)
        {
            // PySCF gen_grid._default_rad: a period x level lookup, not a
            // heuristic. The old ORCA-style formula
            //   (15*int_acc - 40) + radial_row_factor*row
            // gave 44 points for a second-row atom at "ultrafine" where PySCF
            // uses 75 at its default level, which is why Planck's grid did not
            // converge (2.9e-5 Eh still moving at ultrafine vs PySCF's 1.4e-7).
            if (Z <= 0)
                return std::unexpected("radial_point_count: atomic number must be positive");
            return pyscf_radial_count(Z, preset.pyscf_level);
        }

        // ORCA-like five-region pruning. The exact ORCA cutoffs are not published
        // in the manual, so we use a symmetric shell-fraction heuristic.
        inline int pruning_region(int radial_index, int radial_count)
        {
            if (radial_count <= 1)
                return 2;

            const double t = static_cast<double>(radial_index) / static_cast<double>(radial_count - 1);
            if (t < 0.15)
                return 0;
            if (t < 0.40)
                return 1;
            if (t < 0.70)
                return 2;
            if (t < 0.90)
                return 3;
            return 4;
        }

        inline std::expected<Eigen::MatrixXd, std::string> coordinates_bohr(const HartreeFock::Molecule &molecule)
        {
            if (static_cast<std::size_t>(molecule._coordinates.rows()) == molecule.natoms)
                return molecule._coordinates;

            if (static_cast<std::size_t>(molecule.coordinates.rows()) != molecule.natoms)
                return std::unexpected(
                    "coordinates_bohr: molecule coordinates are not initialized");

            if (molecule._is_bohr)
                return molecule.coordinates;

            return molecule.coordinates * ANGSTROM_TO_BOHR;
        }

        inline double treutler_becke_adjustment(int Zi, int Zj)
        {
            // PySCF radi.treutler_atomic_radii_adjust:
            //   rad = sqrt(BRAGG[Z]),  a(i,j) = 1/4 * (rad_j/rad_i - rad_i/rad_j)
            // clamped to [-0.5, 0.5]. Note this uses BRAGG radii, NOT the M4 xi
            // -- they are different quantities and PySCF tables them separately.
            // The previous form u/(u^2-1) with u = (ri-rj)/(ri+rj) on the xi
            // table is a different function of different inputs.
            const double ri = std::sqrt(bragg_radius(Zi));
            const double rj = std::sqrt(bragg_radius(Zj));
            if (ri <= 0.0 || rj <= 0.0)
                return 0.0;

            const double raw = 0.25 * (rj / ri - ri / rj);
            return std::clamp(raw, -0.5, 0.5);
        }

        inline double becke_switch(double mu)
        {
            for (int k = 0; k < 3; ++k)
                mu = 1.5 * mu - 0.5 * mu * mu * mu;
            return 0.5 * (1.0 - mu);
        }

        inline double becke_switch_derivative(double mu)
        {
            double derivative = 1.0;
            for (int k = 0; k < 3; ++k)
            {
                derivative *= 1.5 * (1.0 - mu * mu);
                mu = 1.5 * mu - 0.5 * mu * mu * mu;
            }
            return -0.5 * derivative;
        }

        inline double pair_partition(
            const Eigen::Vector3d &point,
            const Eigen::Vector3d &ri,
            const Eigen::Vector3d &rj,
            int Zi,
            int Zj)
        {
            const Eigen::Vector3d rij = ri - rj;
            const double Rij = rij.norm();
            if (Rij < 1e-14)
                return 0.5;

            double mu = (point - ri).norm() - (point - rj).norm();
            mu /= Rij;

            const double aij = treutler_becke_adjustment(Zi, Zj);
            mu += aij * (1.0 - mu * mu);
            mu = std::clamp(mu, -1.0, 1.0);

            return becke_switch(mu);
        }

        inline std::expected<double, std::string> becke_partition_weight(
            int atom_index,
            const Eigen::Vector3d &point,
            const Eigen::VectorXi &atomic_numbers,
            const Eigen::MatrixXd &coordinates)
        {
            const Eigen::Index natoms = atomic_numbers.size();
            std::vector<double> weights(static_cast<std::size_t>(natoms), 1.0);

            for (Eigen::Index i = 0; i < natoms; ++i)
            {
                const Eigen::Vector3d ri = coordinates.row(i).transpose();
                for (Eigen::Index j = i + 1; j < natoms; ++j)
                {
                    const Eigen::Vector3d rj = coordinates.row(j).transpose();
                    const double sij = pair_partition(
                        point,
                        ri,
                        rj,
                        atomic_numbers(i),
                        atomic_numbers(j));
                    weights[static_cast<std::size_t>(i)] *= sij;
                    weights[static_cast<std::size_t>(j)] *= (1.0 - sij);
                }
            }

            double sum = 0.0;
            for (double w : weights)
                sum += w;

            if (sum <= 0.0)
                return std::unexpected("becke_partition_weight: partition weights underflowed");

            return weights[static_cast<std::size_t>(atom_index)] / sum;
        }

    } // namespace detail

    inline std::expected<GridPreset, std::string> grid_preset(GridLevel level)
    {
        switch (level)
        {
        case GridLevel::Coarse:
        {
            auto int_acc = detail::xc_intacc_for_scheme(3);
            if (!int_acc)
                return std::unexpected(int_acc.error());
            return GridPreset{level, 3, *int_acc, 5, true, 1};
        }
        case GridLevel::Normal:
        {
            auto int_acc = detail::xc_intacc_for_scheme(4);
            if (!int_acc)
                return std::unexpected(int_acc.error());
            return GridPreset{level, 4, *int_acc, 5, true, 3};
        }
        case GridLevel::Fine:
        {
            auto int_acc = detail::xc_intacc_for_scheme(5);
            if (!int_acc)
                return std::unexpected(int_acc.error());
            return GridPreset{level, 5, *int_acc, 5, true, 5};
        }
        case GridLevel::UltraFine:
        {
            auto int_acc = detail::xc_intacc_for_scheme(6);
            if (!int_acc)
                return std::unexpected(int_acc.error());
            return GridPreset{level, 6, *int_acc, 5, true, 7};
        }
        }

        return std::unexpected("grid_preset: unsupported grid level");
    }

    inline std::string grid_pruning_policy_name(GridPruningPolicy policy)
    {
        switch (policy)
        {
        case GridPruningPolicy::LegacyOrcaIndexV1: return "legacy-orca-index-v1";
        case GridPruningPolicy::LegacyOrcaNoLightReductionV1: return "legacy-orca-no-light-reduction-v1";
        case GridPruningPolicy::LegacyOrcaUnprunedMaxV1: return "legacy-orca-unpruned-max-v1";
        case GridPruningPolicy::PySCFNWChem213: return "pyscf-nwchem-2.13";
        case GridPruningPolicy::G43V4QualifiedV1: return "g4-3-v4";
        }
        return "unknown";
    }

    inline std::expected<GridPruningPolicy, std::string> grid_pruning_policy_from_name(
        const std::string &name)
    {
        if (name == "legacy") return GridPruningPolicy::LegacyOrcaIndexV1;
        if (name == "legacy-no-light-reduction") return GridPruningPolicy::LegacyOrcaNoLightReductionV1;
        if (name == "legacy-unpruned-max") return GridPruningPolicy::LegacyOrcaUnprunedMaxV1;
        if (name == "pyscf-nwchem-2.13") return GridPruningPolicy::PySCFNWChem213;
        if (name == "g4-3-v4" || name == "g4-3-v4-test-only")
            return GridPruningPolicy::G43V4QualifiedV1;
        return std::unexpected("grid_policy must be legacy, legacy-no-light-reduction, legacy-unpruned-max, pyscf-nwchem-2.13, or g4-3-v4");
    }

    inline std::expected<ResolvedGridSpec, std::string> resolve_grid_spec(GridRequest request)
    {
        auto preset = grid_preset(request.level);
        if (!preset)
            return std::unexpected(preset.error());

        if (request.radial_points < 0)
            return std::unexpected("grid_radial_points must be 0 or a positive shell count");
        if (request.angular_points < 0 ||
            (request.angular_points > 0 &&
             (request.angular_points < 6 ||
             std::none_of(LEBEDEV_ORDER.begin(), LEBEDEV_ORDER.end(),
                          [&request](const auto &entry) { return entry.second == request.angular_points; }))))
            return std::unexpected("grid_angular_points must be 0 or a supported Lebedev size");

        ResolvedGridSpec spec;
        spec.request = request;
        spec.preset = *preset;
        spec.radial_override = request.radial_points;
        spec.pruning_policy = request.pruning_policy;
        spec.policy_name = grid_pruning_policy_name(request.pruning_policy);
        spec.reduce_light_atoms = preset->reduce_light_atoms &&
                                  request.pruning_policy != GridPruningPolicy::LegacyOrcaNoLightReductionV1 &&
                                  request.pruning_policy != GridPruningPolicy::PySCFNWChem213;
        if (request.pruning_policy == GridPruningPolicy::G43V4QualifiedV1 &&
            request.radial_points != 135 && request.radial_points != 0)
            return std::unexpected("g4-3-v4 requires exactly 135 radial shells");
        return spec;
    }

    inline std::expected<int, std::string> angular_scheme(GridLevel level)
    {
        auto preset = grid_preset(level);
        if (!preset)
            return std::unexpected(preset.error());
        return preset->angular_scheme;
    }

    inline std::expected<int, std::string> effective_angular_scheme(int Z, GridLevel level)
    {
        auto preset = grid_preset(level);
        if (!preset)
            return std::unexpected(preset.error());
        return detail::effective_angular_scheme(Z, *preset);
    }

    inline std::expected<int, std::string> radial_point_count(
        int Z, const ResolvedGridSpec &spec)
    {
        if (spec.radial_override > 0)
            return spec.radial_override;
        return detail::radial_point_count(Z, spec.preset);
    }

    inline std::expected<int, std::string> radial_point_count(int Z, GridLevel level)
    {
        auto spec = resolve_grid_spec(GridRequest{level});
        if (!spec)
            return std::unexpected(spec.error());
        return radial_point_count(Z, *spec);
    }

    inline std::expected<std::array<int, 5>, std::string> angular_shell_sizes(
        int Z, const ResolvedGridSpec &spec)
    {
        if (Z <= 0)
            return std::unexpected("angular_shell_sizes: atomic number must be positive");
        if (spec.request.angular_points > 0)
        {
            // Fixed angular refinement: no pruning or H/He reduction. Neither
            // the radial nodes/weights nor the lab-frame grid orientation move.
            return std::array<int, 5>{spec.request.angular_points, spec.request.angular_points,
                                      spec.request.angular_points, spec.request.angular_points,
                                      spec.request.angular_points};
        }
        if (spec.pruning_policy == GridPruningPolicy::PySCFNWChem213)
            return std::unexpected("angular_shell_sizes: unsupported grid pruning policy");
        if (spec.pruning_policy == GridPruningPolicy::G43V4QualifiedV1)
            return std::unexpected("angular_shell_sizes: G4.3 V4 selects by physical radius per shell");
        GridPreset preset = spec.preset;
        preset.reduce_light_atoms = spec.reduce_light_atoms;
        auto shells = detail::angular_shells_for_scheme(detail::effective_angular_scheme(Z, preset));
        if (!shells)
            return std::unexpected(shells.error());
        if (spec.pruning_policy == GridPruningPolicy::LegacyOrcaUnprunedMaxV1)
            return std::array<int, 5>{(*shells)[3], (*shells)[3], (*shells)[3], (*shells)[3], (*shells)[3]};
        return *shells;
    }

    inline std::expected<int, std::string> angular_points_for_shell(
        int Z, const ResolvedGridSpec &spec, int radial_index, int radial_count, double radius)
    {
        if (spec.request.angular_points > 0)
            return spec.request.angular_points;
        if (spec.pruning_policy == GridPruningPolicy::G43V4QualifiedV1)
        {
            if (radial_count != 135)
                return std::unexpected("g4-3-v4 requires exactly 135 radial shells");
            const std::array<double, 4> boundaries = Z <= 2 ? std::array<double, 4>{.25, 1., 2.5, 3.5} :
                                                       Z <= 10 ? std::array<double, 4>{.4, 1., 2.5, 4.5} :
                                                       Z <= 18 ? std::array<double, 4>{.4, .8, 2.5, 4.5} :
                                                       std::array<double, 4>{};
            if (Z <= 0 || Z > 18)
                return std::unexpected("g4-3-v4 supports atomic numbers 1 through 18 only");
            const std::array<int, 5> points = Z <= 2 ? std::array<int, 5>{26, 146, 2030, 974, 590} :
                                              Z <= 10 ? std::array<int, 5>{110, 770, 2030, 590, 170} :
                                                        std::array<int, 5>{86, 590, 974, 590, 26};
            int band = 0;
            for (const double boundary : boundaries) band += radius / bragg_radius(Z) >= boundary;
            return points[band];
        }
        if (spec.pruning_policy != GridPruningPolicy::PySCFNWChem213)
        {
            const auto shells = angular_shell_sizes(Z, spec);
            if (!shells) return std::unexpected(shells.error());
            return (*shells)[detail::pruning_region(radial_index, radial_count)];
        }

        // Literal PySCF 2.13 gen_grid._default_ang + nwchem_prune.  The
        // selection depends on the fixed atom-centred radius r/R_Bragg, never
        // on density or molecular geometry, so it is safe for analytic grids.
        constexpr int orders[10][7] = {
            {11,15,17,17,17,17,17}, {17,23,23,23,23,23,23}, {23,29,29,29,29,29,29},
            {29,29,35,35,35,35,35}, {35,41,41,41,41,41,41}, {41,47,47,47,47,47,47},
            {47,53,53,53,53,53,53}, {53,59,59,59,59,59,59}, {59,59,59,59,59,59,59},
            {65,65,65,65,65,65,65}};
        constexpr int ngrid[] = {38,50,74,86,110,146,170,194,230,266,302,350,434,
                                 590,770,974,1202,1454,1730,2030,2354,2702,3074,
                                 3470,3890,4334,4802,5294,5810};
        const int period = pyscf_period_index(Z);
        const int order = orders[spec.preset.pyscf_level][period];
        const auto maximum = LEBEDEV_ORDER.find(order);
        if (maximum == LEBEDEV_ORDER.end())
            return std::unexpected("PySCF NWChem pruning: unsupported angular order");
        const int n_ang = maximum->second;
        if (n_ang < 50) return n_ang;
        int index = -1;
        for (int i = 0; i < static_cast<int>(std::size(ngrid)); ++i)
            if (ngrid[i] == n_ang) { index = i; break; }
        if (index < 1)
            return std::unexpected("PySCF NWChem pruning: unsupported maximum angular grid");
        const double thresholds[3][4] = {
            {.25,.50,1.00,4.50}, {.1667,.50,.90,3.50}, {.10,.40,.80,2.50}};
        const int family = Z <= 2 ? 0 : (Z <= 10 ? 1 : 2);
        const double scaled = radius / bragg_radius(Z);
        int place = 0;
        while (place < 4 && scaled > thresholds[family][place]) ++place;
        const int levels[5] = {1, 3, index-1, index, index-1};
        return ngrid[levels[place]];
    }

    inline std::expected<std::array<int, 5>, std::string> angular_shell_sizes(
        int Z, GridLevel level, int angular_points = 0)
    {
        auto spec = resolve_grid_spec(GridRequest{level, angular_points});
        if (!spec)
            return std::unexpected(spec.error());
        return angular_shell_sizes(Z, *spec);
    }

    inline std::expected<int, std::string> atomic_point_count(
        int Z, const ResolvedGridSpec &spec)
    {
        const auto nr = radial_point_count(Z, spec);
        if (!nr)
            return std::unexpected(nr.error());
        const Eigen::MatrixXd radial = MakeTreutlerAhlrichsGrid(*nr, treutler_xi(Z));
        int total = 0;
        for (int ir = 0; ir < *nr; ++ir)
        {
            const auto angular = angular_points_for_shell(Z, spec, ir, *nr, radial(ir, 0));
            if (!angular) return std::unexpected(angular.error());
            total += *angular;
        }
        return total;
    }

    inline std::expected<int, std::string> atomic_point_count(
        int Z, GridLevel level, int angular_points = 0)
    {
        auto spec = resolve_grid_spec(GridRequest{level, angular_points});
        if (!spec)
            return std::unexpected(spec.error());
        return atomic_point_count(Z, *spec);
    }

    inline std::expected<Eigen::MatrixXd, std::string> MakeAtomicGrid(
        int Z,
        const Eigen::Vector3d &center,
        const ResolvedGridSpec &spec)
    {
        if (Z <= 0)
            return std::unexpected(
                "MakeAtomicGrid: atomic number must be positive, got " + std::to_string(Z));

        const auto nr = radial_point_count(Z, spec);
        if (!nr)
            return std::unexpected(nr.error());
        const Eigen::MatrixXd radial = MakeTreutlerAhlrichsGrid(*nr, treutler_xi(Z));

        std::vector<int> angular_sizes(static_cast<std::size_t>(*nr));
        std::map<int, Eigen::MatrixXd> angular_cache;
        for (int ir = 0; ir < *nr; ++ir)
        {
            const auto size = angular_points_for_shell(Z, spec, ir, *nr, radial(ir, 0));
            if (!size) return std::unexpected(size.error());
            angular_sizes[static_cast<std::size_t>(ir)] = *size;
            if (angular_cache.contains(*size)) continue;
            auto angular_grid = MakeLebedevGrid(*size);
            if (!angular_grid)
                return std::unexpected(angular_grid.error());
            angular_cache.emplace(*size, std::move(*angular_grid));
        }

        const auto total_points = atomic_point_count(Z, spec);
        if (!total_points)
            return std::unexpected(total_points.error());
        Eigen::MatrixXd grid(*total_points, 4);
        Eigen::Index row = 0;

        for (int ir = 0; ir < *nr; ++ir)
        {
            const double radius = radial(ir, 0);
            const double wr = radial(ir, 1);
            const Eigen::MatrixXd &angular = angular_cache.at(angular_sizes[static_cast<std::size_t>(ir)]);

            for (Eigen::Index ia = 0; ia < angular.rows(); ++ia, ++row)
            {
                const Eigen::Vector3d direction = angular.row(ia).head<3>().transpose();
                const Eigen::Vector3d point = center + radius * direction;

                grid(row, 0) = point.x();
                grid(row, 1) = point.y();
                grid(row, 2) = point.z();
                grid(row, 3) = wr * angular(ia, 3);
            }
        }

        return grid;
    }

    inline std::expected<Eigen::MatrixXd, std::string> MakeAtomicGrid(
        int Z,
        const Eigen::Vector3d &center,
        GridLevel level,
        int angular_points = 0)
    {
        auto spec = resolve_grid_spec(GridRequest{level, angular_points});
        if (!spec)
            return std::unexpected(spec.error());
        return MakeAtomicGrid(Z, center, *spec);
    }

    inline std::expected<MolecularGrid, std::string> MakeMolecularGrid(
        const Eigen::VectorXi &atomic_numbers,
        const Eigen::MatrixXd &coordinates_bohr,
        const ResolvedGridSpec &spec)
    {
        const Eigen::Index natoms = atomic_numbers.size();
        if (coordinates_bohr.rows() != natoms || coordinates_bohr.cols() != 3)
            return std::unexpected(
                "MakeMolecularGrid: coordinates must have shape natoms x 3");

        std::vector<Eigen::MatrixXd> atomic_grids(static_cast<std::size_t>(natoms));
        Eigen::Index total_points = 0;

        for (Eigen::Index atom = 0; atom < natoms; ++atom)
        {
            auto atomic_grid = MakeAtomicGrid(
                atomic_numbers(atom),
                coordinates_bohr.row(atom).transpose(),
                spec);
            if (!atomic_grid)
                return std::unexpected(atomic_grid.error());
            atomic_grids[static_cast<std::size_t>(atom)] = std::move(*atomic_grid);
            total_points += atomic_grids[static_cast<std::size_t>(atom)].rows();
        }

        MolecularGrid result;
        result.points.resize(total_points, 4);
        result.owner.resize(total_points);
        result.atomic_weights.resize(total_points);
        result.partition_weights.resize(total_points);

        Eigen::Index row = 0;
        for (Eigen::Index atom = 0; atom < natoms; ++atom)
        {
            const Eigen::MatrixXd &atomic_grid = atomic_grids[static_cast<std::size_t>(atom)];
            for (Eigen::Index i = 0; i < atomic_grid.rows(); ++i, ++row)
            {
                const Eigen::Vector3d point = atomic_grid.row(i).head<3>().transpose();
                const auto partition = detail::becke_partition_weight(
                    static_cast<int>(atom),
                    point,
                    atomic_numbers,
                    coordinates_bohr);
                if (!partition)
                    return std::unexpected(partition.error());

                result.points.row(row).head<3>() = point.transpose();
                result.points(row, 3) = atomic_grid(i, 3) * (*partition);
                result.owner(row) = static_cast<int>(atom);
                result.atomic_weights(row) = atomic_grid(i, 3);
                result.partition_weights(row) = *partition;
            }
        }

        return result;
    }

    inline std::expected<MolecularGrid, std::string> MakeMolecularGrid(
        const Eigen::VectorXi &atomic_numbers,
        const Eigen::MatrixXd &coordinates_bohr,
        GridLevel level,
        int angular_points = 0)
    {
        auto spec = resolve_grid_spec(GridRequest{level, angular_points});
        if (!spec)
            return std::unexpected(spec.error());
        return MakeMolecularGrid(atomic_numbers, coordinates_bohr, *spec);
    }

    inline std::expected<MolecularGrid, std::string> MakeMolecularGrid(
        const HartreeFock::Molecule &molecule,
        const ResolvedGridSpec &spec)
    {
        auto coords = detail::coordinates_bohr(molecule);
        if (!coords)
            return std::unexpected(coords.error());
        return MakeMolecularGrid(molecule.atomic_numbers, *coords, spec);
    }

    inline std::expected<MolecularGrid, std::string> MakeMolecularGrid(
        const HartreeFock::Molecule &molecule,
        GridLevel level,
        int angular_points = 0)
    {
        auto spec = resolve_grid_spec(GridRequest{level, angular_points});
        if (!spec)
            return std::unexpected(spec.error());
        return MakeMolecularGrid(molecule, *spec);
    }

} // namespace DFT

#endif // DFT_GRID_H
