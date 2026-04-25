#ifndef HF_SCF_POPULATION_H
#define HF_SCF_POPULATION_H

#include <Eigen/Core>

#include <expected>
#include <string>
#include <vector>

#include "base/types.h"

namespace HartreeFock::SCF
{
    struct AtomicPopulation
    {
        std::size_t atom_index = 0;
        int atomic_number = 0;
        double electron_population = 0.0;
        double net_charge = 0.0;
        double spin_population = 0.0;
    };

    struct PopulationAnalysis
    {
        std::vector<AtomicPopulation> atoms;
        Eigen::VectorXd ao_population;
        double total_electrons = 0.0;
        double total_charge = 0.0;
        double total_spin_population = 0.0;
        bool has_spin_population = false;
    };

    struct MayerBondOrderAnalysis
    {
        Eigen::MatrixXd bond_orders;
    };

    std::expected<PopulationAnalysis, std::string> mulliken_population_analysis(
        const Molecule &molecule,
        const Basis &basis,
        const Eigen::MatrixXd &overlap,
        const Eigen::MatrixXd &total_density,
        const Eigen::MatrixXd *spin_density = nullptr);

    std::expected<PopulationAnalysis, std::string> lowdin_population_analysis(
        const Molecule &molecule,
        const Basis &basis,
        const Eigen::MatrixXd &overlap,
        const Eigen::MatrixXd &total_density,
        const Eigen::MatrixXd *spin_density = nullptr);

    std::expected<MayerBondOrderAnalysis, std::string> mayer_bond_order_analysis(
        const Molecule &molecule,
        const Basis &basis,
        const Eigen::MatrixXd &overlap,
        const Eigen::MatrixXd &total_density,
        const Eigen::MatrixXd *alpha_density = nullptr,
        const Eigen::MatrixXd *beta_density = nullptr);
} // namespace HartreeFock::SCF

#endif // HF_SCF_POPULATION_H
