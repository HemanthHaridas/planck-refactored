#ifndef DFT_XC_GRID_H
#define DFT_XC_GRID_H

#include <expected>
#include <string>

#include <Eigen/Dense>

#include "ao_grid.h"
#include "base/grid.h"
#include "base/wrapper.h"

namespace DFT
{

    struct DensityChannelOnGrid
    {
        Eigen::VectorXd rho;
        Eigen::VectorXd grad_x;
        Eigen::VectorXd grad_y;
        Eigen::VectorXd grad_z;

        [[nodiscard]] Eigen::Index npoints() const noexcept
        {
            return rho.size();
        }

        [[nodiscard]] Eigen::VectorXd gradient_squared() const;

        [[nodiscard]] std::expected<double, std::string> integrated_density(const MolecularGrid &molecular_grid) const;
    };

    struct DensityOnGrid
    {
        bool polarized = false;
        DensityChannelOnGrid total;
        DensityChannelOnGrid alpha;
        DensityChannelOnGrid beta;

        [[nodiscard]] Eigen::Index npoints() const noexcept
        {
            return total.npoints();
        }

        [[nodiscard]] std::expected<double, std::string> integrated_electrons(const MolecularGrid &molecular_grid) const;
    };

    struct XCFunctionalGridResult
    {
        std::string name;
        int family = 0;
        bool polarized = false;
        bool uses_gradients = false;

        Eigen::VectorXd exc;            // per-particle XC energy density epsilon_xc(r)
        Eigen::MatrixXd vrho;           // npoints x (1 or 2)
        Eigen::MatrixXd vsigma;         // npoints x (0, 1, or 3)
        Eigen::VectorXd energy_density; // rho_total(r) * epsilon_xc(r)

        double energy = 0.0;
    };

    struct XCGridEvaluation
    {
        DensityOnGrid density;

        XCFunctionalGridResult exchange;
        XCFunctionalGridResult correlation;

        Eigen::VectorXd exc;
        Eigen::MatrixXd vrho;
        Eigen::MatrixXd vsigma;
        Eigen::VectorXd energy_density;

        double exchange_energy = 0.0;
        double correlation_energy = 0.0;
        double total_energy = 0.0;
        double exact_exchange_coefficient = 0.0;
        double integrated_electrons = 0.0;
    };

    std::expected<DensityOnGrid, std::string>
    evaluate_density_on_grid(
        const AOGridEvaluation &ao_grid,
        const Eigen::Ref<const Eigen::MatrixXd> &restricted_density);

    std::expected<DensityOnGrid, std::string>
    evaluate_density_on_grid(
        const AOGridEvaluation &ao_grid,
        const Eigen::Ref<const Eigen::MatrixXd> &alpha_density,
        const Eigen::Ref<const Eigen::MatrixXd> &beta_density);

    std::expected<XCGridEvaluation, std::string>
    evaluate_xc_on_grid(
        const MolecularGrid &molecular_grid,
        const DensityOnGrid &density,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional);

    std::expected<XCGridEvaluation, std::string>
    evaluate_xc_on_grid(
        const MolecularGrid &molecular_grid,
        const AOGridEvaluation &ao_grid,
        const Eigen::Ref<const Eigen::MatrixXd> &restricted_density,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional);

    std::expected<XCGridEvaluation, std::string>
    evaluate_xc_on_grid(
        const MolecularGrid &molecular_grid,
        const AOGridEvaluation &ao_grid,
        const Eigen::Ref<const Eigen::MatrixXd> &alpha_density,
        const Eigen::Ref<const Eigen::MatrixXd> &beta_density,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional);

} // namespace DFT

#endif // DFT_XC_GRID_H
