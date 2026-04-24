#include "xc_grid.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace DFT
{

    namespace
    {

        std::expected<void, std::string> validate_ao_grid(const AOGridEvaluation &ao_grid)
        {
            const Eigen::Index npoints = ao_grid.values.rows();
            const Eigen::Index nbasis = ao_grid.values.cols();

            if (npoints < 0 || nbasis < 0)
                return std::unexpected("AO grid dimensions must be non-negative");

            if (ao_grid.grad_x.rows() != npoints || ao_grid.grad_y.rows() != npoints || ao_grid.grad_z.rows() != npoints)
                return std::unexpected("AO grid gradient arrays must match the value-grid point count");

            if (ao_grid.grad_x.cols() != nbasis || ao_grid.grad_y.cols() != nbasis || ao_grid.grad_z.cols() != nbasis)
                return std::unexpected("AO grid gradient arrays must match the value-grid basis count");

            return {};
        }

        std::expected<void, std::string> validate_density_matrix(
            const Eigen::Ref<const Eigen::MatrixXd> &density,
            Eigen::Index nbasis,
            const std::string &label)
        {
            if (density.rows() != nbasis || density.cols() != nbasis)
                return std::unexpected(
                    label + " density matrix must be " +
                    std::to_string(nbasis) + "x" + std::to_string(nbasis));

            return {};
        }

        DensityChannelOnGrid evaluate_density_channel(
            const AOGridEvaluation &ao_grid,
            const Eigen::Ref<const Eigen::MatrixXd> &density)
        {
            const Eigen::MatrixXd symmetric_density = 0.5 * (density + density.transpose());
            const Eigen::MatrixXd density_times_values = ao_grid.values * symmetric_density;

            DensityChannelOnGrid channel;
            channel.rho = (ao_grid.values.array() * density_times_values.array()).rowwise().sum().matrix();
            channel.grad_x = (2.0 * ao_grid.grad_x.array() * density_times_values.array()).rowwise().sum().matrix();
            channel.grad_y = (2.0 * ao_grid.grad_y.array() * density_times_values.array()).rowwise().sum().matrix();
            channel.grad_z = (2.0 * ao_grid.grad_z.array() * density_times_values.array()).rowwise().sum().matrix();
            return channel;
        }

        std::vector<double> pack_rho(const DensityOnGrid &density)
        {
            const Eigen::Index npoints = density.npoints();
            std::vector<double> rho(static_cast<std::size_t>(npoints * (density.polarized ? 2 : 1)));

            if (!density.polarized)
            {
                for (Eigen::Index point = 0; point < npoints; ++point)
                    rho[static_cast<std::size_t>(point)] = density.total.rho(point);
                return rho;
            }

            for (Eigen::Index point = 0; point < npoints; ++point)
            {
                const std::size_t offset = static_cast<std::size_t>(2 * point);
                rho[offset] = density.alpha.rho(point);
                rho[offset + 1] = density.beta.rho(point);
            }

            return rho;
        }

        std::vector<double> pack_sigma(const DensityOnGrid &density)
        {
            const Eigen::Index npoints = density.npoints();
            std::vector<double> sigma(static_cast<std::size_t>(npoints * (density.polarized ? 3 : 1)));

            if (!density.polarized)
            {
                const Eigen::VectorXd sigma_total = density.total.gradient_squared();
                for (Eigen::Index point = 0; point < npoints; ++point)
                    sigma[static_cast<std::size_t>(point)] = sigma_total(point);
                return sigma;
            }

            const Eigen::VectorXd sigma_aa = density.alpha.gradient_squared();
            const Eigen::VectorXd sigma_bb = density.beta.gradient_squared();
            const Eigen::VectorXd sigma_ab =
                (density.alpha.grad_x.array() * density.beta.grad_x.array() + density.alpha.grad_y.array() * density.beta.grad_y.array() + density.alpha.grad_z.array() * density.beta.grad_z.array()).matrix();

            for (Eigen::Index point = 0; point < npoints; ++point)
            {
                const std::size_t offset = static_cast<std::size_t>(3 * point);
                sigma[offset] = sigma_aa(point);
                sigma[offset + 1] = sigma_ab(point);
                sigma[offset + 2] = sigma_bb(point);
            }

            return sigma;
        }

        Eigen::MatrixXd unpack_point_major(
            const std::vector<double> &data,
            Eigen::Index npoints,
            Eigen::Index ncols)
        {
            Eigen::MatrixXd matrix(npoints, ncols);
            for (Eigen::Index point = 0; point < npoints; ++point)
            {
                for (Eigen::Index col = 0; col < ncols; ++col)
                {
                    matrix(point, col) = data[static_cast<std::size_t>(point * ncols + col)];
                }
            }
            return matrix;
        }

        std::expected<void, std::string> validate_functional_compatibility(
            const DensityOnGrid &density,
            const XC::Functional &functional,
            const std::string &label)
        {
            const bool functional_polarized = functional.spin() == XC::Spin::Polarized;
            if (functional_polarized != density.polarized)
                return std::unexpected(label + " spin mode does not match the density spin structure");

            if (functional.is_meta_gga_like())
            {
                return std::unexpected(
                    label + " meta-GGA functionals are not supported yet because tau/laplacian terms are not wired");
            }

            if (functional.is_hybrid() && !functional.is_global_hybrid())
            {
                return std::unexpected(
                    label + " range-separated and double-hybrid functionals are not supported yet");
            }

            if (!functional.is_supported_semilocal())
            {
                return std::unexpected(
                    label + " functional family is not supported by the current semilocal DFT grid layer");
            }

            return {};
        }

        std::expected<XCFunctionalGridResult, std::string> evaluate_functional_on_grid(
            const MolecularGrid &molecular_grid,
            const DensityOnGrid &density,
            const XC::Functional &functional,
            const std::string &label)
        {
            if (auto compatible = validate_functional_compatibility(density, functional, label); !compatible)
                return std::unexpected(compatible.error());

            if (molecular_grid.points.cols() != 4)
                return std::unexpected(label + " evaluation requires a molecular grid with quadrature weights");

            const Eigen::Index npoints = density.npoints();
            if (molecular_grid.points.rows() != npoints)
                return std::unexpected(label + " evaluation received inconsistent density/grid sizes");

            XCFunctionalGridResult result;
            result.name = functional.name();
            result.family = functional.family();
            result.polarized = density.polarized;
            result.uses_gradients = functional.is_gga_like();

            const std::vector<double> rho = pack_rho(density);
            std::vector<double> exc;
            std::vector<double> vrho;
            std::vector<double> vsigma;

            if (functional.is_lda_like())
            {
                if (auto eval = functional.evaluate_lda_exc_vxc(rho, static_cast<int>(npoints), exc, vrho); !eval)
                    return std::unexpected(label + " LDA evaluation failed: " + eval.error());
            }
            else if (functional.is_gga_like())
            {
                const std::vector<double> sigma = pack_sigma(density);
                if (auto eval = functional.evaluate_gga_exc_vxc(
                        rho,
                        sigma,
                        static_cast<int>(npoints),
                        exc,
                        vrho,
                        vsigma);
                    !eval)
                {
                    return std::unexpected(label + " GGA evaluation failed: " + eval.error());
                }
            }
            else
            {
                return std::unexpected(label + " evaluation reached an unsupported functional family");
            }

            result.exc = Eigen::Map<const Eigen::VectorXd>(exc.data(), npoints);
            result.vrho = unpack_point_major(vrho, npoints, functional.spin_components());
            if (vsigma.empty())
                result.vsigma.resize(npoints, 0);
            else
                result.vsigma = unpack_point_major(vsigma, npoints, functional.sigma_components());

            result.energy_density =
                (density.total.rho.array() * result.exc.array()).matrix();
            result.energy = molecular_grid.points.col(3).dot(result.energy_density);
            return result;
        }

        XCFunctionalGridResult zero_functional_result(
            const DensityOnGrid &density,
            const std::string &name)
        {
            const Eigen::Index npoints = density.npoints();
            const Eigen::Index spin_components = density.polarized ? 2 : 1;

            XCFunctionalGridResult result;
            result.name = name;
            result.family = XC_FAMILY_UNKNOWN;
            result.polarized = density.polarized;
            result.uses_gradients = false;
            result.exc = Eigen::VectorXd::Zero(npoints);
            result.vrho = Eigen::MatrixXd::Zero(npoints, spin_components);
            result.vsigma.resize(npoints, 0);
            result.energy_density = Eigen::VectorXd::Zero(npoints);
            result.energy = 0.0;
            return result;
        }

        Eigen::MatrixXd combine_response(
            Eigen::Index npoints,
            const Eigen::MatrixXd &lhs,
            const Eigen::MatrixXd &rhs)
        {
            const Eigen::Index ncols = std::max(lhs.cols(), rhs.cols());
            Eigen::MatrixXd combined = Eigen::MatrixXd::Zero(npoints, ncols);

            if (lhs.cols() > 0)
                combined.leftCols(lhs.cols()) += lhs;
            if (rhs.cols() > 0)
                combined.leftCols(rhs.cols()) += rhs;

            return combined;
        }

    } // namespace

    Eigen::VectorXd DensityChannelOnGrid::gradient_squared() const
    {
        return (grad_x.array().square() + grad_y.array().square() + grad_z.array().square()).matrix();
    }

    std::expected<double, std::string> DensityChannelOnGrid::integrated_density(const MolecularGrid &molecular_grid) const
    {
        if (molecular_grid.points.cols() != 4)
        {
            return std::unexpected(
                "integrated_density requires a molecular grid with quadrature weights");
        }

        if (molecular_grid.points.rows() != npoints())
        {
            return std::unexpected(
                "integrated_density received inconsistent density/grid sizes");
        }

        return molecular_grid.points.col(3).dot(rho);
    }

    std::expected<double, std::string> DensityOnGrid::integrated_electrons(const MolecularGrid &molecular_grid) const
    {
        return total.integrated_density(molecular_grid);
    }

    std::expected<DensityOnGrid, std::string>
    evaluate_density_on_grid(
        const AOGridEvaluation &ao_grid,
        const Eigen::Ref<const Eigen::MatrixXd> &restricted_density)
    {
        if (auto valid = validate_ao_grid(ao_grid); !valid)
            return std::unexpected(valid.error());

        if (auto valid = validate_density_matrix(restricted_density, ao_grid.nbasis(), "restricted"); !valid)
            return std::unexpected(valid.error());

        DensityOnGrid density;
        density.polarized = false;
        density.total = evaluate_density_channel(ao_grid, restricted_density);

        density.alpha.rho = 0.5 * density.total.rho;
        density.alpha.grad_x = 0.5 * density.total.grad_x;
        density.alpha.grad_y = 0.5 * density.total.grad_y;
        density.alpha.grad_z = 0.5 * density.total.grad_z;

        density.beta = density.alpha;
        return density;
    }

    std::expected<DensityOnGrid, std::string>
    evaluate_density_on_grid(
        const AOGridEvaluation &ao_grid,
        const Eigen::Ref<const Eigen::MatrixXd> &alpha_density,
        const Eigen::Ref<const Eigen::MatrixXd> &beta_density)
    {
        if (auto valid = validate_ao_grid(ao_grid); !valid)
            return std::unexpected(valid.error());

        if (auto valid = validate_density_matrix(alpha_density, ao_grid.nbasis(), "alpha"); !valid)
            return std::unexpected(valid.error());

        if (auto valid = validate_density_matrix(beta_density, ao_grid.nbasis(), "beta"); !valid)
            return std::unexpected(valid.error());

        DensityOnGrid density;
        density.polarized = true;
        density.alpha = evaluate_density_channel(ao_grid, alpha_density);
        density.beta = evaluate_density_channel(ao_grid, beta_density);

        density.total.rho = density.alpha.rho + density.beta.rho;
        density.total.grad_x = density.alpha.grad_x + density.beta.grad_x;
        density.total.grad_y = density.alpha.grad_y + density.beta.grad_y;
        density.total.grad_z = density.alpha.grad_z + density.beta.grad_z;

        return density;
    }

    std::expected<XCGridEvaluation, std::string>
    evaluate_xc_on_grid(
        const MolecularGrid &molecular_grid,
        const DensityOnGrid &density,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional)
    {
        if (molecular_grid.points.cols() != 4)
            return std::unexpected("XC grid evaluation requires quadrature weights in the fourth molecular-grid column");

        if (molecular_grid.points.rows() != density.npoints())
            return std::unexpected("XC grid evaluation received inconsistent density/grid sizes");

        auto exchange = evaluate_functional_on_grid(
            molecular_grid,
            density,
            exchange_functional,
            "exchange");
        if (!exchange)
            return std::unexpected(exchange.error());

        std::expected<XCFunctionalGridResult, std::string> correlation =
            exchange_functional.is_combined_exchange_correlation()
                ? std::expected<XCFunctionalGridResult, std::string>(
                      zero_functional_result(density, "included in " + exchange->name))
                : evaluate_functional_on_grid(
                      molecular_grid,
                      density,
                      correlation_functional,
                      "correlation");
        if (!correlation)
            return std::unexpected(correlation.error());

        XCGridEvaluation evaluation;
        evaluation.density = density;
        evaluation.exchange = std::move(*exchange);
        evaluation.correlation = std::move(*correlation);

        const Eigen::Index npoints = density.npoints();
        evaluation.exc = evaluation.exchange.exc + evaluation.correlation.exc;
        evaluation.vrho = evaluation.exchange.vrho + evaluation.correlation.vrho;
        evaluation.vsigma = combine_response(
            npoints,
            evaluation.exchange.vsigma,
            evaluation.correlation.vsigma);
        evaluation.energy_density =
            evaluation.exchange.energy_density + evaluation.correlation.energy_density;

        evaluation.exchange_energy = evaluation.exchange.energy;
        evaluation.correlation_energy = evaluation.correlation.energy;
        evaluation.total_energy = evaluation.exchange_energy + evaluation.correlation_energy;
        evaluation.exact_exchange_coefficient =
            exchange_functional.is_hybrid() ? exchange_functional.exact_exchange_coefficient() : 0.0;
        auto integrated_electrons = density.integrated_electrons(molecular_grid);
        if (!integrated_electrons)
            return std::unexpected(integrated_electrons.error());
        evaluation.integrated_electrons = *integrated_electrons;
        return evaluation;
    }

    std::expected<XCGridEvaluation, std::string>
    evaluate_xc_on_grid(
        const MolecularGrid &molecular_grid,
        const AOGridEvaluation &ao_grid,
        const Eigen::Ref<const Eigen::MatrixXd> &restricted_density,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional)
    {
        auto density = evaluate_density_on_grid(ao_grid, restricted_density);
        if (!density)
            return std::unexpected(density.error());

        return evaluate_xc_on_grid(
            molecular_grid,
            *density,
            exchange_functional,
            correlation_functional);
    }

    std::expected<XCGridEvaluation, std::string>
    evaluate_xc_on_grid(
        const MolecularGrid &molecular_grid,
        const AOGridEvaluation &ao_grid,
        const Eigen::Ref<const Eigen::MatrixXd> &alpha_density,
        const Eigen::Ref<const Eigen::MatrixXd> &beta_density,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional)
    {
        auto density = evaluate_density_on_grid(ao_grid, alpha_density, beta_density);
        if (!density)
            return std::unexpected(density.error());

        return evaluate_xc_on_grid(
            molecular_grid,
            *density,
            exchange_functional,
            correlation_functional);
    }

} // namespace DFT
