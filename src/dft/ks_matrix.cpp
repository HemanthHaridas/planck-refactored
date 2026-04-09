#include "ks_matrix.h"

#include <stdexcept>

namespace DFT
{

    namespace
    {

        std::expected<void, std::string> validate_grid_inputs(
            const MolecularGrid &molecular_grid,
            const AOGridEvaluation &ao_grid,
            const XCGridEvaluation &xc_grid)
        {
            if (molecular_grid.points.cols() != 4)
                return std::unexpected("XC matrix assembly requires quadrature weights in the fourth molecular-grid column");

            if (ao_grid.npoints() != molecular_grid.points.rows())
                return std::unexpected("AO grid and molecular grid point counts do not match");

            if (xc_grid.density.npoints() != molecular_grid.points.rows())
                return std::unexpected("XC grid density and molecular grid point counts do not match");

            if (xc_grid.vrho.rows() != molecular_grid.points.rows())
                return std::unexpected("XC vrho array and molecular grid point counts do not match");

            if (xc_grid.density.polarized)
            {
                if (xc_grid.vrho.cols() != 2)
                    return std::unexpected("Polarized XC matrix assembly requires vrho with two spin columns");
                if (!(xc_grid.vsigma.cols() == 0 || xc_grid.vsigma.cols() == 3))
                    return std::unexpected("Polarized XC matrix assembly requires vsigma with either 0 or 3 columns");
            }
            else
            {
                if (xc_grid.vrho.cols() != 1)
                    return std::unexpected("Unpolarized XC matrix assembly requires vrho with one column");
                if (!(xc_grid.vsigma.cols() == 0 || xc_grid.vsigma.cols() == 1))
                    return std::unexpected("Unpolarized XC matrix assembly requires vsigma with either 0 or 1 column");
            }

            return {};
        }

        Eigen::Vector3d density_gradient_at(
            const DensityChannelOnGrid &density,
            Eigen::Index point)
        {
            return {
                density.grad_x(point),
                density.grad_y(point),
                density.grad_z(point)};
        }

        Eigen::VectorXd gradient_projection(
            const AOGridEvaluation &ao_grid,
            Eigen::Index point,
            const Eigen::Vector3d &coefficient)
        {
            return (coefficient.x() * ao_grid.grad_x.row(point).transpose() + coefficient.y() * ao_grid.grad_y.row(point).transpose() + coefficient.z() * ao_grid.grad_z.row(point).transpose()).eval();
        }

        void accumulate_local_potential(
            Eigen::Ref<Eigen::MatrixXd> matrix,
            double weight,
            double vrho,
            const Eigen::VectorXd &phi,
            const Eigen::VectorXd &gradient_term)
        {
            if (weight == 0.0)
                return;

            matrix.noalias() += weight * vrho * (phi * phi.transpose());
            if (gradient_term.size() == phi.size())
                matrix.noalias() += weight * (phi * gradient_term.transpose() + gradient_term * phi.transpose());
        }

    } // namespace

    std::expected<XCMatrixContribution, std::string>
    assemble_xc_matrix(
        const MolecularGrid &molecular_grid,
        const AOGridEvaluation &ao_grid,
        const XCGridEvaluation &xc_grid)
    {
        if (auto valid = validate_grid_inputs(molecular_grid, ao_grid, xc_grid); !valid)
            return std::unexpected(valid.error());

        const Eigen::Index npoints = ao_grid.npoints();
        const Eigen::Index nbasis = ao_grid.nbasis();

        XCMatrixContribution contribution;
        contribution.polarized = xc_grid.density.polarized;
        contribution.alpha = Eigen::MatrixXd::Zero(nbasis, nbasis);
        contribution.beta = Eigen::MatrixXd::Zero(nbasis, nbasis);

        const bool polarized = xc_grid.density.polarized;

#ifdef USE_OPENMP
#pragma omp parallel
#endif
        {
            Eigen::MatrixXd alpha_local = Eigen::MatrixXd::Zero(nbasis, nbasis);
            Eigen::MatrixXd beta_local = Eigen::MatrixXd::Zero(nbasis, nbasis);

#ifdef USE_OPENMP
#pragma omp for nowait schedule(static)
#endif
            for (Eigen::Index point = 0; point < npoints; ++point)
            {
                const double weight = molecular_grid.points(point, 3);
                if (weight == 0.0)
                    continue;

                const Eigen::VectorXd phi = ao_grid.values.row(point).transpose();

                if (!polarized)
                {
                    Eigen::VectorXd gradient_term = Eigen::VectorXd::Zero(nbasis);
                    if (xc_grid.vsigma.cols() == 1)
                    {
                        const Eigen::Vector3d coefficient =
                            2.0 * xc_grid.vsigma(point, 0) *
                            density_gradient_at(xc_grid.density.total, point);
                        gradient_term = gradient_projection(ao_grid, point, coefficient);
                    }

                    accumulate_local_potential(
                        alpha_local,
                        weight,
                        xc_grid.vrho(point, 0),
                        phi,
                        gradient_term);
                    continue;
                }

                Eigen::VectorXd gradient_term_alpha = Eigen::VectorXd::Zero(nbasis);
                Eigen::VectorXd gradient_term_beta = Eigen::VectorXd::Zero(nbasis);

                if (xc_grid.vsigma.cols() == 3)
                {
                    const Eigen::Vector3d grad_alpha = density_gradient_at(xc_grid.density.alpha, point);
                    const Eigen::Vector3d grad_beta = density_gradient_at(xc_grid.density.beta, point);

                    const Eigen::Vector3d coefficient_alpha =
                        2.0 * xc_grid.vsigma(point, 0) * grad_alpha + xc_grid.vsigma(point, 1) * grad_beta;
                    const Eigen::Vector3d coefficient_beta =
                        xc_grid.vsigma(point, 1) * grad_alpha + 2.0 * xc_grid.vsigma(point, 2) * grad_beta;

                    gradient_term_alpha = gradient_projection(ao_grid, point, coefficient_alpha);
                    gradient_term_beta = gradient_projection(ao_grid, point, coefficient_beta);
                }

                accumulate_local_potential(
                    alpha_local,
                    weight,
                    xc_grid.vrho(point, 0),
                    phi,
                    gradient_term_alpha);
                accumulate_local_potential(
                    beta_local,
                    weight,
                    xc_grid.vrho(point, 1),
                    phi,
                    gradient_term_beta);
            }

#ifdef USE_OPENMP
#pragma omp critical
#endif
            {
                contribution.alpha.noalias() += alpha_local;
                if (polarized)
                    contribution.beta.noalias() += beta_local;
            }
        }

        contribution.alpha = 0.5 * (contribution.alpha + contribution.alpha.transpose());
        if (contribution.polarized)
        {
            contribution.beta = 0.5 * (contribution.beta + contribution.beta.transpose());
        }
        else
        {
            contribution.beta = contribution.alpha;
        }

        return contribution;
    }

    KSPotentialMatrices combine_ks_potential(
        const Eigen::Ref<const Eigen::MatrixXd> &coulomb,
        const XCMatrixContribution &xc_matrix)
    {
        KSPotentialMatrices potential;
        potential.polarized = xc_matrix.polarized;
        potential.coulomb = coulomb;
        potential.xc_alpha = xc_matrix.alpha;
        potential.xc_beta = xc_matrix.beta;
        potential.alpha = coulomb + xc_matrix.alpha;
        potential.beta = xc_matrix.polarized
                             ? (coulomb + xc_matrix.beta)
                             : potential.alpha;
        return potential;
    }

} // namespace DFT
