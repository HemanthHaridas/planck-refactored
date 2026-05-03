#ifndef DFT_AO_GRID_H
#define DFT_AO_GRID_H

#include <cmath>
#include <expected>
#include <functional>
#include <string>

#include <Eigen/Dense>

#include "base/grid.h"
#include "base/types.h"

namespace DFT
{

    namespace detail
    {

        inline double ao_integer_power(double value, int exponent)
        {
            double result = 1.0;
            for (int i = 0; i < exponent; ++i)
                result *= value;
            return result;
        }

        inline void evaluate_contracted_gaussian_on_grid(
            const HartreeFock::ContractedView &basis_function,
            const Eigen::MatrixXd &grid_points,
            Eigen::Ref<Eigen::VectorXd> values,
            Eigen::Ref<Eigen::VectorXd> grad_x,
            Eigen::Ref<Eigen::VectorXd> grad_y,
            Eigen::Ref<Eigen::VectorXd> grad_z)
        {
            const Eigen::Vector3d center = basis_function.center();
            const int lx = basis_function._cartesian.x();
            const int ly = basis_function._cartesian.y();
            const int lz = basis_function._cartesian.z();

            const auto exponents = basis_function.exponents();
            const auto coefficients = basis_function.coefficients();
            const auto normalizations = basis_function.normalizations();
            const double component_norm = basis_function._component_norm;

            for (Eigen::Index point_index = 0; point_index < grid_points.rows(); ++point_index)
            {
                const double dx = grid_points(point_index, 0) - center.x();
                const double dy = grid_points(point_index, 1) - center.y();
                const double dz = grid_points(point_index, 2) - center.z();
                const double r2 = dx * dx + dy * dy + dz * dz;

                const double x_l = ao_integer_power(dx, lx);
                const double y_l = ao_integer_power(dy, ly);
                const double z_l = ao_integer_power(dz, lz);
                const double x_lm1 = (lx > 0) ? ao_integer_power(dx, lx - 1) : 0.0;
                const double y_lm1 = (ly > 0) ? ao_integer_power(dy, ly - 1) : 0.0;
                const double z_lm1 = (lz > 0) ? ao_integer_power(dz, lz - 1) : 0.0;
                const double x_lp1 = x_l * dx;
                const double y_lp1 = y_l * dy;
                const double z_lp1 = z_l * dz;

                double value = 0.0;
                double dphi_dx = 0.0;
                double dphi_dy = 0.0;
                double dphi_dz = 0.0;

                for (std::size_t primitive = 0; primitive < exponents.size(); ++primitive)
                {
                    const double alpha = exponents[primitive];
                    const double prefactor =
                        component_norm *
                        coefficients[primitive] *
                        normalizations[primitive] *
                        std::exp(-alpha * r2);

                    const double yz = y_l * z_l;
                    const double xz = x_l * z_l;
                    const double xy = x_l * y_l;

                    value += prefactor * x_l * yz;

                    dphi_dx += prefactor *
                               ((lx > 0 ? static_cast<double>(lx) * x_lm1 * yz : 0.0) - 2.0 * alpha * x_lp1 * yz);
                    dphi_dy += prefactor *
                               ((ly > 0 ? static_cast<double>(ly) * y_lm1 * xz : 0.0) - 2.0 * alpha * y_lp1 * xz);
                    dphi_dz += prefactor *
                               ((lz > 0 ? static_cast<double>(lz) * z_lm1 * xy : 0.0) - 2.0 * alpha * z_lp1 * xy);
                }

                values(point_index) = value;
                grad_x(point_index) = dphi_dx;
                grad_y(point_index) = dphi_dy;
                grad_z(point_index) = dphi_dz;
            }
        }

        inline void evaluate_contracted_gaussian_hessian_on_grid(
            const HartreeFock::ContractedView &basis_function,
            const Eigen::MatrixXd &grid_points,
            Eigen::Ref<Eigen::VectorXd> h_xx,
            Eigen::Ref<Eigen::VectorXd> h_xy,
            Eigen::Ref<Eigen::VectorXd> h_xz,
            Eigen::Ref<Eigen::VectorXd> h_yy,
            Eigen::Ref<Eigen::VectorXd> h_yz,
            Eigen::Ref<Eigen::VectorXd> h_zz)
        {
            const Eigen::Vector3d center = basis_function.center();
            const int lx = basis_function._cartesian.x();
            const int ly = basis_function._cartesian.y();
            const int lz = basis_function._cartesian.z();

            const auto exponents = basis_function.exponents();
            const auto coefficients = basis_function.coefficients();
            const auto normalizations = basis_function.normalizations();
            const double component_norm = basis_function._component_norm;

            for (Eigen::Index point_index = 0; point_index < grid_points.rows(); ++point_index)
            {
                const double dx = grid_points(point_index, 0) - center.x();
                const double dy = grid_points(point_index, 1) - center.y();
                const double dz = grid_points(point_index, 2) - center.z();
                const double r2 = dx * dx + dy * dy + dz * dz;

                const double x_l = ao_integer_power(dx, lx);
                const double y_l = ao_integer_power(dy, ly);
                const double z_l = ao_integer_power(dz, lz);
                const double x_lm1 = (lx > 0) ? ao_integer_power(dx, lx - 1) : 0.0;
                const double y_lm1 = (ly > 0) ? ao_integer_power(dy, ly - 1) : 0.0;
                const double z_lm1 = (lz > 0) ? ao_integer_power(dz, lz - 1) : 0.0;
                const double x_lm2 = (lx > 1) ? ao_integer_power(dx, lx - 2) : 0.0;
                const double y_lm2 = (ly > 1) ? ao_integer_power(dy, ly - 2) : 0.0;
                const double z_lm2 = (lz > 1) ? ao_integer_power(dz, lz - 2) : 0.0;
                const double x_lp1 = x_l * dx;
                const double y_lp1 = y_l * dy;
                const double z_lp1 = z_l * dz;

                double dxx = 0.0;
                double dxy = 0.0;
                double dxz = 0.0;
                double dyy = 0.0;
                double dyz = 0.0;
                double dzz = 0.0;

                for (std::size_t primitive = 0; primitive < exponents.size(); ++primitive)
                {
                    const double alpha = exponents[primitive];
                    const double prefactor =
                        component_norm *
                        coefficients[primitive] *
                        normalizations[primitive] *
                        std::exp(-alpha * r2);

                    const double yz = y_l * z_l;
                    const double xz = x_l * z_l;
                    const double xy = x_l * y_l;

                    const double T = x_l * yz;

                    const double Tx =
                        (lx > 0 ? static_cast<double>(lx) * x_lm1 * yz : 0.0);
                    const double Ty =
                        (ly > 0 ? static_cast<double>(ly) * y_lm1 * xz : 0.0);
                    const double Tz =
                        (lz > 0 ? static_cast<double>(lz) * z_lm1 * xy : 0.0);

                    const double Txx =
                        (lx > 1 ? static_cast<double>(lx * (lx - 1)) * x_lm2 * yz : 0.0);
                    const double Tyy =
                        (ly > 1 ? static_cast<double>(ly * (ly - 1)) * y_lm2 * xz : 0.0);
                    const double Tzz =
                        (lz > 1 ? static_cast<double>(lz * (lz - 1)) * z_lm2 * xy : 0.0);

                    const double Txy =
                        (lx > 0 && ly > 0)
                            ? static_cast<double>(lx * ly) * x_lm1 * y_lm1 * z_l
                            : 0.0;
                    const double Txz =
                        (lx > 0 && lz > 0)
                            ? static_cast<double>(lx * lz) * x_lm1 * y_l * z_lm1
                            : 0.0;
                    const double Tyz =
                        (ly > 0 && lz > 0)
                            ? static_cast<double>(ly * lz) * x_l * y_lm1 * z_lm1
                            : 0.0;

                    const double pref_x = -2.0 * alpha * dx * prefactor;
                    const double pref_y = -2.0 * alpha * dy * prefactor;
                    const double pref_z = -2.0 * alpha * dz * prefactor;

                    const double pref_xx =
                        (-2.0 * alpha + 4.0 * alpha * alpha * dx * dx) * prefactor;
                    const double pref_yy =
                        (-2.0 * alpha + 4.0 * alpha * alpha * dy * dy) * prefactor;
                    const double pref_zz =
                        (-2.0 * alpha + 4.0 * alpha * alpha * dz * dz) * prefactor;

                    const double pref_xy = 4.0 * alpha * alpha * dx * dy * prefactor;
                    const double pref_xz = 4.0 * alpha * alpha * dx * dz * prefactor;
                    const double pref_yz = 4.0 * alpha * alpha * dy * dz * prefactor;

                    dxx += pref_xx * T + 2.0 * pref_x * Tx + prefactor * Txx;
                    dyy += pref_yy * T + 2.0 * pref_y * Ty + prefactor * Tyy;
                    dzz += pref_zz * T + 2.0 * pref_z * Tz + prefactor * Tzz;

                    dxy += pref_xy * T + pref_x * Ty + pref_y * Tx + prefactor * Txy;
                    dxz += pref_xz * T + pref_x * Tz + pref_z * Tx + prefactor * Txz;
                    dyz += pref_yz * T + pref_y * Tz + pref_z * Ty + prefactor * Tyz;
                }

                h_xx(point_index) = dxx;
                h_xy(point_index) = dxy;
                h_xz(point_index) = dxz;
                h_yy(point_index) = dyy;
                h_yz(point_index) = dyz;
                h_zz(point_index) = dzz;
            }
        }

    } // namespace detail

    struct AOGridEvaluation
    {
        Eigen::MatrixXd values; // [npoints x nbasis] AO values phi_mu(r_p)
        Eigen::MatrixXd grad_x; // [npoints x nbasis] d(phi_mu)/dx at grid points
        Eigen::MatrixXd grad_y; // [npoints x nbasis] d(phi_mu)/dy at grid points
        Eigen::MatrixXd grad_z; // [npoints x nbasis] d(phi_mu)/dz at grid points

        [[nodiscard]] Eigen::Index npoints() const noexcept
        {
            return values.rows();
        }

        [[nodiscard]] Eigen::Index nbasis() const noexcept
        {
            return values.cols();
        }

        [[nodiscard]] std::expected<std::reference_wrapper<const Eigen::MatrixXd>, std::string> gradient(int axis) const
        {
            switch (axis)
            {
            case 0:
                return std::cref(grad_x);
            case 1:
                return std::cref(grad_y);
            case 2:
                return std::cref(grad_z);
            default:
                return std::unexpected("AOGridEvaluation::gradient axis must be 0, 1, or 2");
            }
        }
    };

    struct AOGridHessian
    {
        Eigen::MatrixXd h_xx;
        Eigen::MatrixXd h_xy;
        Eigen::MatrixXd h_xz;
        Eigen::MatrixXd h_yy;
        Eigen::MatrixXd h_yz;
        Eigen::MatrixXd h_zz;

        [[nodiscard]] Eigen::Index npoints() const noexcept { return h_xx.rows(); }
        [[nodiscard]] Eigen::Index nbasis() const noexcept { return h_xx.cols(); }
    };

    inline std::expected<AOGridHessian, std::string>
    evaluate_ao_hessian_on_grid(
        const HartreeFock::Basis &basis,
        const MolecularGrid &molecular_grid)
    {
        if (molecular_grid.points.cols() != 4)
            return std::unexpected("AO Hessian evaluation requires molecular grid points with 4 columns");

        const Eigen::Index npoints = molecular_grid.points.rows();
        const Eigen::Index nbasis = static_cast<Eigen::Index>(basis.nbasis());

        AOGridHessian H;
        H.h_xx.resize(npoints, nbasis);
        H.h_xy.resize(npoints, nbasis);
        H.h_xz.resize(npoints, nbasis);
        H.h_yy.resize(npoints, nbasis);
        H.h_yz.resize(npoints, nbasis);
        H.h_zz.resize(npoints, nbasis);

        if (nbasis == 0 || npoints == 0)
        {
            H.h_xx.setZero();
            H.h_xy.setZero();
            H.h_xz.setZero();
            H.h_yy.setZero();
            H.h_yz.setZero();
            H.h_zz.setZero();
            return H;
        }

        for (Eigen::Index basis_index = 0; basis_index < nbasis; ++basis_index)
        {
            const auto &basis_function = basis._basis_functions[static_cast<std::size_t>(basis_index)];
            if (basis_function._shell == nullptr)
                return std::unexpected("AO Hessian evaluation encountered a basis function with a null shell");

            detail::evaluate_contracted_gaussian_hessian_on_grid(
                basis_function,
                molecular_grid.points,
                H.h_xx.col(basis_index),
                H.h_xy.col(basis_index),
                H.h_xz.col(basis_index),
                H.h_yy.col(basis_index),
                H.h_yz.col(basis_index),
                H.h_zz.col(basis_index));
        }

        return H;
    }

    inline std::expected<AOGridEvaluation, std::string>
    evaluate_ao_basis_on_grid(
        const HartreeFock::Basis &basis,
        const MolecularGrid &molecular_grid)
    {
        if (molecular_grid.points.cols() != 4)
            return std::unexpected("AO grid evaluation requires molecular grid points with 4 columns");

        const Eigen::Index npoints = molecular_grid.points.rows();
        const Eigen::Index nbasis = static_cast<Eigen::Index>(basis.nbasis());

        AOGridEvaluation evaluation;
        evaluation.values.resize(npoints, nbasis);
        evaluation.grad_x.resize(npoints, nbasis);
        evaluation.grad_y.resize(npoints, nbasis);
        evaluation.grad_z.resize(npoints, nbasis);

        if (nbasis == 0 || npoints == 0)
        {
            evaluation.values.setZero();
            evaluation.grad_x.setZero();
            evaluation.grad_y.setZero();
            evaluation.grad_z.setZero();
            return evaluation;
        }

        for (Eigen::Index basis_index = 0; basis_index < nbasis; ++basis_index)
        {
            const auto &basis_function = basis._basis_functions[static_cast<std::size_t>(basis_index)];
            if (basis_function._shell == nullptr)
                return std::unexpected("AO grid evaluation encountered a basis function with a null shell");

            detail::evaluate_contracted_gaussian_on_grid(
                basis_function,
                molecular_grid.points,
                evaluation.values.col(basis_index),
                evaluation.grad_x.col(basis_index),
                evaluation.grad_y.col(basis_index),
                evaluation.grad_z.col(basis_index));
        }

        return evaluation;
    }

} // namespace DFT

#endif // DFT_AO_GRID_H
