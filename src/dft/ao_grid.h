#ifndef DFT_AO_GRID_H
#define DFT_AO_GRID_H

#include <algorithm>
#include <array>
#include <cmath>
#include <expected>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "base/grid.h"
#include "base/types.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

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
            // Evaluate one Cartesian contracted AO and its first derivatives on
            // every quadrature point. The DFT stack reuses this same object for
            // density assembly, KS matrix construction, and analytic XC terms.
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

    // Grid points are processed in fixed blocks of kAOBlock consecutive
    // points, aligned at point 0. The block is the unit of AO screening and of
    // every blocked XC contraction (density, XC matrix, XC gradient).
    constexpr Eigen::Index kAOBlock = 128;

    // An AO column is kept in a block when its value or any first derivative
    // exceeds this anywhere in the block (docs/DFT_DEFAULT_GRID_SCOPE.md, S7).
    constexpr double kAOScreenTolerance = 1e-12;

    // One block's AO values and gradients. cols == nullptr means every AO, in
    // order; otherwise column j of phi/g* is AO (*cols)[j] (sorted), and all
    // other AOs are negligible on this block.
    struct AOBlockView
    {
        Eigen::Index p0;
        Eigen::Index n;
        const std::vector<int> *cols;
        Eigen::Ref<const Eigen::MatrixXd> phi;
        std::array<Eigen::Ref<const Eigen::MatrixXd>, 3> grad;

        [[nodiscard]] Eigen::Index ncols() const noexcept { return phi.cols(); }
    };

    inline Eigen::Index ao_block_count(Eigen::Index npoints) noexcept
    {
        return (npoints + kAOBlock - 1) / kAOBlock;
    }

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

        [[nodiscard]] Eigen::Index nblocks() const noexcept { return ao_block_count(npoints()); }

        [[nodiscard]] AOBlockView block(Eigen::Index b) const
        {
            const Eigen::Index p0 = b * kAOBlock;
            const Eigen::Index n = std::min(kAOBlock, npoints() - p0);
            return {p0, n, nullptr, values.middleRows(p0, n),
                    {grad_x.middleRows(p0, n), grad_y.middleRows(p0, n), grad_z.middleRows(p0, n)}};
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

    // AO values and gradients stored per kAOBlock-point block, keeping only
    // the AO columns that are significant on that block. On extended systems
    // most AOs vanish on most blocks (water12: ~27 % kept), which cuts the
    // stored arrays and the blocked XC GEMMs.
    struct ScreenedAOGrid
    {
        Eigen::Index npts = 0;
        Eigen::Index nb = 0;
        std::vector<std::vector<int>> cols; // per block, sorted AO indices
        // One allocation: block b holds phi, d/dx, d/dy, d/dz (each n x k,
        // column-major) back to back from offset[b].
        std::vector<double> buffer;
        std::vector<std::size_t> offset;

        [[nodiscard]] Eigen::Index npoints() const noexcept { return npts; }
        [[nodiscard]] Eigen::Index nbasis() const noexcept { return nb; }
        [[nodiscard]] Eigen::Index nblocks() const noexcept { return static_cast<Eigen::Index>(cols.size()); }
        [[nodiscard]] Eigen::Index block_rows(Eigen::Index b) const noexcept
        {
            return std::min(kAOBlock, npts - b * kAOBlock);
        }

        [[nodiscard]] AOBlockView block(Eigen::Index b) const
        {
            const auto &c = cols[static_cast<std::size_t>(b)];
            const Eigen::Index n = block_rows(b);
            const Eigen::Index k = static_cast<Eigen::Index>(c.size());
            const double *d = buffer.data() + offset[static_cast<std::size_t>(b)];
            using CMap = Eigen::Map<const Eigen::MatrixXd>;
            return {b * kAOBlock, n, &c, CMap(d, n, k), {CMap(d + n * k, n, k), CMap(d + 2 * n * k, n, k), CMap(d + 3 * n * k, n, k)}};
        }

        // Number of stored (point, AO) entries, for reporting.
        [[nodiscard]] Eigen::Index stored_entries() const noexcept
        {
            return static_cast<Eigen::Index>(buffer.size() / 4);
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

        [[nodiscard]] Eigen::Index npoints() const noexcept
        {
            return h_xx.rows();
        }
        [[nodiscard]] Eigen::Index nbasis() const noexcept
        {
            return h_xx.cols();
        }
    };

    // cols != nullptr: only those AOs, column j = AO (*cols)[j].
    inline std::expected<AOGridHessian, std::string>
    evaluate_ao_hessian_on_grid(
        const HartreeFock::Basis &basis,
        const MolecularGrid &molecular_grid,
        const std::vector<int> *cols = nullptr)
    {
        if (molecular_grid.points.cols() != 4)
            return std::unexpected("AO Hessian evaluation requires molecular grid points with 4 columns");

        const Eigen::Index npoints = molecular_grid.points.rows();
        const Eigen::Index nbasis = cols ? static_cast<Eigen::Index>(cols->size())
                                         : static_cast<Eigen::Index>(basis.nbasis());

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

        for (const auto &basis_function : basis._basis_functions)
            if (basis_function._shell == nullptr)
                return std::unexpected("AO Hessian evaluation encountered a basis function with a null shell");

        // Columns are disjoint, so this is bitwise at any thread count. Serial
        // when called from inside a caller's parallel region.
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (!omp_in_parallel())
#endif
        for (Eigen::Index basis_index = 0; basis_index < nbasis; ++basis_index)
        {
            const auto &basis_function = basis._basis_functions[static_cast<std::size_t>(
                cols ? (*cols)[static_cast<std::size_t>(basis_index)] : basis_index)];
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
        // Point-major storage (rows = grid points, cols = AOs) keeps the later
        // rho/Vxc contractions simple because those routines work one
        // quadrature point at a time.
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

        for (const auto &basis_function : basis._basis_functions)
            if (basis_function._shell == nullptr)
                return std::unexpected("AO grid evaluation encountered a basis function with a null shell");

        // Columns are disjoint, so this is bitwise at any thread count. Serial
        // when called from inside a caller's parallel region.
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (!omp_in_parallel())
#endif
        for (Eigen::Index basis_index = 0; basis_index < nbasis; ++basis_index)
        {
            const auto &basis_function = basis._basis_functions[static_cast<std::size_t>(basis_index)];
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

    // Pass 1 evaluates every AO per block and keeps the columns whose value
    // or gradient exceeds `tolerance` anywhere in the block; pass 2 evaluates
    // the kept columns straight into one contiguous buffer (per-block
    // allocations fragmented the heap by ~15 % of the stored size). Blocks are
    // disjoint, so both passes are bitwise at any thread count.
    // ponytail: pass 1 evaluates every AO only to decide what to keep; prescreen
    // shells by their extent against the block's bounding sphere if AO
    // evaluation ever matters (it is ~1 % of a gradient run today).
    inline std::expected<ScreenedAOGrid, std::string>
    evaluate_screened_ao_basis_on_grid(
        const HartreeFock::Basis &basis,
        const MolecularGrid &molecular_grid,
        double tolerance = kAOScreenTolerance)
    {
        if (molecular_grid.points.cols() != 4)
            return std::unexpected("AO grid evaluation requires molecular grid points with 4 columns");
        for (const auto &basis_function : basis._basis_functions)
            if (basis_function._shell == nullptr)
                return std::unexpected("AO grid evaluation encountered a basis function with a null shell");

        ScreenedAOGrid S;
        S.npts = molecular_grid.points.rows();
        S.nb = static_cast<Eigen::Index>(basis.nbasis());
        const Eigen::Index nblocks = ao_block_count(S.npts);
        S.cols.resize(static_cast<std::size_t>(nblocks));
        const auto bf = [&](Eigen::Index mu) -> const auto & {
            return basis._basis_functions[static_cast<std::size_t>(mu)];
        };

#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (!omp_in_parallel())
#endif
        for (Eigen::Index b = 0; b < nblocks; ++b)
        {
            const Eigen::Index n = S.block_rows(b);
            const Eigen::MatrixXd points = molecular_grid.points.middleRows(b * kAOBlock, n);
            Eigen::VectorXd v(n), gx(n), gy(n), gz(n);
            auto &keep = S.cols[static_cast<std::size_t>(b)];
            for (Eigen::Index mu = 0; mu < S.nb; ++mu)
            {
                detail::evaluate_contracted_gaussian_on_grid(bf(mu), points, v, gx, gy, gz);
                const double m = std::max({v.cwiseAbs().maxCoeff(), gx.cwiseAbs().maxCoeff(),
                                           gy.cwiseAbs().maxCoeff(), gz.cwiseAbs().maxCoeff()});
                if (m > tolerance)
                    keep.push_back(static_cast<int>(mu));
            }
        }

        S.offset.resize(static_cast<std::size_t>(nblocks));
        std::size_t total = 0;
        for (Eigen::Index b = 0; b < nblocks; ++b)
        {
            S.offset[static_cast<std::size_t>(b)] = total;
            total += 4 * static_cast<std::size_t>(S.block_rows(b)) * S.cols[static_cast<std::size_t>(b)].size();
        }
        S.buffer.resize(total);

#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (!omp_in_parallel())
#endif
        for (Eigen::Index b = 0; b < nblocks; ++b)
        {
            const Eigen::Index n = S.block_rows(b);
            const auto &c = S.cols[static_cast<std::size_t>(b)];
            const Eigen::Index k = static_cast<Eigen::Index>(c.size());
            const Eigen::MatrixXd points = molecular_grid.points.middleRows(b * kAOBlock, n);
            double *d = S.buffer.data() + S.offset[static_cast<std::size_t>(b)];
            Eigen::Map<Eigen::MatrixXd> phi(d, n, k), gx(d + n * k, n, k), gy(d + 2 * n * k, n, k),
                gz(d + 3 * n * k, n, k);
            for (Eigen::Index j = 0; j < k; ++j)
                detail::evaluate_contracted_gaussian_on_grid(bf(c[static_cast<std::size_t>(j)]), points,
                                                             phi.col(j), gx.col(j), gy.col(j), gz.col(j));
        }
        return S;
    }

    // Dense arrays for the consumers not yet ported to blocks. Screened-out
    // entries are zero, so they see the same AOs as the blocked SCF.
    inline AOGridEvaluation to_dense(const ScreenedAOGrid &S)
    {
        AOGridEvaluation E;
        E.values = Eigen::MatrixXd::Zero(S.npts, S.nb);
        E.grad_x = Eigen::MatrixXd::Zero(S.npts, S.nb);
        E.grad_y = Eigen::MatrixXd::Zero(S.npts, S.nb);
        E.grad_z = Eigen::MatrixXd::Zero(S.npts, S.nb);
        Eigen::MatrixXd *out[4] = {&E.values, &E.grad_x, &E.grad_y, &E.grad_z};
        for (Eigen::Index b = 0; b < S.nblocks(); ++b)
        {
            const AOBlockView blk = S.block(b);
            out[0]->middleRows(blk.p0, blk.n)(Eigen::all, *blk.cols) = blk.phi;
            for (int k = 0; k < 3; ++k)
                out[k + 1]->middleRows(blk.p0, blk.n)(Eigen::all, *blk.cols) = blk.grad[k];
        }
        return E;
    }

} // namespace DFT

#endif // DFT_AO_GRID_H
