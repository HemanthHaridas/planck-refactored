#include "udh_ks_response.h"

#include <cmath>
#include <exception>
#include <utility>

#include "integrals/base.h"

namespace DFT::Gradient
{
    namespace
    {
        bool valid_matrix(const Eigen::MatrixXd &q, Eigen::Index n)
        {
            return n > 0 && q.rows() == n && q.cols() == n && q.allFinite() &&
                (q - q.transpose()).cwiseAbs().maxCoeff() <=
                    1e-12 * (1.0 + q.cwiseAbs().maxCoeff());
        }

        bool valid_pair(const UDHSpinMatrices &q, Eigen::Index n)
        { return valid_matrix(q.alpha, n) && valid_matrix(q.beta, n); }

        bool valid_config(const UDHKSResponseOperator &op)
        {
            return op.nbasis > 0 && std::isfinite(op.exact_exchange) && op.exact_exchange >= 0.0 &&
                op.coulomb && op.xc && (op.exact_exchange == 0.0 || op.exchange);
        }

        std::expected<Eigen::MatrixXd, std::string> matrix_action(
            const UDHMatrixResponseFn &fn, const Eigen::MatrixXd &q,
            Eigen::Index n, const char *label)
        {
            try
            {
                auto value = fn(q);
                if (!value) return std::unexpected(std::string(label) + ": " + value.error());
                if (!valid_matrix(*value, n))
                    return std::unexpected(std::string(label) + ": invalid response matrix.");
                return value;
            }
            catch (const std::exception &error)
            { return std::unexpected(std::string(label) + ": " + error.what()); }
            catch (...)
            { return std::unexpected(std::string(label) + ": unknown callback exception."); }
        }

        std::expected<UDHSpinMatrices, std::string> xc_action(
            const UDHXCResponseFn &fn, const UDHSpinMatrices &q, Eigen::Index n, const char *label)
        {
            try
            {
                auto value = fn(q);
                if (!value) return std::unexpected(std::string(label) + ": " + value.error());
                if (!valid_pair(*value, n))
                    return std::unexpected(std::string(label) + ": invalid spin response matrices.");
                return value;
            }
            catch (const std::exception &error)
            { return std::unexpected(std::string(label) + ": " + error.what()); }
            catch (...)
            { return std::unexpected(std::string(label) + ": unknown callback exception."); }
        }
    }

    std::expected<UDHKSResponseOperator, std::string>
    make_udh_ks_response_operator(Eigen::Index nbasis, double exact_exchange,
        UDHMatrixResponseFn coulomb, UDHMatrixResponseFn exchange, UDHXCResponseFn xc)
    {
        UDHKSResponseOperator op{nbasis, exact_exchange, std::move(coulomb),
                                std::move(exchange), std::move(xc)};
        if (!valid_config(op))
            return std::unexpected("UDH U2: invalid physical response configuration.");
        return op;
    }

    std::expected<UDHKSResponseChannels, std::string>
    UDHKSResponseOperator::apply_channels(const UDHSpinMatrices &trial) const
    {
        if (!valid_config(*this) || !valid_pair(trial, nbasis))
            return std::unexpected("UDH U2: invalid configuration or symmetric AO trial densities.");
        const Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(nbasis, nbasis);
        const Eigen::MatrixXd sum = trial.alpha + trial.beta;
        if (!sum.allFinite()) return std::unexpected("UDH U2: nonfinite total trial density.");
        auto j = matrix_action(coulomb, sum, nbasis, "UDH U2 J");
        if (!j) return std::unexpected(j.error());
        UDHKSResponseChannels out;
        out.coulomb = std::move(*j);
        out.exchange = {zero, zero};
        if (exact_exchange != 0.0)
        {
            auto ka = matrix_action(exchange, trial.alpha, nbasis, "UDH U2 K-alpha");
            if (!ka) return std::unexpected(ka.error());
            auto kb = matrix_action(exchange, trial.beta, nbasis, "UDH U2 K-beta");
            if (!kb) return std::unexpected(kb.error());
            out.exchange = {-exact_exchange * *ka, -exact_exchange * *kb};
        }
        // Two independent spin-only actions expose aa/ba and ab/bb without
        // imposing equality of the cross matrices. Adjointness is joint-spin.
        auto xa = xc_action(xc, {trial.alpha, zero}, nbasis, "UDH U2 XC-from-alpha");
        if (!xa) return std::unexpected(xa.error());
        auto xb = xc_action(xc, {zero, trial.beta}, nbasis, "UDH U2 XC-from-beta");
        if (!xb) return std::unexpected(xb.error());
        out.xc_from_alpha = std::move(*xa);
        out.xc_from_beta = std::move(*xb);
        out.total = {
            out.coulomb + out.exchange.alpha + out.xc_from_alpha.alpha + out.xc_from_beta.alpha,
            out.coulomb + out.exchange.beta + out.xc_from_alpha.beta + out.xc_from_beta.beta};
        if (!valid_pair(out.exchange, nbasis) || !valid_pair(out.total, nbasis))
            return std::unexpected("UDH U2: invalid assembled physical response.");
        return out;
    }

    std::expected<UDHSpinMatrices, std::string>
    UDHKSResponseOperator::apply(const UDHSpinMatrices &trial) const
    {
        auto result = apply_channels(trial);
        if (!result) return std::unexpected(result.error());
        return std::move(result->total);
    }

    std::expected<UDHKSResponseOperator, std::string>
    make_udh_direct_ks_response_operator(const UDHKSResponseInputs &inputs)
    {
        const auto n = inputs.ground_density.alpha.rows();
        if (!inputs.shell_pairs || inputs.shell_pairs->empty() || !inputs.molecular_grid ||
            !inputs.ao_grid || !inputs.exchange_functional || !inputs.correlation_functional ||
            !valid_pair(inputs.ground_density, n) || !std::isfinite(inputs.tol_eri) || inputs.tol_eri < 0.0)
            return std::unexpected("UDH U2: invalid fixed-geometry inputs.");
        for (const auto &pair : *inputs.shell_pairs)
            if (static_cast<std::size_t>(pair.A._index) >= static_cast<std::size_t>(n) ||
                static_cast<std::size_t>(pair.B._index) >= static_cast<std::size_t>(n))
                return std::unexpected("UDH U2: shell-pair indices do not match the AO basis.");
        const auto &ao = *inputs.ao_grid;
        const auto g = ao.npoints();
        const auto valid_ao = [g, n](const Eigen::MatrixXd &m)
        { return m.rows() == g && m.cols() == n && m.allFinite(); };
        if (g == 0 || !valid_ao(ao.values) || !valid_ao(ao.grad_x) ||
            !valid_ao(ao.grad_y) || !valid_ao(ao.grad_z) ||
            inputs.molecular_grid->points.rows() != g || inputs.molecular_grid->points.cols() != 4 ||
            !inputs.molecular_grid->points.allFinite())
            return std::unexpected("UDH U2: invalid AO/grid dimensions or values.");
        const auto &x = *inputs.exchange_functional;
        const auto &c = *inputs.correlation_functional;
        const auto has_kernel = [](const XC::Functional &f)
        { return f.get()->info && (f.get()->info->flags & XC_FLAGS_HAVE_FXC) &&
                                 (f.get()->info->flags & XC_FLAGS_HAVE_VXC); };
        if (x.spin() != XC::Spin::Polarized || c.spin() != XC::Spin::Polarized ||
            x.is_range_separated() || c.is_range_separated() ||
            !has_kernel(x) || !has_kernel(c) ||
            !((x.is_lda_like() && c.is_lda_like()) || (x.is_gga_like() && c.is_gga_like())))
            return std::unexpected("UDH U2: requires polarized global LDA/GGA functionals.");
        // Copy the ground density and input metadata, not a reference to the
        // caller's temporary Inputs object. Geometry/functionals are borrowed.
        const auto fixed = inputs;
        UDHMatrixResponseFn j = [pairs=inputs.shell_pairs, n, engine=inputs.engine, tol=inputs.tol_eri]
            (const Eigen::Ref<const Eigen::MatrixXd> &q)
            -> std::expected<Eigen::MatrixXd, std::string>
        { return _compute_2e_j_direct(*pairs, q, static_cast<std::size_t>(n),
            engine, HartreeFock::ERIKernel::Coulomb, 0.0, tol, nullptr); };
        UDHMatrixResponseFn k = [pairs=inputs.shell_pairs, n, engine=inputs.engine, tol=inputs.tol_eri]
            (const Eigen::Ref<const Eigen::MatrixXd> &q)
            -> std::expected<Eigen::MatrixXd, std::string>
        { return _compute_2e_k_direct(*pairs, q, static_cast<std::size_t>(n),
            engine, HartreeFock::ERIKernel::Coulomb, 0.0, tol, nullptr); };
        UDHXCResponseFn xc = [fixed](const UDHSpinMatrices &q)
            -> std::expected<UDHSpinMatrices, std::string>
        {
            auto value = Driver::compute_analytic_xc_hessian_vector_product_polarized(
                *fixed.molecular_grid, *fixed.ao_grid, fixed.ground_density.alpha,
                fixed.ground_density.beta, q.alpha, q.beta,
                *fixed.exchange_functional, *fixed.correlation_functional);
            if (!value) return std::unexpected(value.error());
            return UDHSpinMatrices{std::move(value->first), std::move(value->second)};
        };
        return make_udh_ks_response_operator(n, inputs.exact_exchange, std::move(j), std::move(k), std::move(xc));
    }
}
