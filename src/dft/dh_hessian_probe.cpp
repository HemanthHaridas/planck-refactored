#include "dh_hessian_probe.h"
#include "dh_probe_gmres.h"
#include "response_packing.h"

#include <array>
#include <fstream>
#include <iomanip>

namespace DFT::Driver
{
    namespace
    {
        using Channels = std::array<Eigen::VectorXd, 4>;
        const std::array<const char *, 4> names = {"orbital", "J", "K", "XC"};

        Eigen::VectorXd pack(const Eigen::MatrixXd &matrix)
        {
            Eigen::VectorXd vector(matrix.size());
            for (Eigen::Index a = 0; a < matrix.rows(); ++a)
                for (Eigen::Index i = 0; i < matrix.cols(); ++i)
                    vector(a * matrix.cols() + i) = matrix(a, i);
            return vector;
        }

        Eigen::MatrixXd unpack(const Eigen::VectorXd &vector, int no, int nv)
        {
            Eigen::MatrixXd matrix(nv, no);
            for (int a = 0; a < nv; ++a)
                for (int i = 0; i < no; ++i)
                    matrix(a, i) = vector(a * no + i);
            return matrix;
        }

        void matrix(std::ostream &log, const std::string &name, const Eigen::MatrixXd &value)
        {
            log << "MATRIX " << name << ' ' << value.rows() << ' ' << value.cols() << '\n';
            for (Eigen::Index r = 0; r < value.rows(); ++r)
            {
                for (Eigen::Index c = 0; c < value.cols(); ++c)
                    log << (c ? " " : "") << value(r, c);
                log << '\n';
            }
            log.flush();
        }

        double compare(std::ostream &log, const std::string &name,
            const Eigen::MatrixXd &reference, const Eigen::MatrixXd &candidate)
        {
            matrix(log, name + ".dense", reference);
            matrix(log, name + ".candidate", candidate);
            const Eigen::MatrixXd delta = candidate - reference;
            matrix(log, name + ".delta", delta);
            const double error = delta.cwiseAbs().maxCoeff();
            log << "SCALAR " << name << ".max_abs " << error << '\n';
            return error;
        }
    }

    std::expected<Eigen::MatrixXd, std::string> run_dh_hessian_swap_probe(
        const Gradient::DHGradientDriverInputs &inputs,
        const Gradient::DHGradientDriverContract &dense_contract,
        const KsOrbitalHessianInputs &shared_inputs,
        const Eigen::MatrixXd &ks_gradient,
        const std::string &log_path)
    {
        std::ofstream log(log_path);
        if (!log) return std::unexpected("DH Hessian probe: cannot open " + log_path);
        log << std::scientific << std::setprecision(17);
        log << "DH_HESSIAN_SWAP_PROBE 1\n"
            << "NOTE candidate=build_ks_orbital_hessian_op; control=apply_dh_eq27_hessian\n"
            << "NOTE dense columns use Eq27; this is a swap/linearity test, not an independent physics oracle\n"
            << "NOTE packing=a*nocc+i; gradient frame=driver standard; only Z changes\n";
        const auto fail = [&](const std::string &message) -> std::expected<Eigen::MatrixXd, std::string>
        {
            log << "ERROR " << message << "\nSTATUS FAILURE\n";
            return std::unexpected("DH Hessian probe: " + message + "; see " + log_path);
        };
        if (!inputs.pt2_result) return fail("missing PT2 result");
        const int no = inputs.pt2_result->n_occ, nv = inputs.pt2_result->n_virt;
        const int n = no * nv;
        if (no <= 0 || nv <= 0 || n > 128) return fail("probe requires 1..128 orbital pairs");
        const Eigen::MatrixXd co = inputs.mo_coeff.leftCols(no);
        const Eigen::MatrixXd cv = inputs.mo_coeff.rightCols(nv);
        const auto &response = inputs.response_operator;
        const Eigen::VectorXd rhs = -pack(dense_contract.lagrangian_rhs.total_ai);
        const Eigen::VectorXd dense_z = pack(inputs.z_ai);
        Eigen::VectorXd diagonal(n);
        for (int a = 0; a < nv; ++a)
            for (int i = 0; i < no; ++i)
                diagonal(a * no + i) = inputs.orbital_energies(no + a) - inputs.orbital_energies(i);
        log << "SCALAR nocc " << no << "\nSCALAR nvirt " << nv
            << "\nSCALAR kernel_scale " << shared_inputs.kernel_scale
            << "\nSCALAR exact_exchange " << response.exact_exchange << '\n';
        matrix(log, "L_ai", dense_contract.lagrangian_rhs.total_ai);

        const auto eq27_channels = [&](const Eigen::VectorXd &x, Eigen::VectorXd *total = nullptr)
            -> std::expected<Channels, std::string>
        {
            auto value = Gradient::apply_dh_eq27_hessian(
                unpack(x, no, nv), co, cv, inputs.orbital_energies, response);
            if (!value) return std::unexpected(value.error());
            if (total) *total = pack(value->total_ai);
            return Channels{pack(value->orbital_energy_ai), pack(value->coulomb_ai),
                pack(value->exchange_ai), pack(value->xc_ai)};
        };
        const auto sum = [](const Channels &c) -> Eigen::VectorXd { return c[0] + c[1] + c[2] + c[3]; };
        const Probe::Action eq27 = [&](const Eigen::VectorXd &x) -> std::expected<Eigen::VectorXd, std::string>
        {
            Eigen::VectorXd total;
            auto c = eq27_channels(x, &total);
            if (!c) return std::unexpected(c.error());
            return total;
        };

        // Keep shared_inputs alive: the original shared operator captures it by reference.
        // Decompose that convention separately, including its ov projection, to
        // distinguish density factors and packing from the iterative solver.
        const auto shared_op = build_ks_orbital_hessian_op(shared_inputs);
        const auto shared_channels = [&](const Eigen::VectorXd &x) -> std::expected<Channels, std::string>
        {
            const Eigen::MatrixXd raw = cv * unpack(x, no, nv) * co.transpose();
            const Eigen::MatrixXd dp = raw + raw.transpose();
            auto j = response.coulomb(dp);
            auto k = response.exchange(dp);
            auto xc = response.xc(dp);
            if (!j || !k || !xc)
                return std::unexpected("shared channel callback failure (must not become a zero action)");
            const auto project = [&](const Eigen::MatrixXd &m) -> Eigen::VectorXd
            { return shared_inputs.kernel_scale * pack_hessian_vector_product_cphf_order(m, co, cv); };
            return Channels{diagonal.cwiseProduct(x), project(*j),
                project(-0.5 * response.exact_exchange * *k), project(*xc)};
        };
        double reconstruction_error = 0.0;
        const Probe::Action shared = [&](const Eigen::VectorXd &x) -> std::expected<Eigen::VectorXd, std::string>
        {
            // The original callback returns zero on XC failure. Check the
            // channels here as well, so that failure cannot look like convergence.
            auto channels = shared_channels(x);
            if (!channels) return std::unexpected(channels.error());
            Eigen::VectorXd actual = shared_op(x);
            if (actual.size() != n || !actual.allFinite())
                return std::unexpected("invalid shared operator output");
            const double error = (actual - sum(*channels)).cwiseAbs().maxCoeff();
            reconstruction_error = std::max(reconstruction_error, error);
            return actual;
        };

        std::array<Eigen::MatrixXd, 4> hd, hs;
        for (int c = 0; c < 4; ++c)
        {
            hd[c] = Eigen::MatrixXd::Zero(n, n);
            hs[c] = Eigen::MatrixXd::Zero(n, n);
        }
        Eigen::MatrixXd shared_matrix(n, n), dense_matrix(n, n);
        for (int col = 0; col < n; ++col)
        {
            const Eigen::VectorXd unit = Eigen::VectorXd::Unit(n, col);
            Eigen::VectorXd dense_column;
            auto d = eq27_channels(unit, &dense_column), s = shared_channels(unit);
            auto actual = shared(unit);
            if (!d) return fail("Eq27 basis direction: " + d.error());
            if (!s) return fail("shared basis channels: " + s.error());
            if (!actual) return fail("shared basis action: " + actual.error());
            for (int c = 0; c < 4; ++c)
            {
                hd[c].col(col) = (*d)[c];
                hs[c].col(col) = (*s)[c];
            }
            shared_matrix.col(col) = *actual;
            dense_matrix.col(col) = dense_column;
        }
        double action_error = 0.0;
        for (int c = 0; c < 4; ++c)
        {
            action_error = std::max(action_error, compare(log, std::string("H.") + names[c], hd[c], hs[c]));
        }
        action_error = std::max(action_error, compare(log, "H.total", dense_matrix, shared_matrix));
        const Eigen::MatrixXd reconstructed_shared = hs[0] + hs[1] + hs[2] + hs[3];
        action_error = std::max(action_error, compare(log, "H.shared_reconstruction", reconstructed_shared, shared_matrix));
        log << "SCALAR H.dense.asymmetry " << (dense_matrix - dense_matrix.transpose()).cwiseAbs().maxCoeff()
            << "\nSCALAR H.shared.asymmetry " << (shared_matrix - shared_matrix.transpose()).cwiseAbs().maxCoeff() << '\n';

        // Mixed directions are essential: constructing columns from an action
        // cannot reveal density-dependent screening/nonlinearity between columns.
        Eigen::VectorXd mixed(n);
        for (int p = 0; p < n; ++p) mixed(p) = std::sin(0.71 * (p + 1)) + 0.3 * std::cos(1.13 * (p + 1));
        mixed.normalize();
        const std::array<Eigen::VectorXd, 4> probes = {mixed, Eigen::VectorXd(0.013 * mixed),
            Eigen::VectorXd(-1.7 * mixed), dense_z};
        for (std::size_t p = 0; p < probes.size(); ++p)
        {
            auto d = eq27_channels(probes[p]), s = shared_channels(probes[p]);
            auto actual = shared(probes[p]);
            if (!d) return fail("Eq27 mixed direction: " + d.error());
            if (!s) return fail("shared mixed channels: " + s.error());
            if (!actual) return fail("shared mixed action: " + actual.error());
            const std::string prefix = "direction" + std::to_string(p);
            matrix(log, prefix + ".x", probes[p]);
            for (int c = 0; c < 4; ++c)
            {
                action_error = std::max(action_error, compare(log, prefix + "." + names[c], hd[c] * probes[p], (*s)[c]));
                action_error = std::max(action_error, compare(log, prefix + ".control." + names[c], hd[c] * probes[p], (*d)[c]));
            }
            action_error = std::max(action_error, compare(log, prefix + ".shared_linearity", shared_matrix * probes[p], *actual));
        }

        const Probe::Action dense_action = [&](const Eigen::VectorXd &x) -> std::expected<Eigen::VectorXd, std::string>
        { return Eigen::VectorXd(dense_matrix * x); };
        const auto solve = [&](const std::string &name, const Probe::Action &action) -> std::expected<Probe::Solve, std::string>
        {
            auto result = Probe::gmres(action, rhs, diagonal);
            if (!result) { log << "ERROR " << name << ' ' << result.error() << '\n'; return result; }
            for (std::size_t i = 0; i < result->true_residuals.size(); ++i)
                log << "ITERATION " << name << ' ' << i << ' ' << result->true_residuals[i] << '\n';
            log << "SCALAR " << name << ".converged " << result->converged << '\n';
            compare(log, name + ".Z_ai", inputs.z_ai, unpack(result->x, no, nv));
            auto self = action(result->x), reference = eq27(result->x);
            if (!self || !reference) return std::unexpected("final residual callback failed");
            matrix(log, name + ".residual_self", *self - rhs);
            matrix(log, name + ".residual_dense", dense_matrix * result->x - rhs);
            matrix(log, name + ".residual_eq27", *reference - rhs);
            return result;
        };
        auto control_dense = solve("gmres_dense", dense_action);
        auto control_action = solve("gmres_eq27", eq27);
        auto candidate = solve("gmres_shared", shared);
        if (!control_dense || !control_action || !candidate)
            return fail("one or more solves returned an action error");
        if (!control_dense->converged || !control_action->converged || !candidate->converged)
            return fail("one or more GMRES solves did not converge; no dense fallback");

        // Reuse identical amplitudes, RHS, response, geometry, quadrature and
        // derivative arrays. No SCF or PT2 rerun is allowed between the contracts.
        auto swapped_inputs = inputs;
        swapped_inputs.z_ai = unpack(candidate->x, no, nv);
        auto swapped = Gradient::build_dh_gradient_driver_contract(swapped_inputs);
        if (!swapped) return fail(swapped.error());
        auto gd = Gradient::build_dh_eq33_pt2_correction_gradient(dense_contract);
        auto gs = Gradient::build_dh_eq33_pt2_correction_gradient(*swapped);
        if (!gd || !gs) return fail("correction gradient assembly failed");
        const double rhs_error = compare(log, "contract.L_ai", dense_contract.lagrangian_rhs.total_ai, swapped->lagrangian_rhs.total_ai);
        compare(log, "D.raw", dense_contract.relaxed_density.raw_mo, swapped->relaxed_density.raw_mo);
        compare(log, "D.symmetric", dense_contract.relaxed_density.symmetric_mo, swapped->relaxed_density.symmetric_mo);
        compare(log, "W.raw", dense_contract.overlap_density.raw_mo, swapped->overlap_density.raw_mo);
        compare(log, "W.symmetric", dense_contract.overlap_density.symmetric_mo, swapped->overlap_density.symmetric_mo);
        compare(log, "gradient.h", dense_contract.non_xc_gradient.one_electron, swapped->non_xc_gradient.one_electron);
        compare(log, "gradient.S", dense_contract.non_xc_gradient.overlap, swapped->non_xc_gradient.overlap);
        compare(log, "gradient.ERI_separable", dense_contract.non_xc_gradient.two_electron_separable, swapped->non_xc_gradient.two_electron_separable);
        compare(log, "gradient.ERI_nonseparable", dense_contract.non_xc_gradient.two_electron_nonseparable, swapped->non_xc_gradient.two_electron_nonseparable);
        compare(log, "gradient.XC_P", dense_contract.xc_ii_gradient.p_side_fixed, swapped->xc_ii_gradient.p_side_fixed);
        compare(log, "gradient.XC_D", dense_contract.xc_ii_gradient.d_side_ao, swapped->xc_ii_gradient.d_side_ao);
        compare(log, "gradient.XC_partition", dense_contract.xc_ii_gradient.becke_partition, swapped->xc_ii_gradient.becke_partition);
        compare(log, "gradient.XC_point", dense_contract.xc_ii_gradient.point_translation, swapped->xc_ii_gradient.point_translation);
        compare(log, "gradient.PT2", gd->total, gs->total);
        matrix(log, "gradient.KS_shared", ks_gradient);
        const double gradient_error = compare(log, "gradient.total", Eigen::MatrixXd(ks_gradient + gd->total), Eigen::MatrixXd(ks_gradient + gs->total));
        const double z_error = (candidate->x - dense_z).cwiseAbs().maxCoeff();
        const double reference_residual = (dense_matrix * candidate->x - rhs).cwiseAbs().maxCoeff();
        action_error = std::max(action_error, reconstruction_error);
        const double control_z_error = std::max((control_dense->x - dense_z).cwiseAbs().maxCoeff(),
            (control_action->x - dense_z).cwiseAbs().maxCoeff());
        log << "SCALAR audit.action_max_abs " << action_error
            << "\nSCALAR audit.reconstruction_max_abs " << reconstruction_error
            << "\nSCALAR audit.control_z_max_abs " << control_z_error
            << "\nSCALAR audit.z_max_abs " << z_error
            << "\nSCALAR audit.reference_residual " << reference_residual
            << "\nSCALAR audit.gradient_max_abs " << gradient_error << '\n';
        const bool agrees = action_error <= 1e-10 && control_z_error <= 1e-9 && z_error <= 1e-9 &&
            reference_residual <= 1e-9 && rhs_error <= 1e-12 && gradient_error <= 1e-9;
        log << "STATUS " << (agrees ? "PASS" : "DIFFERENCE") << '\n';
        log.flush();
        if (!log) return std::unexpected("DH Hessian probe: failed writing ledger");
        return gs->total;
    }
}
