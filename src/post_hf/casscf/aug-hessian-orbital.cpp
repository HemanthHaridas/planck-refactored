#include "post_hf/casscf/aug-hessian-orbital.h"

#include <algorithm>
#include <cmath>

namespace HartreeFock::Correlation::CASSCF
{
    namespace
    {
        // True when use_sym is on, both pair endpoints carry an irrep label,
        // and those labels disagree. Symmetry-forbidden pairs are excluded
        // from the variational parameter set so the AH solver never sees
        // them as nonzero.
        bool pair_blocked_by_symmetry(
            const RotPair &pair,
            const std::vector<int> &mo_irreps,
            bool use_sym)
        {
            if (!use_sym || mo_irreps.empty())
                return false;
            const int ip = (pair.p < static_cast<int>(mo_irreps.size())) ? mo_irreps[pair.p] : -1;
            const int iq = (pair.q < static_cast<int>(mo_irreps.size())) ? mo_irreps[pair.q] : -1;
            return ip >= 0 && iq >= 0 && ip != iq;
        }

        // Cap kappa element-wise at max_rot, mirroring the trust-region
        // policy already used by augmented_hessian_step and
        // diagonal_preconditioned_orbital_step. The CIAH solver itself does
        // not impose a step cap; that is done here at the boundary.
        Eigen::MatrixXd cap_step_norm(Eigen::MatrixXd kappa, double max_rot)
        {
            if (max_rot <= 0.0)
                return kappa;
            if (!kappa.allFinite())
                return Eigen::MatrixXd::Zero(kappa.rows(), kappa.cols());
            const double max_elem = kappa.cwiseAbs().maxCoeff();
            if (max_elem > max_rot)
                kappa *= max_rot / max_elem;
            return kappa;
        }
    } // namespace

    Eigen::VectorXd pack_rotation_matrix(
        const Eigen::MatrixXd &kappa,
        const std::vector<RotPair> &pairs,
        const std::vector<int> &mo_irreps,
        bool use_sym)
    {
        const int npairs = static_cast<int>(pairs.size());
        Eigen::VectorXd packed = Eigen::VectorXd::Zero(npairs);
        if (kappa.size() == 0)
            return packed;
        for (int k = 0; k < npairs; ++k)
        {
            const auto &pair = pairs[static_cast<std::size_t>(k)];
            if (pair_blocked_by_symmetry(pair, mo_irreps, use_sym))
                continue;
            packed(k) = kappa(pair.p, pair.q);
        }
        return packed;
    }

    Eigen::MatrixXd unpack_rotation_vector(
        const Eigen::VectorXd &x,
        const std::vector<RotPair> &pairs,
        int nbasis)
    {
        Eigen::MatrixXd kappa = Eigen::MatrixXd::Zero(nbasis, nbasis);
        const int npairs = static_cast<int>(pairs.size());
        if (x.size() != npairs)
            return kappa;
        for (int k = 0; k < npairs; ++k)
        {
            const auto &pair = pairs[static_cast<std::size_t>(k)];
            const double value = x(k);
            kappa(pair.p, pair.q) = value;
            kappa(pair.q, pair.p) = -value;
        }
        return kappa;
    }

    AugHessianHopFn make_orbital_hessian_action(
        const OrbitalHessianContext *context,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        int n_core,
        int n_act,
        int n_virt,
        const std::vector<RotPair> &pairs,
        const std::vector<int> &mo_irreps,
        bool use_sym)
    {
        const int nbasis = n_core + n_act + n_virt;
        // Capture by reference; the caller owns lifetime. Inside the closure
        // we materialize the trial kappa, call delta_g_sa_action, then pack
        // the result back into the AH solver's flat layout.
        return [context,
                &F_I_mo,
                &F_A_mo,
                n_core,
                n_act,
                n_virt,
                nbasis,
                &pairs,
                &mo_irreps,
                use_sym](const Eigen::VectorXd &x) -> Eigen::VectorXd
        {
            const Eigen::MatrixXd R = unpack_rotation_vector(x, pairs, nbasis);
            const Eigen::MatrixXd HR = delta_g_sa_action(
                R,
                context,
                F_I_mo,
                F_A_mo,
                n_core,
                n_act,
                n_virt,
                mo_irreps,
                use_sym);
            return pack_rotation_matrix(HR, pairs, mo_irreps, use_sym);
        };
    }

    AugHessianGradFn make_orbital_gradient(
        const Eigen::MatrixXd &g_orb,
        const std::vector<RotPair> &pairs,
        const std::vector<int> &mo_irreps,
        bool use_sym)
    {
        // Snapshot the gradient by value so the closure remains safe even
        // if the caller's source matrix is reassigned between AH micro
        // iterations.
        Eigen::MatrixXd g_copy = g_orb;
        return [g_copy = std::move(g_copy),
                &pairs,
                &mo_irreps,
                use_sym]() -> Eigen::VectorXd
        {
            return pack_rotation_matrix(g_copy, pairs, mo_irreps, use_sym);
        };
    }

    AugHessianPrecondFn make_orbital_preconditioner(
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        const std::vector<RotPair> &pairs,
        const std::vector<int> &mo_irreps,
        bool use_sym,
        double level_shift,
        double denom_floor)
    {
        const int npairs = static_cast<int>(pairs.size());
        // Pre-compute the diagonal Hessian denominators once: the F_sum
        // matrix does not change inside a single CIAH call. The AH Ritz
        // value is folded in per-call via `e`, matching the PySCF
        // precond(x, e) = x / (h_diag - (e - level_shift)) shape.
        Eigen::VectorXd h_diag(npairs);
        const Eigen::MatrixXd F_sum = F_I_mo + F_A_mo;
        for (int k = 0; k < npairs; ++k)
        {
            const auto &pair = pairs[static_cast<std::size_t>(k)];
            if (pair_blocked_by_symmetry(pair, mo_irreps, use_sym))
                h_diag(k) = 1.0;
            else
                h_diag(k) = hess_diag(F_sum, pair.p, pair.q);
        }

        return [h_diag = std::move(h_diag),
                level_shift,
                denom_floor,
                &pairs,
                &mo_irreps,
                use_sym](const Eigen::VectorXd &x, double e) -> Eigen::VectorXd
        {
            Eigen::VectorXd out(x.size());
            for (Eigen::Index k = 0; k < x.size(); ++k)
            {
                const auto &pair = pairs[static_cast<std::size_t>(k)];
                if (pair_blocked_by_symmetry(pair, mo_irreps, use_sym))
                {
                    out(k) = 0.0;
                    continue;
                }
                double denom = (h_diag(k) + level_shift) - e;
                // Mirror the floor used by augmented_hessian_step /
                // diagonal_preconditioned_orbital_step so a near-zero
                // denominator cannot produce an exploding correction.
                if (std::abs(denom) < denom_floor)
                    denom = (denom >= 0.0) ? denom_floor : -denom_floor;
                out(k) = x(k) / denom;
            }
            // Renormalize to unit length: the bordered Davidson grows the
            // Krylov basis from preconditioned residuals, so feeding back a
            // raw direction (rather than its scale) keeps the subspace
            // well-conditioned.
            const double norm = out.norm();
            if (std::isfinite(norm) && norm > 0.0)
                out /= norm;
            return out;
        };
    }

    OrbitalAugHessianStep solve_orbital_augmented_hessian_step(
        const Eigen::MatrixXd &g_orb,
        const Eigen::MatrixXd &F_I_mo,
        const Eigen::MatrixXd &F_A_mo,
        const OrbitalHessianContext *context,
        int n_core,
        int n_act,
        int n_virt,
        double level_shift,
        double max_rot,
        const std::vector<int> &mo_irreps,
        bool use_sym,
        const AugHessianOptions &opts)
    {
        OrbitalAugHessianStep result;
        const int nbasis = n_core + n_act + n_virt;
        result.kappa = Eigen::MatrixXd::Zero(nbasis, nbasis);

        const std::vector<RotPair> pairs = non_redundant_pairs(n_core, n_act, n_virt);
        if (pairs.empty())
            return result;

        AugHessianGradFn g_op = make_orbital_gradient(g_orb, pairs, mo_irreps, use_sym);
        AugHessianHopFn h_op = make_orbital_hessian_action(
            context, F_I_mo, F_A_mo, n_core, n_act, n_virt, pairs, mo_irreps, use_sym);
        AugHessianPrecondFn precond = make_orbital_preconditioner(
            F_I_mo, F_A_mo, pairs, mo_irreps, use_sym, level_shift);

        // PySCF CIAH behavior: apply the AH step in a few "micro" updates,
        // updating the gradient estimate as g <- g + H*dx between solves.
        // This improves robustness in flat / ill-conditioned regions relative
        // to taking one monolithic AH step.
        const Eigen::VectorXd g0 = g_op();
        Eigen::VectorXd g_est = g0;
        const double g0_norm = g0.norm();

        auto cap_packed_step = [&](Eigen::VectorXd x) -> Eigen::VectorXd
        {
            if (!x.allFinite())
                return Eigen::VectorXd::Zero(x.size());
            const double max_elem = x.cwiseAbs().maxCoeff();
            if (max_rot > 0.0 && std::isfinite(max_elem) && max_elem > max_rot)
                x *= max_rot / max_elem;
            return x;
        };

        // Seed with steepest descent (-g) like PySCF's initial x0_guess.
        Eigen::VectorXd x0 = g_est;
        const double x0_norm = x0.norm();
        if (std::isfinite(x0_norm) && x0_norm > 0.0)
            x0 *= (-1.0 / x0_norm);
        else
            x0.setZero();

        Eigen::VectorXd step_total = Eigen::VectorXd::Zero(g_est.size());
        // Keep this small: this is one candidate among several in the macro cascade.
        const int max_micro_updates = 3;
        for (int micro = 0; micro < max_micro_updates; ++micro)
        {
            // Provide a mutable gradient to the generic AH solver.
            const AugHessianGradFn g_live = [&g_est]() -> Eigen::VectorXd
            { return g_est; };
            AugHessianResult ah = solve_augmented_hessian(h_op, g_live, precond, x0, opts);
            if (micro == 0)
                result.ah = ah;

            Eigen::VectorXd dx = cap_packed_step(ah.x);
            if (dx.size() == 0 || dx.cwiseAbs().maxCoeff() < 1e-12)
                break;

            step_total.noalias() += dx;
            // Linearized gradient update: g <- g + H*dx (PySCF: g_orb += hdxi).
            const Eigen::VectorXd Hdx = h_op(dx);
            if (!Hdx.allFinite() || Hdx.size() != g_est.size())
                break;
            g_est.noalias() += Hdx;

            // Next seed: use the last accepted increment, mirroring PySCF's x0_guess update.
            x0 = dx;

            const double g_norm = g_est.norm();
            if (std::isfinite(g_norm) && std::isfinite(g0_norm) && g0_norm > 0.0 &&
                g_norm < 0.2 * g0_norm)
                break;
        }

        Eigen::MatrixXd kappa = unpack_rotation_vector(step_total, pairs, nbasis);
        // Trust-region cap at boundary (same as diagonal augmented_hessian_step).
        result.kappa = cap_step_norm(std::move(kappa), max_rot);
        return result;
    }

} // namespace HartreeFock::Correlation::CASSCF
