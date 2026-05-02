#include "post_hf/casscf/aug-hessian.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <limits>

namespace HartreeFock::Correlation::CASSCF
{
    namespace
    {
        // Solve the bordered subspace eigenproblem
        //     H_eff v = w * ovlp v
        // with H_eff[0,0] = 0, H_eff[0,j] = g . x_j, H_eff[i,j] = x_i . (H x_j),
        // ovlp[0,0] = 1, ovlp[i,j] = x_i . x_j. Both matrices are symmetrized
        // before the eigensolve to absorb any drift from the matrix-free
        // contractions. Returns the lowest eigenpair whose first component
        // exceeds v0_min (or eigenpair 0 if none qualifies).
        struct SubspaceMode
        {
            double          eigenvalue   = 0.0;
            Eigen::VectorXd eigenvector;
            double          v0           = 0.0;
            bool            valid        = false;
            int             chosen_index = -1;
        };

        SubspaceMode regular_step(
            const Eigen::MatrixXd &heff,
            const Eigen::MatrixXd &ovlp,
            double                 v0_min,
            double                 lindep)
        {
            SubspaceMode mode;
            const Eigen::Index n = heff.rows();
            if (n < 2 || ovlp.rows() != n)
                return mode;

            // Symmetrize and shift the overlap below lindep up to a small
            // identity floor to keep the generalized eigensolve well posed.
            Eigen::MatrixXd S = 0.5 * (ovlp + ovlp.transpose());
            Eigen::MatrixXd H = 0.5 * (heff + heff.transpose());

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> overlap_eig(S);
            if (overlap_eig.info() != Eigen::Success)
                return mode;

            Eigen::VectorXd seigs = overlap_eig.eigenvalues();
            Eigen::MatrixXd U     = overlap_eig.eigenvectors();
            const double s_min    = seigs.minCoeff();
            if (!(std::isfinite(s_min)) || s_min < lindep)
                return mode;

            // Whiten the bordered system into a standard symmetric eigenproblem:
            //   X = U diag(1/sqrt(s)),   H_white = X^T H X.
            Eigen::VectorXd inv_sqrt = seigs.array().rsqrt();
            Eigen::MatrixXd X        = U * inv_sqrt.asDiagonal();
            Eigen::MatrixXd H_white  = X.transpose() * H * X;
            H_white                  = 0.5 * (H_white + H_white.transpose());

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H_white);
            if (eig.info() != Eigen::Success)
                return mode;

            const Eigen::VectorXd &w  = eig.eigenvalues();
            const Eigen::MatrixXd  Vg = X * eig.eigenvectors();

            // Pick the smallest eigenvalue with |v[0]| > v0_min. PySCF's
            // ciah._regular_step (pyscf/soscf/ciah.py:295) defends against
            // eigenvectors that would scale the orbital step by 1/v0 -> infty.
            int chosen = -1;
            for (Eigen::Index k = 0; k < Vg.cols(); ++k)
            {
                const double v0 = Vg(0, k);
                if (std::abs(v0) > v0_min)
                {
                    chosen = static_cast<int>(k);
                    break;
                }
            }
            if (chosen < 0)
                chosen = 0;

            mode.eigenvalue   = w(chosen);
            mode.eigenvector  = Vg.col(chosen);
            mode.v0           = mode.eigenvector(0);
            mode.valid        = true;
            mode.chosen_index = chosen;
            return mode;
        }

        // Modified Gram-Schmidt: orthogonalize v against each column of V
        // already in the subspace, then normalize. Returns the norm of the
        // residual; caller should reject the new vector if the norm is below
        // the linear-dependence threshold.
        double orthonormalize_against(
            Eigen::VectorXd       &v,
            const Eigen::MatrixXd &V,
            double                 lindep)
        {
            for (Eigen::Index k = 0; k < V.cols(); ++k)
            {
                const double overlap = V.col(k).dot(v);
                v.noalias() -= overlap * V.col(k);
            }
            const double norm = v.norm();
            if (!(std::isfinite(norm)) || norm < lindep)
                return 0.0;
            v /= norm;
            return norm;
        }
    } // namespace

    AugHessianResult solve_augmented_hessian(
        const AugHessianHopFn     &h_op,
        const AugHessianGradFn    &g_op,
        const AugHessianPrecondFn &precond,
        const Eigen::VectorXd     &x0,
        const AugHessianOptions   &opts)
    {
        AugHessianResult result;
        result.x = Eigen::VectorXd::Zero(x0.size());

        if (!h_op || !g_op || x0.size() == 0)
            return result;

        const Eigen::VectorXd g = g_op();
        if (g.size() != x0.size())
            return result;
        const double g_norm = g.norm();
        if (!(std::isfinite(g_norm)) || g_norm == 0.0)
        {
            result.converged = true;
            return result;
        }

        const int    max_cycle    = std::max(1, opts.ah_max_cycle);
        const int    max_subspace = std::max(1, std::min(opts.max_subspace, max_cycle));
        const double lindep       = opts.ah_lindep;
        const double v0_min       = opts.v0_min;

        auto seed_initial_vector = [&]() -> Eigen::VectorXd {
            // Seed with the supplied trial vector; if it is empty/degenerate
            // fall back to the steepest-descent direction -g (the natural
            // first Krylov vector of the bordered system).
            Eigen::VectorXd seed = x0;
            const double norm    = seed.norm();
            if (!(std::isfinite(norm)) || norm < lindep)
                seed = -g;
            const double final_norm = seed.norm();
            if (final_norm > 0.0 && std::isfinite(final_norm))
                seed /= final_norm;
            return seed;
        };

        Eigen::VectorXd seed = seed_initial_vector();
        if (seed.size() != x0.size() || seed.norm() == 0.0)
            return result;

        Eigen::MatrixXd V (x0.size(), 1);
        Eigen::MatrixXd HV(x0.size(), 1);
        V.col(0)  = seed;
        HV.col(0) = h_op(seed);
        if (!HV.col(0).allFinite())
            return result;

        double prev_eigenvalue = std::numeric_limits<double>::infinity();
        Eigen::VectorXd best_x = Eigen::VectorXd::Zero(x0.size());
        double          best_residual = std::numeric_limits<double>::infinity();
        double          best_eigenvalue = 0.0;
        double          best_v0 = 0.0;

        for (int iter = 1; iter <= max_cycle; ++iter)
        {
            const int m = static_cast<int>(V.cols());

            // Bordered subspace matrices. Index 0 is the implicit "1"
            // component of the augmented vector; indices 1..m correspond to
            // the Krylov basis stored in V/HV.
            Eigen::MatrixXd heff = Eigen::MatrixXd::Zero(m + 1, m + 1);
            Eigen::MatrixXd ovlp = Eigen::MatrixXd::Identity(m + 1, m + 1);

            for (int j = 0; j < m; ++j)
            {
                const double gxj = g.dot(V.col(j));
                heff(0, j + 1)   = gxj;
                heff(j + 1, 0)   = gxj;
                for (int i = 0; i < m; ++i)
                {
                    heff(i + 1, j + 1) = V.col(i).dot(HV.col(j));
                    ovlp(i + 1, j + 1) = V.col(i).dot(V.col(j));
                }
            }

            SubspaceMode mode = regular_step(heff, ovlp, v0_min, lindep);
            if (!mode.valid)
                break;

            // Reconstruct the trial step in the full space:
            //   x_trial = sum_j (v_j / v0) * V[:, j]
            // The 1/v0 normalization is what extracts the orbital step from
            // the bordered eigenvector (ciah.py:_regular_step line 308).
            Eigen::VectorXd coeffs(m);
            const double safe_v0 = (std::abs(mode.v0) > 0.0) ? mode.v0
                                                             : std::numeric_limits<double>::epsilon();
            for (int j = 0; j < m; ++j)
                coeffs(j) = mode.eigenvector(j + 1) / safe_v0;

            Eigen::VectorXd x_trial   = V * coeffs;
            Eigen::VectorXd Hx_trial  = HV * coeffs;

            // Davidson residual for the bordered system:
            //   r = H x + g - w x        (the leading 1 in the augmented
            // vector contributes g; we already factored 1/v0 into x).
            const double w           = mode.eigenvalue;
            Eigen::VectorXd residual = Hx_trial + g - w * x_trial;
            const double residual_norm = residual.norm();

            // Cache the best finite iterate so we can fall back if the next
            // micro-cycle stalls or breaks down.
            if (std::isfinite(residual_norm) && residual_norm < best_residual)
            {
                best_residual    = residual_norm;
                best_x           = x_trial;
                best_eigenvalue  = w;
                best_v0          = mode.v0;
            }

            result.iterations = iter;

            // Convergence: bordered residual or eigenvalue stagnation.
            const double eigenvalue_change = std::abs(w - prev_eigenvalue);
            if (residual_norm < opts.ah_conv_tol || eigenvalue_change < opts.ah_conv_tol)
            {
                result.converged = true;
                result.x         = x_trial;
                result.eigenvalue = w;
                result.v0         = mode.v0;
                result.residual_norm = residual_norm;
                result.orbital_only_fallback = false;
                return result;
            }
            // PySCF early-exit: residual is small enough to feed back into
            // the macro driver even if the eigenproblem is not fully
            // converged yet (mc1step.py:ah_start_tol path).
            if (iter >= opts.ah_start_cycle && residual_norm < opts.ah_start_tol)
            {
                result.converged = true;
                result.x         = x_trial;
                result.eigenvalue = w;
                result.v0         = mode.v0;
                result.residual_norm = residual_norm;
                result.orbital_only_fallback = false;
                return result;
            }

            prev_eigenvalue = w;

            // Build the next Krylov vector from the preconditioned residual.
            // Identity preconditioner is used when the caller supplied none,
            // matching the safest fallback in ciah.davidson_cc.
            Eigen::VectorXd next = precond
                ? precond(residual, w - opts.ah_level_shift)
                : residual;
            if (!next.allFinite())
                break;

            const double next_norm = next.norm();
            if (!(std::isfinite(next_norm)) || next_norm < lindep)
                break;
            next /= next_norm;

            const double residual_after = orthonormalize_against(next, V, lindep);
            if (residual_after == 0.0)
                break;

            // Restart at the subspace cap: keep the best iterate plus the
            // newest correction so we do not lose progress already made.
            if (m >= max_subspace)
            {
                Eigen::MatrixXd V_new (x0.size(), 0);
                Eigen::MatrixXd HV_new(x0.size(), 0);

                auto append_restart = [&](Eigen::VectorXd v) {
                    const double v_norm = v.norm();
                    if (!(std::isfinite(v_norm)) || v_norm < lindep)
                        return;
                    v /= v_norm;
                    if (orthonormalize_against(v, V_new, lindep) == 0.0)
                        return;
                    V_new.conservativeResize(Eigen::NoChange, V_new.cols() + 1);
                    HV_new.conservativeResize(Eigen::NoChange, HV_new.cols() + 1);
                    V_new.col(V_new.cols() - 1)   = v;
                    HV_new.col(HV_new.cols() - 1) = h_op(v);
                };

                append_restart(best_x);
                append_restart(next);
                if (V_new.cols() == 0)
                    break;
                V  = std::move(V_new);
                HV = std::move(HV_new);
                continue;
            }

            V.conservativeResize (Eigen::NoChange, m + 1);
            HV.conservativeResize(Eigen::NoChange, m + 1);
            V.col(m)  = next;
            HV.col(m) = h_op(next);
            if (!HV.col(m).allFinite())
                break;
        }

        // Ran out of iterations (or broke down). Return the best iterate we
        // saw; the caller can decide whether to accept it or fall back.
        result.x             = best_x;
        result.eigenvalue    = best_eigenvalue;
        result.v0            = best_v0;
        result.residual_norm = best_residual;

        // Orbital-only fallback: if the chosen AH eigenvalue dropped below
        // the floor, the bordered system is dominated by the gradient row
        // and PySCF (ciah.py:301-306) drops the bordering and solves the
        // pure orbital block. We signal this so the caller can decide
        // whether to retry with a tighter step cap or switch solvers.
        if (best_eigenvalue < opts.orbital_only_floor)
            result.orbital_only_fallback = true;

        return result;
    }

} // namespace HartreeFock::Correlation::CASSCF
