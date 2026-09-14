#pragma once

// Checked restarted GMRES shared by production DH response and diagnostic oracles.
#include <Eigen/Dense>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <expected>
#include <exception>
#include <limits>
#include <functional>
#include <string>
#include <vector>

namespace DFT::Response
{
    using Action = std::function<std::expected<Eigen::VectorXd, std::string>(const Eigen::VectorXd &)>;
    struct Solve
    {
        Eigen::VectorXd x;
        std::vector<double> true_residuals;
        int iterations = 0;
        int action_count = 0;
        int restarts = 0;
        std::size_t krylov_matrix_bytes = 0; // peak basis/directions/H storage, not process RSS
        bool converged = false;
    };

    // Restarted, right-preconditioned GMRES with two-pass modified Gram-Schmidt.
    // No dense operator, dense solution, or SPD assumption enters this solve.
    // Every acceptance check uses a fresh action and the unpreconditioned residual.
    inline std::expected<Solve, std::string> gmres(
        const Action &action, const Eigen::VectorXd &b, const Eigen::VectorXd &diagonal,
        double tolerance = 1e-12, int restart = 24, int max_iterations = 256)
    {
        const Eigen::Index n = b.size();
        if (!action || n == 0 || n > std::numeric_limits<int>::max() || diagonal.size() != n || !b.allFinite() ||
            !diagonal.allFinite() || !std::isfinite(tolerance) || tolerance <= 0 ||
            restart <= 0 || max_iterations <= 0)
            return std::unexpected("GMRES: invalid inputs");
        Solve out;
        const auto apply = [&](const Eigen::VectorXd &x) -> std::expected<Eigen::VectorXd, std::string>
        {
            ++out.action_count;
            try
            {
                auto y = action(x);
                if (!y) return std::unexpected(y.error());
                if (y->size() != n || !y->allFinite())
                    return std::unexpected("GMRES: nonfinite or wrong-sized action");
                return y;
            }
            catch (const std::exception &e) { return std::unexpected(std::string("GMRES action: ")+e.what()); }
            catch (...) { return std::unexpected("GMRES action: unknown exception"); }
        };
        const Eigen::VectorXd inverse = diagonal.cwiseAbs().cwiseMax(1e-8).cwiseInverse();
        const double target = tolerance * std::max(1.0, b.cwiseAbs().maxCoeff());
        if (!std::isfinite(target)) return std::unexpected("GMRES: nonfinite residual target");
        out.x = Eigen::VectorXd::Zero(n);
        auto ax = apply(out.x);
        if (!ax) return std::unexpected(ax.error());
        Eigen::VectorXd residual = b - *ax;
        out.true_residuals.push_back(residual.cwiseAbs().maxCoeff());
        if (out.true_residuals.back() <= target)
        {
            out.converged = true;
            return out;
        }
        while (out.iterations < max_iterations)
        {
            if (out.iterations) ++out.restarts;
            const int width = std::min({restart, static_cast<int>(n), max_iterations - out.iterations});
            Eigen::MatrixXd basis = Eigen::MatrixXd::Zero(n, width + 1);
            Eigen::MatrixXd directions = Eigen::MatrixXd::Zero(n, width);
            Eigen::MatrixXd h = Eigen::MatrixXd::Zero(width + 1, width);
            out.krylov_matrix_bytes = std::max(out.krylov_matrix_bytes,
                sizeof(double)*static_cast<std::size_t>(basis.size()+directions.size()+h.size()));
            const Eigen::VectorXd start = out.x;
            const double beta = residual.norm();
            if (!std::isfinite(beta) || beta==0.0) return std::unexpected("GMRES: invalid residual norm");
            basis.col(0) = residual / beta;
            for (int j = 0; j < width; ++j)
            {
                directions.col(j) = inverse.cwiseProduct(basis.col(j));
                auto applied = apply(directions.col(j));
                if (!applied) return std::unexpected(applied.error());
                Eigen::VectorXd w = *applied;
                const double initial_norm = w.norm();
                if (!std::isfinite(initial_norm)) return std::unexpected("GMRES: nonfinite action norm");
                for (int pass = 0; pass < 2; ++pass)
                    for (int i = 0; i <= j; ++i)
                    {
                        const double dot = basis.col(i).dot(w);
                        h(i, j) += dot;
                        w.noalias() -= dot * basis.col(i);
                    }
                h(j + 1, j) = w.norm();
                if (!std::isfinite(h(j+1,j))) return std::unexpected("GMRES: nonfinite Arnoldi norm");
                const bool breakdown = h(j + 1, j) <= 1e-14 * std::max(1.0, initial_norm);
                if (!breakdown) basis.col(j + 1) = w / h(j + 1, j);
                Eigen::VectorXd target_vector = Eigen::VectorXd::Zero(j + 2);
                target_vector(0) = beta;
                const Eigen::VectorXd y = h.topLeftCorner(j + 2, j + 1)
                    .colPivHouseholderQr().solve(target_vector);
                out.x = start + directions.leftCols(j + 1) * y;
                if (!out.x.allFinite()) return std::unexpected("GMRES: nonfinite iterate");
                ax = apply(out.x);
                if (!ax) return std::unexpected(ax.error());
                residual = b - *ax;
                ++out.iterations;
                out.true_residuals.push_back(residual.cwiseAbs().maxCoeff());
                if (out.true_residuals.back() <= target)
                {
                    out.converged = true;
                    return out;
                }
                if (breakdown) return out; // Report failure; never substitute dense Z.
            }
        }
        return out;
    }
}
