#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "dft/xc_grid.h"

// O1 micro-oracle: exercise the actual whole-grid method with the retired
// per-point schedule and the hoisted schedule. Timings are diagnostic only;
// exact equality and the counted grid work are deterministic acceptance gates.
bool test_dh_grid_sigma()
{
    bool ok = true;
    using Clock = std::chrono::steady_clock;
    for (const Eigen::Index n : {0, 1, 7, 64, 256, 1024, 4096})
    {
        DFT::DensityChannelOnGrid density;
        density.rho = Eigen::VectorXd::Ones(n);
        density.grad_x.resize(n);
        density.grad_y.resize(n);
        density.grad_z.resize(n);
        for (Eigen::Index p = 0; p < n; ++p)
        {
            density.grad_x(p) = std::sin(0.13 * p);
            density.grad_y(p) = 0.25 * std::cos(0.21 * p);
            density.grad_z(p) = 0.125 * (static_cast<int>(p % 17) - 8);
        }
        std::size_t calls = 0, visited = 0;
        const auto evaluate = [&]() -> Eigen::VectorXd
        {
            ++calls;
            visited += static_cast<std::size_t>(n);
            return density.gradient_squared();
        };
        std::vector<double> old_sigma(static_cast<std::size_t>(n));
        const auto old_start = Clock::now();
        for (Eigen::Index p = 0; p < n; ++p)
            old_sigma[static_cast<std::size_t>(p)] = evaluate()(p);
        const double old_us = std::chrono::duration<double, std::micro>(Clock::now() - old_start).count();
        const auto old_calls = calls, old_visited = visited;

        calls = 0;
        visited = 0;
        std::vector<double> new_sigma(static_cast<std::size_t>(n));
        const auto new_start = Clock::now();
        {
            const Eigen::VectorXd ground_sigma = evaluate();
            for (Eigen::Index p = 0; p < n; ++p)
                new_sigma[static_cast<std::size_t>(p)] = ground_sigma(p);
        }
        const double new_us = std::chrono::duration<double, std::micro>(Clock::now() - new_start).count();
        double max_error = 0.0;
        for (Eigen::Index p = 0; p < n; ++p)
        {
            const auto pi = static_cast<std::size_t>(p);
            max_error = std::max(max_error, std::abs(old_sigma[pi] - new_sigma[pi]));
            ok &= std::isfinite(new_sigma[pi]) && new_sigma[pi] >= 0.0;
        }
        const auto size = static_cast<std::size_t>(n);
        ok &= old_sigma == new_sigma;
        ok &= old_calls == size && old_visited == size * size;
        ok &= calls == 1 && visited == size;
        std::printf("DH O1 sigma G=%zu old_calls=%zu new_calls=%zu old_grid_values=%zu "
                    "new_grid_values=%zu old_us=%.3f new_us=%.3f max_error=%.3e\n",
                    size, old_calls, calls, old_visited, visited, old_us, new_us, max_error);
    }
    std::printf("%s DH O1 sigma: exact values and quadratic-to-linear grid work\n", ok ? "PASS" : "FAIL");
    return ok;
}
