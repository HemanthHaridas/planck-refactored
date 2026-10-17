#ifndef HF_BASE_SPHERE_H
#define HF_BASE_SPHERE_H

#include <Eigen/Core>

#include <cmath>
#include <numbers>
#include <vector>

namespace HartreeFock
{
    // Quasi-uniform tiling of the unit sphere by the golden-angle (Fibonacci)
    // construction. Compared to a lat/long grid this avoids the pole pile-up,
    // and compared to a Lebedev grid it gives a single-parameter density that
    // can be set arbitrarily by the caller. Adequate for PCM cavities and for
    // ESP sampling shells; would not be appropriate for high-accuracy XC
    // integration.
    //
    // Lives here rather than in either consumer because both the C-PCM cavity
    // (src/solvation/pcm.cpp) and the ESP Connolly grid
    // (src/populations/esp.cpp) need exactly this, and two copies would be free
    // to drift apart.
    inline std::vector<Eigen::Vector3d> fibonacci_sphere(int npoints)
    {
        std::vector<Eigen::Vector3d> points;
        if (npoints <= 0)
            return points;

        points.reserve(static_cast<std::size_t>(npoints));

        // pi * (3 - sqrt(5)) is the golden angle in radians.
        const double golden_angle = std::numbers::pi * (3.0 - std::sqrt(5.0));
        for (int i = 0; i < npoints; ++i)
        {
            // z is uniformly distributed on (-1, 1) with the +0.5 shift
            // keeping endpoints away from the poles, so each point sits at
            // the centroid of its strip rather than on a pole.
            const double z = 1.0 - 2.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(npoints);
            const double radial = std::sqrt(std::max(0.0, 1.0 - z * z));
            const double phi = golden_angle * static_cast<double>(i);
            points.emplace_back(radial * std::cos(phi), radial * std::sin(phi), z);
        }

        return points;
    }
} // namespace HartreeFock

#endif // HF_BASE_SPHERE_H
