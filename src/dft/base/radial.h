#ifndef DFT_RADIAL_H
#define DFT_RADIAL_H

// Treutler–Ahlrichs M4 radial quadrature grid for DFT integration.
//
// Maps n Gauss–Chebyshev-of-the-second-kind points on (−1, 1) to radial
// points on (0, ∞) via the M4 transformation:
//
//   r(x) = R / ln(2) · (1 + x)^α · ln(2 / (1 − x))
//
// The weights include the r² volume factor so the full 3-D integral over a
// spherical shell is obtained by combining with a Lebedev angular grid whose
// weights sum to 4π:
//
//   ∫ f dV ≈ Σ_i Σ_j  w_rad[i] · w_ang[j] · f(r_i, θ_j, φ_j)
//
// Reference:
//   O. Treutler and R. Ahlrichs, J. Chem. Phys. 102, 346 (1995).
//   https://doi.org/10.1063/1.469408

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <Eigen/Dense>

namespace DFT
{

    // Treutler–Ahlrichs atomic scaling radii r_A [bohr], Table 2 of
    // O. Treutler and R. Ahlrichs, J. Chem. Phys. 102, 346 (1995).
    // Index by atomic number Z (1-based; index 0 unused). Elements H–Kr (Z = 1–36).
    // For Z > 36 use treutler_radius(Z), which falls back to 2.0 bohr.
    inline constexpr double TREUTLER_RADII[37] = {
        0.00,   // Z = 0  (unused)
        0.80,   // Z =  1  H
        0.50,   // Z =  2  He
        1.80,   // Z =  3  Li
        1.40,   // Z =  4  Be
        1.30,   // Z =  5  B
        1.10,   // Z =  6  C
        1.00,   // Z =  7  N
        0.90,   // Z =  8  O
        0.90,   // Z =  9  F
        0.90,   // Z = 10  Ne
        2.40,   // Z = 11  Na
        2.00,   // Z = 12  Mg
        2.00,   // Z = 13  Al
        1.90,   // Z = 14  Si
        1.80,   // Z = 15  P
        1.70,   // Z = 16  S
        1.60,   // Z = 17  Cl
        1.60,   // Z = 18  Ar
        3.00,   // Z = 19  K
        2.70,   // Z = 20  Ca
        2.40,   // Z = 21  Sc
        2.30,   // Z = 22  Ti
        2.10,   // Z = 23  V
        2.10,   // Z = 24  Cr
        2.00,   // Z = 25  Mn
        1.90,   // Z = 26  Fe
        1.90,   // Z = 27  Co
        1.80,   // Z = 28  Ni
        1.80,   // Z = 29  Cu
        1.70,   // Z = 30  Zn
        2.00,   // Z = 31  Ga
        1.90,   // Z = 32  Ge
        1.80,   // Z = 33  As
        1.70,   // Z = 34  Se
        1.70,   // Z = 35  Br
        1.60,   // Z = 36  Kr
    };

    // Return the Treutler–Ahlrichs atomic scaling radius for element Z [bohr].
    // Falls back to 2.0 bohr for Z > 36 (heavier elements not in the 1995 table).
    inline double treutler_radius(int Z)
    {
        if (Z >= 1 && Z <= 36)
            return TREUTLER_RADII[Z];
        return 2.0;
    }

    // Treutler–Ahlrichs M4 radial quadrature.
    //
    // Constructs n radial grid points on (0, ∞) using the M4 mapping applied
    // to the n-point Gauss–Chebyshev quadrature of the second kind.
    //
    // Returns an (n × 2) Eigen matrix:
    //   column 0 : r[i]  — radial positions in (0, ∞) [bohr], ordered r[0] > r[1] > …
    //   column 1 : w[i]  — weights for  ∫₀^∞ f(r) r² dr  (r² factor is included)
    //
    // Parameters:
    //   n      number of quadrature points (must be > 0)
    //   R      atomic scaling radius [bohr]; pass treutler_radius(Z) for element Z
    //   alpha  shape exponent α (default 0.6 as in the original paper)
    //
    // Derivation of weights:
    //   x_k = cos(k π / (n+1)),  k = 1 … n   (GC-2 abscissae)
    //
    //   r(x) = R/ln(2) · (1+x)^α · ln(2/(1−x))
    //
    //   dr/dx = R/ln(2) · (1+x)^(α−1) · [ α · ln(2/(1−x))  +  (1+x)/(1−x) ]
    //
    //   W_k = π/(n+1) · sin(k π/(n+1)) · r_k² · (dr/dx)|_{x_k}
    //
    //   The GC-2 formula ∫ h dx ≈ π/(n+1) Σ sin(θ_k) h(x_k) is used
    //   after absorbing the √(1−x²) factor into the sum.
    inline Eigen::MatrixXd MakeTreutlerAhlrichsGrid(int n, double R = 1.0, double alpha = 0.6)
    {
        if (n <= 0)
            throw std::invalid_argument(
                "MakeTreutlerAhlrichsGrid: n must be positive, got " + std::to_string(n));

        const double ln2  = std::log(2.0);
        const double step = std::numbers::pi / static_cast<double>(n + 1);

        Eigen::MatrixXd grid(n, 2);

        for (int i = 0; i < n; ++i)
        {
            const double theta   = (i + 1) * step;
            const double x       = std::cos(theta);
            const double sin_t   = std::sin(theta);  // sqrt(1 − x²), positive on (0, π)
            const double one_p_x = 1.0 + x;
            const double one_m_x = 1.0 - x;
            const double log_t   = std::log(2.0 / one_m_x);  // ln(2 / (1 − x))

            // M4 mapping: r = R / ln2 · (1+x)^α · ln(2 / (1−x))
            const double r  = R / ln2 * std::pow(one_p_x, alpha) * log_t;

            // Jacobian: dr/dx = R / ln2 · (1+x)^(α−1) · [α · ln(2/(1−x)) + (1+x)/(1−x)]
            const double dr = R / ln2 * std::pow(one_p_x, alpha - 1.0)
                            * (alpha * log_t + one_p_x / one_m_x);

            // Radial weight: step · sin(θ) · r² · dr/dx
            const double w  = step * sin_t * r * r * dr;

            grid(i, 0) = r;
            grid(i, 1) = w;
        }

        return grid;
    }

} // namespace DFT

#endif // DFT_RADIAL_H
