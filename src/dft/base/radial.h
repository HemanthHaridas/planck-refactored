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

#include <Eigen/Dense>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace DFT
{

    // Treutler-Ahlrichs xi scaling parameters, Table 1 of
    // O. Treutler and R. Ahlrichs, J. Chem. Phys. 102, 346 (1995).
    // Indexed by atomic number Z (index 0 is the ghost-atom entry, 1.0).
    //
    // These are the xi of the M4 mapping, NOT atomic radii. Planck previously
    // used a Bragg-Slater-like radius table here, which differs from xi for 28
    // of the first 36 elements (H, C and O happen to coincide, which is why
    // water test cases never exposed it). Values transcribed from PySCF 2.13.0
    // `pyscf.dft.radi._treutler_ahlrichs_xi` so the two codes build the same
    // radial grid; verified elementwise by tests/test_dft_radial_grid.py.
    inline constexpr int TREUTLER_XI_MAX_Z = 103;
    inline constexpr double TREUTLER_XI[104] = {
        1.000, 0.800, 0.900, 1.800, 1.400, 1.300, 1.100, 0.900, 0.900, 0.900,   // Z = 0..9
        0.900, 1.400, 1.300, 1.300, 1.200, 1.100, 1.000, 1.000, 1.000, 1.500,   // Z = 10..19
        1.400, 1.300, 1.200, 1.200, 1.200, 1.200, 1.200, 1.200, 1.100, 1.100,   // Z = 20..29
        1.100, 1.100, 1.000, 0.900, 0.900, 0.900, 0.900, 2.000, 1.700, 1.500,   // Z = 30..39
        1.500, 1.350, 1.350, 1.250, 1.200, 1.250, 1.300, 1.500, 1.500, 1.300,   // Z = 40..49
        1.200, 1.200, 1.150, 1.150, 1.150, 2.500, 2.200, 2.500, 1.500, 1.500,   // Z = 50..59
        1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500,   // Z = 60..69
        1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500,   // Z = 70..79
        1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 2.500, 2.100, 3.685,   // Z = 80..89
        1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500,   // Z = 90..99
        1.500, 1.500, 1.500, 1.500,   // Z = 100..103
    };

    // Return the Treutler-Ahlrichs xi parameter for element Z.
    // PySCF falls back to the last tabulated value for Z beyond the table.
    inline double treutler_xi(int Z)
    {
        if (Z >= 0 && Z <= TREUTLER_XI_MAX_Z)
            return TREUTLER_XI[Z];
        return TREUTLER_XI[TREUTLER_XI_MAX_Z];
    }

    // Bragg-Slater atomic radii [bohr], used for the Becke partition's atomic
    // size adjustment -- a DIFFERENT quantity from the M4 xi above, and PySCF
    // keeps them in separate tables for that reason. Transcribed from PySCF
    // 2.13.0 `pyscf.dft.radi.BRAGG_RADII` (`pyscf.data.radii.BRAGG`).
    inline constexpr int BRAGG_RADII_MAX_Z = 130;
    inline constexpr double BRAGG_RADII[131] = {
        3.7794503594, 0.6614041436, 2.6456165744, 2.7401028806, 1.9842124308,   // Z = 0..4
        1.6062672059, 1.3228082872, 1.2283219810, 1.1338356747, 0.9448630623,   // Z = 5..9
        2.8345891868, 3.4015070242, 2.8345891868, 2.3621576557, 2.0786987370,   // Z = 10..14
        1.8897261246, 1.8897261246, 1.8897261246, 3.4015070242, 4.1573974740,   // Z = 15..19
        3.4015070242, 3.0235617993, 2.6456165744, 2.5511302682, 2.6456165744,   // Z = 20..24
        2.6456165744, 2.6456165744, 2.5511302682, 2.5511302682, 2.5511302682,   // Z = 25..29
        2.5511302682, 2.4566439619, 2.3621576557, 2.1731850432, 2.1731850432,   // Z = 30..34
        2.1731850432, 3.5904796367, 4.4408563927, 3.7794522491, 3.4015070242,   // Z = 35..39
        2.9290754931, 2.7401028806, 2.7401028806, 2.5511302682, 2.4566439619,   // Z = 40..44
        2.5511302682, 2.6456165744, 3.0235617993, 2.9290754931, 2.9290754931,   // Z = 45..49
        2.7401028806, 2.7401028806, 2.6456165744, 2.6456165744, 3.9684248616,   // Z = 50..54
        4.9132879239, 4.0629111678, 3.6849659429, 3.4959933304, 3.4959933304,   // Z = 55..59
        3.4959933304, 3.4959933304, 3.4959933304, 3.4959933304, 3.4015070242,   // Z = 60..64
        3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180,   // Z = 65..69
        3.3070207180, 3.3070207180, 2.9290754931, 2.7401028806, 2.5511302682,   // Z = 70..74
        2.5511302682, 2.4566439619, 2.5511302682, 2.5511302682, 2.5511302682,   // Z = 75..79
        2.8345891868, 3.5904796367, 3.4015070242, 3.0235617993, 3.5904796367,   // Z = 80..84
        2.7401028806, 3.9684248616, 3.4015070242, 4.0629111678, 3.6849659429,   // Z = 85..89
        3.4015070242, 3.4015070242, 3.3070207180, 3.3070207180, 3.3070207180,   // Z = 90..94
        3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180,   // Z = 95..99
        3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180,   // Z = 100..104
        3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180,   // Z = 105..109
        3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180,   // Z = 110..114
        3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180,   // Z = 115..119
        3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180,   // Z = 120..124
        3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180,   // Z = 125..129
        3.3070207180,   // Z = 130..130
    };

    inline double bragg_radius(int Z)
    {
        if (Z >= 0 && Z <= BRAGG_RADII_MAX_Z)
            return BRAGG_RADII[Z];
        return BRAGG_RADII[BRAGG_RADII_MAX_Z];
    }

    // Number of radial points, PySCF's `gen_grid._default_rad` lookup.
    // Planck previously used an ORCA-style heuristic
    // `(15*int_acc - 40) + radial_row_factor*row` with a fixed row factor of 5,
    // which gave only 44 points for a second-row atom where this table gives 75
    // at the same nominal quality -- the direct cause of the grid failing to
    // converge (energy still moving 2.9e-5 Eh at "ultrafine").
    // Rows are PySCF grid levels 0-9; columns are periods 1-7.
    inline constexpr int RAD_GRIDS[10][7] = {
        { 10, 15, 20, 30, 35, 40, 50},
        { 30, 40, 50, 60, 65, 70, 75},
        { 40, 60, 65, 75, 80, 85, 90},
        { 50, 75, 80, 90, 95,100,105},
        { 60, 90, 95,105,110,115,120},
        { 70,105,110,120,125,130,135},
        { 80,120,125,135,140,145,150},
        { 90,135,140,150,155,160,165},
        {100,150,155,165,170,175,180},
        {200,200,200,200,200,200,200},
    };

    // Period index 0-6, matching PySCF's `(nuc > tab).sum()`.
    inline int pyscf_period_index(int Z)
    {
        constexpr int tab[7] = {2, 10, 18, 36, 54, 86, 118};
        int period = 0;
        for (int t : tab)
            if (Z > t)
                ++period;
        return period > 6 ? 6 : period;
    }

    // Radial point count for element Z at PySCF grid level (0-9).
    inline int pyscf_radial_count(int Z, int level)
    {
        const int lv = level < 0 ? 0 : (level > 9 ? 9 : level);
        return RAD_GRIDS[lv][pyscf_period_index(Z)];
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
    //   R      M4 scaling parameter; pass treutler_xi(Z) for element Z
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
        {
            assert(n > 0 && "MakeTreutlerAhlrichsGrid: n must be positive");
            return Eigen::MatrixXd{};
        }

        const double ln2 = std::log(2.0);
        const double step = std::numbers::pi / static_cast<double>(n + 1);

        Eigen::MatrixXd grid(n, 2);

        for (int i = 0; i < n; ++i)
        {
            const double theta = (i + 1) * step;
            const double x = std::cos(theta);
            const double sin_t = std::sin(theta); // sqrt(1 − x²), positive on (0, π)
            const double one_p_x = 1.0 + x;
            const double one_m_x = 1.0 - x;
            const double log_t = std::log(2.0 / one_m_x); // ln(2 / (1 − x))

            // M4 mapping: r = R / ln2 · (1+x)^α · ln(2 / (1−x))
            const double r = R / ln2 * std::pow(one_p_x, alpha) * log_t;

            // Jacobian: dr/dx = R / ln2 · (1+x)^(α−1) · [α · ln(2/(1−x)) + (1+x)/(1−x)]
            const double dr = R / ln2 * std::pow(one_p_x, alpha - 1.0) * (alpha * log_t + one_p_x / one_m_x);

            // Radial weight: step · sin(θ) · r² · dr/dx
            const double w = step * sin_t * r * r * dr;

            grid(i, 0) = r;
            grid(i, 1) = w;
        }

        return grid;
    }

} // namespace DFT

#endif // DFT_RADIAL_H
