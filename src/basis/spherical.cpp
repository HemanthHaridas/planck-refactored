#include "spherical.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

// Cartesian → real-spherical transform, moved verbatim out of the symmetry
// module (src/symmetry/mo_symmetry.cpp) so it can be shared and independently
// validated. The matrices below are unchanged from the long-shipped, hand-
// verified versions; only their home and the error-message text differ.
//
// See spherical.h for the index conventions and spherical_recurrence.h for the
// closed-form oracle used to cross-check these in tests/spherical_transform.cpp.
std::expected<Eigen::MatrixXd, std::string> HartreeFock::BasisFunctions::cart_to_sph_block(int L)
{
    if (L == 0)
        return Eigen::MatrixXd::Identity(1, 1);

    if (L == 1)
    {
        // Cartesian order: px(1,0,0)=col0, py(0,1,0)=col1, pz(0,0,1)=col2
        // Spherical order: m=-1=py, m=0=pz, m=+1=px
        // T is a 3×3 permutation → T⁺ = Tᵀ
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(3, 3);
        T(0, 1) = 1.0; // m=-1 ← py
        T(1, 2) = 1.0; // m= 0 ← pz
        T(2, 0) = 1.0; // m=+1 ← px
        return T;
    }

    if (L == 2)
    {
        // Cartesian order: xx=0, xy=1, xz=2, yy=3, yz=4, zz=5
        // Spherical order: m=-2=dxy, m=-1=dyz, m=0=dz2, m=+1=dxz, m=+2=dx2y2
        //
        // T (6×5):
        //   col m=-2: T[xy,m=-2] = 1
        //   col m=-1: T[yz,m=-1] = 1
        //   col m= 0: T[xx,m=0]=-1/2, T[yy,m=0]=-1/2, T[zz,m=0]=1
        //   col m=+1: T[xz,m=+1] = 1
        //   col m=+2: T[xx,m=+2]=√3/2, T[yy,m=+2]=-√3/2
        //
        // TᵀT = diag(1, 1, 3/2, 1, 3/2)  →  (TᵀT)⁻¹ = diag(1,1,2/3,1,2/3)
        // T⁺ (5×6) = (TᵀT)⁻¹ Tᵀ:
        //   row m=-2:  [0,    1,  0,    0,    0,   0   ]
        //   row m=-1:  [0,    0,  0,    0,    1,   0   ]
        //   row m= 0:  [-1/3, 0,  0,   -1/3,  0,  2/3 ]
        //   row m=+1:  [0,    0,  1,    0,    0,   0   ]
        //   row m=+2:  [1/√3, 0,  0,  -1/√3,  0,   0  ]
        const double s3 = std::sqrt(3.0);
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(5, 6);
        //            xx      xy  xz    yy      yz  zz
        T(0, 1) = 1.0; // m=-2: dxy
        T(1, 4) = 1.0; // m=-1: dyz
        T(2, 0) = -1.0 / 3.0;
        T(2, 3) = -1.0 / 3.0;
        T(2, 5) = 2.0 / 3.0; // m=0: dz2
        T(3, 2) = 1.0;       // m=+1: dxz
        T(4, 0) = 1.0 / s3;
        T(4, 3) = -1.0 / s3; // m=+2: dx2y2
        return T;
    }

    if (L == 3)
    {
        // n_cart=10, n_sph=7
        // Cartesian: xxx=0 xxy=1 xxz=2 xyy=3 xyz=4 xzz=5 yyy=6 yyz=7 yzz=8 zzz=9
        // Spherical: m=-3..+3 → cols 0..6
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(10, 7);
        T(1, 0) = 3;
        T(6, 0) = -1; // m=-3: y(3x²-y²)
        T(4, 1) = 1;  // m=-2: xyz
        T(8, 2) = 4;
        T(1, 2) = -1;
        T(6, 2) = -1; // m=-1: y(4z²-x²-y²)
        T(9, 3) = 2;
        T(2, 3) = -3;
        T(7, 3) = -3; // m= 0: z(2z²-3x²-3y²)
        T(5, 4) = 4;
        T(0, 4) = -1;
        T(3, 4) = -1; // m=+1: x(4z²-x²-y²)
        T(2, 5) = 1;
        T(7, 5) = -1; // m=+2: z(x²-y²)
        T(0, 6) = 1;
        T(3, 6) = -3; // m=+3: x(x²-3y²)
        return T.completeOrthogonalDecomposition().pseudoInverse();
    }

    if (L == 4)
    {
        // n_cart=15, n_sph=9
        // Cartesian: x⁴=0 x³y=1 x³z=2 x²y²=3 x²yz=4 x²z²=5 xy³=6 xy²z=7 xyz²=8
        //            xz³=9 y⁴=10 y³z=11 y²z²=12 yz³=13 z⁴=14
        // Spherical: m=-4..+4 → cols 0..8
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(15, 9);
        T(1, 0) = 1;
        T(6, 0) = -1; // m=-4: xy(x²-y²)
        T(4, 1) = 3;
        T(11, 1) = -1; // m=-3: yz(3x²-y²)
        T(8, 2) = 6;
        T(1, 2) = -1;
        T(6, 2) = -1; // m=-2: xy(6z²-x²-y²)
        T(13, 3) = 4;
        T(4, 3) = -3;
        T(11, 3) = -3; // m=-1: yz(4z²-3x²-3y²)
        T(0, 4) = 3;
        T(3, 4) = 6;
        T(5, 4) = -24; // m= 0: 3x⁴+6x²y²-24x²z²
        T(10, 4) = 3;
        T(12, 4) = -24;
        T(14, 4) = 8; //       +3y⁴-24y²z²+8z⁴
        T(9, 5) = 4;
        T(2, 5) = -3;
        T(7, 5) = -3; // m=+1: xz(4z²-3x²-3y²)
        T(0, 6) = -1;
        T(10, 6) = 1;
        T(5, 6) = 6;
        T(12, 6) = -6; // m=+2: (x²-y²)(6z²-x²-y²)
        T(2, 7) = 1;
        T(7, 7) = -3; // m=+3: xz(x²-3y²)
        T(0, 8) = 1;
        T(3, 8) = -6;
        T(10, 8) = 1; // m=+4: x⁴-6x²y²+y⁴
        return T.completeOrthogonalDecomposition().pseudoInverse();
    }

    if (L == 5)
    {
        // n_cart=21, n_sph=11
        // Cartesian: x⁵=0 x⁴y=1 x⁴z=2 x³y²=3 x³yz=4 x³z²=5 x²y³=6 x²y²z=7 x²yz²=8 x²z³=9
        //            xy⁴=10 xy³z=11 xy²z²=12 xyz³=13 xz⁴=14
        //            y⁵=15 y⁴z=16 y³z²=17 y²z³=18 yz⁴=19 z⁵=20
        // Spherical: m=-5..+5 → cols 0..10
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(21, 11);
        T(1, 0) = 5;
        T(6, 0) = -10;
        T(15, 0) = 1; // m=-5: y(5x⁴-10x²y²+y⁴)
        T(4, 1) = 4;
        T(11, 1) = -4; // m=-4: 4xyz(x²-y²)
        T(8, 2) = 24;
        T(1, 2) = -3;
        T(6, 2) = -2; // m=-3: y(3x²-y²)(8z²-x²-y²)
        T(17, 2) = -8;
        T(15, 2) = 1;
        T(13, 3) = 2;
        T(4, 3) = -1;
        T(11, 3) = -1; // m=-2: xyz(2z²-x²-y²)
        T(1, 4) = 1;
        T(6, 4) = 2;
        T(15, 4) = 1; // m=-1: y(x⁴+2x²y²+y⁴+8z⁴-12x²z²-12y²z²)
        T(19, 4) = 8;
        T(8, 4) = -12;
        T(17, 4) = -12;
        T(20, 5) = 8;
        T(9, 5) = -40;
        T(18, 5) = -40; // m= 0: z(8z⁴-40x²z²-40y²z²+15x⁴+30x²y²+15y⁴)
        T(2, 5) = 15;
        T(7, 5) = 30;
        T(16, 5) = 15;
        T(0, 6) = 1;
        T(3, 6) = 2;
        T(10, 6) = 1; // m=+1: x(x⁴+2x²y²+y⁴+8z⁴-12x³z²-12xy²z²)
        T(14, 6) = 8;
        T(5, 6) = -12;
        T(12, 6) = -12;
        T(9, 7) = 2;
        T(2, 7) = -1;
        T(16, 7) = 1;
        T(18, 7) = -2; // m=+2: z(x²-y²)(2z²-x²-y²)
        T(0, 8) = -1;
        T(3, 8) = 2;
        T(10, 8) = 3; // m=+3: x(x²-3y²)(8z²-x²-y²)
        T(5, 8) = 8;
        T(12, 8) = -24;
        T(2, 9) = 1;
        T(7, 9) = -6;
        T(16, 9) = 1; // m=+4: z(x⁴-6x²y²+y⁴)
        T(0, 10) = 1;
        T(3, 10) = -10;
        T(10, 10) = 5; // m=+5: x(x⁴-10x²y²+5y⁴)
        return T.completeOrthogonalDecomposition().pseudoInverse();
    }

    return std::unexpected(
        "cart_to_sph_block: Cartesian→Spherical transform not implemented for L=" +
        std::to_string(L) + " (max supported: L=5)");
}

std::expected<Eigen::MatrixXd, std::string>
HartreeFock::BasisFunctions::build_cart_to_sph(const HartreeFock::Basis &basis)
{
    const std::size_t n_cart = basis.nbasis();
    const std::size_t n_sph = basis.nbasis_sph();

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n_sph),
                                              static_cast<Eigen::Index>(n_cart));

    Eigen::Index sph_row = 0;
    Eigen::Index cart_col = 0;
    for (const HartreeFock::Shell &sh : basis._shells)
    {
        const int L = static_cast<int>(sh._shell);
        const Eigen::Index nc = (L + 1) * (L + 2) / 2;
        const Eigen::Index ns = 2 * L + 1;

        auto block = cart_to_sph_block(L);
        if (!block)
            return std::unexpected(block.error());
        if (block->rows() != ns || block->cols() != nc)
            return std::unexpected("build_cart_to_sph: shell block has unexpected shape for L=" +
                                   std::to_string(L));

        C.block(sph_row, cart_col, ns, nc) = *block;
        sph_row += ns;
        cart_col += nc;
    }

    if (sph_row != static_cast<Eigen::Index>(n_sph) || cart_col != static_cast<Eigen::Index>(n_cart))
        return std::unexpected("build_cart_to_sph: assembled block offsets do not match basis sizes");

    return C;
}

std::expected<std::vector<double>, std::string>
HartreeFock::BasisFunctions::transform_eri_cart_to_sph(
    const std::vector<double> &eri_cart,
    const Eigen::MatrixXd &C,
    std::size_t n_cart,
    std::size_t max_n_cart)
{
    if (static_cast<std::size_t>(C.cols()) != n_cart)
        return std::unexpected("transform_eri_cart_to_sph: C.cols() does not match n_cart");
    if (n_cart > max_n_cart)
        return std::unexpected(
            "transform_eri_cart_to_sph: n_cart=" + std::to_string(n_cart) +
            " exceeds the dense spherical-ERI limit (" + std::to_string(max_n_cart) +
            "). Spherical mode currently supports modest systems only.");

    const std::size_t nc = n_cart;
    const std::size_t ns = static_cast<std::size_t>(C.rows());
    if (eri_cart.size() != nc * nc * nc * nc)
        return std::unexpected("transform_eri_cart_to_sph: eri_cart size does not match n_cart^4");

    // Four successive single-index contractions. Each step replaces one Cartesian index
    // (size nc) with a spherical index (size ns); intermediates shrink as we go.
    // Indexing is row-major in the order (a,b,c,d).
    auto Cval = [&](std::size_t s, std::size_t c) -> double {
        return C(static_cast<Eigen::Index>(s), static_cast<Eigen::Index>(c));
    };

    // Step 1: t1[p,ν,λ,σ] = Σ_μ C[p,μ] eri[μ,ν,λ,σ]   shape ns·nc·nc·nc
    std::vector<double> t1(ns * nc * nc * nc, 0.0);
    for (std::size_t p = 0; p < ns; ++p)
        for (std::size_t mu = 0; mu < nc; ++mu)
        {
            const double c = Cval(p, mu);
            if (c == 0.0)
                continue;
            const double *src = &eri_cart[mu * nc * nc * nc];
            double *dst = &t1[p * nc * nc * nc];
            for (std::size_t k = 0; k < nc * nc * nc; ++k)
                dst[k] += c * src[k];
        }

    // Step 2: t2[p,q,λ,σ] = Σ_ν C[q,ν] t1[p,ν,λ,σ]   shape ns·ns·nc·nc
    std::vector<double> t2(ns * ns * nc * nc, 0.0);
    for (std::size_t p = 0; p < ns; ++p)
        for (std::size_t q = 0; q < ns; ++q)
            for (std::size_t nu = 0; nu < nc; ++nu)
            {
                const double c = Cval(q, nu);
                if (c == 0.0)
                    continue;
                const double *src = &t1[(p * nc + nu) * nc * nc];
                double *dst = &t2[(p * ns + q) * nc * nc];
                for (std::size_t k = 0; k < nc * nc; ++k)
                    dst[k] += c * src[k];
            }
    t1.clear();
    t1.shrink_to_fit();

    // Step 3: t3[p,q,r,σ] = Σ_λ C[r,λ] t2[p,q,λ,σ]   shape ns·ns·ns·nc
    std::vector<double> t3(ns * ns * ns * nc, 0.0);
    for (std::size_t pq = 0; pq < ns * ns; ++pq)
        for (std::size_t r = 0; r < ns; ++r)
            for (std::size_t lam = 0; lam < nc; ++lam)
            {
                const double c = Cval(r, lam);
                if (c == 0.0)
                    continue;
                const double *src = &t2[(pq * nc + lam) * nc];
                double *dst = &t3[(pq * ns + r) * nc];
                for (std::size_t s = 0; s < nc; ++s)
                    dst[s] += c * src[s];
            }
    t2.clear();
    t2.shrink_to_fit();

    // Step 4: out[p,q,r,s] = Σ_σ C[s,σ] t3[p,q,r,σ]   shape ns⁴
    std::vector<double> out(ns * ns * ns * ns, 0.0);
    for (std::size_t pqr = 0; pqr < ns * ns * ns; ++pqr)
    {
        const double *src = &t3[pqr * nc];
        double *dst = &out[pqr * ns];
        for (std::size_t s = 0; s < ns; ++s)
        {
            double acc = 0.0;
            for (std::size_t sig = 0; sig < nc; ++sig)
                acc += Cval(s, sig) * src[sig];
            dst[s] = acc;
        }
    }

    return out;
}
