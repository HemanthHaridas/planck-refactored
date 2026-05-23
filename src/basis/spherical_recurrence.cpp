#include "spherical_recurrence.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

namespace
{
    // ── Exact polynomial-in-(x,y,z) arithmetic over rational coefficients ─────────
    //
    // A polynomial is a map from monomial exponent triple (a,b,c) to a rational
    // coefficient. We build the real solid harmonics symbolically from the associated
    // Legendre construction so the result is canonical and self-orthonormal by
    // construction — no fragile closed-form coefficient sum. Coefficients are kept as
    // exact fractions to avoid any rounding before the final normalization.

    struct Rat
    {
        long long num = 0;
        long long den = 1;
    };

    long long gcdll(long long a, long long b)
    {
        a = a < 0 ? -a : a;
        b = b < 0 ? -b : b;
        while (b)
        {
            long long t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    Rat rnorm(Rat r)
    {
        if (r.num == 0)
            return {0, 1};
        if (r.den < 0)
        {
            r.num = -r.num;
            r.den = -r.den;
        }
        long long g = gcdll(r.num, r.den);
        r.num /= g;
        r.den /= g;
        return r;
    }

    Rat radd(Rat a, Rat b) { return rnorm({a.num * b.den + b.num * a.den, a.den * b.den}); }
    Rat rmul(Rat a, Rat b) { return rnorm({a.num * b.num, a.den * b.den}); }
    Rat rint(long long n) { return {n, 1}; }

    using Mono = std::array<int, 3>; // (a,b,c) → x^a y^b z^c
    using Poly = std::map<Mono, Rat>;

    void poly_add_term(Poly &p, const Mono &m, Rat c)
    {
        if (c.num == 0)
            return;
        auto it = p.find(m);
        if (it == p.end())
            p[m] = c;
        else
        {
            it->second = radd(it->second, c);
            if (it->second.num == 0)
                p.erase(it);
        }
    }

    Poly poly_add(const Poly &a, const Poly &b)
    {
        Poly r = a;
        for (auto &[m, c] : b)
            poly_add_term(r, m, c);
        return r;
    }

    Poly poly_scale(const Poly &a, Rat s)
    {
        Poly r;
        for (auto &[m, c] : a)
            poly_add_term(r, m, rmul(c, s));
        return r;
    }

    Poly poly_mul(const Poly &a, const Poly &b)
    {
        Poly r;
        for (auto &[ma, ca] : a)
            for (auto &[mb, cb] : b)
                poly_add_term(r, Mono{ma[0] + mb[0], ma[1] + mb[1], ma[2] + mb[2]}, rmul(ca, cb));
        return r;
    }

    Poly poly_pow(const Poly &a, int n)
    {
        Poly r;
        r[Mono{0, 0, 0}] = rint(1);
        for (int i = 0; i < n; ++i)
            r = poly_mul(r, a);
        return r;
    }

    // ── Building blocks ──────────────────────────────────────────────────────────

    // x, y, z, and r² = x²+y²+z² as polynomials.
    Poly P_x() { return {{Mono{1, 0, 0}, rint(1)}}; }
    Poly P_y() { return {{Mono{0, 1, 0}, rint(1)}}; }
    Poly P_z() { return {{Mono{0, 0, 1}, rint(1)}}; }
    Poly P_r2()
    {
        Poly r;
        r[Mono{2, 0, 0}] = rint(1);
        r[Mono{0, 2, 0}] = rint(1);
        r[Mono{0, 0, 2}] = rint(1);
        return r;
    }

    // Re((x+iy)^μ) and Im((x+iy)^μ) as exact polynomials in x,y.
    // These equal ρ^μ cos(μφ) and ρ^μ sin(μφ), so multiplying the Legendre part by
    // them clears the ρ^μ denominator and leaves a pure polynomial.
    void chebyshev_xy(int mu, Poly &re, Poly &im)
    {
        re.clear();
        im.clear();
        // (x+iy)^μ = Σ_k C(μ,k) x^{μ-k} (iy)^k ; split by k mod 4.
        for (int k = 0; k <= mu; ++k)
        {
            // binomial(μ,k) exactly
            long long b = 1;
            for (int t = 0; t < k; ++t)
                b = b * (mu - t) / (t + 1);
            Mono m{mu - k, k, 0};
            switch (k % 4)
            {
            case 0:
                poly_add_term(re, m, rint(b));
                break; // i^k = +1
            case 1:
                poly_add_term(im, m, rint(b));
                break; // i^k = +i
            case 2:
                poly_add_term(re, m, rint(-b));
                break; // i^k = -1
            case 3:
                poly_add_term(im, m, rint(-b));
                break; // i^k = -i
            }
        }
    }

    // r^l · P_l^μ(z/r) / ρ^μ, expressed as an exact polynomial in z and r².
    // Standard closed form:
    //   P_l^μ(w) = (-1)^μ (1-w²)^{μ/2} d^μ/dw^μ P_l(w),  w = z/r,  1-w² = ρ²/r².
    // After multiplying by r^l and dividing by ρ^μ, the (1-w²)^{μ/2}=ρ^μ/r^μ factor and
    // an r^μ cancel, leaving a polynomial in z and r²:
    //   r^l P_l^μ(z/r)/ρ^μ = (-1)^μ Σ_t a_t z^{l-μ-2t} (r²)^t
    // with a_t the coefficients of the μ-th derivative of the Legendre polynomial,
    // re-homogenized by r². We build the Legendre polynomial coefficients exactly,
    // differentiate μ times, then re-homogenize. The leading (-1)^μ is dropped (an
    // overall sign is irrelevant — rows are compared up to sign).
    Poly legendre_part(int l, int mu)
    {
        // Legendre polynomial P_l(w) coefficients: P_l(w) = Σ_s p_s w^s.
        // Rodrigues / explicit: P_l(w) = 2^{-l} Σ_{i} (-1)^i C(l,i) C(2l-2i,l) w^{l-2i}.
        std::map<int, Rat> p; // power of w → coeff
        for (int i = 0; 2 * i <= l; ++i)
        {
            int poww = l - 2 * i;
            // C(l,i)
            long long cli = 1;
            for (int t = 0; t < i; ++t)
                cli = cli * (l - t) / (t + 1);
            // C(2l-2i, l)
            long long c2 = 1;
            {
                int N = 2 * l - 2 * i, K = l;
                for (int t = 0; t < K; ++t)
                    c2 = c2 * (N - t) / (t + 1);
            }
            Rat coeff = rmul(rint(((i % 2) == 0 ? 1 : -1) * cli * c2), Rat{1, 1ll << l});
            auto it = p.find(poww);
            if (it == p.end())
                p[poww] = coeff;
            else
                it->second = radd(it->second, coeff);
        }

        // Differentiate μ times w.r.t. w.
        for (int d = 0; d < mu; ++d)
        {
            std::map<int, Rat> q;
            for (auto &[poww, c] : p)
            {
                if (poww == 0)
                    continue;
                q[poww - 1] = radd(q.count(poww - 1) ? q[poww - 1] : Rat{0, 1},
                                   rmul(c, rint(poww)));
            }
            p.swap(q);
        }

        // Now p holds d^μ P_l/dw^μ as Σ a_s w^s, with s ≤ l-μ and s ≡ l-μ (mod 2).
        // Re-homogenize to degree (l-μ): each w^s term → z^s · (r²)^{(l-μ-s)/2}.
        Poly out;
        const Poly r2 = P_r2();
        const Poly z = P_z();
        for (auto &[s, c] : p)
        {
            int tpow = (l - mu - s) / 2;
            Poly term = poly_scale(poly_pow(z, s), c);
            term = poly_mul(term, poly_pow(r2, tpow));
            out = poly_add(out, term);
        }
        return out;
    }
} // namespace

std::expected<Eigen::MatrixXd, std::string>
HartreeFock::BasisFunctions::cart_to_sph_block_recurrence(int L)
{
    if (L < 0)
        return std::unexpected("cart_to_sph_block_recurrence: L must be ≥ 0");

    const int n_cart = (L + 1) * (L + 2) / 2;
    const int n_sph = 2 * L + 1;

    // Cartesian component order: lx descending, then ly (== _cartesian_shell_order).
    std::vector<Mono> carts;
    carts.reserve(n_cart);
    for (int lx = L; lx >= 0; --lx)
        for (int ly = L - lx; ly >= 0; --ly)
            carts.push_back(Mono{lx, ly, L - lx - ly});

    auto col_of = [&](const Mono &m) -> int {
        for (int c = 0; c < n_cart; ++c)
            if (carts[c] == m)
                return c;
        return -1;
    };

    // Bare-monomial coefficient matrix, rows m = −L … +L.
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(n_sph, n_cart);

    for (int row = 0; row < n_sph; ++row)
    {
        const int m = row - L;
        const int mu = std::abs(m);

        const Poly leg = legendre_part(L, mu); // polynomial in z, r²

        Poly harmonic;
        if (m == 0)
        {
            harmonic = leg; // ρ^0 cos(0) = 1
        }
        else
        {
            Poly re, im;
            chebyshev_xy(mu, re, im);
            harmonic = poly_mul(leg, (m > 0) ? re : im);
        }

        // Place exact coefficients (as doubles) into the row.
        for (auto &[mono, c] : harmonic)
        {
            const int col = col_of(mono);
            if (col < 0)
                continue; // should not happen: degree is exactly L
            T(row, col) = static_cast<double>(c.num) / static_cast<double>(c.den);
        }
    }

    // Convert from bare-monomial coefficients to the unit-normalized Cartesian-Gaussian
    // basis (the integral basis): the self-overlap of monomial x^a y^b z^c is
    //   s = (2a−1)!!(2b−1)!!(2c−1)!!,  unit-normalized function = monomial/√s,
    // so the coefficient on the normalized function is c_bare · √s.
    auto dfact = [](int n) {
        double r = 1.0;
        for (int k = n; k > 1; k -= 2)
            r *= k;
        return r;
    };
    for (int c = 0; c < n_cart; ++c)
    {
        const double s =
            dfact(2 * carts[c][0] - 1) * dfact(2 * carts[c][1] - 1) * dfact(2 * carts[c][2] - 1);
        T.col(c) *= std::sqrt(s);
    }

    // Unit-normalize each spherical row.
    for (int row = 0; row < n_sph; ++row)
    {
        const double nrm = T.row(row).norm();
        if (nrm > 1e-300)
            T.row(row) /= nrm;
    }

    return T;
}
