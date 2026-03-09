#include "os.h"
#include "base.h"
#include "boys.h"

// Local copy of the Boys function so the IDE and compiler both see it
// unambiguously within this translation unit.
// F_n(x) = integral_0^1 t^(2n) exp(-x*t^2) dt
static double _boys(const int n, const double x)
{
    if (x < 1e-7)
        return 1.0 / (2 * n + 1);

    if (x > 20.0)
        return std::tgamma(n + 0.5) / (2.0 * std::pow(x, n + 0.5));

    const double sqrt_x = std::sqrt(x);
    const double ex     = std::exp(-x);
    double f = 0.5 * std::sqrt(M_PI / x) * std::erf(sqrt_x);

    for (int m = 0; m < n; m++)
        f = ((2 * m + 1) * f - ex) / (2.0 * x);

    return f;
}

inline double HartreeFock::ObaraSaika::_os_1d(const double gamma, const double distPA, const double distPB, const int lA, const int lB)
{
    double S[MAX_L + 1][MAX_L + 1] = {};
    
    S[0][0] = 1.0;  // Set base factor
    
    // First build angular momentum in lA
    for (int i = 1; i <= lA; i++)
    {
        S[i][0] = distPA * S[i - 1][0];
        
        if (i > 1)
        {
            S[i][0] = (i - 1) * gamma * S[i - 2][0] + S[i][0];
        }
    }

    // Then build angular momentum in lA
    for (int j = 1; j <= lB; j++)
    {
        S[0][j] = distPB * S[0][j - 1];
        
        if (j > 1)
        {
            S[0][j] = (j - 1) * gamma * S[0][j - 2] + S[0][j];
        }
    }
    
    // Now build the full table
    for (int i = 1; i <= lA; i++)
    {
        for (int j = 1; j <= lB; j++)
        {
            S[i][j] = distPA * S[i - 1][j] +
                      j * gamma * S[i - 1][j - 1];

            if (i > 1)
                S[i][j] += (i - 1) * gamma * S[i - 2][j];
        }
    }
    
    return S[lA][lB];
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> HartreeFock::ObaraSaika::_compute_1e(const std::vector<HartreeFock::ShellPair> &shell_pairs, const std::size_t nbasis)
{
    const std::size_t npairs = shell_pairs.size();
    Eigen::MatrixXd overlap  = Eigen::MatrixXd::Zero(nbasis, nbasis);
    Eigen::MatrixXd kinetic  = Eigen::MatrixXd::Zero(nbasis, nbasis);

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < npairs; i++)
    {
        const std::size_t ii = shell_pairs[i].A._index;
        const std::size_t jj = shell_pairs[i].B._index;
        const auto [s, t]    = _compute_3d_overlap_kinetic(shell_pairs[i]);

        overlap(ii, jj) = s;
        overlap(jj, ii) = s;
        kinetic(ii, jj) = t;
        kinetic(jj, ii) = t;
    }

    return {overlap, kinetic};
}

std::tuple<double, double> HartreeFock::ObaraSaika::_compute_3d_overlap_kinetic(const ShellPair &shell_pair)
{
    const auto &cartA = shell_pair.A._cartesian;
    const auto &cartB = shell_pair.B._cartesian;

    const auto &primitive_pairs = shell_pair.primitive_pairs;
    const std::size_t size_pp   = primitive_pairs.size();

    double S = 0.0; // Overlap integral
    double T = 0.0; // Kinetic energy integral

    for (std::size_t i = 0; i < size_pp; i++)
    {
        const double half_inv_zeta  = primitive_pairs[i].inv_zeta * 0.5;
        const double beta           = primitive_pairs[i].beta;
        const double beta2          = beta * beta;
        const double scale          = primitive_pairs[i].prefactor * primitive_pairs[i].coeff_product;

        // Base 1D overlaps S(lAx, lBx), S(lAy, lBy), S(lAz, lBz)
        const double Sx = _os_1d(half_inv_zeta, primitive_pairs[i].pA[0], primitive_pairs[i].pB[0], cartA[0], cartB[0]);
        const double Sy = _os_1d(half_inv_zeta, primitive_pairs[i].pA[1], primitive_pairs[i].pB[1], cartA[1], cartB[1]);
        const double Sz = _os_1d(half_inv_zeta, primitive_pairs[i].pA[2], primitive_pairs[i].pB[2], cartA[2], cartB[2]);

        S += Sx * Sy * Sz * scale;

        // Angular momenta on B
        const int lbx = cartB[0], lby = cartB[1], lbz = cartB[2];

        // +2 shifted overlaps: S(lA, lB+2)
        const double Sxp = _os_1d(half_inv_zeta, primitive_pairs[i].pA[0], primitive_pairs[i].pB[0], cartA[0], lbx + 2);
        const double Syp = _os_1d(half_inv_zeta, primitive_pairs[i].pA[1], primitive_pairs[i].pB[1], cartA[1], lby + 2);
        const double Szp = _os_1d(half_inv_zeta, primitive_pairs[i].pA[2], primitive_pairs[i].pB[2], cartA[2], lbz + 2);

        // -2 shifted overlaps: S(lA, lB-2) — only valid when lB >= 2
        const double Sxm = (lbx >= 2) ? _os_1d(half_inv_zeta, primitive_pairs[i].pA[0], primitive_pairs[i].pB[0], cartA[0], lbx - 2) : 0.0;
        const double Sym = (lby >= 2) ? _os_1d(half_inv_zeta, primitive_pairs[i].pA[1], primitive_pairs[i].pB[1], cartA[1], lby - 2) : 0.0;
        const double Szm = (lbz >= 2) ? _os_1d(half_inv_zeta, primitive_pairs[i].pA[2], primitive_pairs[i].pB[2], cartA[2], lbz - 2) : 0.0;

        // 1D kinetic contributions:
        // T_x(a,b) = -b*(b-1)/2 * S(a,b-2) + beta*(2b+1) * S(a,b) - 2*beta^2 * S(a,b+2)
        const double Tx = -0.5 * lbx * (lbx - 1) * Sxm + beta * (2 * lbx + 1) * Sx - 2.0 * beta2 * Sxp;
        const double Ty = -0.5 * lby * (lby - 1) * Sym + beta * (2 * lby + 1) * Sy - 2.0 * beta2 * Syp;
        const double Tz = -0.5 * lbz * (lbz - 1) * Szm + beta * (2 * lbz + 1) * Sz - 2.0 * beta2 * Szp;

        // T = Tx*Sy*Sz + Sx*Ty*Sz + Sx*Sy*Tz
        T += (Tx * Sy * Sz + Sx * Ty * Sz + Sx * Sy * Tz) * scale;
    }

    return {S, T};
}

// ─── Nuclear attraction ────────────────────────────────────────────────────────

// Recursive HRR: transfer angular momentum from A-centre to B-centre.
// [a | b+1_q] = [a+1_q | b] + (A-B)_q * [a | b]
// V0[ax][ay][az] contains the VRR result at m=0.
static double _nuclear_hrr(
    const double V0[MAX_L + 1][MAX_L + 1][MAX_L + 1],
    const int ax, const int ay, const int az,
    const int bx, const int by, const int bz,
    const double ABx, const double ABy, const double ABz)
{
    if (bx == 0 && by == 0 && bz == 0)
        return V0[ax][ay][az];

    if (bx > 0)
        return _nuclear_hrr(V0, ax + 1, ay, az, bx - 1, by, bz, ABx, ABy, ABz)
             + ABx * _nuclear_hrr(V0, ax, ay, az, bx - 1, by, bz, ABx, ABy, ABz);

    if (by > 0)
        return _nuclear_hrr(V0, ax, ay + 1, az, bx, by - 1, bz, ABx, ABy, ABz)
             + ABy * _nuclear_hrr(V0, ax, ay, az, bx, by - 1, bz, ABx, ABy, ABz);

    return _nuclear_hrr(V0, ax, ay, az + 1, bx, by, bz - 1, ABx, ABy, ABz)
         + ABz * _nuclear_hrr(V0, ax, ay, az, bx, by, bz - 1, ABx, ABy, ABz);
}

// Compute <A| -Z/|r-C| |B> for a single primitive pair and a single nucleus.
// Returns the unnormalized primitive integral (coeff_product applied by caller).
static double _os_nuclear_primitive(
    const HartreeFock::PrimitivePair& pp,
    const int lAx, const int lAy, const int lAz,
    const int lBx, const int lBy, const int lBz,
    const double ABx, const double ABy, const double ABz,
    const Eigen::Vector3d& nuc_pos)
{
    const int L = lAx + lAy + lAz + lBx + lBy + lBz;

    // P - C
    const double pCx = pp.center[0] - nuc_pos[0];
    const double pCy = pp.center[1] - nuc_pos[1];
    const double pCz = pp.center[2] - nuc_pos[2];
    const double T   = pp.zeta * (pCx * pCx + pCy * pCy + pCz * pCz);

    // pp.prefactor = (pi/zeta)^1.5 * exp(-alpha*beta/zeta * |AB|^2)
    // Nuclear prefactor = 2*pi/zeta * exp(-alpha*beta/zeta * |AB|^2)
    //                   = pp.prefactor * 2*pi/zeta * (zeta/pi)^1.5
    //                   = pp.prefactor * 2 * sqrt(zeta/pi)
    const double nuc_pref = pp.prefactor * 2.0 * std::sqrt(pp.zeta / M_PI);

    // ── VRR table V[ix][iy][iz][m] ──────────────────────────────────────────
    constexpr int MMAX  = 2 * MAX_L + 2;
    double V[MAX_L + 1][MAX_L + 1][MAX_L + 1][MMAX] = {};

    for (int m = 0; m <= L; m++)
        V[0][0][0][m] = nuc_pref * _boys(m, T);

    const double pAx       = pp.pA[0], pAy = pp.pA[1], pAz = pp.pA[2];
    const double hiz       = pp.inv_zeta * 0.5;
    const int    lx_max    = lAx + lBx;
    const int    ly_max    = lAy + lBy;
    const int    lz_max    = lAz + lBz;

    // x-VRR: V[ix][0][0][m]
    for (int ix = 1; ix <= lx_max; ix++) {
        for (int m = 0; m <= L - ix; m++) {
            V[ix][0][0][m] = pAx * V[ix - 1][0][0][m] - pCx * V[ix - 1][0][0][m + 1];
            if (ix > 1)
                V[ix][0][0][m] += (ix - 1) * hiz * (V[ix - 2][0][0][m] - V[ix - 2][0][0][m + 1]);
        }
    }

    // y-VRR: V[ix][iy][0][m]
    for (int ix = 0; ix <= lx_max; ix++) {
        for (int iy = 1; iy <= ly_max; iy++) {
            const int mmax = L - ix - iy;
            if (mmax < 0) continue;
            for (int m = 0; m <= mmax; m++) {
                V[ix][iy][0][m] = pAy * V[ix][iy - 1][0][m] - pCy * V[ix][iy - 1][0][m + 1];
                if (iy > 1)
                    V[ix][iy][0][m] += (iy - 1) * hiz * (V[ix][iy - 2][0][m] - V[ix][iy - 2][0][m + 1]);
            }
        }
    }

    // z-VRR: V[ix][iy][iz][m]
    for (int ix = 0; ix <= lx_max; ix++) {
        for (int iy = 0; iy <= ly_max; iy++) {
            for (int iz = 1; iz <= lz_max; iz++) {
                const int mmax = L - ix - iy - iz;
                if (mmax < 0) continue;
                for (int m = 0; m <= mmax; m++) {
                    V[ix][iy][iz][m] = pAz * V[ix][iy][iz - 1][m] - pCz * V[ix][iy][iz - 1][m + 1];
                    if (iz > 1)
                        V[ix][iy][iz][m] += (iz - 1) * hiz * (V[ix][iy][iz - 2][m] - V[ix][iy][iz - 2][m + 1]);
                }
            }
        }
    }

    // ── Extract m=0 slice for HRR ───────────────────────────────────────────
    double V0[MAX_L + 1][MAX_L + 1][MAX_L + 1] = {};
    for (int ix = 0; ix <= lx_max; ix++)
        for (int iy = 0; iy <= ly_max; iy++)
            for (int iz = 0; iz <= lz_max; iz++)
                V0[ix][iy][iz] = V[ix][iy][iz][0];

    // ── HRR: transfer angular momentum to B ─────────────────────────────────
    return _nuclear_hrr(V0, lAx, lAy, lAz, lBx, lBy, lBz, ABx, ABy, ABz);
}

Eigen::MatrixXd HartreeFock::ObaraSaika::_compute_nuclear_attraction(
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    const std::size_t nbasis,
    const HartreeFock::Molecule& molecule)
{
    const std::size_t npairs = shell_pairs.size();
    Eigen::MatrixXd V        = Eigen::MatrixXd::Zero(nbasis, nbasis);

#pragma omp parallel for schedule(dynamic)
    for (std::size_t p = 0; p < npairs; p++)
    {
        const auto& sp = shell_pairs[p];
        const std::size_t ii = sp.A._index;
        const std::size_t jj = sp.B._index;

        const int lAx = sp.A._cartesian[0], lAy = sp.A._cartesian[1], lAz = sp.A._cartesian[2];
        const int lBx = sp.B._cartesian[0], lBy = sp.B._cartesian[1], lBz = sp.B._cartesian[2];

        // AB = A - B (used by HRR)
        const double ABx = sp.R[0], ABy = sp.R[1], ABz = sp.R[2];

        double v_elem = 0.0;

        for (std::size_t a = 0; a < molecule.natoms; a++)
        {
            const double Z_C = static_cast<double>(molecule.atomic_numbers[a]);
            // Nucleus position in Bohr — molecule._standard is always in Bohr
            const Eigen::Vector3d C(molecule._standard(a, 0),
                                    molecule._standard(a, 1),
                                    molecule._standard(a, 2));

            double v_nuc = 0.0;
            for (const auto& pp : sp.primitive_pairs)
                v_nuc += _os_nuclear_primitive(pp, lAx, lAy, lAz, lBx, lBy, lBz, ABx, ABy, ABz, C)
                         * pp.coeff_product;

            v_elem -= Z_C * v_nuc; // nuclear attraction is negative
        }

        V(ii, jj) = v_elem;
        V(jj, ii) = v_elem;
    }

    return V;
}
