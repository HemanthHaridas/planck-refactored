#include "os.h"
#include "base.h"
#include "boys.h"

inline double HartreeFock::ObaraSaika::_os_1d(const double gamma, const double distPA, const double distPB, const int lA, const int lB)
{
    // +2 shifted overlaps are needed by the kinetic energy formula (lB+2),
    // so the second dimension must accommodate lB = MAX_L + 2.
    double S[MAX_L + 3][MAX_L + 3] = {};
    
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

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> HartreeFock::ObaraSaika::_compute_1e(const std::vector<HartreeFock::ShellPair> &shell_pairs, const std::size_t nbasis, const std::vector<HartreeFock::SignedAOSymOp>*)
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

// ─Nuclear attraction ─────────────

// Iterative HRR: transfer angular momentum from A-centre to B-centre.
// Recurrence: [a | b+1_q] = [a+1_q | b] + (A-B)_q * [a | b]
//
// Three sequential passes (z → y → x) each sweep the working array
// in ascending index order.  Because each update reads W[i+1] before
// writing W[i], W[i+1] is always the old (pre-pass) value — so the
// scan is safe in-place without an auxiliary buffer.
//
// Complexity: O(L^3 * (bx+by+bz))  vs  O(2^(bx+by+bz)) for the
// recursive version.
// VRR spatial dimension: each axis of the VRR table runs from 0 to
// lAx+lBx (or y,z equivalents).  lAx can be MAX_L and lBx can be
// MAX_L independently, so the table needs 2*MAX_L+1 entries per axis.
static constexpr int VRR_DIM  = 2 * MAX_L + 1;  // = 13; per-axis bound for 1-pair VRR
static constexpr int MMAX_4C  = 4 * MAX_L + 2;  // = 26; Boys m upper bound for 4-center ERI

// Thread-local scratch buffers for 4-center ERI (too large for the stack).
// VRR buffer: V[ax][ay][az][cx][cy][cz][m]
// HRR buffer: W[ax][ay][az][cx][cy][cz]  (m=0 slice used during HRR)
static thread_local double _vrr_buf[VRR_DIM][VRR_DIM][VRR_DIM]
                                    [VRR_DIM][VRR_DIM][VRR_DIM][MMAX_4C];
static thread_local double _hrr_buf[VRR_DIM][VRR_DIM][VRR_DIM]
                                    [VRR_DIM][VRR_DIM][VRR_DIM];

static double _nuclear_hrr(
    const double V0[VRR_DIM][VRR_DIM][VRR_DIM],
    const int ax, const int ay, const int az,
    const int bx, const int by, const int bz,
    const double ABx, const double ABy, const double ABz)
{
    if (bx == 0 && by == 0 && bz == 0)
        return V0[ax][ay][az];

    // Working copy — same footprint as V0.
    double W[VRR_DIM][VRR_DIM][VRR_DIM];
    for (int ix = 0; ix <= ax + bx; ix++)
        for (int iy = 0; iy <= ay + by; iy++)
            for (int iz = 0; iz <= az + bz; iz++)
                W[ix][iy][iz] = V0[ix][iy][iz];

    // Phase 1 — transfer bz quanta to az.
    // After k steps: W[ix][iy][iz] = [ix, iy, iz | 0, 0, k]
    // for iz in [0, az + bz - k].
    for (int kz = 0; kz < bz; kz++)
        for (int ix = 0; ix <= ax + bx; ix++)
            for (int iy = 0; iy <= ay + by; iy++)
                for (int iz = 0; iz <= az + bz - kz - 1; iz++)
                    W[ix][iy][iz] = W[ix][iy][iz + 1] + ABz * W[ix][iy][iz];

    // Phase 2 — transfer by quanta to ay (iz range now [0, az]).
    // After k steps: W[ix][iy][iz] = [ix, iy, iz | 0, k, bz]
    // for iy in [0, ay + by - k].
    for (int ky = 0; ky < by; ky++)
        for (int ix = 0; ix <= ax + bx; ix++)
            for (int iy = 0; iy <= ay + by - ky - 1; iy++)
                for (int iz = 0; iz <= az; iz++)
                    W[ix][iy][iz] = W[ix][iy + 1][iz] + ABy * W[ix][iy][iz];

    // Phase 3 — transfer bx quanta to ax (iy range now [0, ay]).
    // After k steps: W[ix][iy][iz] = [ix, iy, iz | k, by, bz]
    // for ix in [0, ax + bx - k].
    for (int kx = 0; kx < bx; kx++)
        for (int ix = 0; ix <= ax + bx - kx - 1; ix++)
            for (int iy = 0; iy <= ay; iy++)
                for (int iz = 0; iz <= az; iz++)
                    W[ix][iy][iz] = W[ix + 1][iy][iz] + ABx * W[ix][iy][iz];

    return W[ax][ay][az];
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

    // VRR table V[ix][iy][iz][m]
    // Spatial dims: each axis runs 0..lAq+lBq ≤ 2*MAX_L, so VRR_DIM = 2*MAX_L+1.
    // m dimension: m+1 is accessed up to L+1 ≤ 2*MAX_L+1, so MMAX = 2*MAX_L+2.
    constexpr int MMAX = 2 * MAX_L + 2;
    double V[VRR_DIM][VRR_DIM][VRR_DIM][MMAX] = {};

    // Compute all Boys values
    for (int m = 0; m <= L; m++)
    {
        V[0][0][0][m] = nuc_pref * HartreeFock::Lookup::boys(m, T);
    }
    
    const double pAx       = pp.pA[0], pAy = pp.pA[1], pAz = pp.pA[2];
    const double hiz       = pp.inv_zeta * 0.5;
    const int    lx_max    = lAx + lBx;
    const int    ly_max    = lAy + lBy;
    const int    lz_max    = lAz + lBz;

    // x-VRR: V[ix][0][0][m]
    for (int ix = 1; ix <= lx_max; ix++)
    {
        for (int m = 0; m <= L - ix; m++)
        {
            V[ix][0][0][m] = pAx * V[ix - 1][0][0][m] - pCx * V[ix - 1][0][0][m + 1];
            if (ix > 1)
            {
                V[ix][0][0][m] += (ix - 1) * hiz * (V[ix - 2][0][0][m] - V[ix - 2][0][0][m + 1]);
            }
        }
    }

    // y-VRR: V[ix][iy][0][m]
    for (int ix = 0; ix <= lx_max; ix++)
    {
        for (int iy = 1; iy <= ly_max; iy++)
        {
            const int mmax = L - ix - iy;
            if (mmax < 0) continue;
            for (int m = 0; m <= mmax; m++)
            {
                V[ix][iy][0][m] = pAy * V[ix][iy - 1][0][m] - pCy * V[ix][iy - 1][0][m + 1];
                if (iy > 1)
                {
                    V[ix][iy][0][m] += (iy - 1) * hiz * (V[ix][iy - 2][0][m] - V[ix][iy - 2][0][m + 1]);
                }
            }
        }
    }

    // z-VRR: V[ix][iy][iz][m]
    for (int ix = 0; ix <= lx_max; ix++)
    {
        for (int iy = 0; iy <= ly_max; iy++)
        {
            for (int iz = 1; iz <= lz_max; iz++)
            {
                const int mmax = L - ix - iy - iz;
                if (mmax < 0) continue;
                for (int m = 0; m <= mmax; m++)
                {
                    V[ix][iy][iz][m] = pAz * V[ix][iy][iz - 1][m] - pCz * V[ix][iy][iz - 1][m + 1];
                    if (iz > 1)
                    {
                        V[ix][iy][iz][m] += (iz - 1) * hiz * (V[ix][iy][iz - 2][m] - V[ix][iy][iz - 2][m + 1]);
                    }
                }
            }
        }
    }

    // Extract m=0 slice for HRR
    double V0[VRR_DIM][VRR_DIM][VRR_DIM] = {};
    for (int ix = 0; ix <= lx_max; ix++)
    {
        for (int iy = 0; iy <= ly_max; iy++)
        {
            for (int iz = 0; iz <= lz_max; iz++)
            {
                V0[ix][iy][iz] = V[ix][iy][iz][0];
            }
        }
    }

    // HRR: transfer angular momentum to B 
    return _nuclear_hrr(V0, lAx, lAy, lAz, lBx, lBy, lBz, ABx, ABy, ABz);
}

Eigen::MatrixXd HartreeFock::ObaraSaika::_compute_nuclear_attraction(
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    const std::size_t nbasis,
    const HartreeFock::Molecule& molecule,
    const std::vector<HartreeFock::SignedAOSymOp>*)
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

// ─── 4-center ERI: VRR ───────────────────────────────────────────────────────
//
// Builds V[ax][ay][az][cx][cy][cz][m] for a single primitive quartet.
// ax runs 0..lABx (= lAx+lBx), cy runs 0..lCDy, etc.
// On return, V[...][0] holds (a 0 | c 0)^{m=0} ready for the HRR stage.
//
// Recurrences (Obara-Saika 1986, Eq. 6):
//   (a+1_q 0|c 0)^m = PA_q (a0|c0)^m + WP_q (a0|c0)^{m+1}
//                   + (a_q/2ζ)[(a-1_q 0|c0)^m - (ρ/ζ)(a-1_q 0|c0)^{m+1}]
//                   + (c_q/2δ)(a-1_q 0|c-1_q 0)^{m+1}
//
//   (a 0|c+1_q 0)^m = QC_q (a0|c0)^m + WQ_q (a0|c0)^{m+1}
//                   + (c_q/2ζ')[(a0|c-1_q 0)^m - (ρ/ζ')(a0|c-1_q 0)^{m+1}]
//                   + (a_q/2δ)(a-1_q 0|c-1_q 0)^{m+1}
static void _eri_vrr(
    const HartreeFock::PrimitivePair& ppAB,
    const HartreeFock::PrimitivePair& ppCD,
    const int lABx, const int lABy, const int lABz,
    const int lCDx, const int lCDy, const int lCDz,
    double V[VRR_DIM][VRR_DIM][VRR_DIM][VRR_DIM][VRR_DIM][VRR_DIM][MMAX_4C])
{
    const double zetaAB = ppAB.zeta;
    const double zetaCD = ppCD.zeta;
    const double delta  = zetaAB + zetaCD;
    const double rho    = zetaAB * zetaCD / delta;

    const double inv_2_zetaAB    = 0.5 / zetaAB;
    const double inv_2_zetaCD    = 0.5 / zetaCD;
    const double inv_2_delta     = 0.5 / delta;
    const double rho_over_zetaAB = rho / zetaAB;
    const double rho_over_zetaCD = rho / zetaCD;

    const auto& P = ppAB.center;
    const auto& Q = ppCD.center;

    // W = weighted average of Gaussian product centers
    const double Wx = (zetaAB * P[0] + zetaCD * Q[0]) / delta;
    const double Wy = (zetaAB * P[1] + zetaCD * Q[1]) / delta;
    const double Wz = (zetaAB * P[2] + zetaCD * Q[2]) / delta;

    // WP = W - P,  WQ = W - Q
    const double WPx = Wx - P[0],  WPy = Wy - P[1],  WPz = Wz - P[2];
    const double WQx = Wx - Q[0],  WQy = Wy - Q[1],  WQz = Wz - Q[2];

    // PA = P - A = ppAB.pA;  QC = Q - C = ppCD.pA (since ppCD.A is shell C)
    const double PAx = ppAB.pA[0], PAy = ppAB.pA[1], PAz = ppAB.pA[2];
    const double QCx = ppCD.pA[0], QCy = ppCD.pA[1], QCz = ppCD.pA[2];

    const double PQx = P[0] - Q[0], PQy = P[1] - Q[1], PQz = P[2] - Q[2];
    const double T   = rho * (PQx*PQx + PQy*PQy + PQz*PQz);

    const int MMAX = lABx + lABy + lABz + lCDx + lCDy + lCDz;

    const double prefac = ppAB.prefactor * ppCD.prefactor * 2.0 * std::sqrt(rho / M_PI);

    // ── Seed ─────────────────────────────────────────────────────────────────
    for (int m = 0; m <= MMAX; ++m)
        V[0][0][0][0][0][0][m] = prefac * HartreeFock::Lookup::boys(m, T);

    // ── A-VRR: x-axis ─────────────────────────────────────────────────────
    for (int ax = 1; ax <= lABx; ++ax)
    {
        const int mlim = MMAX - ax;
        for (int m = 0; m <= mlim; ++m)
        {
            V[ax][0][0][0][0][0][m] =
                PAx * V[ax-1][0][0][0][0][0][m]
              + WPx * V[ax-1][0][0][0][0][0][m+1];
            if (ax > 1)
                V[ax][0][0][0][0][0][m] +=
                    (ax-1) * inv_2_zetaAB *
                    (V[ax-2][0][0][0][0][0][m] - rho_over_zetaAB * V[ax-2][0][0][0][0][0][m+1]);
        }
    }

    // ── A-VRR: y-axis ─────────────────────────────────────────────────────
    for (int ax = 0; ax <= lABx; ++ax)
    {
        for (int ay = 1; ay <= lABy; ++ay)
        {
            const int mlim = MMAX - ax - ay;
            if (mlim < 0) continue;
            for (int m = 0; m <= mlim; ++m)
            {
                V[ax][ay][0][0][0][0][m] =
                    PAy * V[ax][ay-1][0][0][0][0][m]
                  + WPy * V[ax][ay-1][0][0][0][0][m+1];
                if (ay > 1)
                    V[ax][ay][0][0][0][0][m] +=
                        (ay-1) * inv_2_zetaAB *
                        (V[ax][ay-2][0][0][0][0][m] - rho_over_zetaAB * V[ax][ay-2][0][0][0][0][m+1]);
            }
        }
    }

    // ── A-VRR: z-axis ─────────────────────────────────────────────────────
    for (int ax = 0; ax <= lABx; ++ax)
    {
        for (int ay = 0; ay <= lABy; ++ay)
        {
            for (int az = 1; az <= lABz; ++az)
            {
                const int mlim = MMAX - ax - ay - az;
                if (mlim < 0) continue;
                for (int m = 0; m <= mlim; ++m)
                {
                    V[ax][ay][az][0][0][0][m] =
                        PAz * V[ax][ay][az-1][0][0][0][m]
                      + WPz * V[ax][ay][az-1][0][0][0][m+1];
                    if (az > 1)
                        V[ax][ay][az][0][0][0][m] +=
                            (az-1) * inv_2_zetaAB *
                            (V[ax][ay][az-2][0][0][0][m] - rho_over_zetaAB * V[ax][ay][az-2][0][0][0][m+1]);
                }
            }
        }
    }

    // ── C-VRR: x-axis ─────────────────────────────────────────────────────
    for (int ax = 0; ax <= lABx; ++ax)
    {
        for (int ay = 0; ay <= lABy; ++ay)
        {
            for (int az = 0; az <= lABz; ++az)
            {
                for (int cx = 1; cx <= lCDx; ++cx)
                {
                    const int mlim = MMAX - ax - ay - az - cx;
                    if (mlim < 0) continue;
                    for (int m = 0; m <= mlim; ++m)
                    {
                        V[ax][ay][az][cx][0][0][m] =
                            QCx * V[ax][ay][az][cx-1][0][0][m]
                          + WQx * V[ax][ay][az][cx-1][0][0][m+1];
                        if (cx > 1)
                            V[ax][ay][az][cx][0][0][m] +=
                                (cx-1) * inv_2_zetaCD *
                                (V[ax][ay][az][cx-2][0][0][m] - rho_over_zetaCD * V[ax][ay][az][cx-2][0][0][m+1]);
                        if (ax > 0)
                            V[ax][ay][az][cx][0][0][m] +=
                                ax * inv_2_delta * V[ax-1][ay][az][cx-1][0][0][m+1];
                    }
                }
            }
        }
    }

    // ── C-VRR: y-axis ─────────────────────────────────────────────────────
    for (int ax = 0; ax <= lABx; ++ax)
    {
        for (int ay = 0; ay <= lABy; ++ay)
        {
            for (int az = 0; az <= lABz; ++az)
            {
                for (int cx = 0; cx <= lCDx; ++cx)
                {
                    for (int cy = 1; cy <= lCDy; ++cy)
                    {
                        const int mlim = MMAX - ax - ay - az - cx - cy;
                        if (mlim < 0) continue;
                        for (int m = 0; m <= mlim; ++m)
                        {
                            V[ax][ay][az][cx][cy][0][m] =
                                QCy * V[ax][ay][az][cx][cy-1][0][m]
                              + WQy * V[ax][ay][az][cx][cy-1][0][m+1];
                            if (cy > 1)
                                V[ax][ay][az][cx][cy][0][m] +=
                                    (cy-1) * inv_2_zetaCD *
                                    (V[ax][ay][az][cx][cy-2][0][m] - rho_over_zetaCD * V[ax][ay][az][cx][cy-2][0][m+1]);
                            if (ay > 0)
                                V[ax][ay][az][cx][cy][0][m] +=
                                    ay * inv_2_delta * V[ax][ay-1][az][cx][cy-1][0][m+1];
                        }
                    }
                }
            }
        }
    }

    // ── C-VRR: z-axis ─────────────────────────────────────────────────────
    for (int ax = 0; ax <= lABx; ++ax)
    {
        for (int ay = 0; ay <= lABy; ++ay)
        {
            for (int az = 0; az <= lABz; ++az)
            {
                for (int cx = 0; cx <= lCDx; ++cx)
                {
                    for (int cy = 0; cy <= lCDy; ++cy)
                    {
                        for (int cz = 1; cz <= lCDz; ++cz)
                        {
                            const int mlim = MMAX - ax - ay - az - cx - cy - cz;
                            if (mlim < 0) continue;
                            for (int m = 0; m <= mlim; ++m)
                            {
                                V[ax][ay][az][cx][cy][cz][m] =
                                    QCz * V[ax][ay][az][cx][cy][cz-1][m]
                                  + WQz * V[ax][ay][az][cx][cy][cz-1][m+1];
                                if (cz > 1)
                                    V[ax][ay][az][cx][cy][cz][m] +=
                                        (cz-1) * inv_2_zetaCD *
                                        (V[ax][ay][az][cx][cy][cz-2][m] - rho_over_zetaCD * V[ax][ay][az][cx][cy][cz-2][m+1]);
                                if (az > 0)
                                    V[ax][ay][az][cx][cy][cz][m] +=
                                        az * inv_2_delta * V[ax][ay][az-1][cx][cy][cz-1][m+1];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ─── 4-center ERI: A→B HRR on 6D array ──────────────────────────────────────
//
// Modifies W[ax][ay][az][cx][cy][cz] in-place using the same 3-phase iterative
// sweep as _nuclear_hrr, but keeping the CD indices in the inner loop so all
// (cx,cy,cz) entries are updated simultaneously.
//
// After this call, W[lAx][lAy][lAz][cx][cy][cz] = (lA lB | cx cy cz 0).
static void _eri_hrr_ab(
    double W[VRR_DIM][VRR_DIM][VRR_DIM][VRR_DIM][VRR_DIM][VRR_DIM],
    const int lAx, const int lAy, const int lAz,
    const int lBx, const int lBy, const int lBz,
    const int lCDx, const int lCDy, const int lCDz,
    const double ABx, const double ABy, const double ABz)
{
    // Phase 1: transfer lBz quanta from az to bz
    for (int kz = 0; kz < lBz; ++kz)
        for (int ax = 0; ax <= lAx + lBx; ++ax)
            for (int ay = 0; ay <= lAy + lBy; ++ay)
                for (int az = 0; az <= lAz + lBz - kz - 1; ++az)
                    for (int cx = 0; cx <= lCDx; ++cx)
                        for (int cy = 0; cy <= lCDy; ++cy)
                            for (int cz = 0; cz <= lCDz; ++cz)
                                W[ax][ay][az][cx][cy][cz] =
                                    W[ax][ay][az+1][cx][cy][cz]
                                  + ABz * W[ax][ay][az][cx][cy][cz];

    // Phase 2: transfer lBy quanta (az range now [0, lAz])
    for (int ky = 0; ky < lBy; ++ky)
        for (int ax = 0; ax <= lAx + lBx; ++ax)
            for (int ay = 0; ay <= lAy + lBy - ky - 1; ++ay)
                for (int az = 0; az <= lAz; ++az)
                    for (int cx = 0; cx <= lCDx; ++cx)
                        for (int cy = 0; cy <= lCDy; ++cy)
                            for (int cz = 0; cz <= lCDz; ++cz)
                                W[ax][ay][az][cx][cy][cz] =
                                    W[ax][ay+1][az][cx][cy][cz]
                                  + ABy * W[ax][ay][az][cx][cy][cz];

    // Phase 3: transfer lBx quanta (ay range now [0, lAy])
    for (int kx = 0; kx < lBx; ++kx)
        for (int ax = 0; ax <= lAx + lBx - kx - 1; ++ax)
            for (int ay = 0; ay <= lAy; ++ay)
                for (int az = 0; az <= lAz; ++az)
                    for (int cx = 0; cx <= lCDx; ++cx)
                        for (int cy = 0; cy <= lCDy; ++cy)
                            for (int cz = 0; cz <= lCDz; ++cz)
                                W[ax][ay][az][cx][cy][cz] =
                                    W[ax+1][ay][az][cx][cy][cz]
                                  + ABx * W[ax][ay][az][cx][cy][cz];
}

// ─── 4-center ERI: single primitive quartet ──────────────────────────────────
static double _os_eri_primitive(
    const HartreeFock::PrimitivePair& ppAB,
    const HartreeFock::PrimitivePair& ppCD,
    const int lAx, const int lAy, const int lAz,
    const int lBx, const int lBy, const int lBz,
    const int lCx, const int lCy, const int lCz,
    const int lDx, const int lDy, const int lDz,
    const double ABx, const double ABy, const double ABz,
    const double CDx, const double CDy, const double CDz)
{
    const int lABx = lAx + lBx, lABy = lAy + lBy, lABz = lAz + lBz;
    const int lCDx = lCx + lDx, lCDy = lCy + lDy, lCDz = lCz + lDz;

    // Zero the needed sub-region of the thread-local VRR buffer
    for (int ax = 0; ax <= lABx; ++ax)
        for (int ay = 0; ay <= lABy; ++ay)
            for (int az = 0; az <= lABz; ++az)
                for (int cx = 0; cx <= lCDx; ++cx)
                    for (int cy = 0; cy <= lCDy; ++cy)
                        for (int cz = 0; cz <= lCDz; ++cz)
                            for (int m = 0; m < MMAX_4C; ++m)
                                _vrr_buf[ax][ay][az][cx][cy][cz][m] = 0.0;

    // Build VRR table
    _eri_vrr(ppAB, ppCD, lABx, lABy, lABz, lCDx, lCDy, lCDz, _vrr_buf);

    // Extract m=0 slice into HRR buffer and zero unused entries
    for (int ax = 0; ax <= lABx; ++ax)
        for (int ay = 0; ay <= lABy; ++ay)
            for (int az = 0; az <= lABz; ++az)
                for (int cx = 0; cx <= lCDx; ++cx)
                    for (int cy = 0; cy <= lCDy; ++cy)
                        for (int cz = 0; cz <= lCDz; ++cz)
                            _hrr_buf[ax][ay][az][cx][cy][cz] = _vrr_buf[ax][ay][az][cx][cy][cz][0];

    // A→B HRR: modifies _hrr_buf in-place
    _eri_hrr_ab(_hrr_buf, lAx, lAy, lAz, lBx, lBy, lBz, lCDx, lCDy, lCDz, ABx, ABy, ABz);

    // Extract C-side slice at (lAx, lAy, lAz) for C→D HRR
    double V0_CD[VRR_DIM][VRR_DIM][VRR_DIM] = {};
    for (int cx = 0; cx <= lCDx; ++cx)
        for (int cy = 0; cy <= lCDy; ++cy)
            for (int cz = 0; cz <= lCDz; ++cz)
                V0_CD[cx][cy][cz] = _hrr_buf[lAx][lAy][lAz][cx][cy][cz];

    // C→D HRR reusing the existing _nuclear_hrr (same 3-phase sweep)
    return _nuclear_hrr(V0_CD, lCx, lCy, lCz, lDx, lDy, lDz, CDx, CDy, CDz);
}

// ─── 4-center ERI: contracted shell quartet ──────────────────────────────────
static double _contracted_eri(
    const HartreeFock::ShellPair& spAB,
    const HartreeFock::ShellPair& spCD,
    const int lAx, const int lAy, const int lAz,
    const int lBx, const int lBy, const int lBz,
    const int lCx, const int lCy, const int lCz,
    const int lDx, const int lDy, const int lDz)
{
    const double ABx = spAB.R[0], ABy = spAB.R[1], ABz = spAB.R[2];
    const double CDx = spCD.R[0], CDy = spCD.R[1], CDz = spCD.R[2];

    double eri = 0.0;
    for (const auto& ppAB : spAB.primitive_pairs)
        for (const auto& ppCD : spCD.primitive_pairs)
            eri += ppAB.coeff_product * ppCD.coeff_product
                 * _os_eri_primitive(ppAB, ppCD,
                                     lAx, lAy, lAz, lBx, lBy, lBz,
                                     lCx, lCy, lCz, lDx, lDy, lDz,
                                     ABx, ABy, ABz, CDx, CDy, CDz);
    return eri;
}

// ─── Gradient: derivative integral helpers ────────────────────────────────────
//
// Thread-local scratch for nuclear dVRR (two arrays: V and dV per direction).
// MMAX_NUC_D = 2*MAX_L+4 ensures F_{m+1} is always available at the base case.
static constexpr int MMAX_NUC_D = 2 * MAX_L + 4;
static thread_local double _nuc_vrr_d [VRR_DIM][VRR_DIM][VRR_DIM][MMAX_NUC_D];
static thread_local double _nuc_dvrr_d[VRR_DIM][VRR_DIM][VRR_DIM][MMAX_NUC_D];

// Raw (S3d, T3d) primitive products at given AM without scale factor.
// Identical formula to _compute_3d_overlap_kinetic but accepts explicit AM.
static std::pair<double,double> _st_raw(
    double hiz, const Eigen::Vector3d& pA, const Eigen::Vector3d& pB, double beta,
    int lAx, int lAy, int lAz, int lBx, int lBy, int lBz)
{
    using HartreeFock::ObaraSaika::_os_1d;
    const double Sx = _os_1d(hiz, pA[0], pB[0], lAx, lBx);
    const double Sy = _os_1d(hiz, pA[1], pB[1], lAy, lBy);
    const double Sz = _os_1d(hiz, pA[2], pB[2], lAz, lBz);

    const double Sxp = _os_1d(hiz, pA[0], pB[0], lAx, lBx+2);
    const double Syp = _os_1d(hiz, pA[1], pB[1], lAy, lBy+2);
    const double Szp = _os_1d(hiz, pA[2], pB[2], lAz, lBz+2);
    const double Sxm = (lBx>=2) ? _os_1d(hiz, pA[0], pB[0], lAx, lBx-2) : 0.0;
    const double Sym = (lBy>=2) ? _os_1d(hiz, pA[1], pB[1], lAy, lBy-2) : 0.0;
    const double Szm = (lBz>=2) ? _os_1d(hiz, pA[2], pB[2], lAz, lBz-2) : 0.0;

    const double b2 = beta * beta;
    const double Tx = -0.5*lBx*(lBx-1)*Sxm + beta*(2*lBx+1)*Sx - 2.0*b2*Sxp;
    const double Ty = -0.5*lBy*(lBy-1)*Sym + beta*(2*lBy+1)*Sy - 2.0*b2*Syp;
    const double Tz = -0.5*lBz*(lBz-1)*Szm + beta*(2*lBz+1)*Sz - 2.0*b2*Szp;

    return {Sx*Sy*Sz, Tx*Sy*Sz + Sx*Ty*Sz + Sx*Sy*Tz};
}

// dV_μν^{C}/dC_{direction} for one primitive pair.
// Runs VRR+dVRR in parallel, then applies HRR to the dV m=0 slice.
// Returns value without -Z or coeff_product (caller applies those).
static double _os_nuclear_primitive_dC(
    const HartreeFock::PrimitivePair& pp,
    const int lAx, const int lAy, const int lAz,
    const int lBx, const int lBy, const int lBz,
    const double ABx, const double ABy, const double ABz,
    const Eigen::Vector3d& nuc_pos,
    const int direction)
{
    const int L      = lAx + lAy + lAz + lBx + lBy + lBz;
    const int lx_max = lAx + lBx, ly_max = lAy + lBy, lz_max = lAz + lBz;

    const double pCx = pp.center[0] - nuc_pos[0];
    const double pCy = pp.center[1] - nuc_pos[1];
    const double pCz = pp.center[2] - nuc_pos[2];
    const double T   = pp.zeta * (pCx*pCx + pCy*pCy + pCz*pCz);

    const double nuc_pref = pp.prefactor * 2.0 * std::sqrt(pp.zeta / M_PI);
    const double pAx = pp.pA[0], pAy = pp.pA[1], pAz = pp.pA[2];
    const double hiz = pp.inv_zeta * 0.5;
    const double PC_dir = (direction == 0) ? pCx : (direction == 1) ? pCy : pCz;

    auto& V  = _nuc_vrr_d;
    auto& dV = _nuc_dvrr_d;

    // Zero needed region (+2 in m for m+1 accesses)
    const int mclr = L + 3;
    for (int ix = 0; ix <= lx_max; ix++)
        for (int iy = 0; iy <= ly_max; iy++)
            for (int iz = 0; iz <= lz_max; iz++)
                for (int m = 0; m < mclr; m++)
                    V[ix][iy][iz][m] = dV[ix][iy][iz][m] = 0.0;

    // Seed Boys values
    for (int m = 0; m <= L + 2; m++)
        V[0][0][0][m] = nuc_pref * HartreeFock::Lookup::boys(m, T);

    // dVRR base case: 2*zeta*PC_dir * F_{m+1}
    for (int m = 0; m <= L + 1; m++)
        dV[0][0][0][m] = 2.0 * pp.zeta * PC_dir * nuc_pref * HartreeFock::Lookup::boys(m + 1, T);

    // x-VRR + x-dVRR
    for (int ix = 1; ix <= lx_max; ix++) {
        for (int m = 0; m <= L - ix; m++) {
            V[ix][0][0][m] = pAx*V[ix-1][0][0][m] - pCx*V[ix-1][0][0][m+1];
            dV[ix][0][0][m] = pAx*dV[ix-1][0][0][m] - pCx*dV[ix-1][0][0][m+1];
            if (direction == 0) dV[ix][0][0][m] += V[ix-1][0][0][m+1];
            if (ix > 1) {
                V[ix][0][0][m]  += (ix-1)*hiz*(V[ix-2][0][0][m]  - V[ix-2][0][0][m+1]);
                dV[ix][0][0][m] += (ix-1)*hiz*(dV[ix-2][0][0][m] - dV[ix-2][0][0][m+1]);
            }
        }
    }

    // y-VRR + y-dVRR
    for (int ix = 0; ix <= lx_max; ix++) {
        for (int iy = 1; iy <= ly_max; iy++) {
            const int mmax = L - ix - iy;
            if (mmax < 0) continue;
            for (int m = 0; m <= mmax; m++) {
                V[ix][iy][0][m] = pAy*V[ix][iy-1][0][m] - pCy*V[ix][iy-1][0][m+1];
                dV[ix][iy][0][m] = pAy*dV[ix][iy-1][0][m] - pCy*dV[ix][iy-1][0][m+1];
                if (direction == 1) dV[ix][iy][0][m] += V[ix][iy-1][0][m+1];
                if (iy > 1) {
                    V[ix][iy][0][m]  += (iy-1)*hiz*(V[ix][iy-2][0][m]  - V[ix][iy-2][0][m+1]);
                    dV[ix][iy][0][m] += (iy-1)*hiz*(dV[ix][iy-2][0][m] - dV[ix][iy-2][0][m+1]);
                }
            }
        }
    }

    // z-VRR + z-dVRR
    for (int ix = 0; ix <= lx_max; ix++) {
        for (int iy = 0; iy <= ly_max; iy++) {
            for (int iz = 1; iz <= lz_max; iz++) {
                const int mmax = L - ix - iy - iz;
                if (mmax < 0) continue;
                for (int m = 0; m <= mmax; m++) {
                    V[ix][iy][iz][m] = pAz*V[ix][iy][iz-1][m] - pCz*V[ix][iy][iz-1][m+1];
                    dV[ix][iy][iz][m] = pAz*dV[ix][iy][iz-1][m] - pCz*dV[ix][iy][iz-1][m+1];
                    if (direction == 2) dV[ix][iy][iz][m] += V[ix][iy][iz-1][m+1];
                    if (iz > 1) {
                        V[ix][iy][iz][m]  += (iz-1)*hiz*(V[ix][iy][iz-2][m]  - V[ix][iy][iz-2][m+1]);
                        dV[ix][iy][iz][m] += (iz-1)*hiz*(dV[ix][iy][iz-2][m] - dV[ix][iy][iz-2][m+1]);
                    }
                }
            }
        }
    }

    // Extract m=0 slice of dV for HRR
    double dV0[VRR_DIM][VRR_DIM][VRR_DIM] = {};
    for (int ix = 0; ix <= lx_max; ix++)
        for (int iy = 0; iy <= ly_max; iy++)
            for (int iz = 0; iz <= lz_max; iz++)
                dV0[ix][iy][iz] = dV[ix][iy][iz][0];

    return _nuclear_hrr(dV0, lAx, lAy, lAz, lBx, lBy, lBz, ABx, ABy, ABz);
}

// ─── Public: 1e GTO-centre derivatives ───────────────────────────────────────
//
// AM shift rule: ∂φ(α,lA,A)/∂A_q = +2α φ(lA+ê_q) − lA_q φ(lA−ê_q)
// Returns {dS/dAx, dS/dAy, dS/dAz, dT/dAx, dT/dAy, dT/dAz}
std::array<double,6> HartreeFock::ObaraSaika::_compute_1e_deriv_A(
    const HartreeFock::ShellPair& sp)
{
    const int lAx = sp.A._cartesian[0], lAy = sp.A._cartesian[1], lAz = sp.A._cartesian[2];
    const int lBx = sp.B._cartesian[0], lBy = sp.B._cartesian[1], lBz = sp.B._cartesian[2];

    std::array<double,6> result{};

    for (int q = 0; q < 3; ++q) {
        const int lAq = sp.A._cartesian[q];
        double dS = 0.0, dT = 0.0;

        for (const auto& pp : sp.primitive_pairs) {
            const double hiz  = pp.inv_zeta * 0.5;
            const double w    = pp.prefactor * pp.coeff_product;
            const double w2al = 2.0 * pp.alpha * w;

            // +1 shift: 2α * raw_st(lA+ê_q, lB)
            const int axp = lAx + (q == 0), ayp = lAy + (q == 1), azp = lAz + (q == 2);
            auto [Sp, Tp] = _st_raw(hiz, pp.pA, pp.pB, pp.beta, axp, ayp, azp, lBx, lBy, lBz);
            dS += w2al * Sp;
            dT += w2al * Tp;

            // -1 shift: lAq * raw_st(lA-ê_q, lB)
            if (lAq > 0) {
                const int axm = lAx - (q == 0), aym = lAy - (q == 1), azm = lAz - (q == 2);
                auto [Sm, Tm] = _st_raw(hiz, pp.pA, pp.pB, pp.beta, axm, aym, azm, lBx, lBy, lBz);
                dS -= static_cast<double>(lAq) * w * Sm;
                dT -= static_cast<double>(lAq) * w * Tm;
            }
        }
        result[q]     = dS;
        result[q + 3] = dT;
    }
    return result;
}

// ─── Public: nuclear-attraction GTO-centre derivative ────────────────────────
//
// Returns {dV/dAx, dV/dAy, dV/dAz} summed over all nuclei (AM shift rule).
std::array<double,3> HartreeFock::ObaraSaika::_compute_nuclear_deriv_A_elem(
    const HartreeFock::ShellPair& sp,
    const HartreeFock::Molecule& mol)
{
    const int lAx = sp.A._cartesian[0], lAy = sp.A._cartesian[1], lAz = sp.A._cartesian[2];
    const int lBx = sp.B._cartesian[0], lBy = sp.B._cartesian[1], lBz = sp.B._cartesian[2];
    const double ABx = sp.R[0], ABy = sp.R[1], ABz = sp.R[2];

    std::array<double,3> result{};

    for (int q = 0; q < 3; ++q) {
        const int lAq = sp.A._cartesian[q];
        double dV = 0.0;

        for (std::size_t a = 0; a < mol.natoms; ++a) {
            const double Z = static_cast<double>(mol.atomic_numbers[a]);
            const Eigen::Vector3d C(mol._standard(a, 0),
                                    mol._standard(a, 1),
                                    mol._standard(a, 2));

            for (const auto& pp : sp.primitive_pairs) {
                const double w    = pp.coeff_product;
                const double w2al = 2.0 * pp.alpha * w;

                // +1 shift: -Z * 2α * V_prim(lA+ê_q, lB)
                {
                    const int axp = lAx+(q==0), ayp = lAy+(q==1), azp = lAz+(q==2);
                    double Vp = _os_nuclear_primitive(pp, axp, ayp, azp,
                                                     lBx, lBy, lBz, ABx, ABy, ABz, C);
                    dV -= Z * w2al * Vp;
                }
                // -1 shift: +Z * lAq * V_prim(lA-ê_q, lB)
                if (lAq > 0) {
                    const int axm = lAx-(q==0), aym = lAy-(q==1), azm = lAz-(q==2);
                    double Vm = _os_nuclear_primitive(pp, axm, aym, azm,
                                                     lBx, lBy, lBz, ABx, ABy, ABz, C);
                    dV += Z * static_cast<double>(lAq) * w * Vm;
                }
            }
        }
        result[q] = dV;
    }
    return result;
}

// ─── Public: nuclear-position derivative dV/dC ───────────────────────────────
//
// Returns contracted dV_μν/dC_{direction} for one nucleus at C with charge Z.
double HartreeFock::ObaraSaika::_compute_nuclear_deriv_C_elem(
    const HartreeFock::ShellPair& sp,
    const Eigen::Vector3d& C, const double Z, const int direction)
{
    const int lAx = sp.A._cartesian[0], lAy = sp.A._cartesian[1], lAz = sp.A._cartesian[2];
    const int lBx = sp.B._cartesian[0], lBy = sp.B._cartesian[1], lBz = sp.B._cartesian[2];
    const double ABx = sp.R[0], ABy = sp.R[1], ABz = sp.R[2];

    double dV = 0.0;
    for (const auto& pp : sp.primitive_pairs) {
        double dv = _os_nuclear_primitive_dC(pp, lAx, lAy, lAz, lBx, lBy, lBz,
                                              ABx, ABy, ABz, C, direction);
        dV += pp.coeff_product * dv;
    }
    return -Z * dV;   // -Z factor (V includes nuclear charge sign)
}

// ─── Public: ERI derivatives for one contracted (μν|λσ) quartet ──────────────
//
// AM shift rule applied to each of the four centres.
// result[cen*3 + dir], cen∈{0=A,1=B,2=C,3=D}, dir∈{0,1,2}
std::array<double,12> HartreeFock::ObaraSaika::_compute_eri_deriv_elem(
    const HartreeFock::ShellPair& spAB,
    const HartreeFock::ShellPair& spCD)
{
    const int lAx = spAB.A._cartesian[0], lAy = spAB.A._cartesian[1], lAz = spAB.A._cartesian[2];
    const int lBx = spAB.B._cartesian[0], lBy = spAB.B._cartesian[1], lBz = spAB.B._cartesian[2];
    const int lCx = spCD.A._cartesian[0], lCy = spCD.A._cartesian[1], lCz = spCD.A._cartesian[2];
    const int lDx = spCD.B._cartesian[0], lDy = spCD.B._cartesian[1], lDz = spCD.B._cartesian[2];

    const double ABx = spAB.R[0], ABy = spAB.R[1], ABz = spAB.R[2];
    const double CDx = spCD.R[0], CDy = spCD.R[1], CDz = spCD.R[2];

    std::array<double,12> result{};

    // We need per-primitive 2α weighting. Compute a weighted contracted ERI.
    // weighted_ceri_A(ax,ay,az,...) = Σ_{k,l,m,n} 2α_k * cAB_{kl} * cCD_{mn} * ERI_prim
    auto wceri_A = [&](int ax, int ay, int az, int bx, int by, int bz,
                       int cx, int cy, int cz, int dx, int dy, int dz) -> double {
        double eri = 0.0;
        for (const auto& ppAB : spAB.primitive_pairs)
            for (const auto& ppCD : spCD.primitive_pairs)
                eri += (2.0 * ppAB.alpha) * ppAB.coeff_product * ppCD.coeff_product
                     * _os_eri_primitive(ppAB, ppCD,
                                        ax, ay, az, bx, by, bz,
                                        cx, cy, cz, dx, dy, dz,
                                        ABx, ABy, ABz, CDx, CDy, CDz);
        return eri;
    };

    auto wceri_B = [&](int ax, int ay, int az, int bx, int by, int bz,
                       int cx, int cy, int cz, int dx, int dy, int dz) -> double {
        double eri = 0.0;
        for (const auto& ppAB : spAB.primitive_pairs)
            for (const auto& ppCD : spCD.primitive_pairs)
                eri += ppAB.coeff_product * (2.0 * ppAB.beta) * ppCD.coeff_product
                     * _os_eri_primitive(ppAB, ppCD,
                                        ax, ay, az, bx, by, bz,
                                        cx, cy, cz, dx, dy, dz,
                                        ABx, ABy, ABz, CDx, CDy, CDz);
        return eri;
    };

    auto wceri_C = [&](int ax, int ay, int az, int bx, int by, int bz,
                       int cx, int cy, int cz, int dx, int dy, int dz) -> double {
        double eri = 0.0;
        for (const auto& ppAB : spAB.primitive_pairs)
            for (const auto& ppCD : spCD.primitive_pairs)
                eri += ppAB.coeff_product * (2.0 * ppCD.alpha) * ppCD.coeff_product
                     * _os_eri_primitive(ppAB, ppCD,
                                        ax, ay, az, bx, by, bz,
                                        cx, cy, cz, dx, dy, dz,
                                        ABx, ABy, ABz, CDx, CDy, CDz);
        return eri;
    };

    auto wceri_D = [&](int ax, int ay, int az, int bx, int by, int bz,
                       int cx, int cy, int cz, int dx, int dy, int dz) -> double {
        double eri = 0.0;
        for (const auto& ppAB : spAB.primitive_pairs)
            for (const auto& ppCD : spCD.primitive_pairs)
                eri += ppAB.coeff_product * ppCD.coeff_product * (2.0 * ppCD.beta)
                     * _os_eri_primitive(ppAB, ppCD,
                                        ax, ay, az, bx, by, bz,
                                        cx, cy, cz, dx, dy, dz,
                                        ABx, ABy, ABz, CDx, CDy, CDz);
        return eri;
    };

    auto nceri = [&](int ax, int ay, int az, int bx, int by, int bz,
                     int cx, int cy, int cz, int dx, int dy, int dz) -> double {
        return _contracted_eri(spAB, spCD,
                               ax, ay, az, bx, by, bz,
                               cx, cy, cz, dx, dy, dz);
    };

    for (int q = 0; q < 3; ++q) {
        // Centre A: +2α·ERI(lA+ê_q) − lAq·ERI(lA−ê_q)
        {
            const int lAq = spAB.A._cartesian[q];
            const int axp = lAx+(q==0), ayp = lAy+(q==1), azp = lAz+(q==2);
            result[0*3 + q] += wceri_A(axp, ayp, azp, lBx, lBy, lBz,
                                       lCx, lCy, lCz, lDx, lDy, lDz);
            if (lAq > 0) {
                const int axm = lAx-(q==0), aym = lAy-(q==1), azm = lAz-(q==2);
                result[0*3 + q] -= static_cast<double>(lAq) *
                    nceri(axm, aym, azm, lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz);
            }
        }
        // Centre B: +2β·ERI(lB+ê_q) − lBq·ERI(lB−ê_q)
        {
            const int lBq = spAB.B._cartesian[q];
            const int bxp = lBx+(q==0), byp = lBy+(q==1), bzp = lBz+(q==2);
            result[1*3 + q] += wceri_B(lAx, lAy, lAz, bxp, byp, bzp,
                                       lCx, lCy, lCz, lDx, lDy, lDz);
            if (lBq > 0) {
                const int bxm = lBx-(q==0), bym = lBy-(q==1), bzm = lBz-(q==2);
                result[1*3 + q] -= static_cast<double>(lBq) *
                    nceri(lAx, lAy, lAz, bxm, bym, bzm, lCx, lCy, lCz, lDx, lDy, lDz);
            }
        }
        // Centre C: +2γ·ERI(lC+ê_q) − lCq·ERI(lC−ê_q)
        {
            const int lCq = spCD.A._cartesian[q];
            const int cxp = lCx+(q==0), cyp = lCy+(q==1), czp = lCz+(q==2);
            result[2*3 + q] += wceri_C(lAx, lAy, lAz, lBx, lBy, lBz,
                                       cxp, cyp, czp, lDx, lDy, lDz);
            if (lCq > 0) {
                const int cxm = lCx-(q==0), cym = lCy-(q==1), czm = lCz-(q==2);
                result[2*3 + q] -= static_cast<double>(lCq) *
                    nceri(lAx, lAy, lAz, lBx, lBy, lBz, cxm, cym, czm, lDx, lDy, lDz);
            }
        }
        // Centre D: +2δ·ERI(lD+ê_q) − lDq·ERI(lD−ê_q)
        {
            const int lDq = spCD.B._cartesian[q];
            const int dxp = lDx+(q==0), dyp = lDy+(q==1), dzp = lDz+(q==2);
            result[3*3 + q] += wceri_D(lAx, lAy, lAz, lBx, lBy, lBz,
                                       lCx, lCy, lCz, dxp, dyp, dzp);
            if (lDq > 0) {
                const int dxm = lDx-(q==0), dym = lDy-(q==1), dzm = lDz-(q==2);
                result[3*3 + q] -= static_cast<double>(lDq) *
                    nceri(lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, dxm, dym, dzm);
            }
        }
    }
    return result;
}

// Forward declaration — defined later in this file before _compute_2e.
static Eigen::MatrixXd _compute_schwarz_table(
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    std::size_t nbasis);

// ─── Public: build 2e Fock (G = J - 0.5*K) ──────────────────────────────────
//
// Phase 1: build the full (μν|λσ) ERI tensor by iterating over unique shell-pair
//          quartets (p ≤ q) and filling all 8 permutation-symmetry slots.
// Phase 2: contract with density to form G[μν] = Σ_{λσ} P[λσ]·[(μν|λσ) − ½(μλ|νσ)].
//
// This avoids all symmetry-factor edge-cases in the scatter approach.
Eigen::MatrixXd HartreeFock::ObaraSaika::_compute_2e_fock(
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    const Eigen::MatrixXd& density,
    const std::size_t nbasis,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp>*)
{
    const std::size_t nb  = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;

    const Eigen::MatrixXd Q = _compute_schwarz_table(shell_pairs, nb);

    // ── Phase 1: build ERI tensor ─────────────────────────────────────────────
    std::vector<double> eri(nb * nb * nb * nb, 0.0);

    const std::size_t npairs = shell_pairs.size();

    // Disjoint writes per (p,q) pair → lock-free parallelism.
    // thread_local _vrr_buf / _hrr_buf give each thread private scratch space.
#pragma omp parallel for schedule(dynamic)
    for (std::size_t p = 0; p < npairs; ++p)
    {
        const auto& spAB = shell_pairs[p];
        const std::size_t i = spAB.A._index;
        const std::size_t j = spAB.B._index;
        const int lAx = spAB.A._cartesian[0], lAy = spAB.A._cartesian[1], lAz = spAB.A._cartesian[2];
        const int lBx = spAB.B._cartesian[0], lBy = spAB.B._cartesian[1], lBz = spAB.B._cartesian[2];

        for (std::size_t q = p; q < npairs; ++q)
        {
            const auto& spCD = shell_pairs[q];
            const std::size_t k = spCD.A._index;
            const std::size_t l = spCD.B._index;

            // Schwarz screening
            if (Q(i, j) * Q(k, l) < tol_eri) continue;

            const int lCx = spCD.A._cartesian[0], lCy = spCD.A._cartesian[1], lCz = spCD.A._cartesian[2];
            const int lDx = spCD.B._cartesian[0], lDy = spCD.B._cartesian[1], lDz = spCD.B._cartesian[2];

            const double val = _contracted_eri(spAB, spCD,
                                               lAx, lAy, lAz, lBx, lBy, lBz,
                                               lCx, lCy, lCz, lDx, lDy, lDz);

            // Fill all 8 permutation-symmetry equivalent slots:
            //   (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
            //           = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)
            eri[i*nb3 + j*nb2 + k*nb + l] = val;
            eri[j*nb3 + i*nb2 + k*nb + l] = val;
            eri[i*nb3 + j*nb2 + l*nb + k] = val;
            eri[j*nb3 + i*nb2 + l*nb + k] = val;
            eri[k*nb3 + l*nb2 + i*nb + j] = val;
            eri[l*nb3 + k*nb2 + i*nb + j] = val;
            eri[k*nb3 + l*nb2 + j*nb + i] = val;
            eri[l*nb3 + k*nb2 + j*nb + i] = val;
        }
    }

    // ── Phase 2: contract with density ────────────────────────────────────────
    // G[μ][ν] = Σ_{λσ} P[λσ] · ( ERI[μ][ν][λ][σ]  −  0.5 · ERI[μ][λ][ν][σ] )
    // Parallel over μ: thread i owns row i of G → no shared writes.
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nb, nb);

#pragma omp parallel for schedule(static)
    for (std::size_t mu = 0; mu < nb; ++mu)
        for (std::size_t nu = 0; nu < nb; ++nu)
            for (std::size_t lam = 0; lam < nb; ++lam)
                for (std::size_t sig = 0; sig < nb; ++sig)
                    G(mu, nu) += density(lam, sig) *
                                 (eri[mu*nb3 + nu*nb2 + lam*nb + sig]
                                  - 0.5 * eri[mu*nb3 + lam*nb2 + nu*nb + sig]);

    return G;
}

// ─── Public: UHF 2e Fock (G_alpha and G_beta) ────────────────────────────────
//
// UHF Fock formula (per spin σ ∈ {α, β}):
//   G_σ(μν) = Σ_{λσ} P_total(λσ)·(μν|λσ) − P_σ(λσ)·(μλ|νσ)
//
// Phase 1 is identical to _compute_2e_fock — the ERI tensor is spin-independent.
// Phase 2 contracts once for each spin simultaneously, avoiding a second O(N⁴) build.
std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
HartreeFock::ObaraSaika::_compute_2e_fock_uhf(
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    const Eigen::MatrixXd& Pa,
    const Eigen::MatrixXd& Pb,
    const std::size_t nbasis,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp>*)
{
    const std::size_t nb  = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;

    const Eigen::MatrixXd Q = _compute_schwarz_table(shell_pairs, nb);

    // ── Phase 1: build ERI tensor (identical to _compute_2e_fock) ────────────
    std::vector<double> eri(nb * nb * nb * nb, 0.0);

    const std::size_t npairs = shell_pairs.size();

#pragma omp parallel for schedule(dynamic)
    for (std::size_t p = 0; p < npairs; ++p)
    {
        const auto& spAB = shell_pairs[p];
        const std::size_t i = spAB.A._index;
        const std::size_t j = spAB.B._index;
        const int lAx = spAB.A._cartesian[0], lAy = spAB.A._cartesian[1], lAz = spAB.A._cartesian[2];
        const int lBx = spAB.B._cartesian[0], lBy = spAB.B._cartesian[1], lBz = spAB.B._cartesian[2];

        for (std::size_t q = p; q < npairs; ++q)
        {
            const auto& spCD = shell_pairs[q];
            const std::size_t k = spCD.A._index;
            const std::size_t l = spCD.B._index;

            // Schwarz screening
            if (Q(i, j) * Q(k, l) < tol_eri) continue;

            const int lCx = spCD.A._cartesian[0], lCy = spCD.A._cartesian[1], lCz = spCD.A._cartesian[2];
            const int lDx = spCD.B._cartesian[0], lDy = spCD.B._cartesian[1], lDz = spCD.B._cartesian[2];

            const double val = _contracted_eri(spAB, spCD,
                                               lAx, lAy, lAz, lBx, lBy, lBz,
                                               lCx, lCy, lCz, lDx, lDy, lDz);

            eri[i*nb3 + j*nb2 + k*nb + l] = val;
            eri[j*nb3 + i*nb2 + k*nb + l] = val;
            eri[i*nb3 + j*nb2 + l*nb + k] = val;
            eri[j*nb3 + i*nb2 + l*nb + k] = val;
            eri[k*nb3 + l*nb2 + i*nb + j] = val;
            eri[l*nb3 + k*nb2 + i*nb + j] = val;
            eri[k*nb3 + l*nb2 + j*nb + i] = val;
            eri[l*nb3 + k*nb2 + j*nb + i] = val;
        }
    }

    // ── Phase 2: spin-resolved contraction ───────────────────────────────────
    // Ga(μ,ν) += Pt(λ,σ)·(μν|λσ) − Pa(λ,σ)·(μλ|νσ)
    // Gb(μ,ν) += Pt(λ,σ)·(μν|λσ) − Pb(λ,σ)·(μλ|νσ)
    // Parallel over μ: each thread owns a full row of Ga and Gb → no races.
    const Eigen::MatrixXd Pt = Pa + Pb;
    Eigen::MatrixXd Ga = Eigen::MatrixXd::Zero(nb, nb);
    Eigen::MatrixXd Gb = Eigen::MatrixXd::Zero(nb, nb);

#pragma omp parallel for schedule(static)
    for (std::size_t mu = 0; mu < nb; ++mu)
        for (std::size_t nu = 0; nu < nb; ++nu)
            for (std::size_t lam = 0; lam < nb; ++lam)
                for (std::size_t sig = 0; sig < nb; ++sig)
                {
                    const double coulomb = eri[mu*nb3 + nu*nb2 + lam*nb + sig];
                    const double exch    = eri[mu*nb3 + lam*nb2 + nu*nb + sig];
                    Ga(mu, nu) += Pt(lam, sig) * coulomb - Pa(lam, sig) * exch;
                    Gb(mu, nu) += Pt(lam, sig) * coulomb - Pb(lam, sig) * exch;
                }

    return {Ga, Gb};
}

// ─── Cross-overlap between two basis sets ────────────────────────────────────
//
// Computes S_cross(μ, ν) = <χ_μ^large | χ_ν^small> using the same Obara-Saika
// 1D recursion as the standard overlap.  The two bases are defined on the same
// molecule but may differ in size (e.g. large = 6-31G, small = STO-3G).
//
// Used for basis-set projection: projecting small-basis MOs onto a larger basis.
Eigen::MatrixXd HartreeFock::ObaraSaika::_compute_cross_overlap(
    const HartreeFock::Basis& large_basis,
    const HartreeFock::Basis& small_basis)
{
    const std::size_t nb_large = large_basis.nbasis();
    const std::size_t nb_small = small_basis.nbasis();
    Eigen::MatrixXd S_cross = Eigen::MatrixXd::Zero(nb_large, nb_small);

    const auto& large_bfs = large_basis._basis_functions;
    const auto& small_bfs = small_basis._basis_functions;

    for (std::size_t mu = 0; mu < nb_large; ++mu)
    {
        const HartreeFock::ContractedView& cvA = large_bfs[mu];
        const HartreeFock::Shell& shellA = *cvA._shell;
        const int lAx = cvA._cartesian[0];
        const int lAy = cvA._cartesian[1];
        const int lAz = cvA._cartesian[2];
        const Eigen::Vector3d& A = shellA._center;

        for (std::size_t nu = 0; nu < nb_small; ++nu)
        {
            const HartreeFock::ContractedView& cvB = small_bfs[nu];
            const HartreeFock::Shell& shellB = *cvB._shell;
            const int lBx = cvB._cartesian[0];
            const int lBy = cvB._cartesian[1];
            const int lBz = cvB._cartesian[2];
            const Eigen::Vector3d& B = shellB._center;

            const double R2 = (A - B).squaredNorm();

            double s = 0.0;
            for (int i = 0; i < static_cast<int>(shellA.nprimitives()); ++i)
            {
                const double alpha  = shellA._primitives[i];
                const double cA     = shellA._coefficients[i] * shellA._normalizations[i];

                for (int j = 0; j < static_cast<int>(shellB.nprimitives()); ++j)
                {
                    const double beta      = shellB._primitives[j];
                    const double cB        = shellB._coefficients[j] * shellB._normalizations[j];
                    const double zeta      = alpha + beta;
                    const double inv_zeta  = 1.0 / zeta;
                    const double half_inv_zeta = inv_zeta * 0.5;

                    const Eigen::Vector3d P  = (alpha * A + beta * B) * inv_zeta;
                    const Eigen::Vector3d pA = P - A;
                    const Eigen::Vector3d pB = P - B;

                    const double prefactor = std::pow(M_PI * inv_zeta, 1.5)
                                           * std::exp(-alpha * beta * inv_zeta * R2);

                    const double Sx = _os_1d(half_inv_zeta, pA[0], pB[0], lAx, lBx);
                    const double Sy = _os_1d(half_inv_zeta, pA[1], pB[1], lAy, lBy);
                    const double Sz = _os_1d(half_inv_zeta, pA[2], pB[2], lAz, lBz);

                    s += cA * cB * prefactor * Sx * Sy * Sz;
                }
            }
            S_cross(mu, nu) = s;
        }
    }

    return S_cross;
}

// ─── Schwarz screening table ─────────────────────────────────────────────────
//
// Q(i,j) = sqrt(|(ij|ij)|) for each basis-function pair covered by a shell pair.
// Bounds any quartet: |(ij|kl)| ≤ Q(i,j)·Q(k,l)  (Cauchy-Schwarz inequality).
static Eigen::MatrixXd _compute_schwarz_table(
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    std::size_t nbasis)
{
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nbasis, nbasis);

    for (const auto& sp : shell_pairs)
    {
        const std::size_t i = sp.A._index;
        const std::size_t j = sp.B._index;
        const int lAx = sp.A._cartesian[0], lAy = sp.A._cartesian[1], lAz = sp.A._cartesian[2];
        const int lBx = sp.B._cartesian[0], lBy = sp.B._cartesian[1], lBz = sp.B._cartesian[2];

        const double diag = _contracted_eri(sp, sp,
                                            lAx, lAy, lAz, lBx, lBy, lBz,
                                            lAx, lAy, lAz, lBx, lBy, lBz);
        const double q = std::sqrt(std::abs(diag));
        Q(i, j) = q;
        Q(j, i) = q;
    }

    return Q;
}

// Compute ERI and store it for conventional SCF
std::vector <double> HartreeFock::ObaraSaika::_compute_2e(
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    const std::size_t nbasis,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp>*)
{
    const std::size_t nb  = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;

    const Eigen::MatrixXd Q = _compute_schwarz_table(shell_pairs, nb);

    // ── Phase 1: build ERI tensor ─────────────────────────────────────────────
    std::vector<double> eri(nb * nb * nb * nb, 0.0);

    const std::size_t npairs = shell_pairs.size();

    // Disjoint writes per (p,q) pair → lock-free parallelism.
    // thread_local _vrr_buf / _hrr_buf give each thread private scratch space.
#pragma omp parallel for schedule(dynamic)
    for (std::size_t p = 0; p < npairs; ++p)
    {
        const auto& spAB = shell_pairs[p];
        const std::size_t i = spAB.A._index;
        const std::size_t j = spAB.B._index;
        const int lAx = spAB.A._cartesian[0], lAy = spAB.A._cartesian[1], lAz = spAB.A._cartesian[2];
        const int lBx = spAB.B._cartesian[0], lBy = spAB.B._cartesian[1], lBz = spAB.B._cartesian[2];

        for (std::size_t q = p; q < npairs; ++q)
        {
            const auto& spCD = shell_pairs[q];
            const std::size_t k = spCD.A._index;
            const std::size_t l = spCD.B._index;

            // Schwarz screening: |(ij|kl)| ≤ Q(i,j)·Q(k,l)
            if (Q(i, j) * Q(k, l) < tol_eri) continue;

            const int lCx = spCD.A._cartesian[0], lCy = spCD.A._cartesian[1], lCz = spCD.A._cartesian[2];
            const int lDx = spCD.B._cartesian[0], lDy = spCD.B._cartesian[1], lDz = spCD.B._cartesian[2];

            const double val = _contracted_eri(spAB, spCD,
                                               lAx, lAy, lAz, lBx, lBy, lBz,
                                               lCx, lCy, lCz, lDx, lDy, lDz);

            // Fill all 8 permutation-symmetry equivalent slots:
            //   (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
            //           = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)
            eri[i*nb3 + j*nb2 + k*nb + l] = val;
            eri[j*nb3 + i*nb2 + k*nb + l] = val;
            eri[i*nb3 + j*nb2 + l*nb + k] = val;
            eri[j*nb3 + i*nb2 + l*nb + k] = val;
            eri[k*nb3 + l*nb2 + i*nb + j] = val;
            eri[l*nb3 + k*nb2 + i*nb + j] = val;
            eri[k*nb3 + l*nb2 + j*nb + i] = val;
            eri[l*nb3 + k*nb2 + j*nb + i] = val;
        }
    }

    return eri;
}

Eigen::MatrixXd HartreeFock::ObaraSaika::_compute_fock_rhf(const std::vector <double>& _eri, const Eigen::MatrixXd& density, const std::size_t nbasis)
{
    const std::size_t nb  = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nb, nb);
    
#pragma omp parallel for schedule(static)
    for (std::size_t mu = 0; mu < nb; ++mu)
        for (std::size_t nu = 0; nu < nb; ++nu)
            for (std::size_t lam = 0; lam < nb; ++lam)
                for (std::size_t sig = 0; sig < nb; ++sig)
                    G(mu, nu) += density(lam, sig) *
                                 (_eri[mu*nb3 + nu*nb2 + lam*nb + sig]
                                  - 0.5 * _eri[mu*nb3 + lam*nb2 + nu*nb + sig]);
    
    return G;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> HartreeFock::ObaraSaika::_compute_fock_uhf(
    const std::vector<double>& _eri,
    const Eigen::MatrixXd& Pa,
    const Eigen::MatrixXd& Pb,
    std::size_t nbasis)
{
    const std::size_t nb  = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;
    
    const Eigen::MatrixXd Pt = Pa + Pb;
    Eigen::MatrixXd Ga = Eigen::MatrixXd::Zero(nb, nb);
    Eigen::MatrixXd Gb = Eigen::MatrixXd::Zero(nb, nb);

#pragma omp parallel for schedule(static)
    for (std::size_t mu = 0; mu < nb; ++mu)
        for (std::size_t nu = 0; nu < nb; ++nu)
            for (std::size_t lam = 0; lam < nb; ++lam)
                for (std::size_t sig = 0; sig < nb; ++sig)
                {
                    const double coulomb = _eri[mu*nb3 + nu*nb2 + lam*nb + sig];
                    const double exch    = _eri[mu*nb3 + lam*nb2 + nu*nb + sig];
                    Ga(mu, nu) += Pt(lam, sig) * coulomb - Pa(lam, sig) * exch;
                    Gb(mu, nu) += Pt(lam, sig) * coulomb - Pb(lam, sig) * exch;
                }

    return {Ga, Gb};
}
