#include "rys.h"
#include "rys_roots.h"
#include "os.h"      // for _compute_2e_fock (Auto path falls back to OS)

#include <cstring>
#include <cmath>

// ─── Scratch buffer dimensions ────────────────────────────────────────────────
//
// VRR_DIM = 2*MAX_L+1 = 13  (per-axis, matches os.cpp convention)
// Per-root 1D VRR table: _rys_1d[3][VRR_DIM][VRR_DIM] — stack allocated per call.
// Accumulated 6D sum: _rys_sum_buf[VRR_DIM]^6 — thread-local (37 MB, same as _hrr_buf).

static constexpr int VRR_DIM = 2 * MAX_L + 1;  // 13

// Accumulated Rys 6D intermediate: sum_{roots} w_r * Ix[ax][cx] * Iy[ay][cy] * Iz[az][cz]
// Same footprint as the _hrr_buf in os.cpp.
static thread_local double _rys_sum_buf[VRR_DIM][VRR_DIM][VRR_DIM]
                                        [VRR_DIM][VRR_DIM][VRR_DIM];

// ─── Schwarz screening table ──────────────────────────────────────────────────
//
// Q(i,j) = sqrt((ij|ij)) — identical to os.cpp's _compute_schwarz_table.
// Declared here; uses _rys_contracted_eri internally once rys.cpp is complete.
// For now, forward-declare and implement after the ERI functions.

static Eigen::MatrixXd _rys_schwarz_table(
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    std::size_t nbasis);

static bool _auto_prefers_rys(const HartreeFock::ShellPair& spAB,
                              const HartreeFock::ShellPair& spCD) noexcept
{
    const int lAx = spAB.A._cartesian[0], lAy = spAB.A._cartesian[1], lAz = spAB.A._cartesian[2];
    const int lBx = spAB.B._cartesian[0], lBy = spAB.B._cartesian[1], lBz = spAB.B._cartesian[2];
    const int lCx = spCD.A._cartesian[0], lCy = spCD.A._cartesian[1], lCz = spCD.A._cartesian[2];
    const int lDx = spCD.B._cartesian[0], lDy = spCD.B._cartesian[1], lDz = spCD.B._cartesian[2];

    const int lABx = lAx + lBx, lABy = lAy + lBy, lABz = lAz + lBz;
    const int lCDx = lCx + lDx, lCDy = lCy + lDy, lCDz = lCz + lDz;
    const int L = lABx + lABy + lABz + lCDx + lCDy + lCDz;
    const int nroots = L / 2 + 1;

    const double six_d = static_cast<double>(lABx + 1) * static_cast<double>(lABy + 1) *
                         static_cast<double>(lABz + 1) * static_cast<double>(lCDx + 1) *
                         static_cast<double>(lCDy + 1) * static_cast<double>(lCDz + 1);
    const double os_work =
        six_d * static_cast<double>(L + 1) +
        static_cast<double>((lBx + lBy + lBz + lDx + lDy + lDz) + 1) * six_d * 0.25;
    const double rys_work =
        six_d * static_cast<double>(nroots) +
        static_cast<double>((lBx + lBy + lBz + lDx + lDy + lDz) + 1) * six_d * 0.20 +
        24.0 * static_cast<double>(nroots);

    return rys_work < os_work;
}

static double _auto_contracted_eri(
    const HartreeFock::ShellPair& spAB,
    const HartreeFock::ShellPair& spCD,
    int lAx, int lAy, int lAz,
    int lBx, int lBy, int lBz,
    int lCx, int lCy, int lCz,
    int lDx, int lDy, int lDz) noexcept
{
    if (_auto_prefers_rys(spAB, spCD))
        return HartreeFock::RysQuad::_rys_contracted_eri(
            spAB, spCD, lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz);

    return HartreeFock::ObaraSaika::_contracted_eri_elem(
        spAB, spCD, lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz);
}

// ─── 1D Rys VRR ───────────────────────────────────────────────────────────────
//
// Fills vrr[a][c] for a in [0, lAB], c in [0, lCD] using the recurrences:
//
//   [0,0]     = 1.0                                       (seed)
//   [a+1, 0]  = C00 * [a,0]  + a*B10 * [a-1,0]
//   [a, c+1]  = D00 * [a,c]  + c*B01 * [a,c-1]  + a*B00 * [a-1,c]
//
// where for Rys root u = t_r^2:
//   C00 = PA_q + u*WP_q,  D00 = QC_q + u*WQ_q
//   B00 = u/(2*delta),  B10 = 1/(2*zeta) - u/(2*delta),  B01 = 1/(2*eta) - u/(2*delta)

static void _rys_vrr_1d(
    double vrr[VRR_DIM][VRR_DIM],
    const int lAB, const int lCD,
    const double C00, const double D00,
    const double B00, const double B10, const double B01) noexcept
{
    vrr[0][0] = 1.0;

    // Build A-axis (c=0)
    for (int a = 1; a <= lAB; ++a) {
        vrr[a][0] = C00 * vrr[a-1][0];
        if (a >= 2) vrr[a][0] += (a - 1) * B10 * vrr[a-2][0];
    }

    // Build C-axis and mixed (increment c for each a)
    for (int c = 1; c <= lCD; ++c) {
        vrr[0][c] = D00 * vrr[0][c-1];
        if (c >= 2) vrr[0][c] += (c - 1) * B01 * vrr[0][c-2];

        for (int a = 1; a <= lAB; ++a) {
            vrr[a][c] = D00 * vrr[a][c-1]
                      + a * B00 * vrr[a-1][c-1];
            if (c >= 2) vrr[a][c] += (c - 1) * B01 * vrr[a][c-2];
        }
    }
}

// ─── HRR (reused from OS logic) ───────────────────────────────────────────────
//
// AB-HRR: transfer angular momentum from A to B using displacement AB.
// Operates in-place on the 6D W[ax][ay][az][cx][cy][cz] buffer.
// After the sweep, W[lAx][lAy][lAz][cx][cy][cz] holds (lA,lB | cx,cy,cz, 0).
//
// Transfer rule: [a, b+1 | c, d] = [a+1, b | c, d] + AB_q * [a, b | c, d]
//
// We apply the sweep independently for x, then y, then z, mirroring os.cpp's
// _eri_hrr_ab function.

static void _rys_hrr_ab(
    double W[VRR_DIM][VRR_DIM][VRR_DIM][VRR_DIM][VRR_DIM][VRR_DIM],
    const int lAx, const int lAy, const int lAz,
    const int lBx, const int lBy, const int lBz,
    const int lCDx, const int lCDy, const int lCDz,
    const double ABx, const double ABy, const double ABz) noexcept
{
    for (int kz = 0; kz < lBz; ++kz)
        for (int ax = 0; ax <= lAx + lBx; ++ax)
            for (int ay = 0; ay <= lAy + lBy; ++ay)
                for (int az = 0; az <= lAz + lBz - kz - 1; ++az)
                    for (int cx = 0; cx <= lCDx; ++cx)
                        for (int cy = 0; cy <= lCDy; ++cy)
                            for (int cz = 0; cz <= lCDz; ++cz)
                                W[ax][ay][az][cx][cy][cz] =
                                    W[ax][ay][az + 1][cx][cy][cz]
                                  + ABz * W[ax][ay][az][cx][cy][cz];

    for (int ky = 0; ky < lBy; ++ky)
        for (int ax = 0; ax <= lAx + lBx; ++ax)
            for (int ay = 0; ay <= lAy + lBy - ky - 1; ++ay)
                for (int az = 0; az <= lAz; ++az)
                    for (int cx = 0; cx <= lCDx; ++cx)
                        for (int cy = 0; cy <= lCDy; ++cy)
                            for (int cz = 0; cz <= lCDz; ++cz)
                                W[ax][ay][az][cx][cy][cz] =
                                    W[ax][ay + 1][az][cx][cy][cz]
                                  + ABy * W[ax][ay][az][cx][cy][cz];

    for (int kx = 0; kx < lBx; ++kx)
        for (int ax = 0; ax <= lAx + lBx - kx - 1; ++ax)
            for (int ay = 0; ay <= lAy; ++ay)
                for (int az = 0; az <= lAz; ++az)
                    for (int cx = 0; cx <= lCDx; ++cx)
                        for (int cy = 0; cy <= lCDy; ++cy)
                            for (int cz = 0; cz <= lCDz; ++cz)
                                W[ax][ay][az][cx][cy][cz] =
                                    W[ax + 1][ay][az][cx][cy][cz]
                                  + ABx * W[ax][ay][az][cx][cy][cz];
}

// CD-HRR: transfer C→D using a 3D slice V0[cx][cy][cz].
// Mirrors _nuclear_hrr in os.cpp exactly.

static double _rys_hrr_cd(
    double V0[VRR_DIM][VRR_DIM][VRR_DIM],
    const int lCx, const int lCy, const int lCz,
    const int lDx, const int lDy, const int lDz,
    const double CDx, const double CDy, const double CDz) noexcept
{
    // Working copy
    double W[VRR_DIM][VRR_DIM][VRR_DIM];
    for (int cx = 0; cx <= lCx + lDx; ++cx)
        for (int cy = 0; cy <= lCy + lDy; ++cy)
            for (int cz = 0; cz <= lCz + lDz; ++cz)
                W[cx][cy][cz] = V0[cx][cy][cz];

    for (int kx = 0; kx < lDx; ++kx)
        for (int cx = 0; cx <= lCx + lDx - kx - 1; ++cx)
            for (int cy = 0; cy <= lCy + lDy; ++cy)
                for (int cz = 0; cz <= lCz + lDz; ++cz)
                    W[cx][cy][cz] = W[cx + 1][cy][cz] + CDx * W[cx][cy][cz];

    for (int ky = 0; ky < lDy; ++ky)
        for (int cy = 0; cy <= lCy + lDy - ky - 1; ++cy)
            for (int cz = 0; cz <= lCz + lDz; ++cz)
                W[lCx][cy][cz] = W[lCx][cy + 1][cz] + CDy * W[lCx][cy][cz];

    for (int kz = 0; kz < lDz; ++kz)
        for (int cz = 0; cz <= lCz + lDz - kz - 1; ++cz)
            W[lCx][lCy][cz] = W[lCx][lCy][cz + 1] + CDz * W[lCx][lCy][cz];

    return W[lCx][lCy][lCz];
}

// ─── Primitive ERI ────────────────────────────────────────────────────────────

double HartreeFock::RysQuad::_rys_eri_primitive(
    const HartreeFock::PrimitivePair& ppAB,
    const HartreeFock::PrimitivePair& ppCD,
    const int lAx, const int lAy, const int lAz,
    const int lBx, const int lBy, const int lBz,
    const int lCx, const int lCy, const int lCz,
    const int lDx, const int lDy, const int lDz,
    const double ABx, const double ABy, const double ABz,
    const double CDx, const double CDy, const double CDz) noexcept
{
    // Derived quantities
    const double zeta     = ppAB.zeta;
    const double eta      = ppCD.zeta;
    const double delta    = zeta + eta;
    const double inv_delta = 1.0 / delta;
    const double rho      = zeta * eta * inv_delta;
    const double rho_over_zeta = rho * ppAB.inv_zeta;
    const double rho_over_eta  = rho * ppCD.inv_zeta;

    // Gaussian product centers P and Q
    const double Px = ppAB.center[0], Py = ppAB.center[1], Pz = ppAB.center[2];
    const double Qx = ppCD.center[0], Qy = ppCD.center[1], Qz = ppCD.center[2];

    // PQ displacement and Boys argument T = rho * |PQ|^2
    const double PQx = Px - Qx, PQy = Py - Qy, PQz = Pz - Qz;
    const double T   = rho * (PQx*PQx + PQy*PQy + PQz*PQz);

    // PA and QC vectors (stored in PrimitivePair.pA)
    const double PAx = ppAB.pA[0], PAy = ppAB.pA[1], PAz = ppAB.pA[2];
    const double QCx = ppCD.pA[0], QCy = ppCD.pA[1], QCz = ppCD.pA[2];

    // W = (zeta*P + eta*Q) / delta
    const double Wx = (zeta*Px + eta*Qx) * inv_delta;
    const double Wy = (zeta*Py + eta*Qy) * inv_delta;
    const double Wz = (zeta*Pz + eta*Qz) * inv_delta;

    // WP = W - P,  WQ = W - Q
    const double WPx = Wx - Px, WPy = Wy - Py, WPz = Wz - Pz;
    const double WQx = Wx - Qx, WQy = Wy - Qy, WQz = Wz - Qz;

    // Overall prefactor: K_AB * K_CD * 2*sqrt(rho/pi)
    const double prefac = ppAB.prefactor * ppCD.prefactor
                        * 2.0 * std::sqrt(rho / M_PI);

    // Number of Rys roots
    const int lABx = lAx + lBx, lABy = lAy + lBy, lABz = lAz + lBz;
    const int lCDx = lCx + lDx, lCDy = lCy + lDy, lCDz = lCz + lDz;
    const int L    = lABx + lABy + lABz + lCDx + lCDy + lCDz;
    const int n    = L / 2 + 1;

    // Fetch Rys roots and weights
    double t2[HartreeFock::Rys::RYS_MAX_ROOTS];
    double w [HartreeFock::Rys::RYS_MAX_ROOTS];
    HartreeFock::Rys::rys_roots_weights(n, T, t2, w);

    // Zero the 6D accumulation buffer for the active slice
    for (int ax = 0; ax <= lABx; ++ax)
        for (int ay = 0; ay <= lABy; ++ay)
            for (int az = 0; az <= lABz; ++az)
                for (int cx = 0; cx <= lCDx; ++cx)
                    for (int cy = 0; cy <= lCDy; ++cy)
                        for (int cz = 0; cz <= lCDz; ++cz)
                            _rys_sum_buf[ax][ay][az][cx][cy][cz] = 0.0;

    // Per-root VRR + accumulation
    for (int r = 0; r < n; ++r) {
        const double u  = t2[r];
        const double wr = w[r];

        // Root-dependent scalars (same for all axes)
        const double B00 = 0.5 * inv_delta * u;
        const double B10 = 0.5 * ppAB.inv_zeta * (1.0 - rho_over_zeta * u);
        const double B01 = 0.5 * ppCD.inv_zeta * (1.0 - rho_over_eta * u);

        // 1D VRR tables for each Cartesian component
        double Ix[VRR_DIM][VRR_DIM];
        double Iy[VRR_DIM][VRR_DIM];
        double Iz[VRR_DIM][VRR_DIM];

        _rys_vrr_1d(Ix, lABx, lCDx, PAx + u*WPx, QCx + u*WQx, B00, B10, B01);
        _rys_vrr_1d(Iy, lABy, lCDy, PAy + u*WPy, QCy + u*WQy, B00, B10, B01);
        _rys_vrr_1d(Iz, lABz, lCDz, PAz + u*WPz, QCz + u*WQz, B00, B10, B01);

        // Accumulate outer product into _rys_sum_buf
        for (int ax = 0; ax <= lABx; ++ax)
            for (int ay = 0; ay <= lABy; ++ay)
                for (int az = 0; az <= lABz; ++az)
                    for (int cx = 0; cx <= lCDx; ++cx)
                        for (int cy = 0; cy <= lCDy; ++cy)
                            for (int cz = 0; cz <= lCDz; ++cz)
                                _rys_sum_buf[ax][ay][az][cx][cy][cz] +=
                                    wr * Ix[ax][cx] * Iy[ay][cy] * Iz[az][cz];
    }

    // Scale by overall prefactor
    for (int ax = 0; ax <= lABx; ++ax)
        for (int ay = 0; ay <= lABy; ++ay)
            for (int az = 0; az <= lABz; ++az)
                for (int cx = 0; cx <= lCDx; ++cx)
                    for (int cy = 0; cy <= lCDy; ++cy)
                        for (int cz = 0; cz <= lCDz; ++cz)
                            _rys_sum_buf[ax][ay][az][cx][cy][cz] *= prefac;

    // AB-HRR (in-place on _rys_sum_buf)
    _rys_hrr_ab(_rys_sum_buf,
                lAx, lAy, lAz, lBx, lBy, lBz,
                lCDx, lCDy, lCDz,
                ABx, ABy, ABz);

    // Extract CD slice at (lAx, lAy, lAz)
    double V0_CD[VRR_DIM][VRR_DIM][VRR_DIM];
    for (int cx = 0; cx <= lCDx; ++cx)
        for (int cy = 0; cy <= lCDy; ++cy)
            for (int cz = 0; cz <= lCDz; ++cz)
                V0_CD[cx][cy][cz] = _rys_sum_buf[lAx][lAy][lAz][cx][cy][cz];

    // CD-HRR
    return _rys_hrr_cd(V0_CD, lCx, lCy, lCz, lDx, lDy, lDz, CDx, CDy, CDz);
}

// ─── Contracted ERI ───────────────────────────────────────────────────────────

double HartreeFock::RysQuad::_rys_contracted_eri(
    const HartreeFock::ShellPair& spAB,
    const HartreeFock::ShellPair& spCD,
    const int lAx, const int lAy, const int lAz,
    const int lBx, const int lBy, const int lBz,
    const int lCx, const int lCy, const int lCz,
    const int lDx, const int lDy, const int lDz) noexcept
{
    const double ABx = spAB.R[0], ABy = spAB.R[1], ABz = spAB.R[2];
    const double CDx = spCD.R[0], CDy = spCD.R[1], CDz = spCD.R[2];

    double eri = 0.0;
    for (const auto& ppAB : spAB.primitive_pairs)
        for (const auto& ppCD : spCD.primitive_pairs)
            eri += ppAB.coeff_product * ppCD.coeff_product
                 * _rys_eri_primitive(ppAB, ppCD,
                                      lAx, lAy, lAz, lBx, lBy, lBz,
                                      lCx, lCy, lCz, lDx, lDy, lDz,
                                      ABx, ABy, ABz, CDx, CDy, CDz);
    return eri;
}

// ─── Schwarz screening (mirrors os.cpp _compute_schwarz_table) ────────────────

static Eigen::MatrixXd _rys_schwarz_table(
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    const std::size_t nbasis)
{
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nbasis, nbasis);

    for (const auto& sp : shell_pairs) {
        const std::size_t i = sp.A._index;
        const std::size_t j = sp.B._index;
        const int lAx = sp.A._cartesian[0], lAy = sp.A._cartesian[1], lAz = sp.A._cartesian[2];
        const int lBx = sp.B._cartesian[0], lBy = sp.B._cartesian[1], lBz = sp.B._cartesian[2];

        const double val = HartreeFock::RysQuad::_rys_contracted_eri(
            sp, sp, lAx, lAy, lAz, lBx, lBy, lBz, lAx, lAy, lAz, lBx, lBy, lBz);

        Q(i, j) = Q(j, i) = std::sqrt(std::abs(val));
    }
    return Q;
}

std::vector<double> HartreeFock::RysQuad::_compute_2e(
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    const std::size_t nbasis,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp>*)
{
    const std::size_t nb  = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;

    const Eigen::MatrixXd Q = _rys_schwarz_table(shell_pairs, nb);
    std::vector<double> eri(nb * nb * nb * nb, 0.0);

    const std::size_t npairs = shell_pairs.size();
#pragma omp parallel for schedule(dynamic)
    for (std::size_t p = 0; p < npairs; ++p) {
        const auto& spAB = shell_pairs[p];
        const std::size_t i = spAB.A._index;
        const std::size_t j = spAB.B._index;
        const int lAx = spAB.A._cartesian[0], lAy = spAB.A._cartesian[1], lAz = spAB.A._cartesian[2];
        const int lBx = spAB.B._cartesian[0], lBy = spAB.B._cartesian[1], lBz = spAB.B._cartesian[2];

        for (std::size_t q = p; q < npairs; ++q) {
            const auto& spCD = shell_pairs[q];
            const std::size_t k = spCD.A._index;
            const std::size_t l = spCD.B._index;
            if (Q(i, j) * Q(k, l) < tol_eri) continue;

            const int lCx = spCD.A._cartesian[0], lCy = spCD.A._cartesian[1], lCz = spCD.A._cartesian[2];
            const int lDx = spCD.B._cartesian[0], lDy = spCD.B._cartesian[1], lDz = spCD.B._cartesian[2];
            const double val = HartreeFock::RysQuad::_rys_contracted_eri(
                spAB, spCD,
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

    return eri;
}

std::vector<double> HartreeFock::RysQuad::_compute_2e_auto(
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    const std::size_t nbasis,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp>*)
{
    const std::size_t nb  = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;

    const Eigen::MatrixXd Q = _rys_schwarz_table(shell_pairs, nb);
    std::vector<double> eri(nb * nb * nb * nb, 0.0);

    const std::size_t npairs = shell_pairs.size();
#pragma omp parallel for schedule(dynamic)
    for (std::size_t p = 0; p < npairs; ++p) {
        const auto& spAB = shell_pairs[p];
        const std::size_t i = spAB.A._index;
        const std::size_t j = spAB.B._index;
        const int lAx = spAB.A._cartesian[0], lAy = spAB.A._cartesian[1], lAz = spAB.A._cartesian[2];
        const int lBx = spAB.B._cartesian[0], lBy = spAB.B._cartesian[1], lBz = spAB.B._cartesian[2];

        for (std::size_t q = p; q < npairs; ++q) {
            const auto& spCD = shell_pairs[q];
            const std::size_t k = spCD.A._index;
            const std::size_t l = spCD.B._index;
            if (Q(i, j) * Q(k, l) < tol_eri) continue;

            const int lCx = spCD.A._cartesian[0], lCy = spCD.A._cartesian[1], lCz = spCD.A._cartesian[2];
            const int lDx = spCD.B._cartesian[0], lDy = spCD.B._cartesian[1], lDz = spCD.B._cartesian[2];
            const double val = _auto_contracted_eri(spAB, spCD,
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

    return eri;
}

// ─── Public: RHF 2e Fock (direct SCF) ────────────────────────────────────────

Eigen::MatrixXd HartreeFock::RysQuad::_compute_2e_fock(
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    const Eigen::MatrixXd& density,
    const std::size_t nbasis,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp>*)
{
    const std::size_t nb  = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;

    std::vector<double> eri = _compute_2e(shell_pairs, nbasis, tol_eri);

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

// ─── Public: UHF 2e Fock (direct SCF) ────────────────────────────────────────

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
HartreeFock::RysQuad::_compute_2e_fock_uhf(
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

    const Eigen::MatrixXd Pt = Pa + Pb;
    std::vector<double> eri = _compute_2e(shell_pairs, nbasis, tol_eri);

    Eigen::MatrixXd Ga = Eigen::MatrixXd::Zero(nb, nb);
    Eigen::MatrixXd Gb = Eigen::MatrixXd::Zero(nb, nb);

#pragma omp parallel for schedule(static)
    for (std::size_t mu = 0; mu < nb; ++mu)
        for (std::size_t nu = 0; nu < nb; ++nu)
            for (std::size_t lam = 0; lam < nb; ++lam)
                for (std::size_t sig = 0; sig < nb; ++sig) {
                    const double coulomb = eri[mu*nb3 + nu*nb2 + lam*nb + sig];
                    const double exch    = eri[mu*nb3 + lam*nb2 + nu*nb + sig];
                    Ga(mu, nu) += Pt(lam, sig) * coulomb - Pa(lam, sig) * exch;
                    Gb(mu, nu) += Pt(lam, sig) * coulomb - Pb(lam, sig) * exch;
                }

    return {Ga, Gb};
}

// ─── Auto-dispatch variants ───────────────────────────────────────────────────
//
// Per-quartet: if L = la+lb+lc+ld >= RYS_CROSSOVER_L use Rys, else OS.
// The _contracted_eri from OS is a static function in os.cpp, so we cannot call
// it directly. Instead, the Auto path builds the full Fock matrix using a
// hybrid loop: call _rys_contracted_eri for all quartets (since the crossover
// only matters for performance, not correctness).  A future revision will link
// in the OS contracted ERI for the low-L path once os.cpp exposes it.

Eigen::MatrixXd HartreeFock::RysQuad::_compute_2e_fock_auto(
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    const Eigen::MatrixXd& density,
    const std::size_t nbasis,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp>* sym_ops)
{
    const std::size_t nb  = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;
    std::vector<double> eri = _compute_2e_auto(shell_pairs, nbasis, tol_eri, sym_ops);
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

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
HartreeFock::RysQuad::_compute_2e_fock_uhf_auto(
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    const Eigen::MatrixXd& Pa,
    const Eigen::MatrixXd& Pb,
    const std::size_t nbasis,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp>* sym_ops)
{
    const std::size_t nb  = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;
    const Eigen::MatrixXd Pt = Pa + Pb;
    std::vector<double> eri = _compute_2e_auto(shell_pairs, nbasis, tol_eri, sym_ops);
    Eigen::MatrixXd Ga = Eigen::MatrixXd::Zero(nb, nb);
    Eigen::MatrixXd Gb = Eigen::MatrixXd::Zero(nb, nb);

#pragma omp parallel for schedule(static)
    for (std::size_t mu = 0; mu < nb; ++mu)
        for (std::size_t nu = 0; nu < nb; ++nu)
            for (std::size_t lam = 0; lam < nb; ++lam)
                for (std::size_t sig = 0; sig < nb; ++sig) {
                    const double coulomb = eri[mu*nb3 + nu*nb2 + lam*nb + sig];
                    const double exch    = eri[mu*nb3 + lam*nb2 + nu*nb + sig];
                    Ga(mu, nu) += Pt(lam, sig) * coulomb - Pa(lam, sig) * exch;
                    Gb(mu, nu) += Pt(lam, sig) * coulomb - Pb(lam, sig) * exch;
                }
    return {Ga, Gb};
}
