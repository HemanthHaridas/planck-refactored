#include "intcoords.h"
#include "base/tables.h"

#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <cmath>
#include <stdexcept>

// ── Covalent radii (Angstrom) from Alvarez (2008) ────────────────────────────
// Indexed by atomic number Z (1-based; index 0 unused).
// Elements beyond index 36 fall back to 1.5 Å.
static double covalent_radius_ang(int Z)
{
    // Alvarez (2008), Dalton Trans. 2832, Table 2, single-bond radii.
    static constexpr double rcov[] = {
        0.00, // 0 — unused
        0.31, // H
        0.28, // He
        1.28, // Li
        0.96, // Be
        0.84, // B
        0.76, // C
        0.71, // N
        0.66, // O
        0.57, // F
        0.58, // Ne
        1.66, // Na
        1.41, // Mg
        1.21, // Al
        1.11, // Si
        1.07, // P
        1.05, // S
        1.02, // Cl
        1.06, // Ar
        2.03, // K
        1.76, // Ca
        1.70, // Sc
        1.60, // Ti
        1.53, // V
        1.39, // Cr
        1.61, // Mn (LS)
        1.52, // Fe (LS)
        1.50, // Co (LS)
        1.24, // Ni
        1.32, // Cu
        1.22, // Zn
        1.22, // Ga
        1.20, // Ge
        1.19, // As
        1.20, // Se
        1.20, // Br
        1.16, // Kr
    };
    if (Z >= 1 && Z <= 36)
        return rcov[Z];
    return 1.50; // fallback
}

// ── Anonymous helpers (B-matrix s-vector calculations) ───────────────────────

namespace
{
    using Vec3 = Eigen::Vector3d;

    // ── Stretch R(A,B) ────────────────────────────────────────────────────────────

    static double stretch_value(const Eigen::MatrixXd &xyz, int A, int B)
    {
        return (xyz.row(A) - xyz.row(B)).norm();
    }

    static void stretch_brow(const Eigen::MatrixXd &xyz, int A, int B,
                             Vec3 &sA, Vec3 &sB)
    {
        Vec3 rAB = xyz.row(A) - xyz.row(B);
        double R = rAB.norm();
        if (R < 1e-10)
        {
            sA.setZero();
            sB.setZero();
            return;
        }
        sA = rAB / R;
        sB = -sA;
    }

    // ── Bend θ(A,B,C)  — B is the central atom ───────────────────────────────────
    // e1 = (A-B)/|A-B|, e2 = (C-B)/|C-B|,  cos θ = e1·e2
    // ∂θ/∂r_A = (cos θ · e1 - e2) / (sin θ · |A-B|)
    // ∂θ/∂r_C = (cos θ · e2 - e1) / (sin θ · |C-B|)
    // ∂θ/∂r_B = -(∂θ/∂r_A + ∂θ/∂r_C)

    static double bend_value(const Eigen::MatrixXd &xyz, int A, int B, int C)
    {
        Vec3 e1 = xyz.row(A) - xyz.row(B);
        Vec3 e2 = xyz.row(C) - xyz.row(B);
        double d1 = e1.norm(), d2 = e2.norm();
        if (d1 < 1e-10 || d2 < 1e-10)
            return 0.0;
        double cos_t = e1.dot(e2) / (d1 * d2);
        cos_t = std::max(-1.0, std::min(1.0, cos_t));
        return std::acos(cos_t);
    }

    static void bend_brow(const Eigen::MatrixXd &xyz, int A, int B, int C,
                          Vec3 &sA, Vec3 &sB, Vec3 &sC)
    {
        Vec3 rBA = xyz.row(A) - xyz.row(B);
        Vec3 rBC = xyz.row(C) - xyz.row(B);
        double dBA = rBA.norm(), dBC = rBC.norm();
        if (dBA < 1e-10 || dBC < 1e-10)
        {
            sA.setZero();
            sB.setZero();
            sC.setZero();
            return;
        }
        Vec3 e1 = rBA / dBA;
        Vec3 e2 = rBC / dBC;
        double cos_t = e1.dot(e2);
        cos_t = std::max(-1.0 + 1e-10, std::min(1.0 - 1e-10, cos_t));
        double sin_t = std::sqrt(1.0 - cos_t * cos_t);
        if (sin_t < 1e-8)
        {
            sA.setZero();
            sB.setZero();
            sC.setZero();
            return;
        }

        sA = (cos_t * e1 - e2) / (sin_t * dBA);
        sC = (cos_t * e2 - e1) / (sin_t * dBC);
        sB = -(sA + sC);
    }

    // ── Torsion φ(A,B,C,D)  — dihedral about bond B–C ────────────────────────────
    //
    // b1 = r_B - r_A,  b2 = r_C - r_B,  b3 = r_D - r_C
    // m  = b1 × b2,   n  = b2 × b3
    // φ  = atan2( (m×n)·b̂2,  m·n )
    //
    // Wilson s-vectors (Pulay/Schlegel/Bakken-Helgaker):
    //   s_A = -|b2|/|m|² · m
    //   s_D = +|b2|/|n|² · n
    //   s_B = (b1·b2/|b2|² - 1)·|b2|/|m|² · m  +  b3·b2/|b2|²·|b2|/|n|² · n
    //   s_C = -(s_A + s_B + s_D)   [translational invariance]

    static double torsion_value(const Eigen::MatrixXd &xyz,
                                int A, int B, int C, int D)
    {
        Vec3 b1 = xyz.row(B) - xyz.row(A);
        Vec3 b2 = xyz.row(C) - xyz.row(B);
        Vec3 b3 = xyz.row(D) - xyz.row(C);
        Vec3 m = b1.cross(b2);
        Vec3 n = b2.cross(b3);
        double mm = m.squaredNorm(), nn = n.squaredNorm();
        if (mm < 1e-20 || nn < 1e-20)
            return 0.0;
        double b2n = b2.norm();
        double x = m.dot(n);
        double y = b2.dot(m.cross(n)) / b2n;
        return std::atan2(y, x);
    }

    static void torsion_brow(const Eigen::MatrixXd &xyz,
                             int A, int B, int C, int D,
                             Vec3 &sA, Vec3 &sB, Vec3 &sC, Vec3 &sD)
    {
        Vec3 b1 = xyz.row(B) - xyz.row(A);
        Vec3 b2 = xyz.row(C) - xyz.row(B);
        Vec3 b3 = xyz.row(D) - xyz.row(C);
        Vec3 m = b1.cross(b2);
        Vec3 n = b2.cross(b3);
        double mm = m.squaredNorm(), nn = n.squaredNorm();
        double b2n2 = b2.squaredNorm();
        double b2n = std::sqrt(b2n2);

        if (mm < 1e-20 || nn < 1e-20)
        {
            sA.setZero();
            sB.setZero();
            sC.setZero();
            sD.setZero();
            return;
        }

        sA = -b2n / mm * m;
        sD = b2n / nn * n;
        sB = (b1.dot(b2) / b2n2 - 1.0) * (b2n / mm) * m + (b3.dot(b2) / b2n2) * (b2n / nn) * n;
        sC = -(sA + sB + sD);
    }

    // ── Pseudoinverse of a symmetric PSD matrix via Jacobi SVD ───────────────────
    static Eigen::MatrixXd pinv_sym(const Eigen::MatrixXd &G, double eps = 1e-8)
    {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(G,
                                              Eigen::ComputeFullU | Eigen::ComputeFullV);
        const double tol = eps * svd.singularValues().maxCoeff();
        Eigen::VectorXd sv_inv = svd.singularValues().unaryExpr(
            [tol](double s)
            { return (s > tol) ? 1.0 / s : 0.0; });
        return svd.matrixV() * sv_inv.asDiagonal() * svd.matrixU().transpose();
    }

} // anonymous namespace

// ── InternalCoord::value ──────────────────────────────────────────────────────

double HartreeFock::Opt::InternalCoord::value(const Eigen::MatrixXd &xyz) const
{
    switch (type)
    {
    case ICType::Stretch:
        return stretch_value(xyz, atoms[0], atoms[1]);
    case ICType::Bend:
        return bend_value(xyz, atoms[0], atoms[1], atoms[2]);
    case ICType::Torsion:
        return torsion_value(xyz, atoms[0], atoms[1], atoms[2], atoms[3]);
    }
    return 0.0;
}

// ── InternalCoord::brow ───────────────────────────────────────────────────────

Eigen::VectorXd HartreeFock::Opt::InternalCoord::brow(
    const Eigen::MatrixXd &xyz, int natoms) const
{
    Eigen::VectorXd row = Eigen::VectorXd::Zero(3 * natoms);

    switch (type)
    {
    case ICType::Stretch:
    {
        Vec3 sA, sB;
        stretch_brow(xyz, atoms[0], atoms[1], sA, sB);
        row.segment(3 * atoms[0], 3) = sA;
        row.segment(3 * atoms[1], 3) = sB;
        break;
    }
    case ICType::Bend:
    {
        Vec3 sA, sB, sC;
        bend_brow(xyz, atoms[0], atoms[1], atoms[2], sA, sB, sC);
        row.segment(3 * atoms[0], 3) = sA;
        row.segment(3 * atoms[1], 3) = sB;
        row.segment(3 * atoms[2], 3) = sC;
        break;
    }
    case ICType::Torsion:
    {
        Vec3 sA, sB, sC, sD;
        torsion_brow(xyz, atoms[0], atoms[1], atoms[2], atoms[3],
                     sA, sB, sC, sD);
        row.segment(3 * atoms[0], 3) = sA;
        row.segment(3 * atoms[1], 3) = sB;
        row.segment(3 * atoms[2], 3) = sC;
        row.segment(3 * atoms[3], 3) = sD;
        break;
    }
    }
    return row;
}

// ── IntCoordSystem::build ─────────────────────────────────────────────────────

HartreeFock::Opt::IntCoordSystem HartreeFock::Opt::IntCoordSystem::build(
    const Eigen::MatrixXd &xyz_bohr, const Eigen::VectorXi &Z)
{
    const int N = static_cast<int>(xyz_bohr.rows());
    IntCoordSystem ics;
    ics.natoms = N;

    // Covalent radii in Bohr
    std::vector<double> rcov(N);
    for (int i = 0; i < N; ++i)
        rcov[i] = covalent_radius_ang(Z[i]) * ANGSTROM_TO_BOHR;

    // ── Connectivity ─────────────────────────────────────────────────────────
    std::vector<std::vector<int>> adj(N);
    std::vector<std::pair<int, int>> bond_list;

    for (int i = 0; i < N; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            double d = (xyz_bohr.row(i) - xyz_bohr.row(j)).norm();
            if (d < 1.3 * (rcov[i] + rcov[j]))
            {
                bond_list.emplace_back(i, j);
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }

    // ── Stretches ─────────────────────────────────────────────────────────────
    for (const auto &[i, j] : bond_list)
        ics.coords.push_back({ICType::Stretch, {i, j, -1, -1}});

    // ── Bends (all A–B–C with B central, A-B and B-C bonded) ─────────────────
    constexpr double ang_lo = 5.0 * M_PI / 180.0;
    constexpr double ang_hi = 175.0 * M_PI / 180.0;

    for (int B = 0; B < N; ++B)
    {
        const auto &nb = adj[B];
        for (int ia = 0; ia < static_cast<int>(nb.size()); ++ia)
        {
            for (int ic = ia + 1; ic < static_cast<int>(nb.size()); ++ic)
            {
                int A = nb[ia], C = nb[ic];
                double ang = bend_value(xyz_bohr, A, B, C);
                if (ang < ang_lo || ang > ang_hi)
                    continue;
                ics.coords.push_back({ICType::Bend, {A, B, C, -1}});
            }
        }
    }

    // ── Torsions (A–B–C–D about bond B–C) ────────────────────────────────────
    for (const auto &[B, C] : bond_list)
    {
        for (int A : adj[B])
        {
            if (A == C)
                continue;
            double ang1 = bend_value(xyz_bohr, A, B, C);
            if (ang1 < ang_lo || ang1 > ang_hi)
                continue;

            for (int D : adj[C])
            {
                if (D == B || D == A)
                    continue;
                double ang2 = bend_value(xyz_bohr, B, C, D);
                if (ang2 < ang_lo || ang2 > ang_hi)
                    continue;
                ics.coords.push_back({ICType::Torsion, {A, B, C, D}});
            }
        }
    }

    return ics;
}

// ── IntCoordSystem::add_coord ─────────────────────────────────────────────────
//
// Inserts ic into the coordinate list if no equivalent IC is already present,
// checking both forward and reverse orderings.  Returns the 0-based index of
// the (possibly pre-existing) coordinate.

int HartreeFock::Opt::IntCoordSystem::add_coord(const InternalCoord &ic)
{
    for (int k = 0; k < nics(); ++k)
    {
        const auto &c = coords[k];
        if (c.type != ic.type)
            continue;

        if (ic.type == ICType::Stretch)
        {
            // {i,j} == {j,i}
            if ((c.atoms[0] == ic.atoms[0] && c.atoms[1] == ic.atoms[1]) ||
                (c.atoms[0] == ic.atoms[1] && c.atoms[1] == ic.atoms[0]))
                return k;
        }
        else if (ic.type == ICType::Bend)
        {
            // {i,j,k} == {k,j,i}  (same central atom j)
            if (c.atoms[1] == ic.atoms[1] &&
                ((c.atoms[0] == ic.atoms[0] && c.atoms[2] == ic.atoms[2]) ||
                 (c.atoms[0] == ic.atoms[2] && c.atoms[2] == ic.atoms[0])))
                return k;
        }
        else if (ic.type == ICType::Torsion)
        {
            // {i,j,k,l} == {l,k,j,i}
            if ((c.atoms[0] == ic.atoms[0] && c.atoms[1] == ic.atoms[1] &&
                 c.atoms[2] == ic.atoms[2] && c.atoms[3] == ic.atoms[3]) ||
                (c.atoms[0] == ic.atoms[3] && c.atoms[1] == ic.atoms[2] &&
                 c.atoms[2] == ic.atoms[1] && c.atoms[3] == ic.atoms[0]))
                return k;
        }
    }
    // Not present — insert
    coords.push_back(ic);
    return static_cast<int>(coords.size()) - 1;
}

// ── IntCoordSystem::values ────────────────────────────────────────────────────

Eigen::VectorXd HartreeFock::Opt::IntCoordSystem::values(
    const Eigen::MatrixXd &xyz) const
{
    Eigen::VectorXd q(nics());
    for (int i = 0; i < nics(); ++i)
        q[i] = coords[i].value(xyz);
    return q;
}

// ── IntCoordSystem::bmatrix ───────────────────────────────────────────────────

Eigen::MatrixXd HartreeFock::Opt::IntCoordSystem::bmatrix(
    const Eigen::MatrixXd &xyz) const
{
    const int n = nics(), m = 3 * natoms;
    Eigen::MatrixXd B(n, m);
    for (int i = 0; i < n; ++i)
        B.row(i) = coords[i].brow(xyz, natoms);
    return B;
}

// ── IntCoordSystem::cart_to_ic_grad ──────────────────────────────────────────
//
// g_cart is atom-major: g_cart[a*3 + k] = ∂E/∂(coord k of atom a).
// B-matrix is also atom-major: B[i, 3*a + k].  No permutation needed.

Eigen::VectorXd HartreeFock::Opt::IntCoordSystem::cart_to_ic_grad(
    const Eigen::MatrixXd &xyz, const Eigen::VectorXd &g_cart) const
{
    Eigen::MatrixXd B = bmatrix(xyz);
    Eigen::MatrixXd G = B * B.transpose();
    return pinv_sym(G) * (B * g_cart);
}

// ── IntCoordSystem::ic_to_cart_step ──────────────────────────────────────────
//
// Iterative back-transform following Schlegel (1984):
//   x_new = x0 + B^T G^+ dq         (first-order)
//   then iterate:  x += B(x)^T G(x)^+ residual
//   until ‖residual‖ < 1e-10 or max_iter reached.

Eigen::MatrixXd HartreeFock::Opt::IntCoordSystem::ic_to_cart_step(
    const Eigen::MatrixXd &xyz0,
    const Eigen::VectorXd &dq,
    int max_iter) const
{
    Eigen::VectorXd q0 = values(xyz0);

    // First-order step
    Eigen::MatrixXd B0 = bmatrix(xyz0);
    Eigen::MatrixXd G0 = B0 * B0.transpose();
    Eigen::MatrixXd G0_pi = pinv_sym(G0);
    Eigen::VectorXd dx = B0.transpose() * (G0_pi * dq);

    Eigen::MatrixXd xyz = xyz0;
    for (int a = 0; a < natoms; ++a)
        xyz.row(a) += dx.segment(3 * a, 3).transpose();

    // Microiterations
    for (int iter = 0; iter < max_iter; ++iter)
    {
        Eigen::VectorXd q_new = values(xyz);
        Eigen::VectorXd dq_actual(nics());
        for (int i = 0; i < nics(); ++i)
        {
            dq_actual[i] = q_new[i] - q0[i];
            // Wrap torsions to [-π, π]
            if (coords[i].type == ICType::Torsion)
            {
                while (dq_actual[i] > M_PI)
                    dq_actual[i] -= 2.0 * M_PI;
                while (dq_actual[i] < -M_PI)
                    dq_actual[i] += 2.0 * M_PI;
            }
        }
        Eigen::VectorXd residual = dq - dq_actual;
        if (residual.norm() < 1e-10)
            break;

        Eigen::MatrixXd Bn = bmatrix(xyz);
        Eigen::MatrixXd Gn = Bn * Bn.transpose();
        Eigen::VectorXd dcorr = Bn.transpose() * (pinv_sym(Gn) * residual);
        for (int a = 0; a < natoms; ++a)
            xyz.row(a) += dcorr.segment(3 * a, 3).transpose();
    }

    return xyz;
}
