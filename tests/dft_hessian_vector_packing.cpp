// F3.5 (docs/SOSCF_DFT_ANALYTIC_FXC_SCOPE.md): MO projection and (a,i)
// packing -- pure plumbing, no new physics, but a genuinely distinct
// convention risk from F3.1-4's own algebra.
//
// This codebase has TWO different flat-index conventions for an (occ,virt)
// pair, confirmed by reading both implementations rather than assumed:
//   - HartreeFock::Correlation::build_rhf_cphf_matrix / build_uhf_cphf_matrix
//     (the orbital-Hessian linear part solve_augmented_hessian's h_op must
//     eventually match): idx(a,i) = a*n_occ + i -- VIRTUAL-major.
//   - DFT::Driver::ResponseExcitationSpace::flat_index(i,a) = i*n_virt + a
//     -- OCCUPIED-major (the FD-kernel oracle's own packing, used by TDDFT
//     and by every F3.1-4 whole-molecule probe that has since been removed
//     from production).
//
// pack_hessian_vector_product_cphf_order (src/dft/response_packing.{h,cpp},
// kept out of driver.cpp specifically so this test does not have to link
// the whole KS-loop driver and its transitive SCF/post-HF/gradient
// dependencies) projects an AO-basis delta_V_xc into the (a,i) MO block
// and packs it in the FIRST (CPHF) convention. This test verifies that
// packing against a hand-computed dense (n_occ x n_virt) MO block, on
// synthetic (non-square, n_occ != n_virt) matrices so a row/column
// transposition or an i<->a swap cannot hide by coincidence -- the exact
// class of bug this step exists to catch, per the scope doc's own verify
// note.
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

#include "dft/response_packing.h"

namespace
{
    bool g_ok = true;

    void require(bool condition, const std::string &message)
    {
        if (!condition)
        {
            std::cerr << message << '\n';
            g_ok = false;
        }
    }

    // Independent reference: project delta_V_xc into the (n_occ x n_virt)
    // MO block by hand (no reuse of the production helper's own matrix
    // multiply order) and flatten it in the CPHF (a*n_occ+i) convention
    // element-by-element with an explicit loop -- deliberately not the
    // same code shape as pack_hessian_vector_product_cphf_order, so a
    // shared bug in "how to multiply three matrices together" can't hide
    // by being copied into both places.
    Eigen::VectorXd reference_pack_cphf_order(
        const Eigen::MatrixXd &delta_v_xc_ao,
        const Eigen::MatrixXd &c_occ,
        const Eigen::MatrixXd &c_virt)
    {
        const int n_occ = static_cast<int>(c_occ.cols());
        const int n_virt = static_cast<int>(c_virt.cols());
        const int nbasis = static_cast<int>(c_occ.rows());
        Eigen::VectorXd packed(n_occ * n_virt);
        for (int a = 0; a < n_virt; ++a)
        {
            for (int i = 0; i < n_occ; ++i)
            {
                double sum = 0.0;
                for (int mu = 0; mu < nbasis; ++mu)
                    for (int nu = 0; nu < nbasis; ++nu)
                        sum += c_occ(mu, i) * delta_v_xc_ao(mu, nu) * c_virt(nu, a);
                packed(a * n_occ + i) = sum;
            }
        }
        return packed;
    }

    void check_packing(int nbasis, int n_occ, int n_virt, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        Eigen::MatrixXd delta_v_xc_ao(nbasis, nbasis);
        for (int mu = 0; mu < nbasis; ++mu)
            for (int nu = 0; nu < nbasis; ++nu)
                delta_v_xc_ao(mu, nu) = dist(rng);
        // delta_V_xc is a real symmetric operator (it comes from a
        // symmetric rank-2 AO update in every F3.1-4 formula) -- enforce
        // that here too, so this fixture matches what production code
        // actually produces rather than an arbitrary matrix.
        delta_v_xc_ao = (0.5 * (delta_v_xc_ao + delta_v_xc_ao.transpose())).eval();

        Eigen::MatrixXd c_occ(nbasis, n_occ);
        Eigen::MatrixXd c_virt(nbasis, n_virt);
        for (int mu = 0; mu < nbasis; ++mu)
        {
            for (int i = 0; i < n_occ; ++i)
                c_occ(mu, i) = dist(rng);
            for (int a = 0; a < n_virt; ++a)
                c_virt(mu, a) = dist(rng);
        }

        const Eigen::VectorXd packed = DFT::Driver::pack_hessian_vector_product_cphf_order(
            delta_v_xc_ao, c_occ, c_virt);
        const Eigen::VectorXd reference = reference_pack_cphf_order(delta_v_xc_ao, c_occ, c_virt);

        require(packed.size() == n_occ * n_virt,
                "packed vector has wrong size: nbasis=" + std::to_string(nbasis) +
                    " n_occ=" + std::to_string(n_occ) + " n_virt=" + std::to_string(n_virt));

        for (Eigen::Index k = 0; k < packed.size(); ++k)
        {
            const double diff = std::abs(packed(k) - reference(k));
            if (diff > 1e-10)
            {
                std::cerr << "packing mismatch at flat index " << k << ": production="
                          << packed(k) << " reference=" << reference(k) << " diff=" << diff
                          << " (nbasis=" << nbasis << " n_occ=" << n_occ << " n_virt=" << n_virt
                          << ")\n";
                g_ok = false;
            }
        }
    }

    // Confirms the two conventions this file's header comment names are
    // GENUINELY different (not a documentation error) -- if n_occ==n_virt
    // this check would be vacuous (a square matrix in the wrong convention
    // can still equal itself along the diagonal), so this deliberately uses
    // a non-square (n_occ != n_virt) case. ResponseExcitationSpace itself
    // lives in dft/driver.h, which this file deliberately does not link
    // (see the file header comment); its flat_index formula
    // (i*n_virt + a, driver.h) is reproduced literally here rather than
    // called, since the point of this check is only to confirm the two
    // FORMULAS disagree, not to exercise the struct.
    void check_conventions_genuinely_differ()
    {
        const int n_occ = 2;
        const int n_virt = 3;
        // ResponseExcitationSpace::flat_index(i,a) = i*n_virt + a (occupied-major).
        // CPHF idx(a,i) = a*n_occ + i (virtual-major).
        // Pick (i=0, a=1), where they differ: occupied-major: 0*3+1 = 1;
        // virtual-major: 1*2+0 = 2. ((i=1,a=2) would coincide at 5==5 by
        // accident, which is why this specific pair was chosen deliberately.)
        const int occ_major = 0 * n_virt + 1;
        const int virt_major = 1 * n_occ + 0;
        require(occ_major != virt_major,
                "sanity check failed: chosen (i,a) pair does not actually distinguish the "
                "two packing conventions -- fixture needs a different pair");
        require(occ_major == 1, "ResponseExcitationSpace::flat_index(0,1) should be 1");
        require(virt_major == 2, "CPHF idx(a=1,i=0) should be 2");
    }
} // namespace

int main()
{
    check_conventions_genuinely_differ();

    // Non-square (n_occ != n_virt) in every case, per the file's own
    // discipline: a square fixture cannot distinguish a row/column
    // transposition (i<->a swap) from correct packing.
    check_packing(/*nbasis=*/7, /*n_occ=*/2, /*n_virt=*/5, /*seed=*/1);
    check_packing(/*nbasis=*/10, /*n_occ=*/6, /*n_virt=*/4, /*seed=*/2);
    check_packing(/*nbasis=*/6, /*n_occ=*/1, /*n_virt=*/5, /*seed=*/3);
    check_packing(/*nbasis=*/12, /*n_occ=*/4, /*n_virt=*/8, /*seed=*/4);

    return g_ok ? 0 : 1;
}
