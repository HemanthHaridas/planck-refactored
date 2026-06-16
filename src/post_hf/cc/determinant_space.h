#ifndef HF_POSTHF_CC_DETERMINANT_SPACE_H
#define HF_POSTHF_CC_DETERMINANT_SPACE_H

#include <expected>
#include <string>
#include <vector>

#include "integrals/shellpair.h"
#include "post_hf/cc/common.h"
#include "post_hf/cc/mo_blocks.h"

namespace HartreeFock::Correlation::CC
{
    // The determinant-space teaching solvers operate on one compact
    // spin-orbital Hamiltonian representation regardless of whether the
    // reference came from RHF or UHF. Keeping that representation explicit makes
    // the subsequent second-quantized algebra easier to explain.
    struct SpinOrbitalSystem
    {
        int n_spin_orb = 0;
        int n_electrons = 0;
        int n_occ = 0;
        int n_virt = 0;

        Eigen::VectorXd eps_occ;
        Eigen::VectorXd eps_virt;

        // H = sum_pq h_pq a_p^dagger a_q
        //   + 1/4 sum_pqrs <pq||rs> a_p^dagger a_q^dagger a_s a_r
        Tensor2D h1;
        Tensor4D g2;
    };

    // The hybrid moderate-size RCCSDT path can warm-start the determinant-space
    // solver from amplitudes produced by the tensor backend. The tensors use the
    // same spin-orbital occupied/virtual ordering as the production workspace.
    struct DeterminantCCSpinOrbitalSeed
    {
        const Tensor2D *t1 = nullptr;
        const Tensor4D *t2 = nullptr;
        const Tensor6D *t3 = nullptr;
    };

    std::expected<SpinOrbitalSystem, std::string> build_rhf_spin_orbital_system(
        const HartreeFock::Calculator &calculator,
        const RHFReference &reference,
        const MOBlockCache &mo_blocks);

    std::expected<SpinOrbitalSystem, std::string> build_uhf_spin_orbital_system(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const UHFReference &reference,
        const std::string &tag);

    // Solve the projected CC equations by building the similarity-transformed
    // Hamiltonian in the determinant basis and projecting it onto all unique
    // excitations up to `max_rank`. `max_rank=2` gives CCSD; `max_rank=3`
    // gives CCSDT.
    std::expected<double, std::string> solve_determinant_cc(
        HartreeFock::Calculator &calculator,
        const SpinOrbitalSystem &system,
        int max_rank,
        const std::string &log_tag,
        const DeterminantCCSpinOrbitalSeed *seed = nullptr);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_DETERMINANT_SPACE_H
