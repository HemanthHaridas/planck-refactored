#ifndef HF_POST_HF_ORBITAL_HESSIAN_H
#define HF_POST_HF_ORBITAL_HESSIAN_H

#include <vector>

#include <Eigen/Dense>

#include "base/types.h"

namespace HartreeFock::Correlation
{
    // The HF orbital Hessian (real rotations, A + B) as an operator: one Fock
    // build per action, through SCF::two_electron_fock{,_uhf}, the SCF's own
    // two-electron operator (fitted under scf_ri_jk, exact otherwise). The one
    // implementation behind the MP2 Z-vector (solve_rhf_cphf, solve_uhf_cphf)
    // and HF SOSCF (docs/SCF_ORBITAL_HESSIAN_OPERATOR.md).
    //
    // Rotations are packed a-major, x(a*nocc + i); UHF stacks [alpha | beta].
    //   RHF: (A x)_ai = (e_a - e_i) x_ai + [Cvᵀ · 2 G(D + Dᵀ) · Co]_ai,
    //        D = Cv x Coᵀ, G = J − ½K
    //        (= (e_a - e_i)δ + 4(ai|jb) − (ab|ji) − (aj|bi) applied to x).
    //   UHF: (A x)^σ_ai = (e_a - e_i) x^σ_ai + [Cv_σᵀ · G_σ · Co_σ]_ai,
    //        G_σ = J(Dα_s + Dβ_s) − K(Dσ_s), Dσ_s = Dσ + Dσᵀ.
    // Densities and MOs are in the working basis. The Calculator and shell
    // pairs are held by reference and must outlive the operator.
    class OrbitalHessian
    {
    public:
        static OrbitalHessian rhf(const HartreeFock::Calculator &calculator,
                                  const std::vector<HartreeFock::ShellPair> &shell_pairs,
                                  const Eigen::MatrixXd &C, const Eigen::VectorXd &eps, int n_occ);
        static OrbitalHessian uhf(const HartreeFock::Calculator &calculator,
                                  const std::vector<HartreeFock::ShellPair> &shell_pairs,
                                  const Eigen::MatrixXd &Ca, const Eigen::MatrixXd &Cb,
                                  const Eigen::VectorXd &eps_a, const Eigen::VectorXd &eps_b,
                                  int n_occ_a, int n_occ_b);

        // A x. A failed Fock build returns NaN, which poisons the caller's
        // solve visibly rather than silently.
        Eigen::VectorXd operator()(const Eigen::VectorXd &x) const;
        // The orbital-gap diagonal (e_a - e_i), the GMRES preconditioner.
        const Eigen::VectorXd &diagonal() const { return diagonal_; }
        Eigen::Index size() const { return diagonal_.size(); }
        // A, one action per column. Tests and debug dumps only.
        Eigen::MatrixXd materialize() const;

    private:
        OrbitalHessian(const HartreeFock::Calculator &calculator,
                       const std::vector<HartreeFock::ShellPair> &shell_pairs)
            : calculator_(calculator), shell_pairs_(shell_pairs) {}

        const HartreeFock::Calculator &calculator_;
        const std::vector<HartreeFock::ShellPair> &shell_pairs_;
        bool restricted_ = true;
        Eigen::MatrixXd Ca_occ_, Ca_virt_, Cb_occ_, Cb_virt_;
        Eigen::VectorXd diagonal_;
    };
} // namespace HartreeFock::Correlation

#endif // HF_POST_HF_ORBITAL_HESSIAN_H
