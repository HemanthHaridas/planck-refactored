#ifndef HF_THO_H
#define HF_THO_H

#include <utility>
#include <vector>
#include <Eigen/Core>

#include "shellpair.h"
#include "base/types.h"

// ─── Huzinaga direct-expansion integral engine ────────────────────────────────
//
// Computes molecular integrals via explicit binomial expansion of Cartesian
// Gaussian angular-momentum factors, as described by Huzinaga (1965) and
// Tanaka, Huzinaga & Omata (1976).
//
// Algorithm overview
// ──────────────────
// Gaussian product theorem:
//   (x−Ax)^lA exp(−α(x−Ax)²) · (x−Bx)^lB exp(−β(x−Bx)²)
//     = K_AB · Σᵢ Σⱼ C(lA,i)·PAˡᴬ⁻ⁱ · C(lB,j)·PBˡᴮ⁻ʲ · (x−Px)^(i+j) exp(−ζ(x−Px)²)
//
// Integrating over all x, odd powers vanish and even powers give the
// normalised Gaussian moment  M_norm(n,ζ) = (n−1)!! / (2ζ)^(n/2).
//
// The full 3-D overlap primitive is then:
//   S_prim = coeff_product × pp.prefactor × Sx × Sy × Sz
// where Sx = _tho_1d_overlap(lAx, lBx, PAx, PBx, ζ)  [= 1 for s-s].
//
// Kinetic energy uses the differentiation identity applied to the B Gaussian:
//   T_x = β(2lB+1)·S(lA,lB) − 2β²·S(lA,lB+2) − ½lB(lB−1)·S(lA,lB−2)
//
// Nuclear attraction and ERI are implemented in Phases 3–4 (stubs below).

namespace HartreeFock
{
    namespace Huzinaga
    {
        // ── Exposed helpers ───────────────────────────────────────────────────

        // Full Gaussian moment: M(n,ζ) = ∫_{-∞}^{+∞} u^n exp(−ζu²) du
        //   = (n−1)!! / (2ζ)^(n/2) · sqrt(π/ζ)   for even n  ((-1)!! ≡ 1)
        //   = 0                                     for odd n
        double _gaussian_moment(int n, double zeta);

        // 1D overlap via explicit binomial expansion.
        // Returns the *reduced* integral (S(0,0) = 1), consistent with the OS
        // convention — the physical prefactor K_AB·(π/ζ)^(3/2) is in pp.prefactor.
        //
        //   result = Σᵢ₌₀^lA Σⱼ₌₀^lB  C(lA,i)·PA^(lA−i) · C(lB,j)·PB^(lB−j)
        //            · M_norm(i+j, ζ)
        // where M_norm(n,ζ) = (n−1)!! / (2ζ)^(n/2)  for even n, 0 for odd n.
        double _tho_1d_overlap(int lA, int lB, double PA, double PB, double zeta);

        // ── Public interface (mirrors os.h exactly) ───────────────────────────

        std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
            _compute_1e(const std::vector<HartreeFock::ShellPair>& shell_pairs,
                        std::size_t nbasis);

        // Phase 3 stub — nuclear attraction via Boys function + R-integral recursion
        Eigen::MatrixXd
            _compute_nuclear_attraction(const std::vector<HartreeFock::ShellPair>& shell_pairs,
                                        std::size_t nbasis,
                                        const HartreeFock::Molecule& molecule);

        // Phase 4 stub — 4-centre ERI via Boys function + binomial expansion
        Eigen::MatrixXd
            _compute_2e_fock(const std::vector<HartreeFock::ShellPair>& shell_pairs,
                             const Eigen::MatrixXd& density,
                             std::size_t nbasis);

        // Phase 4 UHF stub
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
            _compute_2e_fock_uhf(const std::vector<HartreeFock::ShellPair>& shell_pairs,
                                 const Eigen::MatrixXd& Pa,
                                 const Eigen::MatrixXd& Pb,
                                 std::size_t nbasis);
    }
}

#endif // !HF_THO_H
