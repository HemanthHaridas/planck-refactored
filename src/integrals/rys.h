#ifndef HF_RYS_H
#define HF_RYS_H

#include <utility>
#include <vector>
#include <Eigen/Core>

#include "shellpair.h"
#include "base/types.h"

namespace HartreeFock 
{ 
    namespace RysQuad 
    {

    // L threshold: use Rys when (la+lb+lc+ld) >= this value.
    static constexpr int RYS_CROSSOVER_L = 4;

    // ── Primitive-level ───────────────────────────────────────────────────────
    //
    // Compute a single primitive (uncontracted) ERI (ab|cd) using Rys quadrature.
    // The caller is responsible for multiplying by ppAB.coeff_product * ppCD.coeff_product.
    double _rys_eri_primitive(
        const HartreeFock::PrimitivePair& ppAB,
        const HartreeFock::PrimitivePair& ppCD,
        int lAx, int lAy, int lAz,
        int lBx, int lBy, int lBz,
        int lCx, int lCy, int lCz,
        int lDx, int lDy, int lDz,
        double ABx, double ABy, double ABz,
        double CDx, double CDy, double CDz) noexcept;

    // ── Contracted shell quartet ───────────────────────────────────────────────
    //
    // Sum over all primitive pairs in spAB × spCD.
    double _rys_contracted_eri(
        const HartreeFock::ShellPair& spAB,
        const HartreeFock::ShellPair& spCD,
        int lAx, int lAy, int lAz,
        int lBx, int lBy, int lBz,
        int lCx, int lCy, int lCz,
        int lDx, int lDy, int lDz) noexcept;

    // ── Public API — mirrors ObaraSaika:: signatures ───────────────────────────

    // Build 2e Fock contribution G = J - 0.5*K (direct SCF, RHF).
    Eigen::MatrixXd _compute_2e_fock(
        const std::vector<HartreeFock::ShellPair>& shell_pairs,
        const Eigen::MatrixXd& density,
        std::size_t nbasis,
        double tol_eri = 1e-10,
        const std::vector<HartreeFock::SignedAOSymOp>* sym_ops = nullptr);

    // Build 2e Fock contribution (direct SCF, UHF). Returns {G_alpha, G_beta}.
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
    _compute_2e_fock_uhf(
        const std::vector<HartreeFock::ShellPair>& shell_pairs,
        const Eigen::MatrixXd& Pa,
        const Eigen::MatrixXd& Pb,
        std::size_t nbasis,
        double tol_eri = 1e-10,
        const std::vector<HartreeFock::SignedAOSymOp>* sym_ops = nullptr);

    // ── Auto-dispatch variant ──────────────────────────────────────────────────
    //
    // Selects OS for L < RYS_CROSSOVER_L, Rys for L >= RYS_CROSSOVER_L,
    // at the contracted shell-quartet level.

    Eigen::MatrixXd _compute_2e_fock_auto(
        const std::vector<HartreeFock::ShellPair>& shell_pairs,
        const Eigen::MatrixXd& density,
        std::size_t nbasis,
        double tol_eri = 1e-10,
        const std::vector<HartreeFock::SignedAOSymOp>* sym_ops = nullptr);

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
    _compute_2e_fock_uhf_auto(
        const std::vector<HartreeFock::ShellPair>& shell_pairs,
        const Eigen::MatrixXd& Pa,
        const Eigen::MatrixXd& Pb,
        std::size_t nbasis,
        double tol_eri = 1e-10,
        const std::vector<HartreeFock::SignedAOSymOp>* sym_ops = nullptr);

}} // namespace HartreeFock::RysQuad

#endif // HF_RYS_H
