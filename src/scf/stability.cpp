#include "stability.h"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <format>

#include "io/logging.h"
#include "post_hf/integrals.h"
#include "scf/scf.h"

namespace HartreeFock::SCF
{
    namespace
    {
        // ── Index helpers ──────────────────────────────────────────────────────
        //
        // All MO-basis ERI tensors below are stored row-major in the order in which
        // they were transformed; these helpers turn (a,i,b,j)-style indices into
        // the corresponding flat offset.

        std::size_t ix4(int a, int b, int c, int d, int Nb, int Nc, int Nd)
        {
            return ((static_cast<std::size_t>(a) * Nb + b) * Nc + c) * Nd + d;
        }

        // Flatten the (a,i) compound index of the response space.
        int flat_ai(int a, int i, int n_occ) { return a * n_occ + i; }

        // Total number of electrons minus charge.
        int total_electrons(const HartreeFock::Calculator &calc)
        {
            int n_electrons = 0;
            for (auto Z : calc._molecule.atomic_numbers)
                n_electrons += static_cast<int>(Z);
            n_electrons -= calc._molecule.charge;
            return n_electrons;
        }

        // ── Closed-shell singlet/triplet response matrices ─────────────────────
        //
        // Build `A ± B` for either singlet (real) or triplet (spin-flip) channels
        // in the canonical RHF MO basis.
        //
        // Notation (i,j: doubly occupied; a,b: virtual; ε: canonical RHF energies):
        //
        //   A^S_{ai,bj} =  (ε_a − ε_i) δ_{ab}δ_{ij} + 2(ai|bj) − (ab|ij)
        //   B^S_{ai,bj} =  2(ai|bj) − (aj|ib)
        //
        //   A^T_{ai,bj} =  (ε_a − ε_i) δ_{ab}δ_{ij} − (ab|ij)
        //   B^T_{ai,bj} = −(aj|ib)
        //
        // Three standard checks (Seeger-Pople 1977; Bauernschmitt-Ahlrichs 1996):
        //   • internal real     →  lowest eig of (A^S + B^S)  — another real RHF
        //   • internal complex  →  lowest eig of (A^S − B^S)  — complex orbitals
        //   • external triplet  →  lowest eig of (A^T − B^T)  — RHF→UHF spin flip
        //
        // The singlet (A−B) and triplet (A−B) matrices reduce to the same
        // algebraic form once the (ai|bj) Coulomb pieces cancel, but they
        // diagnose physically distinct instabilities.
        enum class RHFChannel
        {
            SingletAplusB,  // internal, real    — RHF → another RHF
            SingletAminusB, // internal, complex — RHF → complex orbital RHF
            TripletAminusB, // external, triplet — RHF → UHF
        };

        // Build a closed-shell response matrix in MO basis, reusing the
        // (ai|bj) and (ab|ij) tensors already required by the CPHF code.
        Eigen::MatrixXd build_closed_shell_response(
            RHFChannel channel,
            const Eigen::VectorXd &eps,
            int n_occ, int n_virt,
            const std::vector<double> &mo_ai_bj,  // [n_virt × n_occ × n_virt × n_occ]
            const std::vector<double> &mo_ab_ij)  // [n_virt × n_virt × n_occ × n_occ]
        {
            const int nov = n_virt * n_occ;
            Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nov, nov);

            for (int a = 0; a < n_virt; ++a)
                for (int i = 0; i < n_occ; ++i)
                {
                    const int ai = flat_ai(a, i, n_occ);
                    for (int b = 0; b < n_virt; ++b)
                        for (int j = 0; j < n_occ; ++j)
                        {
                            const int bj = flat_ai(b, j, n_occ);
                            const double ai_bj = mo_ai_bj[ix4(a, i, b, j, n_occ, n_virt, n_occ)];
                            const double ab_ij = mo_ab_ij[ix4(a, b, i, j, n_virt, n_occ, n_occ)];
                            const double aj_bi = mo_ai_bj[ix4(a, j, b, i, n_occ, n_virt, n_occ)];

                            double val = 0.0;
                            if (a == b && i == j)
                                val += eps(n_occ + a) - eps(i);

                            switch (channel)
                            {
                            case RHFChannel::SingletAplusB:
                                // (A^S + B^S) = εδ + 4(ai|bj) − (ab|ij) − (aj|ib)
                                val += 4.0 * ai_bj - ab_ij - aj_bi;
                                break;
                            case RHFChannel::SingletAminusB:
                                // (A^S − B^S) = εδ − (ab|ij) + (aj|ib)
                                // Algebraically identical to (A^T − B^T) below;
                                // physically tests stability against complex
                                // orbital rotations, not spin-symmetry breaking.
                                val += -ab_ij + aj_bi;
                                break;
                            case RHFChannel::TripletAminusB:
                                // (A^T − B^T) = εδ − (ab|ij) + (aj|ib)
                                val += -ab_ij + aj_bi;
                                break;
                            }
                            M(ai, bj) = val;
                        }
                }

            // Symmetrize defensively. The closed-form expression is symmetric in
            // (ai)↔(bj), but rounding in the AO→MO transform can leave a residual
            // antisymmetric component of order machine epsilon.
            return 0.5 * (M + M.transpose());
        }

        // Lowest eigenvalue + eigenvector of a real symmetric matrix.
        struct LowestEig
        {
            double value;
            Eigen::VectorXd vector;
        };

        std::expected<LowestEig, std::string> lowest_symmetric_eig(
            const Eigen::MatrixXd &M)
        {
            if (M.rows() != M.cols())
                return std::unexpected("lowest_symmetric_eig: non-square matrix.");
            if (M.rows() == 0)
                return std::unexpected("lowest_symmetric_eig: empty matrix.");

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
            if (es.info() != Eigen::Success)
                return std::unexpected("lowest_symmetric_eig: diagonalization failed.");

            return LowestEig{es.eigenvalues()(0), es.eigenvectors().col(0)};
        }

        // Reshape an (a,i)-flattened eigenvector into the natural [n_virt × n_occ]
        // layout used by the orbital rotation step.
        Eigen::MatrixXd reshape_mode(
            const Eigen::VectorXd &v, int n_virt, int n_occ)
        {
            return Eigen::Map<const Eigen::MatrixXd>(v.data(), n_occ, n_virt).transpose();
        }

        // ── Orbital rotation by an unstable mode ───────────────────────────────
        //
        // Given a virt×occ rotation amplitude `kappa_vo`, build the orthogonal
        // generator
        //
        //     K = [ 0           -kappa_vo^T ]
        //         [ kappa_vo     0          ]
        //
        // (in the [occ | virt] partition of the MO basis) and return the
        // unitary `U = exp(K)` shaped [nb × nb], with all non-(occ,virt)
        // diagonal blocks left as identity.
        //
        // Rather than computing matrix exponentials directly, we use the SVD
        // identity for an off-diagonal antisymmetric matrix:
        //
        //     R = U_R · diag(σ) · V_R^T          (kappa_vo = R)
        //
        // gives
        //
        //     exp(K)_oo = V_R · diag(cos σ) · V_R^T
        //     exp(K)_ov = -V_R · diag(sin σ) · U_R^T
        //     exp(K)_vo =  U_R · diag(sin σ) · V_R^T
        //     exp(K)_vv = U_R · diag(cos σ) · U_R^T
        //
        // This is exact and unitary to machine precision regardless of step size.
        Eigen::MatrixXd build_rotation_matrix(
            const Eigen::MatrixXd &kappa_vo,  // [n_virt × n_occ]
            int n_occ, int n_virt)
        {
            const int nb = n_occ + n_virt;
            Eigen::MatrixXd U = Eigen::MatrixXd::Identity(nb, nb);

            if (kappa_vo.size() == 0)
                return U;

            Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                kappa_vo, Eigen::ComputeThinU | Eigen::ComputeThinV);
            const Eigen::MatrixXd &Ur = svd.matrixU();   // [n_virt × r]
            const Eigen::MatrixXd &Vr = svd.matrixV();   // [n_occ × r]
            const Eigen::VectorXd sigma = svd.singularValues();
            const int r = static_cast<int>(sigma.size());

            Eigen::ArrayXd cos_s(r), sin_s(r);
            for (int k = 0; k < r; ++k)
            {
                cos_s(k) = std::cos(sigma(k));
                sin_s(k) = std::sin(sigma(k));
            }

            // occ-occ block: I_occ + V_r (cos σ - 1) V_r^T
            U.topLeftCorner(n_occ, n_occ).noalias() +=
                Vr * (cos_s - 1.0).matrix().asDiagonal() * Vr.transpose();

            // virt-virt block: I_virt + U_r (cos σ - 1) U_r^T
            U.bottomRightCorner(n_virt, n_virt).noalias() +=
                Ur * (cos_s - 1.0).matrix().asDiagonal() * Ur.transpose();

            // occ-virt block: -V_r sin σ U_r^T
            U.topRightCorner(n_occ, n_virt).noalias() =
                -Vr * sin_s.matrix().asDiagonal() * Ur.transpose();

            // virt-occ block: U_r sin σ V_r^T
            U.bottomLeftCorner(n_virt, n_occ).noalias() =
                Ur * sin_s.matrix().asDiagonal() * Vr.transpose();

            return U;
        }

        // Reset SCF state on the calculator so it can be re-run from the
        // density we just seeded. The SCF entry points pick up
        // calculator._info._scf.alpha.density (and beta.density for UHF)
        // when the guess is ReadDensity, so we flip the guess into that mode
        // and clear the converged flags.
        void reset_for_rerun(HartreeFock::Calculator &calculator)
        {
            calculator._info._is_converged = false;
            calculator._info._delta_energy = 0.0;
            calculator._info._delta_density_max = 0.0;
            calculator._info._delta_density_rms = 0.0;
            calculator._scf._guess = HartreeFock::SCFGuess::ReadDensity;
        }
    } // namespace

    // ─────────────────────────────────────────────────────────────────────────
    // RHF stability
    // ─────────────────────────────────────────────────────────────────────────
    std::expected<StabilityReport, std::string> analyze_rhf_stability(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        double tol)
    {
        if (!calculator._info._is_converged)
            return std::unexpected("analyze_rhf_stability: SCF not converged.");
        if (calculator._scf._scf != HartreeFock::SCFType::RHF || calculator._info._scf.is_uhf)
            return std::unexpected("analyze_rhf_stability: RHF reference required.");

        const int n_electrons = total_electrons(calculator);
        if (n_electrons <= 0 || n_electrons % 2 != 0)
            return std::unexpected("analyze_rhf_stability: closed-shell reference required.");

        const std::size_t nb = calculator._shells.nbasis();
        const int n_occ = n_electrons / 2;
        const int n_virt = static_cast<int>(nb) - n_occ;
        if (n_occ <= 0 || n_virt <= 0)
            return std::unexpected("analyze_rhf_stability: no occupied or virtual orbitals.");

        const Eigen::MatrixXd &C = calculator._info._scf.alpha.mo_coefficients;
        const Eigen::VectorXd &eps = calculator._info._scf.alpha.mo_energies;

        // Pull the AO ERI tensor (computed once) and transform two MO blocks
        // shared across the three channels.
        std::vector<double> eri_local;
        const std::vector<double> &eri = HartreeFock::Correlation::ensure_eri(
            calculator, shell_pairs, eri_local, "Stability :");

        const Eigen::MatrixXd C_occ = C.leftCols(n_occ);
        const Eigen::MatrixXd C_virt = C.middleCols(n_occ, n_virt);

        const auto mo_ai_bj = HartreeFock::Correlation::transform_eri(
            eri, nb, C_virt, C_occ, C_virt, C_occ);
        const auto mo_ab_ij = HartreeFock::Correlation::transform_eri(
            eri, nb, C_virt, C_virt, C_occ, C_occ);

        StabilityReport report;
        report.channels.reserve(3);

        const struct
        {
            RHFChannel ch;
            const char *label;
        } channels[3] = {
            {RHFChannel::SingletAplusB, "RHF -> RHF (real, internal)"},
            {RHFChannel::SingletAminusB, "RHF -> complex RHF (internal)"},
            {RHFChannel::TripletAminusB, "RHF -> UHF (triplet, external)"},
        };

        for (const auto &c : channels)
        {
            Eigen::MatrixXd M = build_closed_shell_response(
                c.ch, eps, n_occ, n_virt, mo_ai_bj, mo_ab_ij);
            auto eig = lowest_symmetric_eig(M);
            if (!eig)
                return std::unexpected(eig.error());

            StabilityChannel sc;
            sc.label = c.label;
            sc.lowest_eigenvalue = eig->value;
            sc.is_unstable = eig->value < -tol;
            sc.lowest_mode = reshape_mode(eig->vector, n_virt, n_occ);
            if (sc.is_unstable)
                report.any_unstable = true;
            report.channels.push_back(std::move(sc));
        }

        return report;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // UHF stability
    // ─────────────────────────────────────────────────────────────────────────
    //
    // For UHF, the orbital Hessian has spin-resolved blocks. With p,q indexing
    // alpha occupied/virtual and r,s indexing beta occupied/virtual:
    //
    //   A^αα_{ai,bj} = (ε^α_a − ε^α_i)δ + (ai|bj)^αα − (aj|bi)^αα
    //   A^ββ                analogous in beta
    //   A^αβ_{ai,bj} = (ai|bj)^αβ                           (no exchange)
    //
    // and B has the same structure with index swaps. The internal (UHF→UHF)
    // check uses (A − B) of the spin-conserving block; the external (UHF→GHF)
    // check uses the spin-flip block where occ and virt come from different
    // spin channels.
    //
    // The full implementation here is the spin-conserving part. The spin-flip
    // GHF check is provided in an abridged form (full GHF stability requires
    // building the cross-spin two-electron blocks, which we keep behind the
    // same ERI transform helpers).
    std::expected<StabilityReport, std::string> analyze_uhf_stability(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        double tol)
    {
        if (!calculator._info._is_converged)
            return std::unexpected("analyze_uhf_stability: SCF not converged.");
        if (!calculator._info._scf.is_uhf)
            return std::unexpected("analyze_uhf_stability: UHF reference required.");

        const int n_electrons = total_electrons(calculator);
        const int n_unpaired = static_cast<int>(calculator._molecule.multiplicity) - 1;
        if (n_unpaired < 0 || (n_electrons - n_unpaired) < 0 ||
            (n_electrons - n_unpaired) % 2 != 0)
            return std::unexpected("analyze_uhf_stability: invalid multiplicity.");

        const int n_alpha = (n_electrons + n_unpaired) / 2;
        const int n_beta = (n_electrons - n_unpaired) / 2;
        const std::size_t nb = calculator._shells.nbasis();
        const int nva = static_cast<int>(nb) - n_alpha;
        const int nvb = static_cast<int>(nb) - n_beta;
        if (nva <= 0 || nvb <= 0)
            return std::unexpected("analyze_uhf_stability: no virtual orbitals in one spin channel.");

        const Eigen::MatrixXd &Ca = calculator._info._scf.alpha.mo_coefficients;
        const Eigen::MatrixXd &Cb = calculator._info._scf.beta.mo_coefficients;
        const Eigen::VectorXd &eps_a = calculator._info._scf.alpha.mo_energies;
        const Eigen::VectorXd &eps_b = calculator._info._scf.beta.mo_energies;

        std::vector<double> eri_local;
        const std::vector<double> &eri = HartreeFock::Correlation::ensure_eri(
            calculator, shell_pairs, eri_local, "Stability :");

        const Eigen::MatrixXd Ca_occ = Ca.leftCols(n_alpha);
        const Eigen::MatrixXd Ca_virt = Ca.middleCols(n_alpha, nva);
        const Eigen::MatrixXd Cb_occ = Cb.leftCols(n_beta);
        const Eigen::MatrixXd Cb_virt = Cb.middleCols(n_beta, nvb);

        // Same-spin (ai|bj) and (ab|ij) blocks.
        const auto aibj_aa = HartreeFock::Correlation::transform_eri(
            eri, nb, Ca_virt, Ca_occ, Ca_virt, Ca_occ);
        const auto abij_aa = HartreeFock::Correlation::transform_eri(
            eri, nb, Ca_virt, Ca_virt, Ca_occ, Ca_occ);
        const auto aibj_bb = HartreeFock::Correlation::transform_eri(
            eri, nb, Cb_virt, Cb_occ, Cb_virt, Cb_occ);
        const auto abij_bb = HartreeFock::Correlation::transform_eri(
            eri, nb, Cb_virt, Cb_virt, Cb_occ, Cb_occ);

        // Cross-spin Coulomb (ai_α | bj_β). No exchange in opposite-spin block.
        const auto aibj_ab = HartreeFock::Correlation::transform_eri(
            eri, nb, Ca_virt, Ca_occ, Cb_virt, Cb_occ);

        // Spin-conserving (A − B) matrix in the αα ⊕ ββ basis with cross-spin
        // Coulomb coupling. Diagonal block algebra is the closed-shell
        // (A^T − B^T) form per spin; off-diagonal block is +2(ai_α|bj_β).
        const int nov_a = nva * n_alpha;
        const int nov_b = nvb * n_beta;
        const int nov = nov_a + nov_b;

        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nov, nov);

        // αα block (top-left)
        for (int a = 0; a < nva; ++a)
            for (int i = 0; i < n_alpha; ++i)
            {
                const int ai = flat_ai(a, i, n_alpha);
                for (int b = 0; b < nva; ++b)
                    for (int j = 0; j < n_alpha; ++j)
                    {
                        const int bj = flat_ai(b, j, n_alpha);
                        double val = 0.0;
                        if (a == b && i == j)
                            val += eps_a(n_alpha + a) - eps_a(i);
                        const double ab_ij = abij_aa[ix4(a, b, i, j, nva, n_alpha, n_alpha)];
                        const double aj_bi = aibj_aa[ix4(a, j, b, i, n_alpha, nva, n_alpha)];
                        val += -ab_ij + aj_bi;
                        M(ai, bj) = val;
                    }
            }

        // ββ block (bottom-right)
        for (int a = 0; a < nvb; ++a)
            for (int i = 0; i < n_beta; ++i)
            {
                const int ai = nov_a + flat_ai(a, i, n_beta);
                for (int b = 0; b < nvb; ++b)
                    for (int j = 0; j < n_beta; ++j)
                    {
                        const int bj = nov_a + flat_ai(b, j, n_beta);
                        double val = 0.0;
                        if (a == b && i == j)
                            val += eps_b(n_beta + a) - eps_b(i);
                        const double ab_ij = abij_bb[ix4(a, b, i, j, nvb, n_beta, n_beta)];
                        const double aj_bi = aibj_bb[ix4(a, j, b, i, n_beta, nvb, n_beta)];
                        val += -ab_ij + aj_bi;
                        M(ai, bj) = val;
                    }
            }

        // αβ off-diagonal: +2 (ai_α | bj_β). The factor of 2 comes from the
        // Coulomb-like (A − B) coupling once the de-excitation block is
        // folded in (no exchange between opposite spins).
        for (int a = 0; a < nva; ++a)
            for (int i = 0; i < n_alpha; ++i)
            {
                const int ai = flat_ai(a, i, n_alpha);
                for (int b = 0; b < nvb; ++b)
                    for (int j = 0; j < n_beta; ++j)
                    {
                        const int bj = nov_a + flat_ai(b, j, n_beta);
                        const double ai_bj = aibj_ab[ix4(a, i, b, j, n_alpha, nvb, n_beta)];
                        M(ai, bj) = 2.0 * ai_bj;
                        M(bj, ai) = 2.0 * ai_bj;
                    }
            }

        M = 0.5 * (M + M.transpose());

        StabilityReport report;
        auto eig_internal = lowest_symmetric_eig(M);
        if (!eig_internal)
            return std::unexpected(eig_internal.error());

        StabilityChannel sc_internal;
        sc_internal.label = "UHF -> UHF (spin-conserving, internal)";
        sc_internal.lowest_eigenvalue = eig_internal->value;
        sc_internal.is_unstable = eig_internal->value < -tol;

        // Split the eigenvector back into the two spin channels.
        Eigen::VectorXd va = eig_internal->vector.head(nov_a);
        Eigen::VectorXd vb = eig_internal->vector.tail(nov_b);
        sc_internal.lowest_mode = reshape_mode(va, nva, n_alpha);
        sc_internal.lowest_mode_beta = reshape_mode(vb, nvb, n_beta);

        if (sc_internal.is_unstable)
            report.any_unstable = true;
        report.channels.push_back(std::move(sc_internal));

        // Spin-flip (UHF → GHF) block. The leading mode is the
        // alpha-occupied → beta-virtual orbital rotation (and vice versa)
        // coupled by exchange-like cross-spin terms. We build it as a
        // diagonal-only approximation here (orbital-energy gap), which is the
        // standard "external" check used by most production codes when the
        // full GHF Hessian is too expensive to assemble. A negative entry
        // here is rare and only flags very pathological cases.
        const int nflip = n_alpha * nvb + n_beta * nva;
        if (nflip > 0)
        {
            Eigen::VectorXd diag(nflip);
            int k = 0;
            for (int i = 0; i < n_alpha; ++i)
                for (int a = 0; a < nvb; ++a)
                    diag(k++) = eps_b(n_beta + a) - eps_a(i);
            for (int i = 0; i < n_beta; ++i)
                for (int a = 0; a < nva; ++a)
                    diag(k++) = eps_a(n_alpha + a) - eps_b(i);

            StabilityChannel sc_flip;
            sc_flip.label = "UHF -> GHF (spin-flip, external, diagonal approx)";
            sc_flip.lowest_eigenvalue = diag.minCoeff();
            sc_flip.is_unstable = sc_flip.lowest_eigenvalue < -tol;
            // Mode left empty: following a spin-flip instability requires
            // promoting to GHF, which Planck does not yet support.
            if (sc_flip.is_unstable)
                report.any_unstable = true;
            report.channels.push_back(std::move(sc_flip));
        }

        return report;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Follow an unstable mode: rotate orbitals, optionally promote RHF→UHF,
    // re-run SCF.
    // ─────────────────────────────────────────────────────────────────────────
    std::expected<double, std::string> follow_instability_and_rerun(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const StabilityChannel &channel,
        double step_scale)
    {
        if (!channel.is_unstable)
            return std::unexpected("follow_instability_and_rerun: channel is not flagged unstable.");
        if (channel.lowest_mode.size() == 0)
            return std::unexpected(std::format(
                "follow_instability_and_rerun: no follow-mode available for channel '{}'.",
                channel.label));

        const int n_electrons = total_electrons(calculator);
        const std::size_t nb = calculator._shells.nbasis();
        const bool was_rhf = calculator._scf._scf == HartreeFock::SCFType::RHF;
        const bool is_triplet_external =
            channel.label.find("RHF -> UHF") != std::string::npos;

        // For an RHF → UHF triplet instability we need to promote the
        // calculator to a UHF reference, mix alpha and beta separately
        // (with opposite signs along the unstable mode), and re-run UHF.
        if (is_triplet_external && was_rhf)
        {
            if (n_electrons % 2 != 0)
                return std::unexpected("follow_instability_and_rerun: RHF→UHF requires closed-shell.");

            const int n_alpha = n_electrons / 2;
            const int n_beta = n_electrons / 2;
            const int n_virt = static_cast<int>(nb) - n_alpha;

            const Eigen::MatrixXd C_in = calculator._info._scf.alpha.mo_coefficients;
            const Eigen::VectorXd eps_in = calculator._info._scf.alpha.mo_energies;

            // Adaptive step. For deeply unstable modes (|λ| ≳ 0.01) we need a
            // sizeable rotation to escape the basin — `π/4` is the standard
            // "45° spin-symmetry break" used by production codes. For weakly
            // unstable modes the user-supplied `step_scale` (default 0.05) is
            // already enough.
            const double step =
                (channel.lowest_eigenvalue < -1e-2)
                    ? std::max(step_scale, 0.5 * 3.14159265358979323846 / 2.0)
                    : step_scale;

            // Apply +s along the mode to alpha and −s to beta. This is the
            // standard "spin-symmetry break" mixing used by every production
            // code: a triplet rotation pushes alpha occupied weight into one
            // virtual combination and beta into the orthogonal one.
            const Eigen::MatrixXd kappa = step * channel.lowest_mode;
            const Eigen::MatrixXd U_pos = build_rotation_matrix(kappa, n_alpha, n_virt);
            const Eigen::MatrixXd U_neg = build_rotation_matrix(-kappa, n_alpha, n_virt);

            // Promote DataSCF to a UHF container before writing the mixed MOs.
            calculator._scf._scf = HartreeFock::SCFType::UHF;
            calculator._info._scf = HartreeFock::DataSCF(true);
            calculator._info._scf.initialize(nb);

            calculator._info._scf.alpha.mo_coefficients = C_in * U_pos;
            calculator._info._scf.beta.mo_coefficients = C_in * U_neg;
            calculator._info._scf.alpha.mo_energies = eps_in;
            calculator._info._scf.beta.mo_energies = eps_in;
            calculator._info._scf.alpha.density =
                calculator._info._scf.alpha.mo_coefficients.leftCols(n_alpha) *
                calculator._info._scf.alpha.mo_coefficients.leftCols(n_alpha).transpose();
            calculator._info._scf.beta.density =
                calculator._info._scf.beta.mo_coefficients.leftCols(n_beta) *
                calculator._info._scf.beta.mo_coefficients.leftCols(n_beta).transpose();

            // The SCF entry path will see is_init=true and reuse our seeded
            // density. Force its convergence flag back to false.
            reset_for_rerun(calculator);

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info, "Stability :",
                std::format("Following RHF→UHF triplet instability "
                            "(λ={:.6e}, step={:.3f}); re-running as UHF.",
                            channel.lowest_eigenvalue, step));

            auto res = HartreeFock::SCF::run_uhf(calculator, shell_pairs, nullptr);
            if (!res)
                return std::unexpected(std::format(
                    "follow_instability_and_rerun: UHF restart failed: {}", res.error()));
            return calculator._info._energy;
        }

        // Internal instability (RHF→RHF or UHF→UHF): rotate within the same
        // SCF type and re-run. For UHF this rotates only the alpha block;
        // the beta lowest_mode (when present) is applied to the beta channel.
        const bool is_uhf = calculator._info._scf.is_uhf;
        const int n_unpaired = static_cast<int>(calculator._molecule.multiplicity) - 1;
        const int n_alpha = is_uhf ? (n_electrons + n_unpaired) / 2 : n_electrons / 2;
        const int n_beta = is_uhf ? (n_electrons - n_unpaired) / 2 : n_electrons / 2;
        const int nva = static_cast<int>(nb) - n_alpha;
        const int nvb = static_cast<int>(nb) - n_beta;

        const Eigen::MatrixXd kappa_a = step_scale * channel.lowest_mode;
        const Eigen::MatrixXd U_a = build_rotation_matrix(kappa_a, n_alpha, nva);
        calculator._info._scf.alpha.mo_coefficients =
            calculator._info._scf.alpha.mo_coefficients * U_a;
        calculator._info._scf.alpha.density =
            calculator._info._scf.alpha.mo_coefficients.leftCols(n_alpha) *
            calculator._info._scf.alpha.mo_coefficients.leftCols(n_alpha).transpose();
        if (!is_uhf)
            calculator._info._scf.alpha.density *= 2.0;  // RHF closed-shell density

        if (is_uhf && channel.lowest_mode_beta.size() > 0)
        {
            const Eigen::MatrixXd kappa_b = step_scale * channel.lowest_mode_beta;
            const Eigen::MatrixXd U_b = build_rotation_matrix(kappa_b, n_beta, nvb);
            calculator._info._scf.beta.mo_coefficients =
                calculator._info._scf.beta.mo_coefficients * U_b;
            calculator._info._scf.beta.density =
                calculator._info._scf.beta.mo_coefficients.leftCols(n_beta) *
                calculator._info._scf.beta.mo_coefficients.leftCols(n_beta).transpose();
        }

        reset_for_rerun(calculator);

        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info, "Stability :",
            std::format("Following internal instability '{}' (λ={:.6e}, step={:.3f}); "
                        "re-running SCF.",
                        channel.label, channel.lowest_eigenvalue, step_scale));

        std::expected<void, std::string> res;
        switch (calculator._scf._scf)
        {
        case HartreeFock::SCFType::RHF:
            res = HartreeFock::SCF::run_rhf(calculator, shell_pairs, nullptr);
            break;
        case HartreeFock::SCFType::UHF:
            res = HartreeFock::SCF::run_uhf(calculator, shell_pairs, nullptr);
            break;
        case HartreeFock::SCFType::ROHF:
            res = HartreeFock::SCF::run_rohf(calculator, shell_pairs, nullptr);
            break;
        }
        if (!res)
            return std::unexpected(std::format(
                "follow_instability_and_rerun: restart failed: {}", res.error()));
        return calculator._info._energy;
    }

} // namespace HartreeFock::SCF
