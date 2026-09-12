#ifndef DFT_DH_PT2_GRADIENT_H
#define DFT_DH_PT2_GRADIENT_H

#include <expected>
#include <functional>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "ao_grid.h"
#include "base/grid.h"
#include "base/wrapper.h"
#include "post_hf/mp2.h"

namespace DFT::Gradient
{
    struct DHPT2AmplitudeDensity;

    using DHResponseFn = std::function<std::expected<Eigen::MatrixXd, std::string>(
        const Eigen::Ref<const Eigen::MatrixXd> &)>;

    // Paper Eq. (41), in Planck's AO total-density convention.  `exchange`
    // supplies the raw K[D] channel; apply() explicitly forms K[D]+K[D]^T.
    // The separate channels prevent the closed-shell factors from being
    // hidden in a generic Fock builder.
    struct DHEq41ResponseOperator
    {
        double exact_exchange = 0.0;
        DHResponseFn coulomb;
        DHResponseFn exchange;
        DHResponseFn xc;

        [[nodiscard]] std::expected<Eigen::MatrixXd, std::string>
        apply(const Eigen::Ref<const Eigen::MatrixXd> &density) const;

        struct Channels
        {
            Eigen::MatrixXd coulomb;
            Eigen::MatrixXd exchange;
            Eigen::MatrixXd xc;
            Eigen::MatrixXd total;
        };

        [[nodiscard]] std::expected<Channels, std::string>
        apply_channels(const Eigen::Ref<const Eigen::MatrixXd> &density) const;
    };

    [[nodiscard]] std::expected<DHEq41ResponseOperator, std::string>
    make_dh_eq41_response_operator(
        double exact_exchange,
        DHResponseFn coulomb,
        DHResponseFn exchange,
        DHResponseFn xc);

    // Binds Eq. (41)'s raw J/K channels to Planck's memory-direct ERI
    // builders. The pointed-to shell-pair and symmetry data must outlive the
    // returned operator; XC remains an explicit fixed-geometry kernel action.
    struct DHEq41DirectERIInputs
    {
        const std::vector<HartreeFock::ShellPair> *shell_pairs = nullptr;
        std::size_t nbasis = 0;
        HartreeFock::IntegralMethod engine = HartreeFock::IntegralMethod::ObaraSaika;
        double tol_eri = 1e-10;
        const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr;
        double exact_exchange = 0.0;
        DHResponseFn xc;
    };

    [[nodiscard]] std::expected<DHEq41ResponseOperator, std::string>
    make_dh_eq41_direct_eri_response_operator(const DHEq41DirectERIInputs &inputs);

    // Fixed-geometry R_XC(D') action used by Eq. (41). The grid, AO values,
    // and functionals must outlive the returned callback; ground_density is
    // copied so the callback cannot observe later SCF-density mutation.
    struct DHEq41XCInputs
    {
        const MolecularGrid *molecular_grid = nullptr;
        const AOGridEvaluation *ao_grid = nullptr;
        Eigen::MatrixXd ground_density;
        const XC::Functional *exchange_functional = nullptr;
        const XC::Functional *correlation_functional = nullptr;
    };

    [[nodiscard]] std::expected<DHResponseFn, std::string>
    make_dh_eq41_xc_response_callback(const DHEq41XCInputs &inputs);

    // AO realization of Eq. (41).  `mo_coeff` has shape (nao, nocc+nvirt)
    // and maps the paper D' blocks into the AO density used by the response.
    struct DHEq41ResponseDensity
    {
        Eigen::MatrixXd dprime_ao;
        Eigen::MatrixXd response_ao;
    };

    [[nodiscard]] std::expected<DHEq41ResponseDensity, std::string>
    build_dh_eq41_response_density(
        const DHPT2AmplitudeDensity &dprime,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq41ResponseOperator &response);

    // Paper Eqs. (37)-(39), before response or derivative integrals.
    // All quantities are closed-shell spatial-orbital quantities and carry
    // the requested c_PT2 scale exactly once.
    struct DHPT2AmplitudeDensity
    {
        int n_occ = 0;
        int n_virt = 0;
        std::vector<double> t_tilde; // [i,j,a,b], Eq. (39)
        Eigen::MatrixXd dprime_oo;   // Eq. (37)
        Eigen::MatrixXd dprime_vv;   // Eq. (38)
        Eigen::MatrixXd dprime_mo;   // diagonal oo/vv blocks only
    };

    [[nodiscard]] std::expected<DHPT2AmplitudeDensity, std::string>
    build_dh_pt2_amplitude_density(
        const HartreeFock::Correlation::RMP2Result &result,
        double c_pt2);

    // Amplitude-dependent part of the closed-shell Eq. (40) Lagrangian RHS.
    // The dense MO ERI tensor uses chemists' ordering [p,q,r,s] = (pq|rs).
    // `c_pt2` scales the raw-amplitude second term; t_tilde already carries
    // the same scale from Eq. (39).  Both named parts are returned to keep
    // their signs and index patterns independently testable.
    struct DHEq40AmplitudeRHS
    {
        Eigen::MatrixXd three_external; // sum_jbc t~_ij^cb [(ac|jb)-(ab|jc)]
        // Literal closed-shell reduction of the two negative terms in the
        // unrestricted Eq. (22).  Keep the spin-origin channels distinct
        // until the final sum; their cancellation must not be hidden by an
        // amplitude-index relabeling.
        Eigen::MatrixXd internal_same_spin_exchange; // -c sum_klb t_kl^ab (ki|lb)
        Eigen::MatrixXd internal_opposite_spin_direct; // +c sum_klb t_kl^ab (kb|li)
        Eigen::MatrixXd three_internal; // sum of the two literal channels
        Eigen::MatrixXd total;
    };

    [[nodiscard]] std::expected<DHEq40AmplitudeRHS, std::string>
    build_dh_eq40_amplitude_rhs(
        const HartreeFock::Correlation::RMP2Result &result,
        const DHPT2AmplitudeDensity &amplitudes,
        const std::vector<double> &mo_eri,
        double c_pt2,
        bool include_legacy_internal = false);

    // Closed-shell Eq. (40) RHS in its solver orientation (virtual, occupied).
    // `response_ai` is the Eq. (41) AO response transformed as C_v^T R C_o;
    // Production uses the literal pair derivative without adding its
    // occupied-coefficient/internal contribution a second time. The legacy
    // convention is retained only for explicit standalone reference tests.
    enum class DHRHSConvention { LegacyExternalPlusInternal, LiteralEq47 };

    struct DHLagrangianRHS
    {
        Eigen::MatrixXd response_ai;
        Eigen::MatrixXd three_external_ai;
        Eigen::MatrixXd internal_same_spin_exchange_ai;
        Eigen::MatrixXd internal_opposite_spin_direct_ai;
        Eigen::MatrixXd three_internal_ai;
        // Raw internal spin channels above remain available for comparison.
        // Only this selected internal contribution enters amplitude_ai.
        Eigen::MatrixXd included_internal_ai;
        Eigen::MatrixXd amplitude_ai;
        Eigen::MatrixXd total_ai;
    };

    [[nodiscard]] std::expected<DHLagrangianRHS, std::string>
    build_dh_lagrangian_rhs(
        const DHEq41ResponseDensity &eq41,
        const Eigen::Ref<const Eigen::MatrixXd> &c_occ,
        const Eigen::Ref<const Eigen::MatrixXd> &c_virt,
        const DHEq40AmplitudeRHS &amplitude,
        DHRHSConvention convention = DHRHSConvention::LiteralEq47);

    // Eq. (28) in two non-interchangeable forms. `raw_mo` carries Z_ai only
    // in its virtual-occupied block. `symmetric_mo` halves that block into vo
    // and ov and is valid only for contraction with a symmetric operator.
    struct DHRelaxedDifferenceDensity
    {
        Eigen::MatrixXd z_ai;
        Eigen::MatrixXd raw_mo;
        Eigen::MatrixXd symmetric_mo;
    };

    [[nodiscard]] std::expected<DHRelaxedDifferenceDensity, std::string>
    build_dh_relaxed_difference_density(
        const DHPT2AmplitudeDensity &dprime,
        const Eigen::Ref<const Eigen::MatrixXd> &z_ai);

    // Closed-shell Eq. (27) action in virtual-major Z_ai orientation.  The
    // response density is raw C_v Z C_o^T, never the symmetric Eq. (28)
    // adapter. All returned channel blocks have shape (nvirt, nocc).
    struct DHEq27HessianAction
    {
        Eigen::MatrixXd raw_z_ao;
        Eigen::MatrixXd orbital_energy_ai;
        Eigen::MatrixXd coulomb_ai;
        Eigen::MatrixXd exchange_ai;
        Eigen::MatrixXd xc_ai;
        Eigen::MatrixXd response_ai;
        Eigen::MatrixXd total_ai;
    };

    [[nodiscard]] std::expected<DHEq27HessianAction, std::string>
    apply_dh_eq27_hessian(
        const Eigen::Ref<const Eigen::MatrixXd> &z_ai,
        const Eigen::Ref<const Eigen::MatrixXd> &c_occ,
        const Eigen::Ref<const Eigen::MatrixXd> &c_virt,
        const Eigen::Ref<const Eigen::VectorXd> &orbital_energies,
        const DHEq41ResponseOperator &response);

    // Eq. (42)-(43) diagonal energy-weighted PT2 density blocks.  Eq. (42)'s
    // response uses Eq. (28)'s symmetric relaxed density: the paper
    // symmetrizes D before the ensuing contractions with symmetric matrices.
    // Each term remains named so overlap-gradient assembly cannot hide its
    // prefactors.
    struct DHEq42_43EnergyWeightedDensity
    {
        Eigen::MatrixXd response_oo;
        Eigen::MatrixXd orbital_oo;
        Eigen::MatrixXd amplitude_oo;
        Eigen::MatrixXd w_oo;
        Eigen::MatrixXd orbital_vv;
        Eigen::MatrixXd amplitude_vv;
        Eigen::MatrixXd w_vv;
    };

    [[nodiscard]] std::expected<DHEq42_43EnergyWeightedDensity, std::string>
    build_dh_eq42_43_energy_weighted_density(
        const DHRelaxedDifferenceDensity &relaxed,
        const HartreeFock::Correlation::RMP2Result &result,
        const DHPT2AmplitudeDensity &amplitudes,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const Eigen::Ref<const Eigen::VectorXd> &orbital_energies,
        const std::vector<double> &mo_eri,
        const DHEq41ResponseOperator &response,
        bool include_legacy_pair = true);

    // Eqs. (44)-(45) raw off-diagonal energy-weighted blocks.  W_ia is
    // occupied-by-virtual; W_ai is virtual-by-occupied.  They must remain
    // separate until the dedicated symmetric overlap-contraction adapter.
    struct DHEq44_45EnergyWeightedDensity
    {
        Eigen::MatrixXd w_ia;
        Eigen::MatrixXd w_ai;
    };

    [[nodiscard]] std::expected<DHEq44_45EnergyWeightedDensity, std::string>
    build_dh_eq44_45_energy_weighted_density(
        const DHRelaxedDifferenceDensity &relaxed,
        const HartreeFock::Correlation::RMP2Result &result,
        const DHPT2AmplitudeDensity &amplitudes,
        const Eigen::Ref<const Eigen::VectorXd> &orbital_energies,
        const std::vector<double> &mo_eri,
        bool include_legacy_pair = true);

    // Literal four-coefficient metric derivative of the Eq. (47) pair
    // scalar.  These are raw paper-gauge blocks: U_ia=-S_ia^(x), U_ai=0.
    // This is the production pair-amplitude builder. The compressed blocks
    // above remain available to independent term-level tests; the production
    // contract skips their legacy pair work and uses these blocks instead.
    struct DHEq47PairMetricOverlapDensity
    {
        Eigen::MatrixXd w_oo;
        Eigen::MatrixXd w_vv;
        Eigen::MatrixXd w_ia;
        Eigen::MatrixXd w_ai;
    };

    [[nodiscard]] std::expected<DHEq47PairMetricOverlapDensity, std::string>
    build_dh_eq47_pair_metric_overlap_density(
        const HartreeFock::Correlation::RMP2Result &result,
        const DHPT2AmplitudeDensity &amplitudes,
        const std::vector<double> &mo_eri);

    // The only permitted raw-to-symmetric conversion for the Eq. (42)-(45)
    // overlap density. `symmetric_mo = 1/2 (raw_mo + raw_mo^T)` preserves its
    // contraction with any symmetric overlap derivative.
    struct DHOverlapDensity
    {
        Eigen::MatrixXd raw_mo;
        Eigen::MatrixXd symmetric_mo;
    };

    [[nodiscard]] std::expected<DHOverlapDensity, std::string>
    build_dh_overlap_density_adapter(
        const DHEq42_43EnergyWeightedDensity &diagonal,
        const DHEq44_45EnergyWeightedDensity &off_diagonal);

    // Eqs. (46)-(47), correction-only two-particle PT2 density in AO
    // chemists' ordering [mu,nu,kappa,tau].  The separable tensor contains
    // only D P - a_x/2 D P, with a_x from the same KS reference as the
    // response operator, never the SCF reference terms. The
    // nonseparable tensor is Eq. (47)'s direct T-tilde backtransformation.
    // `*_symmetric_ao` are their separate eightfold ERI-symmetry adapters;
    // use those, not the raw tensors, with a permutationally symmetric AO ERI
    // derivative.  Every tensor has n_ao^4 contiguous elements.
    struct DHEq46_47TwoParticleDensity
    {
        int n_ao = 0;
        Eigen::MatrixXd relaxed_difference_ao;
        std::vector<double> separable_raw_ao;
        std::vector<double> nonseparable_raw_ao;
        std::vector<double> total_raw_ao;
        std::vector<double> separable_symmetric_ao;
        std::vector<double> nonseparable_symmetric_ao;
        std::vector<double> total_symmetric_ao;
    };

    [[nodiscard]] std::expected<DHEq46_47TwoParticleDensity, std::string>
    build_dh_eq46_47_two_particle_density(
        const DHRelaxedDifferenceDensity &relaxed,
        const DHPT2AmplitudeDensity &amplitudes,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_density_ao,
        double exact_exchange);

    // Non-XC part of paper Eq. (33), evaluated coordinate by coordinate in
    // the AO basis.  All derivative vectors share one coordinate order; each
    // ERI derivative is an n_ao^4 tensor in chemists' ordering.  These are
    // correction-only derivatives: normal KS-gradient terms are excluded.
    struct DHEq33NonXCDerivatives
    {
        std::vector<Eigen::MatrixXd> hamiltonian_ao;
        std::vector<Eigen::MatrixXd> overlap_ao;
        std::vector<std::vector<double>> eri_ao;
    };

    struct DHEq33NonXCGradient
    {
        Eigen::VectorXd one_electron;
        Eigen::VectorXd overlap;
        Eigen::VectorXd two_electron_separable;
        Eigen::VectorXd two_electron_nonseparable;
        Eigen::VectorXd two_electron;
        Eigen::VectorXd total;
    };

    [[nodiscard]] std::expected<DHEq33NonXCGradient, std::string>
    build_dh_eq33_non_xc_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const DHOverlapDensity &overlap,
        const DHEq46_47TwoParticleDensity &two_particle,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33NonXCDerivatives &derivatives);

    // Explicit XC-II term in Eq. (33): D_relaxed : d V_XC[P] / dx, with
    // both the relaxed density and quadrature grid held fixed.  This is the
    // post-Z-vector operator derivative; XC-I (a derivative of D_relaxed)
    // and moving-grid/partition terms are intentionally excluded.
    struct DHEq33XCFixedDensityInputs
    {
        const HartreeFock::Molecule *molecule = nullptr;
        const HartreeFock::Basis *basis = nullptr;
        const MolecularGrid *molecular_grid = nullptr;
        const AOGridEvaluation *ao_grid = nullptr;
        const AOGridHessian *ao_hessian = nullptr;
        Eigen::MatrixXd ground_density_ao;
        const XC::Functional *exchange_functional = nullptr;
        const XC::Functional *correlation_functional = nullptr;
    };

    struct DHEq33XCFixedDensityGradient
    {
        Eigen::MatrixXd gradient; // natoms x 3, fixed-grid XC-II only
    };

    [[nodiscard]] std::expected<DHEq33XCFixedDensityGradient, std::string>
    build_dh_eq33_xc_fixed_density_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs);

    // LDA moving-grid part of the XC-II operator derivative.  The two
    // channels correspond, respectively, to derivative of the owner Becke
    // weight and translation of an atom-owned quadrature point.  Fixed-grid
    // XC-II and XC-I are deliberately not included here.
    struct DHEq33XCLDAMovingGridGradient
    {
        Eigen::MatrixXd becke_partition;
        Eigen::MatrixXd point_translation;
        Eigen::MatrixXd total;
    };

    [[nodiscard]] std::expected<DHEq33XCLDAMovingGridGradient, std::string>
    build_dh_eq33_xc_lda_moving_grid_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs);

    // LDA fixed-coefficient AO-centre derivative of the relaxed-D side of
    // D:V_XC[P].  D itself remains fixed, so this is not XC-I response.
    struct DHEq33XCLDADifferenceAOGradient
    {
        Eigen::MatrixXd gradient;
    };

    [[nodiscard]] std::expected<DHEq33XCLDADifferenceAOGradient, std::string>
    build_dh_eq33_xc_lda_difference_ao_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs);

    // GGA counterpart of the moving-grid XC-II term.  The point-translation
    // channel includes the spatial Hessians of rho_P and rho_D; it is kept
    // distinct from the Becke-weight derivative for independent validation.
    struct DHEq33XCGGAMovingGridGradient
    {
        Eigen::MatrixXd becke_partition;
        Eigen::MatrixXd point_translation;
        Eigen::MatrixXd total;
    };

    [[nodiscard]] std::expected<DHEq33XCGGAMovingGridGradient, std::string>
    build_dh_eq33_xc_gga_moving_grid_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs);

    // Fixed-coefficient AO-centre derivative of the relaxed-D side of
    // D:V_XC[P].  This is not XC-I: D itself is held fixed, while its AO
    // realization changes with nuclear geometry.  It completes a literal
    // geometry derivative together with the P-side fixed term and grid terms.
    struct DHEq33XCGGADifferenceAOGradient
    {
        Eigen::MatrixXd gradient;
    };

    [[nodiscard]] std::expected<DHEq33XCGGADifferenceAOGradient, std::string>
    build_dh_eq33_xc_gga_difference_ao_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs);

    // Complete, fixed-coefficient geometry derivative of D:V_XC[P].  Each
    // term remains observable for paper-level and finite-difference audits;
    // this object still excludes XC-I (the derivative of the relaxed D
    // matrix) and is not yet a final driver contribution.
    struct DHEq33CompleteXCIIGradient
    {
        Eigen::MatrixXd p_side_fixed;
        Eigen::MatrixXd d_side_ao;
        Eigen::MatrixXd becke_partition;
        Eigen::MatrixXd point_translation;
        Eigen::MatrixXd total;
    };

    [[nodiscard]] std::expected<DHEq33CompleteXCIIGradient, std::string>
    build_dh_eq33_complete_xc_ii_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs);

    // Literal fixed-total-density AO derivative of the restricted KS Fock
    // matrix, F^(x)_op = h^(x) + J^(x)[P] - 1/2 a_x K^(x)[P] +
    // (V_XC[P])^(x)_geom.  This is the explicit term of the co-moving-Fock
    // oracle, not an additional Eq. (33) gradient contribution.  The XC
    // matrix is reconstructed analytically from the complete XC-II scalar
    // derivative with symmetric AO test densities; this retains AO-centre,
    // Becke-partition, and moving-grid channels exactly, at O(n_AO^2) XC
    // contractions per Cartesian coordinate.  It is consequently an
    // audit/oracle primitive rather than the production large-basis path.
    struct DHKSFixedDensityFockDerivative
    {
        std::vector<Eigen::MatrixXd> hamiltonian;
        std::vector<Eigen::MatrixXd> coulomb;
        std::vector<Eigen::MatrixXd> exact_exchange;
        std::vector<Eigen::MatrixXd> xc_geometry;
        std::vector<Eigen::MatrixXd> total;
    };

    [[nodiscard]] std::expected<DHKSFixedDensityFockDerivative, std::string>
    build_dh_ks_fixed_density_fock_derivative(
        const Eigen::Ref<const Eigen::MatrixXd> &ground_density_ao,
        double exact_exchange_coefficient,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33NonXCDerivatives &derivatives,
        const DHEq33XCFixedDensityInputs &xc_inputs);

    // Driver boundary for the paper-defined, correction-only DH gradient
    // objects.  The caller owns the KS solve, direct response operator,
    // Z-vector solve, and derivative-integral preparation; this builder
    // validates those products and constructs every downstream Eq. (28),
    // (33), and (37)-(47) object exactly once.
    struct DHGradientDriverInputs
    {
        const HartreeFock::Correlation::RMP2Result *pt2_result = nullptr;
        double c_pt2 = 0.0;
        Eigen::MatrixXd mo_coeff;
        Eigen::VectorXd orbital_energies;
        std::vector<double> mo_eri;
        // Single source of a_x for both the KS response and Eq. (46)'s
        // separable D:F_KS two-electron derivative.
        DHEq41ResponseOperator response_operator;
        Eigen::MatrixXd z_ai;
        DHEq33NonXCDerivatives non_xc_derivatives;
        DHEq33XCFixedDensityInputs xc_inputs;
        // The contract always uses the validated literal Eq. (47) RHS and
        // pair metric blocks; there is no production legacy/gated fallback.
    };

    struct DHGradientDriverContract
    {
        DHPT2AmplitudeDensity amplitudes;
        DHEq41ResponseDensity eq41_response;
        DHEq40AmplitudeRHS eq40_amplitude_rhs;
        DHLagrangianRHS lagrangian_rhs;
        DHRelaxedDifferenceDensity relaxed_density;
        DHEq42_43EnergyWeightedDensity eq42_43_overlap;
        DHEq44_45EnergyWeightedDensity eq44_45_overlap;
        DHOverlapDensity overlap_density;
        DHEq46_47TwoParticleDensity two_particle_density;
        DHEq33NonXCGradient non_xc_gradient;
        DHEq33CompleteXCIIGradient xc_ii_gradient;
    };

    [[nodiscard]] std::expected<DHGradientDriverContract, std::string>
    build_dh_gradient_driver_contract(const DHGradientDriverInputs &inputs);

    // Final correction-only Eq. (33) sum. The normal KS gradient and nuclear
    // repulsion remain driver-owned; this object is only the scaled PT2
    // electronic correction to be added after the KS gradient is available.
    struct DHEq33PT2CorrectionGradient
    {
        Eigen::MatrixXd non_xc;
        Eigen::MatrixXd xc_ii;
        Eigen::MatrixXd total;
    };

    [[nodiscard]] std::expected<DHEq33PT2CorrectionGradient, std::string>
    build_dh_eq33_pt2_correction_gradient(const DHGradientDriverContract &contract);
}

#endif
