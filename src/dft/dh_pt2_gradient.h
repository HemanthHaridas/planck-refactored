#ifndef DFT_DH_PT2_GRADIENT_H
#define DFT_DH_PT2_GRADIENT_H

#include <expected>
#include <array>
#include <functional>
#include <memory>
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

    // Fixed-geometry R_XC(D') action used by Eq. (41). Preparation snapshots
    // AO/grid/ground fields and functional derivatives. The returned callback
    // owns its immutable cache; no caller object must remain alive. Construct
    // a new callback after changing geometry, grid, density or functionals.
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

    enum class DHTildeAmplitudeStorage
    {
        // Dense-reference/test adapter. Retains Eq. (39) explicitly.
        Materialized,
        // Production view. Refers to the canonical PT2 T2 storage and
        // evaluates Eq. (39) on demand, avoiding a second O(o^2 v^2) array.
        SourceView
    };

    // Bounded composite-pair view of Eq. (39). p=i*n_virt+a and
    // q=j*n_virt+b. The ordered tile intentionally assumes no amplitude
    // symmetry, so it remains valid for a recoverable source backend.
    struct DHTildeAmplitudePairTile
    {
        int n_virt = 0;
        int p0 = 0, q0 = 0;
        int rows = 0, cols = 0;
        std::vector<double> values; // Eq. (39), row-major [p-p0,q-q0]

        [[nodiscard]] double at(int i, int j, int a, int b) const;
    };

    using DHTildeAmplitudePairTileConsumer =
        std::function<bool(const DHTildeAmplitudePairTile &)>;

    // Paper Eqs. (37)-(39), before response or derivative integrals.
    // All quantities are closed-shell spatial-orbital quantities and carry
    // the requested c_PT2 scale exactly once.
    struct DHPT2AmplitudeDensity
    {
        int n_occ = 0;
        int n_virt = 0;
        std::vector<double> t_tilde; // [i,j,a,b], Eq. (39)
        // Representation is explicit: an empty materialized vector is
        // invalid, not an accidental alias for SourceView (important for a
        // one-amplitude tensor whose corruption can leave it empty).
        DHTildeAmplitudeStorage storage = DHTildeAmplitudeStorage::Materialized;
        const HartreeFock::Correlation::RMP2Result *source_pt2 = nullptr; // non-owning in SourceView
        double c_pt2 = 0.0;
        Eigen::MatrixXd dprime_oo;   // Eq. (37)
        Eigen::MatrixXd dprime_vv;   // Eq. (38)
        Eigen::MatrixXd dprime_mo;   // diagonal oo/vv blocks only

        [[nodiscard]] double tilde(int i, int j, int a, int b) const;
        [[nodiscard]] bool has_tilde_storage() const;
        [[nodiscard]] std::expected<DHTildeAmplitudePairTile, std::string>
        read_pair_tile(int p0, int q0, int rows, int cols) const;
        [[nodiscard]] std::expected<void, std::string>
        for_each_pair_tile(int pair_tile_width,
                           const DHTildeAmplitudePairTileConsumer &consumer) const;
    };

    // O7.6.5 ownership record for the production SourceView handoff. Scratch
    // file bytes are deliberately not resident bytes; the invariant is about
    // live RAM duplicates of canonical/tilde amplitudes.
    struct DHAmplitudeOwnershipAudit
    {
        std::size_t raw_t2_bytes = 0;
        std::size_t packed_t2_bytes = 0;
        std::size_t source_resident_bytes = 0;
        std::size_t tilde_bytes = 0;
        bool source_valid = false;
        [[nodiscard]] bool passes_production() const
        { return raw_t2_bytes==0 && packed_t2_bytes==0 && source_resident_bytes==0 &&
                 tilde_bytes==0 && source_valid; }
    };

    [[nodiscard]] DHAmplitudeOwnershipAudit audit_dh_amplitude_ownership(
        const HartreeFock::Correlation::RMP2Result &result,
        const DHPT2AmplitudeDensity &amplitudes);

    [[nodiscard]] std::expected<DHPT2AmplitudeDensity, std::string>
    build_dh_pt2_amplitude_density(
        const HartreeFock::Correlation::RMP2Result &result,
        double c_pt2, DHTildeAmplitudeStorage storage = DHTildeAmplitudeStorage::Materialized);

    // O7.5.1: exact MO-ERI sectors used by Eq. (40)'s four coefficient
    // slots. This is a lossless adapter over the dense MO tensor today;
    // production will later construct the sectors directly. Indices are
    // local to their named occupied/virtual spaces.
    struct DHMolecularERIBlocks
    {
        int n_occ = 0;
        int n_virt = 0;
        std::vector<double> ovov; // (i a | j b)
        std::vector<double> vvov; // (a b | i c)
        std::vector<double> ooov; // (i j | k a)
        std::vector<double> ovvv; // (i a | b c)
        std::vector<double> ovoo; // (i a | j k)
    };

    [[nodiscard]] std::expected<DHMolecularERIBlocks, std::string>
    build_dh_molecular_eri_blocks(const std::vector<double> &mo_eri, int n_occ, int n_virt);

    // O4: dE_pair = sum_pq G(p,q) U(p,q), with dC=C U and fixed
    // Theta_ijab=(1+delta_ij)t_tilde_ijab. Slots follow (i,a,j,b).
    // Occupied slots have shape (nmo,no), virtual slots (nmo,nv).
    // No ERI or amplitude symmetry is assumed. The reduction tile bounds
    // packing workspace, not the existing dense MO ERI/amplitude storage.
    struct DHPairCoefficientGradient
    {
        std::array<Eigen::MatrixXd, 4> slots;
        Eigen::MatrixXd total;
    };

    [[nodiscard]] std::expected<DHPairCoefficientGradient, std::string>
    build_dh_pair_coefficient_gradient(
        const DHPT2AmplitudeDensity &amplitudes,
        const std::vector<double> &mo_eri,
        int reduction_tile = 128);

    [[nodiscard]] std::expected<DHPairCoefficientGradient, std::string>
    build_dh_pair_coefficient_gradient(
        const DHPT2AmplitudeDensity &amplitudes,
        const DHMolecularERIBlocks &eri_blocks,
        int reduction_tile = 128);

    // Amplitude-dependent part of the closed-shell Eq. (40) Lagrangian RHS.
    // The dense MO ERI tensor uses chemists' ordering [p,q,r,s] = (pq|rs).
    // `c_pt2` scales the raw-amplitude second term; t_tilde already carries
    // the same scale from Eq. (39).  Both named parts are returned to keep
    // their signs and index patterns independently testable.
    struct DHEq40AmplitudeRHS
    {
        DHPairCoefficientGradient pair_coefficients; // reused by pair W
        Eigen::MatrixXd three_external; // G_vo - G_ov^T, all four slots
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

    // Production Eq. (40) uses LiteralEq47 and therefore never needs the
    // excluded legacy-internal bracket. The typed route rejects that bracket
    // rather than silently requiring an unlisted MO-ERI sector.
    [[nodiscard]] std::expected<DHEq40AmplitudeRHS, std::string>
    build_dh_eq40_amplitude_rhs(
        const HartreeFock::Correlation::RMP2Result &result,
        const DHPT2AmplitudeDensity &amplitudes,
        const DHMolecularERIBlocks &eri_blocks,
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

    // Cheap paper-gauge projection of the same O4 coefficient derivatives
    // used by Eq. 40; performs no amplitude/integral contraction.
    [[nodiscard]] std::expected<DHEq47PairMetricOverlapDensity, std::string>
    build_dh_eq47_pair_metric_overlap_density(
        const DHPairCoefficientGradient &coefficients, int n_occ, int n_virt);

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
    // `*_symmetric_ao` are separate eightfold ERI-symmetry adapters retained
    // by ReferenceAll. The direct production derivative can equivalently use
    // literal raw Eq. (47) against its symmetric quartet and Eq. (46)'s
    // quadratic D/P/a_x factorization. Every tensor has n_ao^4 contiguous
    // elements.
    struct DHEq46_47TwoParticleDensity
    {
        int n_ao = 0;
        Eigen::MatrixXd relaxed_difference_ao;
        // Eq. (46)'s production factorization.  In ContractionOnly mode the
        // direct ERI producer uses these O(N^2) objects instead of retaining
        // the separable symmetric O(N^4) tensor.
        Eigen::MatrixXd ground_density_ao;
        double exact_exchange = 0.0;
        std::vector<double> separable_raw_ao;
        std::vector<double> nonseparable_raw_ao;
        std::vector<double> total_raw_ao;
        std::vector<double> separable_symmetric_ao;
        std::vector<double> nonseparable_symmetric_ao;
        std::vector<double> total_symmetric_ao;
    };

    enum class DHGammaStorage { ReferenceAll, ContractionOnly };

    // O5: exact four-stage Eq. 47 backtransformation, (i,a,j,b) ->
    // (mu,nu,kappa,tau), replacing b,j,a,i successively. No symmetry or
    // orthonormality assumption on C. Theta=(1+delta_ij)*t_tilde carries
    // c_PT2 once. Column tiles bound GEMM panels, not quartic tensor storage.
    [[nodiscard]] std::expected<std::vector<double>, std::string>
    build_dh_eq47_nonseparable_raw_ao(
        const DHPT2AmplitudeDensity &amplitudes,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        int column_tile = 128);

    // O7.4.1: emit bounded mu-row tiles of the raw Eq. (47) AO transform.
    // Each tile is row-major [local_mu, nu, kappa, tau] and contains
    // mu_count*n_ao^3 values. The shared [i,nu,kappa,tau] partial transform
    // is built once; no full final [mu,nu,kappa,tau] output is allocated.
    using DHEq47RawAOTileConsumer = std::function<std::expected<void, std::string>(
        int mu_begin, int mu_count, const std::vector<double> &tile)>;

    [[nodiscard]] std::expected<void, std::string>
    for_each_dh_eq47_nonseparable_raw_ao_tile(
        const DHPT2AmplitudeDensity &amplitudes,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        int transform_tile, int mu_tile,
        const DHEq47RawAOTileConsumer &consumer);

    // O7.4.4: bounded four-index output tiles.  The producer streams
    // [local_mu, local_nu, local_kappa, local_tau] blocks and never creates
    // the former [i,nu,kappa,tau] penultimate tensor.
    using DHEq47RawAOBlockConsumer = std::function<std::expected<void, std::string>(
        int mu_begin, int mu_count, int nu_begin, int nu_count,
        int kappa_begin, int kappa_count, int tau_begin, int tau_count,
        const std::vector<double> &tile)>;

    // Optional conservative ledger for the O7.4.4 producer. It counts the
    // largest simultaneous vector payload, including a transform's input and
    // output while that replacement is in flight, plus an emitted mu tile.
    struct DHEq47BlockWorkspace
    {
        std::size_t peak_live_elements = 0;
        std::size_t peak_output_slab_elements = 0;
        std::size_t peak_emitted_tile_elements = 0;
    };

    [[nodiscard]] std::expected<void, std::string>
    for_each_dh_eq47_nonseparable_raw_ao_block(
        const DHPT2AmplitudeDensity &amplitudes,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        int occupied_tile, int mu_tile, int nu_tile, int kappa_tile, int tau_tile,
        const DHEq47RawAOBlockConsumer &consumer,
        DHEq47BlockWorkspace *workspace = nullptr);

    // O7.4.3: this non-owning context is valid only for the synchronous
    // Eq. (33) assembly call that receives it.  It intentionally is not a
    // member of DHEq46_47TwoParticleDensity or DHGradientDriverContract:
    // retaining it would either dangle after contract assembly or require a
    // duplicate O(o^2 v^2) amplitude allocation.
    struct DHEq47TiledExecution
    {
        const DHPT2AmplitudeDensity *amplitudes = nullptr;
        const Eigen::MatrixXd *mo_coeff = nullptr;
        int transform_tile = 128;
        int mu_tile = 32;
        int occupied_tile = 16;
        int nu_tile = 16;
        int kappa_tile = 16;
        int tau_tile = 16;
        // Optional call-scoped O7.6.5 ledger.  The direct consumer records
        // its bounded Eq. (47) workspace here; the pointer never escapes.
        DHEq47BlockWorkspace *workspace = nullptr;
    };

    [[nodiscard]] std::expected<DHEq46_47TwoParticleDensity, std::string>
    build_dh_eq46_47_two_particle_density(
        const DHRelaxedDifferenceDensity &relaxed,
        const DHPT2AmplitudeDensity &amplitudes,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_density_ao,
        double exact_exchange,
        DHGammaStorage storage = DHGammaStorage::ReferenceAll);

    // The h/S part of Eq. (33), after directly contracting AO derivative
    // primitives at every nuclear Cartesian coordinate.  It has the same
    // atom-major layout as the ERI derivative bundle, but deliberately does
    // not retain 3M dense AO matrices.
    struct DHEq33OneElectronOverlapGradient
    {
        Eigen::VectorXd one_electron;
        Eigen::VectorXd overlap;
    };

    using DHEq33OneElectronOverlapContraction = std::function<
        std::expected<DHEq33OneElectronOverlapGradient, std::string>(
            const Eigen::Ref<const Eigen::MatrixXd> &relaxed_ao,
            const Eigen::Ref<const Eigen::MatrixXd> &overlap_ao)>;

    // The separable Eq. (46) and nonseparable Eq. (47) ERI channels after
    // contracting a derivative quartet as it is evaluated.  Each vector has
    // one atom-major Cartesian entry per nuclear coordinate.
    struct DHEq33TwoElectronGradient
    {
        Eigen::VectorXd separable;
        Eigen::VectorXd nonseparable;
    };

    using DHEq33TwoElectronContraction = std::function<
        std::expected<DHEq33TwoElectronGradient, std::string>(
            const DHEq46_47TwoParticleDensity &two_particle,
            const DHEq47TiledExecution *eq47_tiled)>;

    // Non-XC part of paper Eq. (33), evaluated coordinate by coordinate in
    // the AO basis.  All derivative vectors share one coordinate order; each
    // ERI derivative is an n_ao^4 tensor in chemists' ordering.  The legacy
    // dense arrays are retained as a reference/test adapter. Production may
    // instead provide exactly one direct callback for h/S and for ERIs. These
    // are correction-only derivatives: normal KS-gradient terms are excluded.
    struct DHEq33NonXCDerivatives
    {
        std::vector<Eigen::MatrixXd> hamiltonian_ao;
        std::vector<Eigen::MatrixXd> overlap_ao;
        std::vector<std::vector<double>> eri_ao;
        DHEq33OneElectronOverlapContraction direct_one_electron_overlap;
        DHEq33TwoElectronContraction direct_two_electron;
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
        const DHEq33NonXCDerivatives &derivatives,
        const DHEq47TiledExecution *eq47_tiled = nullptr);

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
    // Immutable shared owner for the per-geometry derivative inputs. `mo_eri`
    // is populated only by explicit dense-reference callers; the production
    // literal Eq. (47) path carries no all-MO ERI tensor through this boundary.
    // Copying a driver view/probe never copies these arrays; no nonowning spans
    // escape.
    struct DHGradientGeometryWorkspace
    {
        std::vector<double> mo_eri;
        DHEq33NonXCDerivatives non_xc_derivatives;
    };

    struct DHGradientDriverInputs
    {
        const HartreeFock::Correlation::RMP2Result *pt2_result = nullptr;
        double c_pt2 = 0.0;
        Eigen::MatrixXd mo_coeff;
        Eigen::VectorXd orbital_energies;
        std::shared_ptr<const DHGradientGeometryWorkspace> workspace;
        // Single source of a_x for both the KS response and Eq. (46)'s
        // separable D:F_KS two-electron derivative.
        DHEq41ResponseOperator response_operator;
        Eigen::MatrixXd z_ai;
        DHEq33XCFixedDensityInputs xc_inputs;
        // The contract always uses the validated literal Eq. (47) RHS and
        // pair metric blocks; there is no production legacy/gated fallback.
    };

    struct DHStationaryProducts
    {
        // Snapshot binding for the move-only production handoff. The source
        // PT2/MO data must remain unchanged during this synchronous geometry.
        const HartreeFock::Correlation::RMP2Result *source_pt2 = nullptr;
        double c_pt2 = 0.0, exact_exchange = 0.0;
        Eigen::MatrixXd source_mo_coeff;
        DHPT2AmplitudeDensity amplitudes;
        DHEq41ResponseDensity eq41_response;
        DHEq40AmplitudeRHS eq40_amplitude_rhs;
        DHLagrangianRHS lagrangian_rhs;
    };

    [[nodiscard]] std::expected<DHStationaryProducts, std::string>
    build_dh_stationary_products(const HartreeFock::Correlation::RMP2Result &pt2,
        double c_pt2, const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const std::vector<double> &mo_eri, const DHEq41ResponseOperator &response,
        const DHMolecularERIBlocks *eq40_blocks = nullptr);

    // Compact production entry point. The caller provides exactly the five
    // Eq. (40) sectors; no dense all-MO tensor is retained or required.
    [[nodiscard]] std::expected<DHStationaryProducts, std::string>
    build_dh_stationary_products(const HartreeFock::Correlation::RMP2Result &pt2,
        double c_pt2, const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq41ResponseOperator &response,
        const DHMolecularERIBlocks &eq40_blocks,
        DHTildeAmplitudeStorage storage = DHTildeAmplitudeStorage::SourceView);

    struct DHGradientDriverContract : DHStationaryProducts
    {
        DHRelaxedDifferenceDensity relaxed_density;
        DHEq42_43EnergyWeightedDensity eq42_43_overlap;
        DHEq44_45EnergyWeightedDensity eq44_45_overlap;
        DHOverlapDensity overlap_density;
        DHEq46_47TwoParticleDensity two_particle_density;
        DHEq33NonXCGradient non_xc_gradient;
        DHEq33CompleteXCIIGradient xc_ii_gradient;
        // Snapshot of the production-relevant retained owners after Eq. (33).
        // It is data-only: reporting it cannot extend any workspace lifetime.
        DHAmplitudeOwnershipAudit amplitude_ownership;
        DHEq47BlockWorkspace eq47_workspace;
        std::size_t reference_mo_eri_bytes = 0;
        std::size_t materialized_eq47_bytes = 0;
    };

    [[nodiscard]] std::expected<DHGradientDriverContract, std::string>
    build_dh_gradient_driver_contract(const DHGradientDriverInputs &inputs);

    // Production consumes the same checked stationary products used by the
    // solve, by move. The one-argument overload independently rebuilds them
    // and retains all Gamma arrays as a small-system reference adapter.
    [[nodiscard]] std::expected<DHGradientDriverContract, std::string>
    build_dh_gradient_driver_contract(const DHGradientDriverInputs &inputs,
        DHStationaryProducts &&stationary);

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
