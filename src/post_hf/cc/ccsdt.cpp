#include "post_hf/cc/ccsdt.h"

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <limits>
#include <unordered_map>
#include <vector>

#include "io/logging.h"
#include "post_hf/casscf/strings.h"
#include "post_hf/casscf_internal.h"
#include "post_hf/cc/diis.h"

namespace
{
    using HartreeFock::Correlation::CASSCF::apply_annihilation;
    using HartreeFock::Correlation::CASSCF::apply_creation;
    using HartreeFock::Correlation::CASSCF::generate_strings;
    using HartreeFock::Correlation::CASSCFInternal::CIString;
    using HartreeFock::Correlation::CASSCFInternal::low_bit_mask;
    using HartreeFock::Correlation::CC::AmplitudeDIIS;
    using HartreeFock::Correlation::CC::MOBlockCache;
    using HartreeFock::Correlation::CC::RCCSDTState;
    using HartreeFock::Correlation::CC::RHFReference;
    using HartreeFock::Correlation::CC::Tensor2D;
    using HartreeFock::Correlation::CC::Tensor4D;

    // This RCCSDT path is deliberately aimed at small teaching examples.
    // Instead of hard-coding hundreds of T3 diagram contractions, it evaluates
    // the similarity-transformed Hamiltonian in a compact determinant basis.
    // The result is easy to inspect and mathematically faithful, but it is not
    // intended to scale to production-sized orbital spaces.
    constexpr int kMaxSpinOrbitals = 12;
    constexpr int kMaxDeterminants = 1200;

    struct SpinOrbitalSystem
    {
        int n_spin_orb = 0;
        int n_electrons = 0;
        int n_occ = 0;
        int n_virt = 0;

        // Canonical orbital energies are reused for the diagonal preconditioner.
        Eigen::VectorXd eps_occ;
        Eigen::VectorXd eps_virt;

        // One- and two-body Hamiltonian pieces in the spin-orbital basis:
        // H = sum_pq h_pq a_p^dagger a_q
        //   + 1/4 sum_pqrs <pq||rs> a_p^dagger a_q^dagger a_s a_r
        Tensor2D h1;
        Tensor4D g2;
    };

    struct DeterminantOpResult
    {
        CIString det = 0;
        double phase = 0.0;
        bool valid = false;
    };

    struct DeterminantSpace
    {
        std::vector<CIString> determinants;
        std::unordered_map<CIString, int> lookup;
        Eigen::MatrixXd hamiltonian;
        CIString reference_det = 0;
        int reference_index = -1;
    };

    struct Excitation
    {
        int rank = 0;
        std::vector<int> occupied;
        std::vector<int> virtuals;
        double denominator = 0.0;
        double reference_phase = 0.0;
        int determinant_index = -1;
    };

    [[nodiscard]] int spatial_index(int so_index) noexcept
    {
        return so_index / 2;
    }

    [[nodiscard]] int spin_index(int so_index) noexcept
    {
        return so_index % 2;
    }

    [[nodiscard]] bool same_spin(int lhs, int rhs) noexcept
    {
        return spin_index(lhs) == spin_index(rhs);
    }

    [[nodiscard]] double rms_norm(const Eigen::VectorXd &vec)
    {
        if (vec.size() == 0)
            return 0.0;
        return std::sqrt(vec.squaredNorm() / static_cast<double>(vec.size()));
    }

    // Apply the ordered excitation operator
    //   a_a^dagger a_b^dagger a_c^dagger ... a_k a_j a_i
    // to an arbitrary determinant. The occupied indices are expected in
    // ascending order and the virtual indices in ascending order as well.
    // Creation must then be applied in reverse because operator products act on
    // kets from right to left.
    DeterminantOpResult apply_excitation_operator(
        CIString determinant,
        const std::vector<int> &occupied,
        const std::vector<int> &virtuals)
    {
        DeterminantOpResult current{
            .det = determinant,
            .phase = 1.0,
            .valid = true,
        };

        for (const int i : occupied)
        {
            const auto res = apply_annihilation(current.det, i);
            if (!res.valid)
                return {};
            current.det = res.det;
            current.phase *= res.phase;
        }

        for (auto it = virtuals.rbegin(); it != virtuals.rend(); ++it)
        {
            const auto res = apply_creation(current.det, *it);
            if (!res.valid)
                return {};
            current.det = res.det;
            current.phase *= res.phase;
        }

        return current;
    }

    SpinOrbitalSystem build_spin_orbital_system(
        const HartreeFock::Calculator &calculator,
        const RHFReference &reference,
        const MOBlockCache &mo_blocks)
    {
        SpinOrbitalSystem system;
        system.n_occ = 2 * reference.n_occ;
        system.n_virt = 2 * reference.n_virt;
        system.n_electrons = system.n_occ;
        system.n_spin_orb = 2 * reference.n_mo;
        system.eps_occ = Eigen::VectorXd(system.n_occ);
        system.eps_virt = Eigen::VectorXd(system.n_virt);
        system.h1 = Tensor2D(system.n_spin_orb, system.n_spin_orb, 0.0);
        system.g2 = Tensor4D(system.n_spin_orb, system.n_spin_orb,
                             system.n_spin_orb, system.n_spin_orb, 0.0);

        for (int i = 0; i < system.n_occ; ++i)
            system.eps_occ(i) = reference.eps_occ(spatial_index(i));
        for (int a = 0; a < system.n_virt; ++a)
            system.eps_virt(a) = reference.eps_virt(spatial_index(a));

        Eigen::MatrixXd C_full(reference.n_ao, reference.n_mo);
        C_full.leftCols(reference.n_occ) = reference.C_occ;
        C_full.rightCols(reference.n_virt) = reference.C_virt;

        // The AO core Hamiltonian is transformed to the canonical MO basis and
        // then duplicated across alpha/beta spin sectors.
        const Eigen::MatrixXd h_mo = C_full.transpose() * calculator._hcore * C_full;
        for (int p = 0; p < system.n_spin_orb; ++p)
            for (int q = 0; q < system.n_spin_orb; ++q)
                if (same_spin(p, q))
                    system.h1(p, q) = h_mo(spatial_index(p), spatial_index(q));

        // Expand the spatial `(pq|rs)` tensor into antisymmetrized spin-orbital
        // integrals `<pq||rs> = (pr|qs) - (ps|qr)` with the corresponding spin
        // delta factors made explicit.
        for (int p = 0; p < system.n_spin_orb; ++p)
            for (int q = 0; q < system.n_spin_orb; ++q)
                for (int r = 0; r < system.n_spin_orb; ++r)
                    for (int s = 0; s < system.n_spin_orb; ++s)
                    {
                        const int P = spatial_index(p);
                        const int Q = spatial_index(q);
                        const int R = spatial_index(r);
                        const int S = spatial_index(s);

                        const double coulomb =
                            (same_spin(p, r) && same_spin(q, s))
                                ? mo_blocks.full(P, R, Q, S)
                                : 0.0;
                        const double exchange =
                            (same_spin(p, s) && same_spin(q, r))
                                ? mo_blocks.full(P, S, Q, R)
                                : 0.0;
                        system.g2(p, q, r, s) = coulomb - exchange;
                    }

        return system;
    }

    std::expected<DeterminantSpace, std::string> build_determinant_space(
        const SpinOrbitalSystem &system)
    {
        if (system.n_spin_orb > kMaxSpinOrbitals)
            return std::unexpected(
                std::format("build_determinant_space: prototype RCCSDT is limited to {} spin orbitals, got {}.",
                            kMaxSpinOrbitals, system.n_spin_orb));

        DeterminantSpace space;
        space.determinants = generate_strings(system.n_spin_orb, system.n_electrons);
        if (space.determinants.empty())
            return std::unexpected("build_determinant_space: failed to enumerate determinants.");
        if (static_cast<int>(space.determinants.size()) > kMaxDeterminants)
            return std::unexpected(
                std::format("build_determinant_space: determinant space has {} states; prototype limit is {}.",
                            space.determinants.size(), kMaxDeterminants));

        space.lookup.reserve(space.determinants.size());
        for (std::size_t idx = 0; idx < space.determinants.size(); ++idx)
            space.lookup.emplace(space.determinants[idx], static_cast<int>(idx));

        space.reference_det = low_bit_mask(system.n_electrons);
        const auto ref_it = space.lookup.find(space.reference_det);
        if (ref_it == space.lookup.end())
            return std::unexpected("build_determinant_space: RHF reference determinant not found.");
        space.reference_index = ref_it->second;
        return space;
    }

    Eigen::MatrixXd build_hamiltonian_matrix(
        const SpinOrbitalSystem &system,
        const DeterminantSpace &space)
    {
        const int dim = static_cast<int>(space.determinants.size());
        Eigen::MatrixXd hamiltonian = Eigen::MatrixXd::Zero(dim, dim);

        // For teaching-sized systems it is perfectly acceptable to build the
        // dense Hamiltonian directly by applying second-quantized operators to
        // each ket. This avoids burying the logic in hard-to-read casework.
        for (int ket_idx = 0; ket_idx < dim; ++ket_idx)
        {
            const CIString ket = space.determinants[static_cast<std::size_t>(ket_idx)];

            for (int q = 0; q < system.n_spin_orb; ++q)
                for (int p = 0; p < system.n_spin_orb; ++p)
                {
                    const double hpq = system.h1(p, q);
                    if (std::abs(hpq) < 1e-14)
                        continue;

                    const auto annihilated = apply_annihilation(ket, q);
                    if (!annihilated.valid)
                        continue;
                    const auto created = apply_creation(annihilated.det, p);
                    if (!created.valid)
                        continue;

                    const auto bra_it = space.lookup.find(created.det);
                    if (bra_it == space.lookup.end())
                        continue;

                    hamiltonian(bra_it->second, ket_idx) +=
                        hpq * annihilated.phase * created.phase;
                }

            for (int r = 0; r < system.n_spin_orb; ++r)
                for (int s = 0; s < system.n_spin_orb; ++s)
                    for (int p = 0; p < system.n_spin_orb; ++p)
                        for (int q = 0; q < system.n_spin_orb; ++q)
                        {
                            const double gpqrs = system.g2(p, q, r, s);
                            if (std::abs(gpqrs) < 1e-14)
                                continue;

                            const auto step1 = apply_annihilation(ket, r);
                            if (!step1.valid)
                                continue;
                            const auto step2 = apply_annihilation(step1.det, s);
                            if (!step2.valid)
                                continue;
                            const auto step3 = apply_creation(step2.det, q);
                            if (!step3.valid)
                                continue;
                            const auto step4 = apply_creation(step3.det, p);
                            if (!step4.valid)
                                continue;

                            const auto bra_it = space.lookup.find(step4.det);
                            if (bra_it == space.lookup.end())
                                continue;

                            hamiltonian(bra_it->second, ket_idx) +=
                                0.25 * gpqrs * step1.phase * step2.phase *
                                step3.phase * step4.phase;
                        }
        }

        return hamiltonian;
    }

    std::expected<std::vector<Excitation>, std::string> build_excitation_list(
        const SpinOrbitalSystem &system,
        const DeterminantSpace &space)
    {
        std::vector<Excitation> excitations;

        const auto append_excitation =
            [&](int rank, std::vector<int> occupied, std::vector<int> virtuals)
            -> std::expected<void, std::string>
        {
            const DeterminantOpResult result =
                apply_excitation_operator(space.reference_det, occupied, virtuals);
            if (!result.valid)
                return std::unexpected("build_excitation_list: invalid excitation from RHF reference.");

            const auto det_it = space.lookup.find(result.det);
            if (det_it == space.lookup.end())
                return std::unexpected("build_excitation_list: excited determinant missing from lookup.");

            double denominator = 0.0;
            for (const int i : occupied)
                denominator += system.eps_occ(i);
            for (const int a : virtuals)
                denominator -= system.eps_virt(a - system.n_occ);

            if (std::abs(denominator) < 1e-12)
                return std::unexpected("build_excitation_list: zero orbital-energy denominator encountered.");

            excitations.push_back(Excitation{
                .rank = rank,
                .occupied = std::move(occupied),
                .virtuals = std::move(virtuals),
                .denominator = denominator,
                .reference_phase = result.phase,
                .determinant_index = det_it->second,
            });
            return {};
        };

        for (int i = 0; i < system.n_occ; ++i)
            for (int a = system.n_occ; a < system.n_spin_orb; ++a)
            {
                auto res = append_excitation(1, {i}, {a});
                if (!res)
                    return std::unexpected(res.error());
            }

        for (int i = 0; i < system.n_occ; ++i)
            for (int j = i + 1; j < system.n_occ; ++j)
                for (int a = system.n_occ; a < system.n_spin_orb; ++a)
                    for (int b = a + 1; b < system.n_spin_orb; ++b)
                    {
                        auto res = append_excitation(2, {i, j}, {a, b});
                        if (!res)
                            return std::unexpected(res.error());
                    }

        for (int i = 0; i < system.n_occ; ++i)
            for (int j = i + 1; j < system.n_occ; ++j)
                for (int k = j + 1; k < system.n_occ; ++k)
                    for (int a = system.n_occ; a < system.n_spin_orb; ++a)
                        for (int b = a + 1; b < system.n_spin_orb; ++b)
                            for (int c = b + 1; c < system.n_spin_orb; ++c)
                            {
                                auto res = append_excitation(3, {i, j, k}, {a, b, c});
                                if (!res)
                                    return std::unexpected(res.error());
                            }

        return excitations;
    }

    Eigen::VectorXd build_initial_guess(
        const SpinOrbitalSystem &system,
        const std::vector<Excitation> &excitations)
    {
        Eigen::VectorXd amplitudes = Eigen::VectorXd::Zero(
            static_cast<Eigen::Index>(excitations.size()));

        for (Eigen::Index idx = 0; idx < amplitudes.size(); ++idx)
        {
            const Excitation &exc = excitations[static_cast<std::size_t>(idx)];

            // Canonical RHF singles are zero initially. The doubles guess is the
            // MP2 amplitude written in the same unique-excitation indexing used by
            // the determinant-space cluster operator. Triples start from zero.
            if (exc.rank == 2)
            {
                amplitudes(idx) = system.g2(exc.occupied[0], exc.occupied[1],
                                            exc.virtuals[0], exc.virtuals[1]) /
                                  exc.denominator;
            }
        }

        return amplitudes;
    }

    Eigen::MatrixXd build_cluster_matrix(
        const DeterminantSpace &space,
        const std::vector<Excitation> &excitations,
        const Eigen::VectorXd &amplitudes)
    {
        const int dim = static_cast<int>(space.determinants.size());
        Eigen::MatrixXd cluster = Eigen::MatrixXd::Zero(dim, dim);

        // Each stored amplitude multiplies one ordered excitation operator. Since
        // the excitation list uses unique index tuples only, no factorial
        // prefactors are needed here.
        for (Eigen::Index mu = 0; mu < amplitudes.size(); ++mu)
        {
            const double t_mu = amplitudes(mu);
            if (std::abs(t_mu) < 1e-14)
                continue;

            const Excitation &exc = excitations[static_cast<std::size_t>(mu)];
            for (int ket_idx = 0; ket_idx < dim; ++ket_idx)
            {
                const DeterminantOpResult result = apply_excitation_operator(
                    space.determinants[static_cast<std::size_t>(ket_idx)],
                    exc.occupied,
                    exc.virtuals);
                if (!result.valid)
                    continue;

                const auto bra_it = space.lookup.find(result.det);
                if (bra_it == space.lookup.end())
                    continue;

                cluster(bra_it->second, ket_idx) += t_mu * result.phase;
            }
        }

        return cluster;
    }

    Eigen::VectorXd apply_exponential(
        const Eigen::MatrixXd &cluster,
        const Eigen::VectorXd &input,
        double sign,
        int max_order)
    {
        Eigen::VectorXd output = input;
        Eigen::VectorXd term = input;

        // The cluster operator is nilpotent because it can only move electrons
        // from the RHF occupied block into the virtual block. The exponential
        // series therefore terminates after finitely many terms.
        for (int order = 1; order <= max_order; ++order)
        {
            term = (sign / static_cast<double>(order)) * (cluster * term);
            output += term;
            if (term.norm() < 1e-14)
                break;
        }

        return output;
    }

    Eigen::VectorXd build_projected_residuals(
        const Eigen::VectorXd &bar_h_ref,
        const std::vector<Excitation> &excitations)
    {
        Eigen::VectorXd residuals = Eigen::VectorXd::Zero(
            static_cast<Eigen::Index>(excitations.size()));
        for (Eigen::Index idx = 0; idx < residuals.size(); ++idx)
        {
            const Excitation &exc = excitations[static_cast<std::size_t>(idx)];
            residuals(idx) = exc.reference_phase * bar_h_ref(exc.determinant_index);
        }
        return residuals;
    }
} // namespace

namespace HartreeFock::Correlation::CC
{
    std::expected<RCCSDTState, std::string> prepare_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        if (calculator._calculation != HartreeFock::CalculationType::SinglePoint)
            return std::unexpected("prepare_rccsdt: RCCSDT is currently available only for single-point calculations.");

        auto reference_res = build_rhf_reference(calculator);
        if (!reference_res)
            return std::unexpected(reference_res.error());

        auto block_res = build_mo_block_cache(calculator, shell_pairs, *reference_res, "RCCSDT :");
        if (!block_res)
            return std::unexpected(block_res.error());

        // The current determinant-space solver uses unique excitation denominators
        // internally and does not need a dense spatial T3 tensor during setup.
        auto denom_res = build_denominator_cache(*reference_res, false);
        if (!denom_res)
            return std::unexpected(denom_res.error());

        RCCSDTState state{
            .reference = std::move(*reference_res),
            .mo_blocks = std::move(*block_res),
            .denominators = std::move(*denom_res),
            .amplitudes = {},
        };
        return state;
    }

    std::expected<void, std::string> run_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        auto state_res = prepare_rccsdt(calculator, shell_pairs);
        if (!state_res)
            return std::unexpected(state_res.error());

        RCCSDTState state = std::move(*state_res);
        const SpinOrbitalSystem system = build_spin_orbital_system(
            calculator, state.reference, state.mo_blocks);

        auto det_space_res = build_determinant_space(system);
        if (!det_space_res)
            return std::unexpected(det_space_res.error());
        DeterminantSpace det_space = std::move(*det_space_res);
        det_space.hamiltonian = build_hamiltonian_matrix(system, det_space);

        auto excitation_res = build_excitation_list(system, det_space);
        if (!excitation_res)
            return std::unexpected(excitation_res.error());
        const std::vector<Excitation> excitations = std::move(*excitation_res);

        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "RCCSDT :",
            std::format("Determinant-space prototype: nso={} nelec={} ndet={} nexc={}",
                        system.n_spin_orb, system.n_electrons,
                        det_space.determinants.size(), excitations.size()));
        if (std::none_of(excitations.begin(), excitations.end(), [](const Excitation &exc)
                         { return exc.rank == 3; }))
        {
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "RCCSDT :",
                "Triple-excitation manifold is empty for this system; RCCSDT reduces to CCSD.");
        }
        HartreeFock::Logger::blank();

        Eigen::VectorXd amplitudes = build_initial_guess(system, excitations);
        const Eigen::VectorXd reference_vector = Eigen::VectorXd::Unit(
            static_cast<Eigen::Index>(det_space.determinants.size()),
            det_space.reference_index);

        const unsigned int max_iter = calculator._scf.get_max_cycles(calculator._shells.nbasis());
        const double tol_energy = calculator._scf._tol_energy;
        const double tol_residual = calculator._scf._tol_density;
        const bool use_diis = calculator._scf._use_DIIS;
        const double damping = 0.6;

        AmplitudeDIIS diis(static_cast<int>(std::max(2u, calculator._scf._DIIS_dim)));
        double previous_energy = std::numeric_limits<double>::quiet_NaN();

        for (unsigned int iter = 1; iter <= max_iter; ++iter)
        {
            const auto iter_start = std::chrono::steady_clock::now();

            const Eigen::MatrixXd cluster = build_cluster_matrix(det_space, excitations, amplitudes);
            const Eigen::VectorXd psi = apply_exponential(cluster, reference_vector, +1.0, system.n_electrons);
            const Eigen::VectorXd sigma = det_space.hamiltonian * psi;
            const Eigen::VectorXd bar_h_ref = apply_exponential(cluster, sigma, -1.0, system.n_electrons);

            const double electronic_energy = bar_h_ref(det_space.reference_index);
            const double energy =
                electronic_energy + calculator._nuclear_repulsion - calculator._total_energy;
            const Eigen::VectorXd residuals = build_projected_residuals(bar_h_ref, excitations);
            const double residual_rms = rms_norm(residuals);

            Eigen::VectorXd updated = amplitudes;
            for (Eigen::Index idx = 0; idx < updated.size(); ++idx)
                updated(idx) += damping * residuals(idx) /
                                excitations[static_cast<std::size_t>(idx)].denominator;

            const double update_rms = rms_norm(updated - amplitudes);

            diis.push(updated, residuals);
            if (use_diis && diis.ready())
            {
                auto diis_res = diis.extrapolate();
                if (diis_res)
                    updated = std::move(*diis_res);
            }

            const double delta_energy =
                std::isnan(previous_energy) ? 0.0 : (energy - previous_energy);
            previous_energy = energy;
            amplitudes = std::move(updated);

            const double time_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - iter_start).count();

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "RCCSDT Iter :",
                std::format(
                    "{:3d}  E_corr={:.10f}  dE={:+.3e}  rms(res)={:.3e}  rms(step)={:.3e}  diis={}  t={:.3f}s",
                    iter, energy, delta_energy, residual_rms, update_rms, diis.size(), time_sec));

            if (std::abs(delta_energy) < tol_energy && residual_rms < tol_residual)
            {
                // `bar_h_ref` is the total electronic energy. The surrounding
                // driver stores the post-HF correction separately and adds it to
                // the SCF total later, so convert here.
                calculator._correlation_energy =
                    electronic_energy + calculator._nuclear_repulsion - calculator._total_energy;
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "RCCSDT :",
                    std::format("Converged in {} iterations.", iter));
                return {};
            }
        }

        return std::unexpected(
            std::format("run_rccsdt: failed to converge in {} iterations.", max_iter));
    }
} // namespace HartreeFock::Correlation::CC
