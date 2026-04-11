#include "post_hf/cc/determinant_space.h"

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <format>
#include <limits>
#include <unordered_map>
#include <vector>

#include "io/logging.h"
#include "post_hf/casscf/strings.h"
#include "post_hf/casscf_internal.h"
#include "post_hf/cc/diis.h"
#include "post_hf/integrals.h"

namespace
{
    using HartreeFock::Correlation::CASSCF::apply_annihilation;
    using HartreeFock::Correlation::CASSCF::apply_creation;
    using HartreeFock::Correlation::CASSCF::generate_strings;
    using HartreeFock::Correlation::CASSCFInternal::CIString;
    using HartreeFock::Correlation::CASSCFInternal::low_bit_mask;
    using HartreeFock::Correlation::CC::AmplitudeDIIS;
    using HartreeFock::Correlation::CC::DeterminantCCSpinOrbitalSeed;
    using HartreeFock::Correlation::CC::MOBlockCache;
    using HartreeFock::Correlation::CC::RHFReference;
    using HartreeFock::Correlation::CC::SpinOrbitalSystem;
    using HartreeFock::Correlation::CC::Tensor2D;
    using HartreeFock::Correlation::CC::Tensor4D;
    using HartreeFock::Correlation::CC::UHFReference;

    // The determinant-space solver is still a teaching-oriented exact backend,
    // but it is useful as a validation and completion backstop for moderate
    // systems while the fully tensorized CCSDT engine is under construction.
    // Keep the cap comfortably below "real production" territory so we do not
    // accidentally route very large jobs into an exponential algorithm.
    constexpr int kMaxSpinOrbitals = 16;
    constexpr int kMaxDeterminants = 10000;

    enum class SpinLabel
    {
        Alpha,
        Beta,
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

    struct UHFSpinOrbitalMap
    {
        int n_occ_alpha = 0;
        int n_occ_beta = 0;
        int n_virt_alpha = 0;
        int n_virt_beta = 0;

        [[nodiscard]] SpinLabel spin(int so_index) const noexcept
        {
            if (so_index < n_occ_alpha)
                return SpinLabel::Alpha;
            if (so_index < n_occ_alpha + n_occ_beta)
                return SpinLabel::Beta;

            const int virt_index = so_index - (n_occ_alpha + n_occ_beta);
            if (virt_index < n_virt_alpha)
                return SpinLabel::Alpha;
            return SpinLabel::Beta;
        }

        [[nodiscard]] int mo_index(int so_index) const noexcept
        {
            if (so_index < n_occ_alpha)
                return so_index;
            if (so_index < n_occ_alpha + n_occ_beta)
                return so_index - n_occ_alpha;

            const int virt_index = so_index - (n_occ_alpha + n_occ_beta);
            if (virt_index < n_virt_alpha)
                return n_occ_alpha + virt_index;
            return n_occ_beta + (virt_index - n_virt_alpha);
        }
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

    std::expected<DeterminantSpace, std::string> build_determinant_space(
        const SpinOrbitalSystem &system)
    {
        if (system.n_spin_orb > kMaxSpinOrbitals)
            return std::unexpected(
                std::format("build_determinant_space: prototype determinant-space CC is limited to {} spin orbitals, got {}.",
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

        // All current determinant-space CC prototypes order occupied spin
        // orbitals first and virtual spin orbitals second, so the reference is
        // the contiguous low-bit occupation pattern.
        space.reference_det = low_bit_mask(system.n_electrons);
        const auto ref_it = space.lookup.find(space.reference_det);
        if (ref_it == space.lookup.end())
            return std::unexpected("build_determinant_space: reference determinant not found.");
        space.reference_index = ref_it->second;
        return space;
    }

    Eigen::MatrixXd build_hamiltonian_matrix(
        const SpinOrbitalSystem &system,
        const DeterminantSpace &space)
    {
        const int dim = static_cast<int>(space.determinants.size());
        Eigen::MatrixXd hamiltonian = Eigen::MatrixXd::Zero(dim, dim);

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
        const DeterminantSpace &space,
        int max_rank)
    {
        if (max_rank < 1 || max_rank > 3)
            return std::unexpected("build_excitation_list: determinant-space prototype supports only ranks 1-3.");

        std::vector<Excitation> excitations;

        const auto append_excitation =
            [&](int rank, std::vector<int> occupied, std::vector<int> virtuals)
            -> std::expected<void, std::string>
        {
            const DeterminantOpResult result =
                apply_excitation_operator(space.reference_det, occupied, virtuals);
            if (!result.valid)
                return std::unexpected("build_excitation_list: invalid excitation from reference determinant.");

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

        if (max_rank >= 2)
        {
            for (int i = 0; i < system.n_occ; ++i)
                for (int j = i + 1; j < system.n_occ; ++j)
                    for (int a = system.n_occ; a < system.n_spin_orb; ++a)
                        for (int b = a + 1; b < system.n_spin_orb; ++b)
                        {
                            auto res = append_excitation(2, {i, j}, {a, b});
                            if (!res)
                                return std::unexpected(res.error());
                        }
        }

        if (max_rank >= 3)
        {
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
            if (exc.rank == 2)
            {
                amplitudes(idx) = system.g2(exc.occupied[0], exc.occupied[1],
                                            exc.virtuals[0], exc.virtuals[1]) /
                                  exc.denominator;
            }
        }

        return amplitudes;
    }

    Eigen::VectorXd build_seeded_initial_guess(
        const SpinOrbitalSystem &system,
        const std::vector<Excitation> &excitations,
        const DeterminantCCSpinOrbitalSeed &seed)
    {
        Eigen::VectorXd amplitudes = build_initial_guess(system, excitations);

        for (Eigen::Index idx = 0; idx < amplitudes.size(); ++idx)
        {
            const Excitation &exc = excitations[static_cast<std::size_t>(idx)];

            if (exc.rank == 1 && seed.t1 != nullptr)
            {
                const int i = exc.occupied[0];
                const int a = exc.virtuals[0] - system.n_occ;
                amplitudes(idx) = (*seed.t1)(i, a);
            }
            else if (exc.rank == 2 && seed.t2 != nullptr)
            {
                const int i = exc.occupied[0];
                const int j = exc.occupied[1];
                const int a = exc.virtuals[0] - system.n_occ;
                const int b = exc.virtuals[1] - system.n_occ;
                amplitudes(idx) = (*seed.t2)(i, j, a, b);
            }
            else if (exc.rank == 3 && seed.t3 != nullptr)
            {
                const int i = exc.occupied[0];
                const int j = exc.occupied[1];
                const int k = exc.occupied[2];
                const int a = exc.virtuals[0] - system.n_occ;
                const int b = exc.virtuals[1] - system.n_occ;
                const int c = exc.virtuals[2] - system.n_occ;
                amplitudes(idx) = (*seed.t3)(i, j, k, a, b, c);
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
    std::expected<SpinOrbitalSystem, std::string> build_rhf_spin_orbital_system(
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

        const Eigen::MatrixXd h_mo = C_full.transpose() * calculator._hcore * C_full;
        for (int p = 0; p < system.n_spin_orb; ++p)
            for (int q = 0; q < system.n_spin_orb; ++q)
                if (same_spin(p, q))
                    system.h1(p, q) = h_mo(spatial_index(p), spatial_index(q));

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

    std::expected<SpinOrbitalSystem, std::string> build_uhf_spin_orbital_system(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const UHFReference &reference,
        const std::string &tag)
    {
        std::vector<double> eri_local;
        const std::vector<double> &eri = HartreeFock::Correlation::ensure_eri(
            calculator, shell_pairs, eri_local, tag);

        SpinOrbitalSystem system;
        system.n_occ = reference.n_occ_alpha + reference.n_occ_beta;
        system.n_virt = reference.n_virt_alpha + reference.n_virt_beta;
        system.n_electrons = system.n_occ;
        system.n_spin_orb = system.n_occ + system.n_virt;
        system.eps_occ = Eigen::VectorXd(system.n_occ);
        system.eps_virt = Eigen::VectorXd(system.n_virt);
        system.h1 = Tensor2D(system.n_spin_orb, system.n_spin_orb, 0.0);
        system.g2 = Tensor4D(system.n_spin_orb, system.n_spin_orb,
                             system.n_spin_orb, system.n_spin_orb, 0.0);

        const UHFSpinOrbitalMap so_map{
            .n_occ_alpha = reference.n_occ_alpha,
            .n_occ_beta = reference.n_occ_beta,
            .n_virt_alpha = reference.n_virt_alpha,
            .n_virt_beta = reference.n_virt_beta,
        };

        for (int i = 0; i < reference.n_occ_alpha; ++i)
            system.eps_occ(i) = reference.eps_alpha(i);
        for (int i = 0; i < reference.n_occ_beta; ++i)
            system.eps_occ(reference.n_occ_alpha + i) = reference.eps_beta(i);
        for (int a = 0; a < reference.n_virt_alpha; ++a)
            system.eps_virt(a) = reference.eps_alpha(reference.n_occ_alpha + a);
        for (int a = 0; a < reference.n_virt_beta; ++a)
            system.eps_virt(reference.n_virt_alpha + a) = reference.eps_beta(reference.n_occ_beta + a);

        try
        {
            const std::size_t nb = static_cast<std::size_t>(reference.n_ao);
            const Eigen::MatrixXd h_alpha = reference.C_alpha.transpose() * calculator._hcore * reference.C_alpha;
            const Eigen::MatrixXd h_beta = reference.C_beta.transpose() * calculator._hcore * reference.C_beta;

            const Tensor4D eri_aaaa(
                reference.n_mo, reference.n_mo, reference.n_mo, reference.n_mo,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    reference.C_alpha, reference.C_alpha,
                    reference.C_alpha, reference.C_alpha));
            const Tensor4D eri_aabb(
                reference.n_mo, reference.n_mo, reference.n_mo, reference.n_mo,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    reference.C_alpha, reference.C_alpha,
                    reference.C_beta, reference.C_beta));
            const Tensor4D eri_bbaa(
                reference.n_mo, reference.n_mo, reference.n_mo, reference.n_mo,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    reference.C_beta, reference.C_beta,
                    reference.C_alpha, reference.C_alpha));
            const Tensor4D eri_bbbb(
                reference.n_mo, reference.n_mo, reference.n_mo, reference.n_mo,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    reference.C_beta, reference.C_beta,
                    reference.C_beta, reference.C_beta));

            const auto mo_eri = [&](SpinLabel left_spin, SpinLabel right_spin,
                                    int p, int r, int q, int s) -> double
            {
                if (left_spin == SpinLabel::Alpha && right_spin == SpinLabel::Alpha)
                    return eri_aaaa(p, r, q, s);
                if (left_spin == SpinLabel::Alpha && right_spin == SpinLabel::Beta)
                    return eri_aabb(p, r, q, s);
                if (left_spin == SpinLabel::Beta && right_spin == SpinLabel::Alpha)
                    return eri_bbaa(p, r, q, s);
                return eri_bbbb(p, r, q, s);
            };

            for (int p = 0; p < system.n_spin_orb; ++p)
                for (int q = 0; q < system.n_spin_orb; ++q)
                {
                    const SpinLabel spin_p = so_map.spin(p);
                    const SpinLabel spin_q = so_map.spin(q);
                    if (spin_p != spin_q)
                        continue;

                    const int P = so_map.mo_index(p);
                    const int Q = so_map.mo_index(q);
                    system.h1(p, q) = (spin_p == SpinLabel::Alpha)
                                          ? h_alpha(P, Q)
                                          : h_beta(P, Q);
                }

            for (int p = 0; p < system.n_spin_orb; ++p)
                for (int q = 0; q < system.n_spin_orb; ++q)
                    for (int r = 0; r < system.n_spin_orb; ++r)
                        for (int s = 0; s < system.n_spin_orb; ++s)
                        {
                            const SpinLabel spin_p = so_map.spin(p);
                            const SpinLabel spin_q = so_map.spin(q);
                            const SpinLabel spin_r = so_map.spin(r);
                            const SpinLabel spin_s = so_map.spin(s);
                            const int P = so_map.mo_index(p);
                            const int Q = so_map.mo_index(q);
                            const int R = so_map.mo_index(r);
                            const int S = so_map.mo_index(s);

                            const double coulomb =
                                (spin_p == spin_r && spin_q == spin_s)
                                    ? mo_eri(spin_p, spin_q, P, R, Q, S)
                                    : 0.0;
                            const double exchange =
                                (spin_p == spin_s && spin_q == spin_r)
                                    ? mo_eri(spin_p, spin_q, P, S, Q, R)
                                    : 0.0;
                            system.g2(p, q, r, s) = coulomb - exchange;
                        }
        }
        catch (const std::exception &ex)
        {
            return std::unexpected("build_uhf_spin_orbital_system: " + std::string(ex.what()));
        }

        return system;
    }

    std::expected<double, std::string> solve_determinant_cc(
        HartreeFock::Calculator &calculator,
        const SpinOrbitalSystem &system,
        int max_rank,
        const std::string &log_tag,
        const DeterminantCCSpinOrbitalSeed *seed)
    {
        auto det_space_res = build_determinant_space(system);
        if (!det_space_res)
            return std::unexpected(det_space_res.error());
        DeterminantSpace det_space = std::move(*det_space_res);
        det_space.hamiltonian = build_hamiltonian_matrix(system, det_space);

        auto excitation_res = build_excitation_list(system, det_space, max_rank);
        if (!excitation_res)
            return std::unexpected(excitation_res.error());
        const std::vector<Excitation> excitations = std::move(*excitation_res);

        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            log_tag,
            std::format("Determinant-space prototype: nso={} nelec={} ndet={} nexc={} rank<={}",
                        system.n_spin_orb, system.n_electrons,
                        det_space.determinants.size(), excitations.size(), max_rank));
        if (max_rank == 3 &&
            std::none_of(excitations.begin(), excitations.end(), [](const Excitation &exc)
                         { return exc.rank == 3; }))
        {
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                log_tag,
                "Triple-excitation manifold is empty for this system; CCSDT reduces to CCSD.");
        }
        HartreeFock::Logger::blank();

        Eigen::VectorXd amplitudes =
            (seed == nullptr)
                ? build_initial_guess(system, excitations)
                : build_seeded_initial_guess(system, excitations, *seed);
        if (seed != nullptr)
        {
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                log_tag,
                "Using spin-orbital amplitude warm start from the staged tensor RCCSDT backend.");
        }
        const Eigen::VectorXd reference_vector = Eigen::VectorXd::Unit(
            static_cast<Eigen::Index>(det_space.determinants.size()),
            det_space.reference_index);

        const unsigned int max_iter = calculator._scf.get_max_cycles(calculator._shells.nbasis());
        const double tol_energy = calculator._scf._tol_energy;
        const double tol_residual = calculator._scf._tol_density;
        const bool use_diis = calculator._scf._use_DIIS;
        const double damping = 0.6;

        AmplitudeDIIS diis(static_cast<int>(std::max(2u, calculator._scf._DIIS_dim)));
        double previous_corr_energy = std::numeric_limits<double>::quiet_NaN();

        for (unsigned int iter = 1; iter <= max_iter; ++iter)
        {
            const auto iter_start = std::chrono::steady_clock::now();

            const Eigen::MatrixXd cluster = build_cluster_matrix(det_space, excitations, amplitudes);
            const Eigen::VectorXd psi = apply_exponential(cluster, reference_vector, +1.0, system.n_electrons);
            const Eigen::VectorXd sigma = det_space.hamiltonian * psi;
            const Eigen::VectorXd bar_h_ref = apply_exponential(cluster, sigma, -1.0, system.n_electrons);

            const double electronic_energy = bar_h_ref(det_space.reference_index);
            const double corr_energy =
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
                std::isnan(previous_corr_energy) ? 0.0 : (corr_energy - previous_corr_energy);
            previous_corr_energy = corr_energy;
            amplitudes = std::move(updated);

            const double time_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - iter_start).count();

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                log_tag + " Iter :",
                std::format(
                    "{:3d}  E_corr={:.10f}  dE={:+.3e}  rms(res)={:.3e}  rms(step)={:.3e}  diis={}  t={:.3f}s",
                    iter, corr_energy, delta_energy, residual_rms, update_rms, diis.size(), time_sec));

            if (std::abs(delta_energy) < tol_energy && residual_rms < tol_residual)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    log_tag,
                    std::format("Converged in {} iterations.", iter));
                return corr_energy;
            }
        }

        return std::unexpected(
            std::format("solve_determinant_cc: failed to converge in {} iterations.", max_iter));
    }
} // namespace HartreeFock::Correlation::CC
