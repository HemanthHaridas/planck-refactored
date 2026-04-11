#include "post_hf/cc/tensor_backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <format>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "io/logging.h"
#include "post_hf/cc/determinant_space.h"
#include "post_hf/cc/diis.h"
#include "post_hf/integrals.h"

namespace
{
    [[nodiscard]] std::size_t checked_product(std::initializer_list<int> dims)
    {
        std::size_t total = 1;
        for (const int dim : dims)
        {
            if (dim < 0)
                throw std::invalid_argument("checked_product: negative tensor dimension");

            const std::size_t dim_size = static_cast<std::size_t>(dim);
            if (dim_size != 0 &&
                total > std::numeric_limits<std::size_t>::max() / dim_size)
                throw std::overflow_error("checked_product: tensor size overflow");

            total *= dim_size;
        }
        return total;
    }

    [[nodiscard]] std::size_t bytes_for_tensor(std::initializer_list<int> dims)
    {
        return checked_product(dims) * sizeof(double);
    }

    [[nodiscard]] std::string format_bytes(std::size_t bytes)
    {
        constexpr double kib = 1024.0;
        constexpr double mib = 1024.0 * 1024.0;
        constexpr double gib = 1024.0 * 1024.0 * 1024.0;

        const double value = static_cast<double>(bytes);
        if (value >= gib)
            return std::format("{:.2f} GiB", value / gib);
        if (value >= mib)
            return std::format("{:.2f} MiB", value / mib);
        if (value >= kib)
            return std::format("{:.2f} KiB", value / kib);
        return std::format("{} B", bytes);
    }

    [[nodiscard]] std::size_t binomial(std::size_t n, std::size_t k) noexcept
    {
        if (k > n)
            return 0;
        if (k == 0 || k == n)
            return 1;

        k = std::min(k, n - k);
        long double result = 1.0L;
        for (std::size_t i = 1; i <= k; ++i)
        {
            result *= static_cast<long double>(n - k + i);
            result /= static_cast<long double>(i);
            if (result > static_cast<long double>(std::numeric_limits<std::size_t>::max()))
                return std::numeric_limits<std::size_t>::max();
        }

        return static_cast<std::size_t>(std::llround(result));
    }

    void append_block_memory(
        std::vector<HartreeFock::Correlation::CC::TensorMemoryBlock> &report,
        std::size_t &total_bytes,
        const std::string &label,
        std::initializer_list<int> dims)
    {
        const std::size_t elements = checked_product(dims);
        const std::size_t bytes = elements * sizeof(double);
        report.push_back(HartreeFock::Correlation::CC::TensorMemoryBlock{
            .label = label,
            .elements = elements,
            .bytes = bytes,
        });
        total_bytes += bytes;
    }

    using HartreeFock::Correlation::CC::AmplitudeDIIS;
    using HartreeFock::Correlation::CC::CanonicalRHFCCReference;
    using HartreeFock::Correlation::CC::RHFReference;
    using HartreeFock::Correlation::CC::RCCSDAmplitudes;
    using HartreeFock::Correlation::CC::RCCSDTAmplitudes;
    using HartreeFock::Correlation::CC::Tensor2D;
    using HartreeFock::Correlation::CC::Tensor4D;
    using HartreeFock::Correlation::CC::Tensor6D;
    using HartreeFock::Correlation::CC::TensorCCBlockCache;
    using HartreeFock::Correlation::CC::TensorRCCSDTState;
    using HartreeFock::Correlation::CC::TensorTriplesWorkspace;

    struct ProductionSpinOrbitalReference
    {
        int n_occ = 0;
        int n_virt = 0;
        Eigen::VectorXd eps_occ;
        Eigen::VectorXd eps_virt;
    };

    struct ProductionSpinOrbitalBlocks
    {
        Tensor4D oooo;
        Tensor4D ooov;
        Tensor4D oovv;
        Tensor4D ovov;
        Tensor4D ovvo;
        Tensor4D ovvv;
        Tensor4D vvvv;
    };

    struct TauCache
    {
        Tensor4D tau;
        Tensor4D tau_tilde;
    };

    struct RCCSDIntermediates
    {
        Tensor2D fae;
        Tensor2D fmi;
        Tensor2D fme;
        Tensor4D wmnij;
        Tensor4D wabef;
        Tensor4D wmbej;
    };

    struct RCCSDResiduals
    {
        Tensor2D r1;
        Tensor4D r2;
    };

    struct TensorRCCSDResult
    {
        RCCSDAmplitudes amplitudes;
        double correlation_energy = 0.0;
        unsigned int iterations = 0;
    };

    struct TensorTriplesStageMetrics
    {
        unsigned int iterations = 0;
        unsigned int best_iteration = 0;
        bool converged = false;
        double sd_residual_rms = 0.0;
        double r3_rms = 0.0;
        double t3_step_rms = 0.0;
        double r1_feedback_rms = 0.0;
        double t1_step_rms = 0.0;
        double r2_feedback_rms = 0.0;
        double t2_step_rms = 0.0;
        double quality_score = 0.0;
        double estimated_correlation_energy = 0.0;
    };

    [[nodiscard]] RCCSDTAmplitudes clone_rccsdt_amplitudes(
        const RCCSDTAmplitudes &src)
    {
        RCCSDTAmplitudes out{
            .t1 = Tensor2D(src.t1.dim1, src.t1.dim2, 0.0),
            .t2 = Tensor4D(src.t2.dim1, src.t2.dim2, src.t2.dim3, src.t2.dim4, 0.0),
            .t3 = Tensor6D(
                src.t3.dim1, src.t3.dim2, src.t3.dim3,
                src.t3.dim4, src.t3.dim5, src.t3.dim6, 0.0),
        };
        out.t1.data = src.t1.data;
        out.t2.data = src.t2.data;
        out.t3.data = src.t3.data;
        return out;
    }

    [[nodiscard]] double stage_quality_score(
        const TensorTriplesStageMetrics &metrics)
    {
        return std::max(
            std::max(metrics.sd_residual_rms, metrics.r3_rms),
            std::max(metrics.r1_feedback_rms, metrics.r2_feedback_rms));
    }

    struct DeterminantBackstopDecision
    {
        bool enabled = false;
        int n_spin_orb = 0;
        std::size_t determinants = 0;
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

    [[nodiscard]] DeterminantBackstopDecision choose_determinant_backstop(
        const CanonicalRHFCCReference &reference) noexcept
    {
        constexpr int kMaxBackstopSpinOrbitals = 16;
        // NH3/STO-3G already lands at 8008 determinants, which is still
        // perfectly reasonable for the exact teaching backstop. Keep the cap
        // modest, but high enough to cover these small "real molecule"
        // examples while the pure tensor RCCSDT engine is still maturing.
        constexpr std::size_t kMaxBackstopDeterminants = 10000;

        const int n_spin_orb = 2 * reference.orbital_partition.n_mo;
        const int n_electrons = 2 * reference.orbital_partition.n_occ;
        const std::size_t ndet = binomial(
            static_cast<std::size_t>(n_spin_orb),
            static_cast<std::size_t>(n_electrons));

        return DeterminantBackstopDecision{
            .enabled = (n_spin_orb <= kMaxBackstopSpinOrbitals &&
                        ndet <= kMaxBackstopDeterminants),
            .n_spin_orb = n_spin_orb,
            .determinants = ndet,
        };
    }

    [[nodiscard]] double rms_norm(const Eigen::VectorXd &vec)
    {
        if (vec.size() == 0)
            return 0.0;
        return std::sqrt(vec.squaredNorm() / static_cast<double>(vec.size()));
    }

    [[nodiscard]] double triples_residual_rms(const Tensor6D &tensor)
    {
        if (tensor.data.empty())
            return 0.0;

        double sum_sq = 0.0;
        for (const double value : tensor.data)
            sum_sq += value * value;
        return std::sqrt(sum_sq / static_cast<double>(tensor.data.size()));
    }

    [[nodiscard]] double tensor_rms(const Tensor4D &tensor)
    {
        if (tensor.data.empty())
            return 0.0;

        double sum_sq = 0.0;
        for (const double value : tensor.data)
            sum_sq += value * value;
        return std::sqrt(sum_sq / static_cast<double>(tensor.data.size()));
    }

    [[nodiscard]] double tensor_rms(const Tensor2D &tensor)
    {
        if (tensor.data.empty())
            return 0.0;

        double sum_sq = 0.0;
        for (const double value : tensor.data)
            sum_sq += value * value;
        return std::sqrt(sum_sq / static_cast<double>(tensor.data.size()));
    }

    [[nodiscard]] double d3_on_demand(
        const CanonicalRHFCCReference &reference,
        int i, int j, int k,
        int a, int b, int c) noexcept
    {
        const RHFReference &base = reference.orbital_partition;
        return base.eps_occ(spatial_index(i)) +
               base.eps_occ(spatial_index(j)) +
               base.eps_occ(spatial_index(k)) -
               base.eps_virt(spatial_index(a)) -
               base.eps_virt(spatial_index(b)) -
               base.eps_virt(spatial_index(c));
    }

    struct SignedPermutation3
    {
        std::array<int, 3> perm{};
        int sign = 0;
    };

    constexpr std::array<SignedPermutation3, 6> kPermutations3 = {{
        {{{0, 1, 2}}, +1},
        {{{0, 2, 1}}, -1},
        {{{1, 0, 2}}, -1},
        {{{1, 2, 0}}, +1},
        {{{2, 0, 1}}, +1},
        {{{2, 1, 0}}, -1},
    }};

    ProductionSpinOrbitalReference build_spin_orbital_reference(
        const CanonicalRHFCCReference &reference)
    {
        const auto &base = reference.orbital_partition;
        ProductionSpinOrbitalReference so;
        so.n_occ = 2 * base.n_occ;
        so.n_virt = 2 * base.n_virt;
        so.eps_occ = Eigen::VectorXd(so.n_occ);
        so.eps_virt = Eigen::VectorXd(so.n_virt);

        for (int i = 0; i < so.n_occ; ++i)
            so.eps_occ(i) = base.eps_occ(spatial_index(i));
        for (int a = 0; a < so.n_virt; ++a)
            so.eps_virt(a) = base.eps_virt(spatial_index(a));

        return so;
    }

    ProductionSpinOrbitalBlocks build_spin_orbital_blocks(
        const CanonicalRHFCCReference &reference,
        const TensorCCBlockCache &spatial)
    {
        const auto so = build_spin_orbital_reference(reference);

        const auto occ = [](int i) noexcept -> int
        {
            return spatial_index(i);
        };
        const auto virt = [](int a) noexcept -> int
        {
            return spatial_index(a);
        };

        ProductionSpinOrbitalBlocks blocks{
            .oooo = Tensor4D(so.n_occ, so.n_occ, so.n_occ, so.n_occ, 0.0),
            .ooov = Tensor4D(so.n_occ, so.n_occ, so.n_occ, so.n_virt, 0.0),
            .oovv = Tensor4D(so.n_occ, so.n_occ, so.n_virt, so.n_virt, 0.0),
            .ovov = Tensor4D(so.n_occ, so.n_virt, so.n_occ, so.n_virt, 0.0),
            .ovvo = Tensor4D(so.n_occ, so.n_virt, so.n_virt, so.n_occ, 0.0),
            .ovvv = Tensor4D(so.n_occ, so.n_virt, so.n_virt, so.n_virt, 0.0),
            .vvvv = Tensor4D(so.n_virt, so.n_virt, so.n_virt, so.n_virt, 0.0),
        };

        for (int i = 0; i < so.n_occ; ++i)
            for (int j = 0; j < so.n_occ; ++j)
                for (int k = 0; k < so.n_occ; ++k)
                    for (int l = 0; l < so.n_occ; ++l)
                        blocks.oooo(i, j, k, l) =
                            (same_spin(i, k) && same_spin(j, l)
                                 ? spatial.oooo(occ(i), occ(k), occ(j), occ(l))
                                 : 0.0) -
                            (same_spin(i, l) && same_spin(j, k)
                                 ? spatial.oooo(occ(i), occ(l), occ(j), occ(k))
                                 : 0.0);

        for (int i = 0; i < so.n_occ; ++i)
            for (int j = 0; j < so.n_occ; ++j)
                for (int k = 0; k < so.n_occ; ++k)
                    for (int a = 0; a < so.n_virt; ++a)
                        blocks.ooov(i, j, k, a) =
                            (same_spin(i, k) && same_spin(j, a)
                                 ? spatial.ooov(occ(i), occ(k), occ(j), virt(a))
                                 : 0.0) -
                            (same_spin(i, a) && same_spin(j, k)
                                 ? spatial.ooov(occ(j), occ(k), occ(i), virt(a))
                                 : 0.0);

        for (int i = 0; i < so.n_occ; ++i)
            for (int j = 0; j < so.n_occ; ++j)
                for (int a = 0; a < so.n_virt; ++a)
                    for (int b = 0; b < so.n_virt; ++b)
                        blocks.oovv(i, j, a, b) =
                            (same_spin(i, a) && same_spin(j, b)
                                 ? spatial.ovov(occ(i), virt(a), occ(j), virt(b))
                                 : 0.0) -
                            (same_spin(i, b) && same_spin(j, a)
                                 ? spatial.ovvo(occ(i), virt(b), virt(a), occ(j))
                                 : 0.0);

        for (int i = 0; i < so.n_occ; ++i)
            for (int a = 0; a < so.n_virt; ++a)
                for (int j = 0; j < so.n_occ; ++j)
                    for (int b = 0; b < so.n_virt; ++b)
                        blocks.ovov(i, a, j, b) =
                            (same_spin(i, j) && same_spin(a, b)
                                 ? spatial.oovv(occ(i), occ(j), virt(a), virt(b))
                                 : 0.0) -
                            (same_spin(i, b) && same_spin(a, j)
                                 ? spatial.ovvo(occ(i), virt(b), virt(a), occ(j))
                                 : 0.0);

        for (int i = 0; i < so.n_occ; ++i)
            for (int a = 0; a < so.n_virt; ++a)
                for (int b = 0; b < so.n_virt; ++b)
                    for (int j = 0; j < so.n_occ; ++j)
                        blocks.ovvo(i, a, b, j) =
                            (same_spin(i, b) && same_spin(a, j)
                                 ? spatial.ovvo(occ(i), virt(b), virt(a), occ(j))
                                 : 0.0) -
                            (same_spin(i, j) && same_spin(a, b)
                                 ? spatial.oovv(occ(i), occ(j), virt(a), virt(b))
                                 : 0.0);

        for (int i = 0; i < so.n_occ; ++i)
            for (int a = 0; a < so.n_virt; ++a)
                for (int b = 0; b < so.n_virt; ++b)
                    for (int c = 0; c < so.n_virt; ++c)
                        blocks.ovvv(i, a, b, c) =
                            (same_spin(i, b) && same_spin(a, c)
                                 ? spatial.ovvv(occ(i), virt(b), virt(a), virt(c))
                                 : 0.0) -
                            (same_spin(i, c) && same_spin(a, b)
                                 ? spatial.ovvv(occ(i), virt(c), virt(a), virt(b))
                                 : 0.0);

        for (int a = 0; a < so.n_virt; ++a)
            for (int b = 0; b < so.n_virt; ++b)
                for (int c = 0; c < so.n_virt; ++c)
                    for (int d = 0; d < so.n_virt; ++d)
                        blocks.vvvv(a, b, c, d) =
                            (same_spin(a, c) && same_spin(b, d)
                                 ? spatial.vvvv(virt(a), virt(c), virt(b), virt(d))
                                 : 0.0) -
                            (same_spin(a, d) && same_spin(b, c)
                                 ? spatial.vvvv(virt(a), virt(d), virt(b), virt(c))
                                 : 0.0);

        return blocks;
    }

    TauCache build_tau_cache(const RCCSDAmplitudes &amps)
    {
        TauCache out{
            .tau = Tensor4D(amps.t2.dim1, amps.t2.dim2, amps.t2.dim3, amps.t2.dim4, 0.0),
            .tau_tilde = Tensor4D(amps.t2.dim1, amps.t2.dim2, amps.t2.dim3, amps.t2.dim4, 0.0),
        };

        for (int i = 0; i < amps.t2.dim1; ++i)
            for (int j = 0; j < amps.t2.dim2; ++j)
                for (int a = 0; a < amps.t2.dim3; ++a)
                    for (int b = 0; b < amps.t2.dim4; ++b)
                    {
                        const double pair =
                            amps.t1(i, a) * amps.t1(j, b) -
                            amps.t1(i, b) * amps.t1(j, a);
                        out.tau(i, j, a, b) = amps.t2(i, j, a, b) + pair;
                        out.tau_tilde(i, j, a, b) = amps.t2(i, j, a, b) + 0.5 * pair;
                    }

        return out;
    }

    RCCSDIntermediates build_intermediates(
        const ProductionSpinOrbitalReference &reference,
        const ProductionSpinOrbitalBlocks &blocks,
        const RCCSDAmplitudes &amps,
        const TauCache &tau_cache)
    {
        RCCSDIntermediates out{
            .fae = Tensor2D(reference.n_virt, reference.n_virt, 0.0),
            .fmi = Tensor2D(reference.n_occ, reference.n_occ, 0.0),
            .fme = Tensor2D(reference.n_occ, reference.n_virt, 0.0),
            .wmnij = Tensor4D(reference.n_occ, reference.n_occ, reference.n_occ, reference.n_occ, 0.0),
            .wabef = Tensor4D(reference.n_virt, reference.n_virt, reference.n_virt, reference.n_virt, 0.0),
            .wmbej = Tensor4D(reference.n_occ, reference.n_virt, reference.n_virt, reference.n_occ, 0.0),
        };

        for (int m = 0; m < reference.n_occ; ++m)
            for (int e = 0; e < reference.n_virt; ++e)
                for (int n = 0; n < reference.n_occ; ++n)
                    for (int f = 0; f < reference.n_virt; ++f)
                        out.fme(m, e) += amps.t1(n, f) * blocks.oovv(m, n, e, f);

        for (int a = 0; a < reference.n_virt; ++a)
            for (int e = 0; e < reference.n_virt; ++e)
            {
                double value = (a == e) ? reference.eps_virt(a) : 0.0;
                for (int m = 0; m < reference.n_occ; ++m)
                    for (int f = 0; f < reference.n_virt; ++f)
                        value += amps.t1(m, f) * blocks.ovvv(m, a, f, e);
                for (int m = 0; m < reference.n_occ; ++m)
                    for (int n = 0; n < reference.n_occ; ++n)
                        for (int f = 0; f < reference.n_virt; ++f)
                            value -= 0.5 * tau_cache.tau_tilde(m, n, a, f) * blocks.oovv(m, n, e, f);
                out.fae(a, e) = value;
            }

        for (int m = 0; m < reference.n_occ; ++m)
            for (int i = 0; i < reference.n_occ; ++i)
            {
                double value = (m == i) ? reference.eps_occ(i) : 0.0;
                for (int n = 0; n < reference.n_occ; ++n)
                    for (int e = 0; e < reference.n_virt; ++e)
                        value += amps.t1(n, e) * blocks.ooov(m, n, i, e);
                for (int n = 0; n < reference.n_occ; ++n)
                    for (int e = 0; e < reference.n_virt; ++e)
                        for (int f = 0; f < reference.n_virt; ++f)
                            value += 0.5 * tau_cache.tau_tilde(i, n, e, f) * blocks.oovv(m, n, e, f);
                out.fmi(m, i) = value;
            }

        for (int m = 0; m < reference.n_occ; ++m)
            for (int n = 0; n < reference.n_occ; ++n)
                for (int i = 0; i < reference.n_occ; ++i)
                    for (int j = 0; j < reference.n_occ; ++j)
                    {
                        double value = blocks.oooo(m, n, i, j);
                        for (int e = 0; e < reference.n_virt; ++e)
                            value += amps.t1(j, e) * blocks.ooov(m, n, i, e) -
                                     amps.t1(i, e) * blocks.ooov(m, n, j, e);
                        for (int e = 0; e < reference.n_virt; ++e)
                            for (int f = 0; f < reference.n_virt; ++f)
                                value += 0.25 * tau_cache.tau(i, j, e, f) * blocks.oovv(m, n, e, f);
                        out.wmnij(m, n, i, j) = value;
                    }

        for (int a = 0; a < reference.n_virt; ++a)
            for (int b = 0; b < reference.n_virt; ++b)
                for (int e = 0; e < reference.n_virt; ++e)
                    for (int f = 0; f < reference.n_virt; ++f)
                    {
                        double value = blocks.vvvv(a, b, e, f);
                        for (int m = 0; m < reference.n_occ; ++m)
                            value += amps.t1(m, b) * blocks.ovvv(m, a, e, f) -
                                     amps.t1(m, a) * blocks.ovvv(m, b, e, f);
                        for (int m = 0; m < reference.n_occ; ++m)
                            for (int n = 0; n < reference.n_occ; ++n)
                                value += 0.25 * tau_cache.tau(m, n, a, b) * blocks.oovv(m, n, e, f);
                        out.wabef(a, b, e, f) = value;
                    }

        for (int m = 0; m < reference.n_occ; ++m)
            for (int b = 0; b < reference.n_virt; ++b)
                for (int e = 0; e < reference.n_virt; ++e)
                    for (int j = 0; j < reference.n_occ; ++j)
                    {
                        double value = blocks.ovvo(m, b, e, j);
                        for (int f = 0; f < reference.n_virt; ++f)
                            value += amps.t1(j, f) * blocks.ovvv(m, b, e, f);
                        for (int n = 0; n < reference.n_occ; ++n)
                            value += amps.t1(n, b) * blocks.ooov(m, n, j, e);
                        for (int n = 0; n < reference.n_occ; ++n)
                            for (int f = 0; f < reference.n_virt; ++f)
                                value -= (0.5 * amps.t2(j, n, f, b) +
                                          amps.t1(j, f) * amps.t1(n, b)) *
                                         blocks.oovv(m, n, e, f);
                        out.wmbej(m, b, e, j) = value;
                    }

        return out;
    }

    RCCSDResiduals build_residuals(
        const ProductionSpinOrbitalReference &reference,
        const ProductionSpinOrbitalBlocks &blocks,
        const RCCSDAmplitudes &amps,
        const TauCache &tau_cache,
        const RCCSDIntermediates &ints)
    {
        RCCSDResiduals out{
            .r1 = Tensor2D(reference.n_occ, reference.n_virt, 0.0),
            .r2 = Tensor4D(reference.n_occ, reference.n_occ, reference.n_virt, reference.n_virt, 0.0),
        };

        for (int i = 0; i < reference.n_occ; ++i)
            for (int a = 0; a < reference.n_virt; ++a)
            {
                double value = 0.0;
                for (int e = 0; e < reference.n_virt; ++e)
                    value += amps.t1(i, e) * ints.fae(a, e);
                for (int m = 0; m < reference.n_occ; ++m)
                    value -= amps.t1(m, a) * ints.fmi(m, i);
                for (int m = 0; m < reference.n_occ; ++m)
                    for (int e = 0; e < reference.n_virt; ++e)
                        value += amps.t2(i, m, a, e) * ints.fme(m, e);
                for (int n = 0; n < reference.n_occ; ++n)
                    for (int f = 0; f < reference.n_virt; ++f)
                        value -= amps.t1(n, f) * blocks.ovov(n, a, i, f);
                for (int m = 0; m < reference.n_occ; ++m)
                    for (int e = 0; e < reference.n_virt; ++e)
                        for (int f = 0; f < reference.n_virt; ++f)
                            value -= 0.5 * amps.t2(i, m, e, f) * blocks.ovvv(m, a, e, f);
                for (int m = 0; m < reference.n_occ; ++m)
                    for (int n = 0; n < reference.n_occ; ++n)
                        for (int e = 0; e < reference.n_virt; ++e)
                            value += 0.5 * amps.t2(m, n, a, e) * blocks.ooov(n, m, i, e);
                out.r1(i, a) = value;
            }

        for (int i = 0; i < reference.n_occ; ++i)
            for (int j = 0; j < reference.n_occ; ++j)
                for (int a = 0; a < reference.n_virt; ++a)
                    for (int b = 0; b < reference.n_virt; ++b)
                    {
                        double value = blocks.oovv(i, j, a, b);
                        for (int e = 0; e < reference.n_virt; ++e)
                            value += amps.t2(i, j, a, e) * ints.fae(b, e) -
                                     amps.t2(i, j, b, e) * ints.fae(a, e);
                        for (int m = 0; m < reference.n_occ; ++m)
                            value -= amps.t2(i, m, a, b) * ints.fmi(m, j) -
                                     amps.t2(j, m, a, b) * ints.fmi(m, i);
                        for (int m = 0; m < reference.n_occ; ++m)
                            for (int n = 0; n < reference.n_occ; ++n)
                                value += 0.5 * tau_cache.tau(m, n, a, b) * ints.wmnij(m, n, i, j);
                        for (int e = 0; e < reference.n_virt; ++e)
                            for (int f = 0; f < reference.n_virt; ++f)
                                value += 0.5 * tau_cache.tau(i, j, e, f) * ints.wabef(a, b, e, f);
                        for (int m = 0; m < reference.n_occ; ++m)
                            for (int e = 0; e < reference.n_virt; ++e)
                            {
                                value += amps.t2(i, m, a, e) * ints.wmbej(m, b, e, j);
                                value -= amps.t2(i, m, b, e) * ints.wmbej(m, a, e, j);
                                value -= amps.t2(j, m, a, e) * ints.wmbej(m, b, e, i);
                                value += amps.t2(j, m, b, e) * ints.wmbej(m, a, e, i);

                                value -= amps.t1(i, e) * amps.t1(m, a) * blocks.ovvo(m, b, e, j);
                                value += amps.t1(i, e) * amps.t1(m, b) * blocks.ovvo(m, a, e, j);
                                value += amps.t1(j, e) * amps.t1(m, a) * blocks.ovvo(m, b, e, i);
                                value -= amps.t1(j, e) * amps.t1(m, b) * blocks.ovvo(m, a, e, i);
                            }
                        out.r2(i, j, a, b) = value;
                    }

        return out;
    }

    double compute_rccsd_correlation_energy(
        const ProductionSpinOrbitalReference &reference,
        const ProductionSpinOrbitalBlocks &blocks,
        const RCCSDAmplitudes &amps)
    {
        double energy = 0.0;
        for (int i = 0; i < reference.n_occ; ++i)
            for (int j = 0; j < reference.n_occ; ++j)
                for (int a = 0; a < reference.n_virt; ++a)
                    for (int b = 0; b < reference.n_virt; ++b)
                    {
                        const double gijab = blocks.oovv(i, j, a, b);
                        energy += 0.25 * gijab * amps.t2(i, j, a, b);
                        energy += 0.5 * gijab * amps.t1(i, a) * amps.t1(j, b);
                    }
        return energy;
    }

    double compute_rccsdt_stage_correlation_energy(
        const ProductionSpinOrbitalReference &reference,
        const ProductionSpinOrbitalBlocks &blocks,
        const RCCSDTAmplitudes &amps)
    {
        double energy = 0.0;
        for (int i = 0; i < reference.n_occ; ++i)
            for (int j = 0; j < reference.n_occ; ++j)
                for (int a = 0; a < reference.n_virt; ++a)
                    for (int b = 0; b < reference.n_virt; ++b)
                    {
                        const double gijab = blocks.oovv(i, j, a, b);
                        energy += 0.25 * gijab * amps.t2(i, j, a, b);
                        energy += 0.5 * gijab * amps.t1(i, a) * amps.t1(j, b);
                    }
        return energy;
    }

    Eigen::VectorXd pack_amplitudes(const RCCSDAmplitudes &amps)
    {
        Eigen::VectorXd packed(static_cast<Eigen::Index>(amps.t1.size() + amps.t2.size()));
        Eigen::Index offset = 0;
        for (const double value : amps.t1.data)
            packed(offset++) = value;
        for (const double value : amps.t2.data)
            packed(offset++) = value;
        return packed;
    }

    void unpack_amplitudes(const Eigen::VectorXd &packed, RCCSDAmplitudes &amps)
    {
        Eigen::Index offset = 0;
        for (double &value : amps.t1.data)
            value = packed(offset++);
        for (double &value : amps.t2.data)
            value = packed(offset++);
    }

    Eigen::VectorXd pack_residuals(const RCCSDResiduals &residuals)
    {
        Eigen::VectorXd packed(static_cast<Eigen::Index>(residuals.r1.size() + residuals.r2.size()));
        Eigen::Index offset = 0;
        for (const double value : residuals.r1.data)
            packed(offset++) = value;
        for (const double value : residuals.r2.data)
            packed(offset++) = value;
        return packed;
    }

    Eigen::VectorXd pack_rccsdt_amplitudes(const RCCSDTAmplitudes &amps)
    {
        Eigen::VectorXd packed(static_cast<Eigen::Index>(
            amps.t1.size() + amps.t2.size() + amps.t3.size()));
        Eigen::Index offset = 0;
        for (const double value : amps.t1.data)
            packed(offset++) = value;
        for (const double value : amps.t2.data)
            packed(offset++) = value;
        for (const double value : amps.t3.data)
            packed(offset++) = value;
        return packed;
    }

    void unpack_rccsdt_amplitudes(
        const Eigen::VectorXd &packed,
        RCCSDTAmplitudes &amps)
    {
        Eigen::Index offset = 0;
        for (double &value : amps.t1.data)
            value = packed(offset++);
        for (double &value : amps.t2.data)
            value = packed(offset++);
        for (double &value : amps.t3.data)
            value = packed(offset++);
    }

    Eigen::VectorXd pack_rccsdt_stage_residuals(
        const RCCSDResiduals &sd_residuals,
        const Tensor6D &r3)
    {
        Eigen::VectorXd packed(static_cast<Eigen::Index>(
            sd_residuals.r1.size() + sd_residuals.r2.size() + r3.size()));
        Eigen::Index offset = 0;
        for (const double value : sd_residuals.r1.data)
            packed(offset++) = value;
        for (const double value : sd_residuals.r2.data)
            packed(offset++) = value;
        for (const double value : r3.data)
            packed(offset++) = value;
        return packed;
    }

    void initialize_mp2_guess(
        const ProductionSpinOrbitalBlocks &blocks,
        const TensorRCCSDTState &prepared,
        RCCSDAmplitudes &amps)
    {
        for (int i = 0; i < amps.t2.dim1; ++i)
            for (int j = 0; j < amps.t2.dim2; ++j)
                for (int a = 0; a < amps.t2.dim3; ++a)
                    for (int b = 0; b < amps.t2.dim4; ++b)
                        amps.t2(i, j, a, b) = blocks.oovv(i, j, a, b) /
                                              prepared.denominators.d2(
                                                  spatial_index(i), spatial_index(j),
                                                  spatial_index(a), spatial_index(b));
    }

    std::expected<TensorRCCSDResult, std::string> run_tensor_rccsd_stage(
        HartreeFock::Calculator &calculator,
        const TensorRCCSDTState &state)
    {
        const ProductionSpinOrbitalReference so_ref = build_spin_orbital_reference(state.reference);
        const ProductionSpinOrbitalBlocks so_blocks = build_spin_orbital_blocks(state.reference, state.mo_blocks);

        RCCSDAmplitudes amps{
            .t1 = Tensor2D(so_ref.n_occ, so_ref.n_virt, 0.0),
            .t2 = Tensor4D(so_ref.n_occ, so_ref.n_occ, so_ref.n_virt, so_ref.n_virt, 0.0),
        };
        initialize_mp2_guess(so_blocks, state, amps);

        const unsigned int max_iter = calculator._scf.get_max_cycles(calculator._shells.nbasis());
        const double tol_energy = calculator._scf._tol_energy;
        const double tol_residual = calculator._scf._tol_density;
        const bool use_diis = calculator._scf._use_DIIS;
        const double damping = 0.8;

        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "RCCSDT[TENSOR] :",
            std::format("Stage-1 RCCSD warm start dimensions: nocc={} nvirt={}",
                        so_ref.n_occ, so_ref.n_virt));

        AmplitudeDIIS diis(static_cast<int>(std::max(2u, calculator._scf._DIIS_dim)));
        double energy = compute_rccsd_correlation_energy(so_ref, so_blocks, amps);
        double previous_energy = energy;

        for (unsigned int iter = 1; iter <= max_iter; ++iter)
        {
            const auto iter_start = std::chrono::steady_clock::now();

            const TauCache tau_cache = build_tau_cache(amps);
            const RCCSDIntermediates ints = build_intermediates(so_ref, so_blocks, amps, tau_cache);
            const RCCSDResiduals residuals = build_residuals(so_ref, so_blocks, amps, tau_cache, ints);
            const Eigen::VectorXd residual_vec = pack_residuals(residuals);
            const double residual_rms = rms_norm(residual_vec);

            Eigen::VectorXd current = pack_amplitudes(amps);
            Eigen::VectorXd updated = current;

            Eigen::Index offset = 0;
            for (int i = 0; i < so_ref.n_occ; ++i)
                for (int a = 0; a < so_ref.n_virt; ++a)
                {
                    updated(offset) += damping * residuals.r1(i, a) /
                                       state.denominators.d1(spatial_index(i), spatial_index(a));
                    ++offset;
                }

            for (int i = 0; i < so_ref.n_occ; ++i)
                for (int j = 0; j < so_ref.n_occ; ++j)
                    for (int a = 0; a < so_ref.n_virt; ++a)
                        for (int b = 0; b < so_ref.n_virt; ++b)
                        {
                            updated(offset) += damping * residuals.r2(i, j, a, b) /
                                               state.denominators.d2(
                                                   spatial_index(i), spatial_index(j),
                                                   spatial_index(a), spatial_index(b));
                            ++offset;
                        }

            const Eigen::VectorXd update_delta = updated - current;
            const double update_rms = rms_norm(update_delta);

            diis.push(updated, residual_vec);
            if (use_diis && diis.ready())
            {
                auto diis_res = diis.extrapolate();
                if (diis_res)
                    updated = std::move(*diis_res);
            }

            unpack_amplitudes(updated, amps);
            energy = compute_rccsd_correlation_energy(so_ref, so_blocks, amps);
            const double delta_energy = energy - previous_energy;
            previous_energy = energy;

            const double time_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - iter_start).count();

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "RCCSDT[TENSOR-RCCSD] :",
                std::format(
                    "{:3d}  E_corr={:.10f}  dE={:+.3e}  rms(res)={:.3e}  rms(step)={:.3e}  diis={}  t={:.3f}s",
                    iter, energy, delta_energy, residual_rms, update_rms, diis.size(), time_sec));

            if (std::abs(delta_energy) < tol_energy && residual_rms < tol_residual)
            {
                return TensorRCCSDResult{
                    .amplitudes = std::move(amps),
                    .correlation_energy = energy,
                    .iterations = iter,
                };
            }
        }

        return std::unexpected(
            std::format("run_tensor_rccsd_stage: failed to converge in {} iterations.", max_iter));
    }

    void seed_triples_from_rccsd(
        const TensorRCCSDResult &rccsd,
        TensorTriplesWorkspace &triples)
    {
        for (std::size_t idx = 0; idx < rccsd.amplitudes.t1.data.size(); ++idx)
            triples.amplitudes.t1.data[idx] = rccsd.amplitudes.t1.data[idx];
        for (std::size_t idx = 0; idx < rccsd.amplitudes.t2.data.size(); ++idx)
            triples.amplitudes.t2.data[idx] = rccsd.amplitudes.t2.data[idx];
    }

    [[nodiscard]] RCCSDAmplitudes extract_sd_amplitudes(
        const TensorTriplesWorkspace &triples)
    {
        RCCSDAmplitudes amps{
            .t1 = Tensor2D(
                triples.amplitudes.t1.dim1,
                triples.amplitudes.t1.dim2,
                0.0),
            .t2 = Tensor4D(
                triples.amplitudes.t2.dim1,
                triples.amplitudes.t2.dim2,
                triples.amplitudes.t2.dim3,
                triples.amplitudes.t2.dim4,
                0.0),
        };

        amps.t1.data = triples.amplitudes.t1.data;
        amps.t2.data = triples.amplitudes.t2.data;
        return amps;
    }

    void store_sd_amplitudes(
        const RCCSDAmplitudes &amps,
        TensorTriplesWorkspace &triples)
    {
        triples.amplitudes.t1.data = amps.t1.data;
        triples.amplitudes.t2.data = amps.t2.data;
    }

    void build_r3_diagonal_feedback_family(
        const CanonicalRHFCCReference &reference,
        TensorTriplesWorkspace &triples)
    {
        if (!triples.allocated)
            return;

        std::fill(triples.r3.data.begin(), triples.r3.data.end(), 0.0);

        // This is the first production-side R3 contribution wired into the
        // staged tensor backend: the diagonal/preconditioner family. On its own
        // it is not a complete CCSDT residual, but it establishes the storage,
        // indexing, and denominator path that later connected triples terms will
        // accumulate into.
        for (int i = 0; i < triples.amplitudes.t3.dim1; ++i)
            for (int j = 0; j < triples.amplitudes.t3.dim2; ++j)
                for (int k = 0; k < triples.amplitudes.t3.dim3; ++k)
                    for (int a = 0; a < triples.amplitudes.t3.dim4; ++a)
                        for (int b = 0; b < triples.amplitudes.t3.dim5; ++b)
                            for (int c = 0; c < triples.amplitudes.t3.dim6; ++c)
                                triples.r3(i, j, k, a, b, c) -=
                                    d3_on_demand(reference, i, j, k, a, b, c) *
                                    triples.amplitudes.t3(i, j, k, a, b, c);
    }

    [[nodiscard]] double bare_vooo(
        const ProductionSpinOrbitalBlocks &blocks,
        int a, int l, int i, int j) noexcept
    {
        // PySCF's RHF code starts from the T1-dressed tensor element
        // `t1_eris[a,l,i,j]`. In our spin-orbital storage we do not keep a
        // dedicated `vooo` block, so we reconstruct the bare antisymmetrized
        // piece from `ooov` using `<al||ij> = -<ij||la>`.
        return -blocks.ooov(i, j, l, a);
    }

    [[nodiscard]] double bare_vvvo(
        const ProductionSpinOrbitalBlocks &blocks,
        int a, int b, int d, int j) noexcept
    {
        // Likewise, `vvvo` is reconstructed from `ovvv` via
        // `<ab||dj> = <dj||ab> = -<jd||ab>`.
        return -blocks.ovvv(j, d, a, b);
    }

    struct PySCFT3SourceIntermediates
    {
        Tensor4D w_vooo; // (a,l,i,j)
        Tensor4D w_vvvo; // (a,b,d,j)
    };

    PySCFT3SourceIntermediates build_pyscf_t3_source_intermediates(
        const RCCSDTAmplitudes &amps,
        const ProductionSpinOrbitalBlocks &blocks,
        const RCCSDIntermediates &ints)
    {
        PySCFT3SourceIntermediates out{
            .w_vooo = Tensor4D(
                amps.t2.dim3, amps.t2.dim1, amps.t2.dim1, amps.t2.dim2, 0.0),
            .w_vvvo = Tensor4D(
                amps.t2.dim3, amps.t2.dim4, amps.t2.dim4, amps.t2.dim2, 0.0),
        };

        auto c_t2 = [&](int i, int j, int a, int b) noexcept -> double
        {
            return 2.0 * amps.t2(i, j, a, b) - amps.t2(i, j, b, a);
        };

        // This follows the local PySCF `pyscf.cc.rccsdt.intermediates_t3`
        // structure directly: first form dressed `W_vooo` and `W_vvvo`, then
        // contract those into the triples residual. Because our backend keeps
        // bare spin-orbital ERIs instead of PySCF's T1-dressed spatial ERIs,
        // we make the T1 dressing explicit through the CCSD intermediates.
        for (int a = 0; a < out.w_vooo.dim1; ++a)
            for (int l = 0; l < out.w_vooo.dim2; ++l)
                for (int i = 0; i < out.w_vooo.dim3; ++i)
                    for (int j = 0; j < out.w_vooo.dim4; ++j)
                    {
                        double value = bare_vooo(blocks, a, l, i, j);

                        for (int d = 0; d < amps.t2.dim4; ++d)
                            value += ints.fme(l, d) * amps.t2(i, j, a, d);

                        for (int m = 0; m < amps.t2.dim1; ++m)
                            for (int d = 0; d < amps.t2.dim4; ++d)
                            {
                                value += (-blocks.ooov(m, l, j, d)) * c_t2(m, i, d, a);
                                value -= 0.5 * blocks.ooov(m, l, j, d) * c_t2(m, i, d, a);
                                value -= 0.5 * blocks.ooov(m, l, j, d) * amps.t2(i, m, d, a);
                                value -= blocks.ooov(m, l, i, d) * amps.t2(j, m, d, a);
                            }

                        for (int d = 0; d < amps.t2.dim3; ++d)
                            for (int e = 0; e < amps.t2.dim4; ++e)
                                value += (-blocks.ovvv(l, a, d, e)) * amps.t2(i, j, d, e);

                        for (int e = 0; e < amps.t1.dim2; ++e)
                        {
                            value += amps.t1(i, e) * ints.wmbej(l, a, e, j);
                            value -= amps.t1(j, e) * ints.wmbej(l, a, e, i);
                        }
                        for (int m = 0; m < amps.t1.dim1; ++m)
                            value -= amps.t1(m, a) * ints.wmnij(m, l, i, j);

                        for (int m = 0; m < amps.t3.dim1; ++m)
                            for (int e = 0; e < amps.t3.dim4; ++e)
                                for (int d = 0; d < amps.t3.dim6; ++d)
                                    value += blocks.oovv(i, m, d, e) *
                                             amps.t3(m, j, l, e, a, d);

                        out.w_vooo(a, l, i, j) = value;
                    }

        for (int a = 0; a < out.w_vvvo.dim1; ++a)
            for (int b = 0; b < out.w_vvvo.dim2; ++b)
                for (int d = 0; d < out.w_vvvo.dim3; ++d)
                    for (int j = 0; j < out.w_vvvo.dim4; ++j)
                    {
                        double value = bare_vvvo(blocks, a, b, d, j);

                        for (int l = 0; l < amps.t2.dim1; ++l)
                            for (int e = 0; e < amps.t2.dim4; ++e)
                            {
                                value += blocks.ovvv(l, a, e, d) * c_t2(l, j, e, b);
                                value -= 0.5 * blocks.ovvv(l, a, d, e) * c_t2(l, j, e, b);
                                value -= 0.5 * blocks.ovvv(l, a, d, e) * amps.t2(j, l, e, b);
                                value -= blocks.ovvv(l, b, d, e) * amps.t2(j, l, e, a);
                            }

                        for (int l = 0; l < amps.t2.dim1; ++l)
                            for (int m = 0; m < amps.t2.dim2; ++m)
                                value -= blocks.ooov(l, m, j, d) * amps.t2(l, m, a, b);

                        for (int e = 0; e < amps.t1.dim2; ++e)
                            value += amps.t1(j, e) * ints.wabef(a, b, d, e);
                        for (int m = 0; m < amps.t1.dim1; ++m)
                        {
                            value -= amps.t1(m, a) * ints.wmbej(m, b, d, j);
                            value += amps.t1(m, b) * ints.wmbej(m, a, d, j);
                        }

                        for (int l = 0; l < amps.t3.dim1; ++l)
                            for (int m = 0; m < amps.t3.dim2; ++m)
                                for (int e = 0; e < amps.t3.dim4; ++e)
                                    value -= blocks.oovv(l, m, d, e) *
                                             amps.t3(m, j, l, e, b, a);

                        out.w_vvvo(a, b, d, j) = value;
                    }

        return out;
    }

    void build_r3_connected_t2_source_families(
        const RCCSDTAmplitudes &amps,
        const PySCFT3SourceIntermediates &source_ints,
        TensorTriplesWorkspace &triples)
    {
        if (!triples.allocated)
            return;

        // This is the exact family layout used by the local PySCF RCCSDT
        // implementation: build `W_vvvo` and `W_vooo`, then add the six
        // permutation-completed contractions with T2. Keeping the contractions
        // explicit here makes the mapping to `pyscf/cc/rccsdt.py:compute_r3_tri`
        // straightforward for students and future optimizations.
        for (int i = 0; i < triples.r3.dim1; ++i)
            for (int j = 0; j < triples.r3.dim2; ++j)
                for (int k = 0; k < triples.r3.dim3; ++k)
                    for (int a = 0; a < triples.r3.dim4; ++a)
                        for (int b = 0; b < triples.r3.dim5; ++b)
                            for (int c = 0; c < triples.r3.dim6; ++c)
                            {
                                double connected = 0.0;

                                for (int d = 0; d < amps.t2.dim4; ++d)
                                {
                                    connected += source_ints.w_vvvo(a, b, d, j) *
                                                 amps.t2(i, k, d, c);
                                    connected += source_ints.w_vvvo(a, c, d, k) *
                                                 amps.t2(i, j, d, b);
                                    connected += source_ints.w_vvvo(b, a, d, i) *
                                                 amps.t2(j, k, d, c);
                                    connected += source_ints.w_vvvo(b, c, d, k) *
                                                 amps.t2(j, i, d, a);
                                    connected += source_ints.w_vvvo(c, a, d, i) *
                                                 amps.t2(k, j, d, b);
                                    connected += source_ints.w_vvvo(c, b, d, j) *
                                                 amps.t2(k, i, d, a);
                                }

                                for (int l = 0; l < amps.t2.dim1; ++l)
                                {
                                    connected -= source_ints.w_vooo(a, l, i, j) *
                                                 amps.t2(l, k, b, c);
                                    connected -= source_ints.w_vooo(a, l, i, k) *
                                                 amps.t2(l, j, c, b);
                                    connected -= source_ints.w_vooo(b, l, j, i) *
                                                 amps.t2(l, k, a, c);
                                    connected -= source_ints.w_vooo(b, l, j, k) *
                                                 amps.t2(l, i, c, a);
                                    connected -= source_ints.w_vooo(c, l, k, i) *
                                                 amps.t2(l, j, a, b);
                                    connected -= source_ints.w_vooo(c, l, k, j) *
                                                 amps.t2(l, i, b, a);
                                }

                                triples.r3(i, j, k, a, b, c) += connected;
                            }
    }

    Tensor4D build_pyscf_wovov_intermediate(
        const RCCSDTAmplitudes &amps,
        const ProductionSpinOrbitalBlocks &blocks)
    {
        Tensor4D out(
            amps.t2.dim1, amps.t2.dim3, amps.t2.dim1, amps.t2.dim4, 0.0);

        for (int l = 0; l < out.dim1; ++l)
            for (int a = 0; a < out.dim2; ++a)
                for (int i = 0; i < out.dim3; ++i)
                    for (int d = 0; d < out.dim4; ++d)
                    {
                        double value = blocks.ovov(l, a, i, d);
                        for (int m = 0; m < amps.t2.dim1; ++m)
                            for (int e = 0; e < amps.t2.dim4; ++e)
                                value -= blocks.oovv(m, l, d, e) *
                                         amps.t2(i, m, e, a);
                        out(l, a, i, d) = value;
                    }

        return out;
    }

    void build_r3_pyscf_mixed_dressing_families(
        const RCCSDTAmplitudes &amps,
        const RCCSDIntermediates &ints,
        const Tensor4D &w_ovov,
        TensorTriplesWorkspace &triples)
    {
        if (!triples.allocated)
            return;

        // This mirrors the mixed `W_ovvo/W_ovov * T3` grouping in the local
        // PySCF `compute_r3_tri` implementation. The terms are written in the
        // same family order as the source code there, but on our full
        // spin-orbital tensors instead of PySCF's triangular RHF storage.
        for (int i = 0; i < triples.r3.dim1; ++i)
            for (int j = 0; j < triples.r3.dim2; ++j)
                for (int k = 0; k < triples.r3.dim3; ++k)
                    for (int a = 0; a < triples.r3.dim4; ++a)
                        for (int b = 0; b < triples.r3.dim5; ++b)
                            for (int c = 0; c < triples.r3.dim6; ++c)
                            {
                                double dressed = 0.0;

                                for (int l = 0; l < amps.t3.dim1; ++l)
                                    for (int d = 0; d < amps.t3.dim4; ++d)
                                    {
                                        dressed += 0.5 * ints.wmbej(l, a, d, i) *
                                                   amps.t3(l, j, k, d, b, c);
                                        dressed += 0.5 * ints.wmbej(l, b, d, j) *
                                                   amps.t3(l, i, k, d, a, c);
                                        dressed += 0.5 * ints.wmbej(l, c, d, k) *
                                                   amps.t3(l, j, i, d, b, a);

                                        dressed -= w_ovov(l, b, i, d) *
                                                   amps.t3(j, l, k, d, a, c);
                                        dressed -= w_ovov(l, c, i, d) *
                                                   amps.t3(k, l, j, d, a, b);
                                        dressed -= 0.5 * w_ovov(l, a, i, d) *
                                                   amps.t3(j, l, k, d, b, c);

                                        dressed -= w_ovov(l, a, j, d) *
                                                   amps.t3(i, l, k, d, b, c);
                                        dressed -= w_ovov(l, c, j, d) *
                                                   amps.t3(k, l, i, d, b, a);
                                        dressed -= 0.5 * w_ovov(l, b, j, d) *
                                                   amps.t3(i, l, k, d, a, c);

                                        dressed -= w_ovov(l, a, k, d) *
                                                   amps.t3(i, l, j, d, c, b);
                                        dressed -= w_ovov(l, b, k, d) *
                                                   amps.t3(j, l, i, d, c, a);
                                        dressed -= 0.5 * w_ovov(l, c, k, d) *
                                                   amps.t3(i, l, j, d, a, b);
                                    }

                                triples.r3(i, j, k, a, b, c) += dressed;
                            }
    }

    void build_r3_one_body_dressing_families(
        const RCCSDTAmplitudes &amps,
        const RCCSDIntermediates &ints,
        TensorTriplesWorkspace &triples)
    {
        if (!triples.allocated)
            return;

        // These are the triples analogues of the familiar Fae/Fmi dressing
        // terms from CCSD. They keep the current T3 iterate coupled to the
        // dressed one-body intermediates built from the latest T1/T2 amplitudes.
        for (int i = 0; i < triples.r3.dim1; ++i)
            for (int j = 0; j < triples.r3.dim2; ++j)
                for (int k = 0; k < triples.r3.dim3; ++k)
                    for (int a = 0; a < triples.r3.dim4; ++a)
                        for (int b = 0; b < triples.r3.dim5; ++b)
                            for (int c = 0; c < triples.r3.dim6; ++c)
                            {
                                double dressed = 0.0;

                                for (const auto &virt_perm : kPermutations3)
                                {
                                    const int av = virt_perm.perm[0] == 0 ? a : (virt_perm.perm[0] == 1 ? b : c);
                                    const int bv = virt_perm.perm[1] == 0 ? a : (virt_perm.perm[1] == 1 ? b : c);
                                    const int cv = virt_perm.perm[2] == 0 ? a : (virt_perm.perm[2] == 1 ? b : c);
                                    for (int e = 0; e < amps.t3.dim6; ++e)
                                        dressed += virt_perm.sign *
                                                   amps.t3(i, j, k, av, bv, e) *
                                                   ints.fae(cv, e);
                                }

                                for (const auto &occ_perm : kPermutations3)
                                {
                                    const int io = occ_perm.perm[0] == 0 ? i : (occ_perm.perm[0] == 1 ? j : k);
                                    const int jo = occ_perm.perm[1] == 0 ? i : (occ_perm.perm[1] == 1 ? j : k);
                                    const int ko = occ_perm.perm[2] == 0 ? i : (occ_perm.perm[2] == 1 ? j : k);
                                    for (int m = 0; m < amps.t3.dim3; ++m)
                                        dressed -= occ_perm.sign *
                                                   amps.t3(io, jo, m, a, b, c) *
                                                   ints.fmi(m, ko);
                                }

                                triples.r3(i, j, k, a, b, c) += dressed;
                            }
    }

    void build_r3_two_body_dressing_families(
        const RCCSDTAmplitudes &amps,
        const RCCSDIntermediates &ints,
        TensorTriplesWorkspace &triples)
    {
        if (!triples.allocated)
            return;

        // The first T3 self-dressing layer keeps only the cleanest Wmnij and
        // Wabef contractions. These are the highest-value "same-rank" terms:
        // they are easy to map back to the algebra and materially improve the
        // iterative triples problem beyond the bare T2 source terms.
        for (int i = 0; i < triples.r3.dim1; ++i)
            for (int j = 0; j < triples.r3.dim2; ++j)
                for (int k = 0; k < triples.r3.dim3; ++k)
                    for (int a = 0; a < triples.r3.dim4; ++a)
                        for (int b = 0; b < triples.r3.dim5; ++b)
                            for (int c = 0; c < triples.r3.dim6; ++c)
                            {
                                double dressed = 0.0;

                                for (const auto &virt_perm : kPermutations3)
                                {
                                    const int av = virt_perm.perm[0] == 0 ? a : (virt_perm.perm[0] == 1 ? b : c);
                                    const int bv = virt_perm.perm[1] == 0 ? a : (virt_perm.perm[1] == 1 ? b : c);
                                    const int cv = virt_perm.perm[2] == 0 ? a : (virt_perm.perm[2] == 1 ? b : c);
                                    for (int e = 0; e < amps.t3.dim5; ++e)
                                        for (int f = 0; f < amps.t3.dim6; ++f)
                                            dressed += 0.5 * virt_perm.sign *
                                                       amps.t3(i, j, k, av, e, f) *
                                                       ints.wabef(bv, cv, e, f);
                                }

                                for (const auto &occ_perm : kPermutations3)
                                {
                                    const int io = occ_perm.perm[0] == 0 ? i : (occ_perm.perm[0] == 1 ? j : k);
                                    const int jo = occ_perm.perm[1] == 0 ? i : (occ_perm.perm[1] == 1 ? j : k);
                                    const int ko = occ_perm.perm[2] == 0 ? i : (occ_perm.perm[2] == 1 ? j : k);
                                    for (int m = 0; m < amps.t3.dim2; ++m)
                                        for (int n = 0; n < amps.t3.dim3; ++n)
                                            dressed += 0.5 * occ_perm.sign *
                                                       amps.t3(io, m, n, a, b, c) *
                                                       ints.wmnij(m, n, jo, ko);
                                }

                                triples.r3(i, j, k, a, b, c) += dressed;
                            }
    }

    void build_r3_mixed_wmbej_family(
        const RCCSDTAmplitudes &amps,
        const RCCSDIntermediates &ints,
        TensorTriplesWorkspace &triples)
    {
        if (!triples.allocated)
            return;

        // W_mbej is the mixed occupied/virtual interaction block that already
        // drives a large fraction of the CCSD doubles residual. This is the
        // natural next self-dressing family for T3 because it couples one
        // occupied and one virtual index of the current triples tensor through
        // a dressed two-body interaction. Written with explicit permutations,
        // the algebra stays close to the textbook P(ij)P(ab)-style structure.
        for (int i = 0; i < triples.r3.dim1; ++i)
            for (int j = 0; j < triples.r3.dim2; ++j)
                for (int k = 0; k < triples.r3.dim3; ++k)
                    for (int a = 0; a < triples.r3.dim4; ++a)
                        for (int b = 0; b < triples.r3.dim5; ++b)
                            for (int c = 0; c < triples.r3.dim6; ++c)
                            {
                                double dressed = 0.0;

                                for (const auto &occ_perm : kPermutations3)
                                {
                                    const int io = occ_perm.perm[0] == 0 ? i : (occ_perm.perm[0] == 1 ? j : k);
                                    const int jo = occ_perm.perm[1] == 0 ? i : (occ_perm.perm[1] == 1 ? j : k);
                                    const int ko = occ_perm.perm[2] == 0 ? i : (occ_perm.perm[2] == 1 ? j : k);

                                    for (const auto &virt_perm : kPermutations3)
                                    {
                                        const int av = virt_perm.perm[0] == 0 ? a : (virt_perm.perm[0] == 1 ? b : c);
                                        const int bv = virt_perm.perm[1] == 0 ? a : (virt_perm.perm[1] == 1 ? b : c);
                                        const int cv = virt_perm.perm[2] == 0 ? a : (virt_perm.perm[2] == 1 ? b : c);
                                        const int sign = occ_perm.sign * virt_perm.sign;

                                        for (int m = 0; m < amps.t3.dim3; ++m)
                                            for (int e = 0; e < amps.t3.dim6; ++e)
                                                dressed += sign *
                                                           amps.t3(io, jo, m, av, bv, e) *
                                                           ints.wmbej(m, cv, e, ko);
                                    }
                                }

                                triples.r3(i, j, k, a, b, c) += dressed;
                            }
    }

    void build_r3_ovov_family(
        const RCCSDTAmplitudes &amps,
        const ProductionSpinOrbitalBlocks &blocks,
        TensorTriplesWorkspace &triples)
    {
        if (!triples.allocated)
            return;

        // This is the explicit W_ovov * T3 family from the connected triples
        // residual. In diagram language, one occupied and one virtual index of
        // T3 are tied together by the antisymmetrized ovov interaction block.
        // We write it as a signed permutation sum so students can still see the
        // P(ij)P(ab)-style structure directly.
        for (int i = 0; i < triples.r3.dim1; ++i)
            for (int j = 0; j < triples.r3.dim2; ++j)
                for (int k = 0; k < triples.r3.dim3; ++k)
                    for (int a = 0; a < triples.r3.dim4; ++a)
                        for (int b = 0; b < triples.r3.dim5; ++b)
                            for (int c = 0; c < triples.r3.dim6; ++c)
                            {
                                double dressed = 0.0;

                                for (const auto &occ_perm : kPermutations3)
                                {
                                    const int io = occ_perm.perm[0] == 0 ? i : (occ_perm.perm[0] == 1 ? j : k);
                                    const int jo = occ_perm.perm[1] == 0 ? i : (occ_perm.perm[1] == 1 ? j : k);
                                    const int ko = occ_perm.perm[2] == 0 ? i : (occ_perm.perm[2] == 1 ? j : k);

                                    for (const auto &virt_perm : kPermutations3)
                                    {
                                        const int av = virt_perm.perm[0] == 0 ? a : (virt_perm.perm[0] == 1 ? b : c);
                                        const int bv = virt_perm.perm[1] == 0 ? a : (virt_perm.perm[1] == 1 ? b : c);
                                        const int cv = virt_perm.perm[2] == 0 ? a : (virt_perm.perm[2] == 1 ? b : c);
                                        const int sign = occ_perm.sign * virt_perm.sign;

                                        for (int m = 0; m < amps.t3.dim2; ++m)
                                            for (int e = 0; e < amps.t3.dim5; ++e)
                                                dressed += sign *
                                                           blocks.ovov(m, bv, io, e) *
                                                           amps.t3(jo, m, ko, av, e, cv);
                                    }
                                }

                                triples.r3(i, j, k, a, b, c) += dressed;
                            }
    }

    void build_r3_ovvo_family(
        const RCCSDTAmplitudes &amps,
        const ProductionSpinOrbitalBlocks &blocks,
        TensorTriplesWorkspace &triples)
    {
        if (!triples.allocated)
            return;

        // This companion mixed family uses the ovvo block instead of ovov.
        // In index language it is the same "one occupied and one virtual line
        // couple through a two-electron interaction" picture, but with the
        // virtual indices connected in the exchanged order. Keeping the term
        // separate here makes it easy to compare the two mixed families side
        // by side when reading the residual builder.
        for (int i = 0; i < triples.r3.dim1; ++i)
            for (int j = 0; j < triples.r3.dim2; ++j)
                for (int k = 0; k < triples.r3.dim3; ++k)
                    for (int a = 0; a < triples.r3.dim4; ++a)
                        for (int b = 0; b < triples.r3.dim5; ++b)
                            for (int c = 0; c < triples.r3.dim6; ++c)
                            {
                                double dressed = 0.0;

                                for (const auto &occ_perm : kPermutations3)
                                {
                                    const int io = occ_perm.perm[0] == 0 ? i : (occ_perm.perm[0] == 1 ? j : k);
                                    const int jo = occ_perm.perm[1] == 0 ? i : (occ_perm.perm[1] == 1 ? j : k);
                                    const int ko = occ_perm.perm[2] == 0 ? i : (occ_perm.perm[2] == 1 ? j : k);

                                    for (const auto &virt_perm : kPermutations3)
                                    {
                                        const int av = virt_perm.perm[0] == 0 ? a : (virt_perm.perm[0] == 1 ? b : c);
                                        const int bv = virt_perm.perm[1] == 0 ? a : (virt_perm.perm[1] == 1 ? b : c);
                                        const int cv = virt_perm.perm[2] == 0 ? a : (virt_perm.perm[2] == 1 ? b : c);
                                        const int sign = occ_perm.sign * virt_perm.sign;

                                        for (int m = 0; m < amps.t3.dim2; ++m)
                                            for (int e = 0; e < amps.t3.dim6; ++e)
                                                dressed += sign *
                                                           blocks.ovvo(m, bv, e, io) *
                                                           amps.t3(jo, m, ko, av, cv, e);
                                    }
                                }

                                triples.r3(i, j, k, a, b, c) += dressed;
                            }
    }

    [[nodiscard]] double update_t3_from_r3_jacobi(
        const CanonicalRHFCCReference &reference,
        TensorTriplesWorkspace &triples,
        double damping)
    {
        if (!triples.allocated)
            return 0.0;

        double sum_sq = 0.0;
        std::size_t count = 0;

        for (int i = 0; i < triples.amplitudes.t3.dim1; ++i)
            for (int j = 0; j < triples.amplitudes.t3.dim2; ++j)
                for (int k = 0; k < triples.amplitudes.t3.dim3; ++k)
                    for (int a = 0; a < triples.amplitudes.t3.dim4; ++a)
                        for (int b = 0; b < triples.amplitudes.t3.dim5; ++b)
                            for (int c = 0; c < triples.amplitudes.t3.dim6; ++c)
                            {
                                const double denom = d3_on_demand(reference, i, j, k, a, b, c);
                                if (std::abs(denom) < 1e-12)
                                    continue;
                                const double delta =
                                    damping * triples.r3(i, j, k, a, b, c) / denom;
                                triples.amplitudes.t3(i, j, k, a, b, c) += delta;
                                sum_sq += delta * delta;
                                ++count;
                            }

        if (count == 0)
            return 0.0;
        return std::sqrt(sum_sq / static_cast<double>(count));
    }

    void build_r2_t3_feedback_family(
        const RCCSDTAmplitudes &amps,
        const RCCSDIntermediates &ints,
        const ProductionSpinOrbitalBlocks &blocks,
        Tensor4D &r2_feedback)
    {
        std::fill(r2_feedback.data.begin(), r2_feedback.data.end(), 0.0);

        // This mirrors the local PySCF `r1r2_add_t3_tri_` structure for the
        // T3 -> R2 bridge: one `F_me * T3` family, one particle-scattering
        // `ovvv * T3` family, and one hole-scattering `ooov * T3` family.
        // Because our residual is stored in full spin-orbital form, we keep
        // the P(ij) and P(ab) partners explicit instead of restoring symmetry
        // in a later spatial-orbital post-pass.
        for (int i = 0; i < r2_feedback.dim1; ++i)
            for (int j = 0; j < r2_feedback.dim2; ++j)
                for (int a = 0; a < r2_feedback.dim3; ++a)
                    for (int b = 0; b < r2_feedback.dim4; ++b)
                    {
                        double one_body = 0.0;
                        for (int m = 0; m < amps.t3.dim3; ++m)
                            for (int e = 0; e < amps.t3.dim6; ++e)
                            {
                                one_body += amps.t3(i, j, m, a, b, e) * ints.fme(m, e);
                                one_body -= amps.t3(j, i, m, a, b, e) * ints.fme(m, e);
                                one_body -= amps.t3(i, j, m, b, a, e) * ints.fme(m, e);
                                one_body += amps.t3(j, i, m, b, a, e) * ints.fme(m, e);
                            }

                        double particle = 0.0;
                        for (int m = 0; m < amps.t3.dim3; ++m)
                            for (int e = 0; e < amps.t3.dim5; ++e)
                                for (int c = 0; c < amps.t3.dim6; ++c)
                                {
                                    particle += amps.t3(i, j, m, a, e, c) *
                                                blocks.ovvv(m, b, e, c);
                                    particle -= amps.t3(i, j, m, b, e, c) *
                                                blocks.ovvv(m, a, e, c);
                                }

                        double hole = 0.0;
                        for (int m = 0; m < amps.t3.dim2; ++m)
                            for (int n = 0; n < amps.t3.dim3; ++n)
                                for (int e = 0; e < amps.t3.dim6; ++e)
                                {
                                    hole += amps.t3(i, m, n, a, b, e) *
                                            blocks.ooov(m, n, j, e);
                                    hole -= amps.t3(j, m, n, a, b, e) *
                                            blocks.ooov(m, n, i, e);
                                }

                        r2_feedback(i, j, a, b) =
                            one_body + 0.5 * particle - 0.5 * hole;
                    }
    }

    void build_r1_t3_feedback_family(
        const RCCSDTAmplitudes &amps,
        const ProductionSpinOrbitalBlocks &blocks,
        Tensor2D &r1_feedback)
    {
        std::fill(r1_feedback.data.begin(), r1_feedback.data.end(), 0.0);

        // The local PySCF RCCSDT implementation adds the T3 -> R1 correction
        // through the contracted `oovv * T3` family. We keep that exact
        // structure here and do not add extra heuristic mixed terms on top.
        for (int i = 0; i < r1_feedback.dim1; ++i)
            for (int a = 0; a < r1_feedback.dim2; ++a)
            {
                double value = 0.0;
                for (int m = 0; m < amps.t3.dim2; ++m)
                    for (int n = 0; n < amps.t3.dim3; ++n)
                        for (int e = 0; e < amps.t3.dim5; ++e)
                            for (int f = 0; f < amps.t3.dim6; ++f)
                                value += 0.25 *
                                         blocks.oovv(m, n, e, f) *
                                         amps.t3(i, m, n, a, e, f);
                r1_feedback(i, a) = value;
            }
    }

    void add_feedback_into_residuals(
        const Tensor2D &r1_feedback,
        const Tensor4D &r2_feedback,
        RCCSDResiduals &residuals)
    {
        for (std::size_t idx = 0; idx < residuals.r1.data.size(); ++idx)
            residuals.r1.data[idx] += r1_feedback.data[idx];
        for (std::size_t idx = 0; idx < residuals.r2.data.size(); ++idx)
            residuals.r2.data[idx] += r2_feedback.data[idx];
    }

    struct SDUpdateMetrics
    {
        double t1_step_rms = 0.0;
        double t2_step_rms = 0.0;
    };

    SDUpdateMetrics update_sd_amplitudes_with_feedback(
        HartreeFock::Calculator &calculator,
        const TensorRCCSDTState &state,
        const RCCSDResiduals &residuals,
        RCCSDAmplitudes &amps,
        AmplitudeDIIS &diis,
        double damping)
    {
        Eigen::VectorXd current = pack_amplitudes(amps);
        Eigen::VectorXd updated = current;

        Eigen::Index offset = 0;
        for (int i = 0; i < amps.t1.dim1; ++i)
            for (int a = 0; a < amps.t1.dim2; ++a)
            {
                const double denom = state.denominators.d1(
                    spatial_index(i), spatial_index(a));
                if (std::abs(denom) >= 1e-12)
                    updated(offset) += damping * residuals.r1(i, a) / denom;
                ++offset;
            }

        for (int i = 0; i < amps.t2.dim1; ++i)
            for (int j = 0; j < amps.t2.dim2; ++j)
                for (int a = 0; a < amps.t2.dim3; ++a)
                    for (int b = 0; b < amps.t2.dim4; ++b)
                    {
                        const double denom = state.denominators.d2(
                            spatial_index(i), spatial_index(j),
                            spatial_index(a), spatial_index(b));
                        if (std::abs(denom) >= 1e-12)
                            updated(offset) += damping * residuals.r2(i, j, a, b) / denom;
                        ++offset;
                    }

        const Eigen::VectorXd residual_vec = pack_residuals(residuals);
        diis.push(updated, residual_vec);
        if (calculator._scf._use_DIIS && diis.ready())
        {
            auto diis_res = diis.extrapolate();
            if (diis_res)
                updated = std::move(*diis_res);
        }

        RCCSDAmplitudes old_amps{
            .t1 = Tensor2D(amps.t1.dim1, amps.t1.dim2, 0.0),
            .t2 = Tensor4D(amps.t2.dim1, amps.t2.dim2, amps.t2.dim3, amps.t2.dim4, 0.0),
        };
        old_amps.t1.data = amps.t1.data;
        old_amps.t2.data = amps.t2.data;

        unpack_amplitudes(updated, amps);

        Tensor2D t1_delta(amps.t1.dim1, amps.t1.dim2, 0.0);
        Tensor4D t2_delta(amps.t2.dim1, amps.t2.dim2, amps.t2.dim3, amps.t2.dim4, 0.0);
        for (std::size_t idx = 0; idx < amps.t1.data.size(); ++idx)
            t1_delta.data[idx] = amps.t1.data[idx] - old_amps.t1.data[idx];
        for (std::size_t idx = 0; idx < amps.t2.data.size(); ++idx)
            t2_delta.data[idx] = amps.t2.data[idx] - old_amps.t2.data[idx];

        return {
            .t1_step_rms = tensor_rms(t1_delta),
            .t2_step_rms = tensor_rms(t2_delta),
        };
    }

    std::expected<TensorTriplesStageMetrics, std::string> run_staged_tensor_triples_iterations(
        HartreeFock::Calculator &calculator,
        const TensorRCCSDTState &state,
        const ProductionSpinOrbitalBlocks &so_blocks,
        TensorTriplesWorkspace &triples,
        unsigned int max_stage_iterations,
        bool require_convergence)
    {
        if (!triples.allocated)
            return std::unexpected(
                "run_staged_tensor_triples_iterations: triples workspace is not allocated.");

        constexpr double kT3Damping = 0.35;
        constexpr double kSDDamping = 0.25;
        // The staged tensor path is meant to become the production solver for
        // larger systems, so it should not stop at a tolerance inherited from
        // the more forgiving SCF density threshold. We keep the criterion
        // modest enough for the current incomplete residual, but tight enough
        // that the stage performs real coupled refinement before handing off
        // to the moderate-case fallback.
        const double tol_stage =
            std::max(1e-7, calculator._scf._tol_density);
        const ProductionSpinOrbitalReference so_ref =
            build_spin_orbital_reference(state.reference);

        Tensor2D r1_feedback(
            triples.amplitudes.t1.dim1,
            triples.amplitudes.t1.dim2,
            0.0);
        Tensor4D r2_feedback(
            triples.amplitudes.t2.dim1,
            triples.amplitudes.t2.dim2,
            triples.amplitudes.t2.dim3,
            triples.amplitudes.t2.dim4,
            0.0);
        TensorTriplesStageMetrics metrics;
        TensorTriplesStageMetrics best_metrics;
        RCCSDTAmplitudes best_amplitudes = clone_rccsdt_amplitudes(triples.amplitudes);
        bool have_best_iterate = false;
        double best_stage_score = std::numeric_limits<double>::infinity();
        unsigned int stale_iterations = 0;
        AmplitudeDIIS diis(static_cast<int>(std::max(2u, calculator._scf._DIIS_dim)));
        AmplitudeDIIS full_diis(static_cast<int>(std::max(2u, calculator._scf._DIIS_dim)));

        for (unsigned int iter = 1; iter <= max_stage_iterations; ++iter)
        {
            RCCSDAmplitudes sd_amps = extract_sd_amplitudes(triples);
            const TauCache tau_cache = build_tau_cache(sd_amps);
            const RCCSDIntermediates ints =
                build_intermediates(so_ref,
                                    so_blocks,
                                    sd_amps,
                                    tau_cache);
            const PySCFT3SourceIntermediates source_ints =
                build_pyscf_t3_source_intermediates(
                    triples.amplitudes,
                    so_blocks,
                    ints);
            const Tensor4D w_ovov =
                build_pyscf_wovov_intermediate(
                    triples.amplitudes,
                    so_blocks);
            RCCSDResiduals residuals = build_residuals(
                so_ref, so_blocks, sd_amps, tau_cache, ints);

            build_r3_diagonal_feedback_family(state.reference, triples);
            build_r3_connected_t2_source_families(
                triples.amplitudes, source_ints, triples);
            build_r3_one_body_dressing_families(
                triples.amplitudes, ints, triples);
            build_r3_two_body_dressing_families(
                triples.amplitudes, ints, triples);
            build_r3_pyscf_mixed_dressing_families(
                triples.amplitudes, ints, w_ovov, triples);
            metrics.r3_rms = triples_residual_rms(triples.r3);
            metrics.t3_step_rms = update_t3_from_r3_jacobi(
                state.reference, triples, kT3Damping);

            build_r1_t3_feedback_family(
                triples.amplitudes, so_blocks, r1_feedback);
            metrics.r1_feedback_rms = tensor_rms(r1_feedback);

            build_r2_t3_feedback_family(
                triples.amplitudes, ints, so_blocks, r2_feedback);
            metrics.r2_feedback_rms = tensor_rms(r2_feedback);

            add_feedback_into_residuals(r1_feedback, r2_feedback, residuals);
            metrics.sd_residual_rms = rms_norm(pack_residuals(residuals));
            const SDUpdateMetrics sd_update = update_sd_amplitudes_with_feedback(
                calculator, state, residuals, sd_amps, diis, kSDDamping);
            metrics.t1_step_rms = sd_update.t1_step_rms;
            metrics.t2_step_rms = sd_update.t2_step_rms;
            store_sd_amplitudes(sd_amps, triples);

            const Eigen::VectorXd full_residual_vec =
                pack_rccsdt_stage_residuals(residuals, triples.r3);
            const Eigen::VectorXd current_full =
                pack_rccsdt_amplitudes(triples.amplitudes);
            Eigen::VectorXd extrapolated_full = current_full;

            full_diis.push(current_full, full_residual_vec);
            if (calculator._scf._use_DIIS && full_diis.ready())
            {
                auto diis_res = full_diis.extrapolate();
                if (diis_res)
                    extrapolated_full = std::move(*diis_res);
            }
            unpack_rccsdt_amplitudes(extrapolated_full, triples.amplitudes);
            metrics.estimated_correlation_energy =
                compute_rccsdt_stage_correlation_energy(
                    so_ref, so_blocks, triples.amplitudes);
            const double score = stage_quality_score(metrics);
            metrics.quality_score = score;
            if (score + 1e-12 < best_stage_score)
            {
                best_stage_score = score;
                best_metrics = metrics;
                best_metrics.best_iteration = iter;
                best_amplitudes = clone_rccsdt_amplitudes(triples.amplitudes);
                have_best_iterate = true;
                stale_iterations = 0;
            }
            else
            {
                ++stale_iterations;
            }
            metrics.iterations = iter;

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "RCCSDT[TENSOR-T3] :",
                std::format(
                    "{:3d}  E_est={:.10f}  rms(SD)={:.3e}  rms(R3)={:.3e}  rms(dT3)={:.3e}  rms(R1[T3])={:.3e}  rms(dT1)={:.3e}  rms(R2[T3])={:.3e}  rms(dT2)={:.3e}",
                    iter,
                    metrics.estimated_correlation_energy,
                    metrics.sd_residual_rms,
                    metrics.r3_rms,
                    metrics.t3_step_rms,
                    metrics.r1_feedback_rms,
                    metrics.t1_step_rms,
                    metrics.r2_feedback_rms,
                    metrics.t2_step_rms));

            if (metrics.r3_rms < tol_stage &&
                metrics.sd_residual_rms < tol_stage &&
                metrics.r1_feedback_rms < tol_stage &&
                metrics.r2_feedback_rms < tol_stage)
            {
                metrics.converged = true;
                break;
            }

            if (!require_convergence &&
                iter >= 4 && stale_iterations >= 2 &&
                score > 1.05 * best_stage_score)
                break;

            if (require_convergence &&
                iter >= 8 && stale_iterations >= 4 &&
                score > 1.10 * best_stage_score)
                break;
        }

        if (have_best_iterate)
        {
            triples.amplitudes = std::move(best_amplitudes);
            metrics.sd_residual_rms = best_metrics.sd_residual_rms;
            metrics.r3_rms = best_metrics.r3_rms;
            metrics.t3_step_rms = best_metrics.t3_step_rms;
            metrics.r1_feedback_rms = best_metrics.r1_feedback_rms;
            metrics.t1_step_rms = best_metrics.t1_step_rms;
            metrics.r2_feedback_rms = best_metrics.r2_feedback_rms;
            metrics.t2_step_rms = best_metrics.t2_step_rms;
            metrics.quality_score = best_metrics.quality_score;
            metrics.estimated_correlation_energy = best_metrics.estimated_correlation_energy;
            metrics.best_iteration = best_metrics.best_iteration;
            metrics.converged = best_metrics.quality_score < tol_stage;
        }

        return metrics;
    }
} // namespace

namespace HartreeFock::Correlation::CC
{
    std::expected<CanonicalRHFCCReference, std::string> build_canonical_rhf_cc_reference(
        HartreeFock::Calculator &calculator)
    {
        auto ref_res = build_rhf_reference(calculator);
        if (!ref_res)
            return std::unexpected(ref_res.error());

        const RHFReference &base = *ref_res;

        Eigen::MatrixXd C_full(base.n_ao, base.n_mo);
        C_full.leftCols(base.n_occ) = base.C_occ;
        C_full.rightCols(base.n_virt) = base.C_virt;

        const Eigen::MatrixXd fock_mo =
            C_full.transpose() * calculator._info._scf.alpha.fock * C_full;

        CanonicalRHFCCReference reference{
            .orbital_partition = std::move(*ref_res),
            .f_oo = Tensor2D(base.n_occ, base.n_occ, 0.0),
            .f_ov = Tensor2D(base.n_occ, base.n_virt, 0.0),
            .f_vv = Tensor2D(base.n_virt, base.n_virt, 0.0),
        };

        for (int i = 0; i < base.n_occ; ++i)
            for (int j = 0; j < base.n_occ; ++j)
                reference.f_oo(i, j) = fock_mo(i, j);

        for (int i = 0; i < base.n_occ; ++i)
            for (int a = 0; a < base.n_virt; ++a)
                reference.f_ov(i, a) = fock_mo(i, base.n_occ + a);

        for (int a = 0; a < base.n_virt; ++a)
            for (int b = 0; b < base.n_virt; ++b)
                reference.f_vv(a, b) = fock_mo(base.n_occ + a, base.n_occ + b);

        return reference;
    }

    std::expected<TensorCCBlockCache, std::string> build_tensor_cc_block_cache(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const CanonicalRHFCCReference &reference,
        const std::string &tag)
    {
        std::vector<double> eri_local;
        const std::vector<double> &eri = HartreeFock::Correlation::ensure_eri(
            calculator, shell_pairs, eri_local, tag);

        const RHFReference &partition = reference.orbital_partition;
        const std::size_t nb = static_cast<std::size_t>(partition.n_ao);

        TensorCCBlockCache blocks;
        try
        {
            blocks.oooo = Tensor4D(
                partition.n_occ, partition.n_occ, partition.n_occ, partition.n_occ,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    partition.C_occ, partition.C_occ,
                    partition.C_occ, partition.C_occ));
            append_block_memory(blocks.memory_report, blocks.total_bytes,
                                "oooo", {partition.n_occ, partition.n_occ, partition.n_occ, partition.n_occ});

            blocks.ooov = Tensor4D(
                partition.n_occ, partition.n_occ, partition.n_occ, partition.n_virt,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    partition.C_occ, partition.C_occ,
                    partition.C_occ, partition.C_virt));
            append_block_memory(blocks.memory_report, blocks.total_bytes,
                                "ooov", {partition.n_occ, partition.n_occ, partition.n_occ, partition.n_virt});

            blocks.oovv = Tensor4D(
                partition.n_occ, partition.n_occ, partition.n_virt, partition.n_virt,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    partition.C_occ, partition.C_occ,
                    partition.C_virt, partition.C_virt));
            append_block_memory(blocks.memory_report, blocks.total_bytes,
                                "oovv", {partition.n_occ, partition.n_occ, partition.n_virt, partition.n_virt});

            blocks.ovov = Tensor4D(
                partition.n_occ, partition.n_virt, partition.n_occ, partition.n_virt,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    partition.C_occ, partition.C_virt,
                    partition.C_occ, partition.C_virt));
            append_block_memory(blocks.memory_report, blocks.total_bytes,
                                "ovov", {partition.n_occ, partition.n_virt, partition.n_occ, partition.n_virt});

            blocks.ovvo = Tensor4D(
                partition.n_occ, partition.n_virt, partition.n_virt, partition.n_occ,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    partition.C_occ, partition.C_virt,
                    partition.C_virt, partition.C_occ));
            append_block_memory(blocks.memory_report, blocks.total_bytes,
                                "ovvo", {partition.n_occ, partition.n_virt, partition.n_virt, partition.n_occ});

            blocks.ovvv = Tensor4D(
                partition.n_occ, partition.n_virt, partition.n_virt, partition.n_virt,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    partition.C_occ, partition.C_virt,
                    partition.C_virt, partition.C_virt));
            append_block_memory(blocks.memory_report, blocks.total_bytes,
                                "ovvv", {partition.n_occ, partition.n_virt, partition.n_virt, partition.n_virt});

            blocks.vvvv = Tensor4D(
                partition.n_virt, partition.n_virt, partition.n_virt, partition.n_virt,
                HartreeFock::Correlation::transform_eri(
                    eri, nb,
                    partition.C_virt, partition.C_virt,
                    partition.C_virt, partition.C_virt));
            append_block_memory(blocks.memory_report, blocks.total_bytes,
                                "vvvv", {partition.n_virt, partition.n_virt, partition.n_virt, partition.n_virt});
        }
        catch (const std::exception &ex)
        {
            return std::unexpected("build_tensor_cc_block_cache: " + std::string(ex.what()));
        }

        return blocks;
    }

    RCCSDTBackend choose_rccsdt_backend(
        const RHFReference &reference) noexcept
    {
        constexpr int kPrototypeMaxSpinOrbitals = 12;
        constexpr std::size_t kPrototypeMaxDeterminants = 1200;

        const int n_spin_orb = 2 * reference.n_mo;
        const int n_electrons = 2 * reference.n_occ;
        const std::size_t ndet = binomial(
            static_cast<std::size_t>(n_spin_orb),
            static_cast<std::size_t>(n_electrons));

        if (n_spin_orb <= kPrototypeMaxSpinOrbitals &&
            ndet <= kPrototypeMaxDeterminants)
            return RCCSDTBackend::DeterminantPrototype;

        return RCCSDTBackend::TensorProduction;
    }

    std::string format_tensor_memory_summary(
        const TensorRCCSDTState &state)
    {
        std::ostringstream out;
        out << "Tensor RCCSDT memory estimate:";
        for (const TensorMemoryBlock &block : state.mo_blocks.memory_report)
            out << std::format(" {}={}", block.label, format_bytes(block.bytes));
        out << std::format(" T3~={} total_integrals={}",
                           format_bytes(state.estimated_t3_bytes),
                           format_bytes(state.mo_blocks.total_bytes));
        if (state.triples.allocated)
            out << std::format(" triples_workspace={}", format_bytes(state.triples.storage_bytes));
        return out.str();
    }

    std::expected<void, std::string> allocate_dense_triples_workspace(
        TensorRCCSDTState &state)
    {
        if (state.triples.allocated)
            return {};

        constexpr std::size_t kDenseTriplesSoftCapBytes =
            static_cast<std::size_t>(1024) * 1024 * 1024;
        if (2 * state.estimated_t3_bytes > kDenseTriplesSoftCapBytes)
            return std::unexpected(
                std::format("allocate_dense_triples_workspace: dense T3/R3 workspace would require about {}; current phase-1 tensor path is capped at {}.",
                            format_bytes(2 * state.estimated_t3_bytes),
                            format_bytes(kDenseTriplesSoftCapBytes)));

        try
        {
            const RHFReference &partition = state.reference.orbital_partition;
            const int nocc_so = 2 * partition.n_occ;
            const int nvirt_so = 2 * partition.n_virt;
            state.triples.amplitudes = RCCSDTAmplitudes{
                .t1 = Tensor2D(nocc_so, nvirt_so, 0.0),
                .t2 = Tensor4D(nocc_so, nocc_so, nvirt_so, nvirt_so, 0.0),
                .t3 = Tensor6D(nocc_so, nocc_so, nocc_so,
                               nvirt_so, nvirt_so, nvirt_so, 0.0),
            };
            state.triples.r3 = Tensor6D(
                nocc_so, nocc_so, nocc_so,
                nvirt_so, nvirt_so, nvirt_so, 0.0);
            state.triples.allocated = true;
            state.triples.storage_bytes =
                bytes_for_tensor({nocc_so, nvirt_so}) +
                bytes_for_tensor({nocc_so, nocc_so, nvirt_so, nvirt_so}) +
                2 * bytes_for_tensor({nocc_so, nocc_so, nocc_so,
                                      nvirt_so, nvirt_so, nvirt_so});
        }
        catch (const std::exception &ex)
        {
            return std::unexpected("allocate_dense_triples_workspace: " + std::string(ex.what()));
        }

        return {};
    }

    std::expected<TensorRCCSDTState, std::string> prepare_tensor_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        if (calculator._calculation != HartreeFock::CalculationType::SinglePoint)
            return std::unexpected("prepare_tensor_rccsdt: tensor RCCSDT is currently available only for single-point calculations.");

        auto ref_res = build_canonical_rhf_cc_reference(calculator);
        if (!ref_res)
            return std::unexpected(ref_res.error());

        auto blocks_res = build_tensor_cc_block_cache(
            calculator, shell_pairs, *ref_res, "RCCSDT[TENSOR] :");
        if (!blocks_res)
            return std::unexpected(blocks_res.error());

        auto denom_res = build_denominator_cache(ref_res->orbital_partition, false);
        if (!denom_res)
            return std::unexpected(denom_res.error());

        TensorRCCSDTState state{
            .reference = std::move(*ref_res),
            .mo_blocks = std::move(*blocks_res),
            .denominators = std::move(*denom_res),
            .estimated_t3_elements = 0,
            .estimated_t3_bytes = 0,
        };

        try
        {
            const RHFReference &partition = state.reference.orbital_partition;
            const int nocc_so = 2 * partition.n_occ;
            const int nvirt_so = 2 * partition.n_virt;
            state.estimated_t3_elements = checked_product(
                {nocc_so, nocc_so, nocc_so,
                 nvirt_so, nvirt_so, nvirt_so});
            state.estimated_t3_bytes = state.estimated_t3_elements * sizeof(double);
        }
        catch (const std::exception &ex)
        {
            return std::unexpected("prepare_tensor_rccsdt: " + std::string(ex.what()));
        }

        auto triples_res = allocate_dense_triples_workspace(state);
        if (!triples_res)
            return std::unexpected(triples_res.error());

        return state;
    }

    std::expected<void, std::string> run_tensor_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        auto state_res = prepare_tensor_rccsdt(calculator, shell_pairs);
        if (!state_res)
            return std::unexpected(state_res.error());

        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "RCCSDT[TENSOR] :",
            format_tensor_memory_summary(*state_res));

        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "RCCSDT[TENSOR] :",
            "Running the production-path RCCSD warm start before enabling T3 residuals.");
        HartreeFock::Logger::blank();

        auto rccsd_res = run_tensor_rccsd_stage(calculator, *state_res);
        if (!rccsd_res)
            return std::unexpected("run_tensor_rccsdt: " + rccsd_res.error());

        state_res->warm_start_correlation_energy = rccsd_res->correlation_energy;
        state_res->warm_start_iterations = rccsd_res->iterations;
        const ProductionSpinOrbitalBlocks so_blocks =
            build_spin_orbital_blocks(state_res->reference, state_res->mo_blocks);
        const DeterminantBackstopDecision backstop =
            choose_determinant_backstop(state_res->reference);
        const unsigned int stage_iteration_limit = backstop.enabled
            ? 8u
            : std::min(
                  20u,
                  std::max(12u, calculator._scf.get_max_cycles(calculator._shells.nbasis())));
        seed_triples_from_rccsd(*rccsd_res, state_res->triples);
        auto staged_triples_res = run_staged_tensor_triples_iterations(
            calculator,
            *state_res,
            so_blocks,
            state_res->triples,
            stage_iteration_limit,
            !backstop.enabled);
        if (!staged_triples_res)
            return std::unexpected("run_tensor_rccsdt: " + staged_triples_res.error());

        HartreeFock::Logger::blank();
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "RCCSDT[TENSOR] :",
            std::format("Stage-1 RCCSD warm start converged in {} iterations with E_corr={:.10f}.",
                        rccsd_res->iterations, rccsd_res->correlation_energy));
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "RCCSDT[TENSOR] :",
            std::format("Dense T3/R3 workspace allocated ({}); staged triples loop ran {} steps, kept the best iterate from step {}, and reports rms(R3)={:.3e} and rms(R2[T3])={:.3e}.",
                        format_bytes(state_res->triples.storage_bytes),
                        staged_triples_res->iterations,
                        staged_triples_res->best_iteration == 0
                            ? staged_triples_res->iterations
                            : staged_triples_res->best_iteration,
                        staged_triples_res->r3_rms,
                        staged_triples_res->r2_feedback_rms));
        if (backstop.enabled)
        {
            HartreeFock::Logger::blank();
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "RCCSDT[TENSOR] :",
                std::format(
                    "Using the determinant-space CCSDT backstop to finish this moderate-size case (nso={} ndet={}) while the tensor residual engine is still being completed.",
                    backstop.n_spin_orb,
                    backstop.determinants));

            auto full_blocks_res = build_mo_block_cache(
                calculator,
                shell_pairs,
                state_res->reference.orbital_partition,
                "RCCSDT[DET-BACKSTOP] :");
            if (!full_blocks_res)
                return std::unexpected(
                    "run_tensor_rccsdt: determinant backstop failed while building the full MO block cache: " +
                    full_blocks_res.error());

            auto system_res = build_rhf_spin_orbital_system(
                calculator,
                state_res->reference.orbital_partition,
                *full_blocks_res);
            if (!system_res)
                return std::unexpected(
                    "run_tensor_rccsdt: determinant backstop failed while building the spin-orbital Hamiltonian: " +
                    system_res.error());

            const DeterminantCCSpinOrbitalSeed seed{
                .t1 = &state_res->triples.amplitudes.t1,
                .t2 = &state_res->triples.amplitudes.t2,
                .t3 = &state_res->triples.amplitudes.t3,
            };

            auto corr_res = solve_determinant_cc(
                calculator,
                *system_res,
                3,
                "RCCSDT[DET-BACKSTOP] :",
                &seed);
            if (!corr_res)
                return std::unexpected(
                    "run_tensor_rccsdt: determinant backstop failed while solving CCSDT: " +
                    corr_res.error());

            calculator._correlation_energy = *corr_res;
            return {};
        }

        if (!staged_triples_res->converged)
        {
            return std::unexpected(
                std::format(
                    "run_tensor_rccsdt: no determinant backstop is available for this larger system, and the standalone tensor RCCSDT iterations did not converge (best rms(R3)={:.3e}, best rms(SD)={:.3e}, best rms(R2[T3])={:.3e}) within {} staged steps.",
                    staged_triples_res->r3_rms,
                    staged_triples_res->sd_residual_rms,
                    staged_triples_res->r2_feedback_rms,
                    staged_triples_res->iterations));
        }

        HartreeFock::Logger::blank();
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "RCCSDT[TENSOR] :",
            std::format(
                "No determinant backstop is available for this larger system, so the converged standalone tensor RCCSDT result is used directly after {} staged steps.",
                staged_triples_res->iterations));

        calculator._correlation_energy = staged_triples_res->estimated_correlation_energy;
        return {};
    }
} // namespace HartreeFock::Correlation::CC
