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

    struct ProductionSpinOrbitalChemistsBlocks
    {
        Tensor4D oooo;
        Tensor4D ooov;
        Tensor4D oovv;
        Tensor4D ovov;
        Tensor4D ovvo;
        Tensor4D ovvv;
        Tensor4D vvvv;
    };

    // These full-MO chemists-notation containers mirror the local PySCF
    // RCCSDT data flow closely enough that we can rebuild the larger-system
    // tensor solver around the same dressed-intermediate sequence later,
    // while still keeping the storage explicit and teachable in Planck.
    struct ProductionSpinOrbitalChemistsSystem
    {
        int n_mo = 0;
        int n_occ = 0;
        int n_virt = 0;
        Tensor2D fock;  // [p,q] chemists/MO ordering
        Tensor4D eri;   // [p,r,q,s] to match PySCF eris.pppp storage
    };

    struct DressedSpinOrbitalSystem
    {
        Tensor2D fock;  // [p,q]
        Tensor4D eri;   // [p,r,q,s] to match PySCF's t1_eris storage
    };

    struct DressedSinglesDoublesIntermediates
    {
        Tensor2D f_oo;
        Tensor2D f_vv;
        Tensor4D w_oooo;
        Tensor4D w_ovvo;
        Tensor4D w_ovov;
    };

    struct DressedTriplesIntermediates
    {
        Tensor2D f_oo;
        Tensor2D f_vv;
        Tensor4D w_oooo;
        Tensor4D w_ovvo;
        Tensor4D w_ovov;
        Tensor4D w_vooo;
        Tensor4D w_vvvo;
        Tensor4D w_vvvv;
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
        double energy_change = 0.0;
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
        // The raw T3 -> R1/R2 correction norms are useful diagnostics, but
        // they are not standalone convergence criteria: in the full CCSDT
        // equations those terms are just one part of the SD residual. Use the
        // actual SD and T3 residual magnitudes, together with the step norms,
        // to rank staged iterates.
        return std::max(
            std::max(metrics.sd_residual_rms, metrics.r3_rms),
            std::max(
                std::max(metrics.t1_step_rms, metrics.t2_step_rms),
                metrics.t3_step_rms));
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

    constexpr std::array<std::array<int, 3>, 6> kPermutations3 = {{
        {{0, 1, 2}},
        {{0, 2, 1}},
        {{1, 0, 2}},
        {{1, 2, 0}},
        {{2, 0, 1}},
        {{2, 1, 0}},
    }};

    [[nodiscard]] double t3_p201(
        const Tensor6D &t3,
        int i, int j, int k,
        int a, int b, int c) noexcept
    {
        return 2.0 * t3(i, j, k, a, b, c) -
               t3(i, j, k, b, a, c) -
               t3(i, j, k, c, b, a);
    }

    [[nodiscard]] double t3_p422(
        const Tensor6D &t3,
        int i, int j, int k,
        int a, int b, int c) noexcept
    {
        return 4.0 * t3(i, j, k, a, b, c) -
               2.0 * t3(i, j, k, a, c, b) -
               2.0 * t3(i, j, k, b, a, c) +
               t3(i, j, k, b, c, a) +
               t3(i, j, k, c, a, b) -
               2.0 * t3(i, j, k, c, b, a);
    }

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

    ProductionSpinOrbitalChemistsBlocks build_spin_orbital_chemists_blocks(
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

        ProductionSpinOrbitalChemistsBlocks blocks{
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
                                 : 0.0);

        for (int i = 0; i < so.n_occ; ++i)
            for (int j = 0; j < so.n_occ; ++j)
                for (int k = 0; k < so.n_occ; ++k)
                    for (int a = 0; a < so.n_virt; ++a)
                        blocks.ooov(i, j, k, a) =
                            (same_spin(i, k) && same_spin(j, a)
                                 ? spatial.ooov(occ(i), occ(k), occ(j), virt(a))
                                 : 0.0);

        for (int i = 0; i < so.n_occ; ++i)
            for (int j = 0; j < so.n_occ; ++j)
                for (int a = 0; a < so.n_virt; ++a)
                    for (int b = 0; b < so.n_virt; ++b)
                        blocks.oovv(i, j, a, b) =
                            (same_spin(i, a) && same_spin(j, b)
                                 ? spatial.ovov(occ(i), virt(a), occ(j), virt(b))
                                 : 0.0);

        for (int i = 0; i < so.n_occ; ++i)
            for (int a = 0; a < so.n_virt; ++a)
                for (int j = 0; j < so.n_occ; ++j)
                    for (int b = 0; b < so.n_virt; ++b)
                        blocks.ovov(i, a, j, b) =
                            (same_spin(i, j) && same_spin(a, b)
                                 ? spatial.oovv(occ(i), occ(j), virt(a), virt(b))
                                 : 0.0);

        for (int i = 0; i < so.n_occ; ++i)
            for (int a = 0; a < so.n_virt; ++a)
                for (int b = 0; b < so.n_virt; ++b)
                    for (int j = 0; j < so.n_occ; ++j)
                        blocks.ovvo(i, a, b, j) =
                            (same_spin(i, b) && same_spin(a, j)
                                 ? spatial.ovvo(occ(i), virt(b), virt(a), occ(j))
                                 : 0.0);

        for (int i = 0; i < so.n_occ; ++i)
            for (int a = 0; a < so.n_virt; ++a)
                for (int b = 0; b < so.n_virt; ++b)
                    for (int c = 0; c < so.n_virt; ++c)
                        blocks.ovvv(i, a, b, c) =
                            (same_spin(i, b) && same_spin(a, c)
                                 ? spatial.ovvv(occ(i), virt(b), virt(a), virt(c))
                                 : 0.0);

        for (int a = 0; a < so.n_virt; ++a)
            for (int b = 0; b < so.n_virt; ++b)
                for (int c = 0; c < so.n_virt; ++c)
                    for (int d = 0; d < so.n_virt; ++d)
                        blocks.vvvv(a, b, c, d) =
                            (same_spin(a, c) && same_spin(b, d)
                                 ? spatial.vvvv(virt(a), virt(c), virt(b), virt(d))
                                 : 0.0);

        return blocks;
    }

    std::expected<ProductionSpinOrbitalChemistsSystem, std::string>
    build_spin_orbital_chemists_system(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const CanonicalRHFCCReference &reference)
    {
        std::vector<double> eri_local;
        const std::vector<double> &eri_ao =
            HartreeFock::Correlation::ensure_eri(
                calculator, shell_pairs, eri_local, "RCCSDT[TENSOR] :");

        const RHFReference &partition = reference.orbital_partition;
        const int nmo_spatial = partition.n_mo;
        const int nmo_so = 2 * nmo_spatial;
        const int nocc_so = 2 * partition.n_occ;
        const int nvirt_so = 2 * partition.n_virt;

        Eigen::MatrixXd c_full(partition.n_ao, partition.n_mo);
        c_full.leftCols(partition.n_occ) = partition.C_occ;
        c_full.rightCols(partition.n_virt) = partition.C_virt;
        const Eigen::MatrixXd fock_mo =
            c_full.transpose() * calculator._info._scf.alpha.fock * c_full;

        try
        {
            const std::vector<double> spatial_mo_eri =
                HartreeFock::Correlation::transform_eri(
                    eri_ao,
                    static_cast<std::size_t>(partition.n_ao),
                    c_full, c_full, c_full, c_full);
            Tensor4D spatial_pppp(
                nmo_spatial, nmo_spatial, nmo_spatial, nmo_spatial,
                spatial_mo_eri);

            ProductionSpinOrbitalChemistsSystem system{
                .n_mo = nmo_so,
                .n_occ = nocc_so,
                .n_virt = nvirt_so,
                .fock = Tensor2D(nmo_so, nmo_so, 0.0),
                .eri = Tensor4D(nmo_so, nmo_so, nmo_so, nmo_so, 0.0),
            };

            for (int p = 0; p < nmo_so; ++p)
                for (int q = 0; q < nmo_so; ++q)
                    system.fock(p, q) =
                        same_spin(p, q)
                            ? fock_mo(spatial_index(p), spatial_index(q))
                            : 0.0;

            // PySCF stores `eris.pppp` as `(p r | q s)`, i.e. the full
            // chemists tensor transposed to `[p,r,q,s]`.  Keeping that layout
            // here lets the later dressed-system builders follow the PySCF
            // equations directly without hidden index swaps.
            for (int p = 0; p < nmo_so; ++p)
                for (int r = 0; r < nmo_so; ++r)
                    for (int q = 0; q < nmo_so; ++q)
                        for (int s = 0; s < nmo_so; ++s)
                            system.eri(p, r, q, s) =
                                (same_spin(p, q) && same_spin(r, s))
                                    ? spatial_pppp(
                                          spatial_index(p),
                                          spatial_index(q),
                                          spatial_index(r),
                                          spatial_index(s))
                                    : 0.0;

            return system;
        }
        catch (const std::exception &ex)
        {
            return std::unexpected(
                "build_spin_orbital_chemists_system: " + std::string(ex.what()));
        }
    }

    std::expected<ProductionSpinOrbitalChemistsSystem, std::string>
    build_restricted_spatial_system(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const CanonicalRHFCCReference &reference)
    {
        std::vector<double> eri_local;
        const std::vector<double> &eri_ao =
            HartreeFock::Correlation::ensure_eri(
                calculator, shell_pairs, eri_local, "RCCSDT[TENSOR] :");

        const RHFReference &partition = reference.orbital_partition;
        Eigen::MatrixXd c_full(partition.n_ao, partition.n_mo);
        c_full.leftCols(partition.n_occ) = partition.C_occ;
        c_full.rightCols(partition.n_virt) = partition.C_virt;
        const Eigen::MatrixXd fock_mo =
            c_full.transpose() * calculator._info._scf.alpha.fock * c_full;

        try
        {
            const std::vector<double> spatial_mo_eri =
                HartreeFock::Correlation::transform_eri(
                    eri_ao,
                    static_cast<std::size_t>(partition.n_ao),
                    c_full, c_full, c_full, c_full);
            Tensor4D chemists(
                partition.n_mo, partition.n_mo, partition.n_mo, partition.n_mo,
                spatial_mo_eri);

            ProductionSpinOrbitalChemistsSystem system{
                .n_mo = partition.n_mo,
                .n_occ = partition.n_occ,
                .n_virt = partition.n_virt,
                .fock = Tensor2D(partition.n_mo, partition.n_mo, 0.0),
                .eri = Tensor4D(partition.n_mo, partition.n_mo, partition.n_mo, partition.n_mo, 0.0),
            };

            for (int p = 0; p < partition.n_mo; ++p)
                for (int q = 0; q < partition.n_mo; ++q)
                    system.fock(p, q) = fock_mo(p, q);

            // Store the restricted MO ERIs in the same `[p,r,q,s]` layout that
            // PySCF uses for `eris.pppp`, so the later dressed builders can
            // follow the local RCCSDT equations directly.
            for (int p = 0; p < partition.n_mo; ++p)
                for (int r = 0; r < partition.n_mo; ++r)
                    for (int q = 0; q < partition.n_mo; ++q)
                        for (int s = 0; s < partition.n_mo; ++s)
                            system.eri(p, r, q, s) = chemists(p, q, r, s);

            return system;
        }
        catch (const std::exception &ex)
        {
            return std::unexpected(
                "build_restricted_spatial_system: " + std::string(ex.what()));
        }
    }

    [[nodiscard]] DressedSpinOrbitalSystem build_dressed_spin_orbital_system(
        const ProductionSpinOrbitalChemistsSystem &system,
        const RCCSDTAmplitudes &amps)
    {
        Eigen::MatrixXd x = Eigen::MatrixXd::Identity(system.n_mo, system.n_mo);
        Eigen::MatrixXd y = Eigen::MatrixXd::Identity(system.n_mo, system.n_mo);
        for (int i = 0; i < system.n_occ; ++i)
            for (int a = 0; a < system.n_virt; ++a)
            {
                x(system.n_occ + a, i) -= amps.t1(i, a);
                y(i, system.n_occ + a) += amps.t1(i, a);
            }

        Tensor2D undressed_fock(system.n_mo, system.n_mo, 0.0);
        for (int r = 0; r < system.n_mo; ++r)
            for (int s = 0; s < system.n_mo; ++s)
            {
                double value = system.fock(r, s);
                for (int i = 0; i < system.n_occ; ++i)
                    for (int a = 0; a < system.n_virt; ++a)
                    {
                        const int va = system.n_occ + a;
                        value += 2.0 * system.eri(r, i, s, va) * amps.t1(i, a);
                        value -= system.eri(r, i, va, s) * amps.t1(i, a);
                    }
                undressed_fock(r, s) = value;
            }

        DressedSpinOrbitalSystem dressed{
            .fock = Tensor2D(system.n_mo, system.n_mo, 0.0),
            .eri = Tensor4D(system.n_mo, system.n_mo, system.n_mo, system.n_mo, 0.0),
        };

        for (int p = 0; p < system.n_mo; ++p)
            for (int q = 0; q < system.n_mo; ++q)
            {
                double value = 0.0;
                for (int r = 0; r < system.n_mo; ++r)
                    for (int s = 0; s < system.n_mo; ++s)
                        value += x(p, r) * undressed_fock(r, s) * y(q, s);
                dressed.fock(p, q) = value;
            }

        Tensor4D stage1(system.n_mo, system.n_mo, system.n_mo, system.n_mo, 0.0);
        Tensor4D stage2(system.n_mo, system.n_mo, system.n_mo, system.n_mo, 0.0);
        Tensor4D stage3(system.n_mo, system.n_mo, system.n_mo, system.n_mo, 0.0);

        for (int p = 0; p < system.n_mo; ++p)
            for (int v = 0; v < system.n_mo; ++v)
                for (int u = 0; u < system.n_mo; ++u)
                    for (int w = 0; w < system.n_mo; ++w)
                    {
                        double value = 0.0;
                        for (int t = 0; t < system.n_mo; ++t)
                            value += x(p, t) * system.eri(t, v, u, w);
                        stage1(p, v, u, w) = value;
                    }

        for (int p = 0; p < system.n_mo; ++p)
            for (int r = 0; r < system.n_mo; ++r)
                for (int u = 0; u < system.n_mo; ++u)
                    for (int w = 0; w < system.n_mo; ++w)
                    {
                        double value = 0.0;
                        for (int v = 0; v < system.n_mo; ++v)
                            value += x(r, v) * stage1(p, v, u, w);
                        stage2(p, r, u, w) = value;
                    }

        for (int p = 0; p < system.n_mo; ++p)
            for (int r = 0; r < system.n_mo; ++r)
                for (int q = 0; q < system.n_mo; ++q)
                    for (int w = 0; w < system.n_mo; ++w)
                    {
                        double value = 0.0;
                        for (int u = 0; u < system.n_mo; ++u)
                            value += y(q, u) * stage2(p, r, u, w);
                        stage3(p, r, q, w) = value;
                    }

        for (int p = 0; p < system.n_mo; ++p)
            for (int r = 0; r < system.n_mo; ++r)
                for (int q = 0; q < system.n_mo; ++q)
                    for (int s = 0; s < system.n_mo; ++s)
                    {
                        double value = 0.0;
                        for (int w = 0; w < system.n_mo; ++w)
                            value += y(s, w) * stage3(p, r, q, w);
                        dressed.eri(p, r, q, s) = value;
                    }

        return dressed;
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

                                // Match PySCF GCCSD's P(ij)P(ab) antisymmetrized
                                // T1*T1*ovov correction exactly in the spin-orbital
                                // warm-start residual.
                                value += amps.t1(i, e) * amps.t1(m, a) * blocks.ovov(m, b, j, e);
                                value -= amps.t1(i, e) * amps.t1(m, b) * blocks.ovov(m, a, j, e);
                                value -= amps.t1(j, e) * amps.t1(m, a) * blocks.ovov(m, b, i, e);
                                value += amps.t1(j, e) * amps.t1(m, b) * blocks.ovov(m, a, i, e);
                            }
                        // GCCSD keeps the singles-driven ovvv and ooov pieces
                        // explicit in R2 rather than absorbing them into Wmbej.
                        for (int e = 0; e < reference.n_virt; ++e)
                        {
                            value += amps.t1(i, e) * blocks.ovvv(j, e, b, a);
                            value -= amps.t1(j, e) * blocks.ovvv(i, e, b, a);
                        }
                        for (int m = 0; m < reference.n_occ; ++m)
                        {
                            value -= amps.t1(m, a) * blocks.ooov(i, j, m, b);
                            value += amps.t1(m, b) * blocks.ooov(i, j, m, a);
                        }
                        out.r2(i, j, a, b) = value;
                    }

        return out;
    }

    [[nodiscard]] DressedSinglesDoublesIntermediates
    build_dressed_sd_intermediates(
        const ProductionSpinOrbitalChemistsSystem &system,
        const DressedSpinOrbitalSystem &dressed,
        const Tensor4D &t2)
    {
        DressedSinglesDoublesIntermediates ints{
            .f_oo = Tensor2D(system.n_occ, system.n_occ, 0.0),
            .f_vv = Tensor2D(system.n_virt, system.n_virt, 0.0),
            .w_oooo = Tensor4D(system.n_occ, system.n_occ, system.n_occ, system.n_occ, 0.0),
            .w_ovvo = Tensor4D(system.n_occ, system.n_virt, system.n_virt, system.n_occ, 0.0),
            .w_ovov = Tensor4D(system.n_occ, system.n_virt, system.n_occ, system.n_virt, 0.0),
        };
        const auto virt = [&system](int a) noexcept
        {
            return system.n_occ + a;
        };

        for (int b = 0; b < system.n_virt; ++b)
            for (int c = 0; c < system.n_virt; ++c)
            {
                double value = dressed.fock(virt(b), virt(c));
                for (int k = 0; k < system.n_occ; ++k)
                    for (int l = 0; l < system.n_occ; ++l)
                        for (int d = 0; d < system.n_virt; ++d)
                        {
                            value -= 2.0 * dressed.eri(k, l, virt(d), virt(c)) * t2(k, l, d, b);
                            value += dressed.eri(k, l, virt(c), virt(d)) * t2(k, l, d, b);
                        }
                ints.f_vv(b, c) = value;
            }

        for (int k = 0; k < system.n_occ; ++k)
            for (int j = 0; j < system.n_occ; ++j)
            {
                double value = dressed.fock(k, j);
                for (int l = 0; l < system.n_occ; ++l)
                    for (int c = 0; c < system.n_virt; ++c)
                        for (int d = 0; d < system.n_virt; ++d)
                        {
                            value += 2.0 * dressed.eri(l, k, virt(c), virt(d)) * t2(l, j, c, d);
                            value -= dressed.eri(l, k, virt(d), virt(c)) * t2(l, j, c, d);
                        }
                ints.f_oo(k, j) = value;
            }

        for (int k = 0; k < system.n_occ; ++k)
            for (int l = 0; l < system.n_occ; ++l)
                for (int i = 0; i < system.n_occ; ++i)
                    for (int j = 0; j < system.n_occ; ++j)
                    {
                        double value = dressed.eri(k, l, i, j);
                        for (int c = 0; c < system.n_virt; ++c)
                            for (int d = 0; d < system.n_virt; ++d)
                                value += dressed.eri(k, l, virt(c), virt(d)) * t2(i, j, c, d);
                        ints.w_oooo(k, l, i, j) = value;
                    }

        for (int k = 0; k < system.n_occ; ++k)
            for (int a = 0; a < system.n_virt; ++a)
                for (int c = 0; c < system.n_virt; ++c)
                    for (int i = 0; i < system.n_occ; ++i)
                    {
                        double value = -dressed.eri(k, virt(a), virt(c), i);
                        for (int l = 0; l < system.n_occ; ++l)
                            for (int d = 0; d < system.n_virt; ++d)
                            {
                                value -= dressed.eri(k, l, virt(c), virt(d)) * t2(i, l, a, d);
                                value += 0.5 * dressed.eri(k, l, virt(d), virt(c)) * t2(i, l, a, d);
                                value += 0.5 * dressed.eri(k, l, virt(c), virt(d)) * t2(i, l, d, a);
                            }
                        ints.w_ovvo(k, a, c, i) = value;
                    }

        for (int k = 0; k < system.n_occ; ++k)
            for (int a = 0; a < system.n_virt; ++a)
                for (int i = 0; i < system.n_occ; ++i)
                    for (int c = 0; c < system.n_virt; ++c)
                    {
                        double value = -dressed.eri(k, virt(a), i, virt(c));
                        for (int l = 0; l < system.n_occ; ++l)
                            for (int d = 0; d < system.n_virt; ++d)
                                value += 0.5 * dressed.eri(k, l, virt(d), virt(c)) * t2(l, i, a, d);
                        ints.w_ovov(k, a, i, c) = value;
                    }

        return ints;
    }

    [[nodiscard]] RCCSDResiduals build_dressed_sd_residuals(
        const ProductionSpinOrbitalChemistsSystem &system,
        const DressedSpinOrbitalSystem &dressed,
        const DressedSinglesDoublesIntermediates &ints,
        const RCCSDAmplitudes &amps)
    {
        RCCSDResiduals residuals{
            .r1 = Tensor2D(system.n_occ, system.n_virt, 0.0),
            .r2 = Tensor4D(system.n_occ, system.n_occ, system.n_virt, system.n_virt, 0.0),
        };
        const auto virt = [&system](int a) noexcept
        {
            return system.n_occ + a;
        };

        Tensor4D c_t2(amps.t2.dim1, amps.t2.dim2, amps.t2.dim3, amps.t2.dim4, 0.0);
        for (int i = 0; i < amps.t2.dim1; ++i)
            for (int j = 0; j < amps.t2.dim2; ++j)
                for (int a = 0; a < amps.t2.dim3; ++a)
                    for (int b = 0; b < amps.t2.dim4; ++b)
                        c_t2(i, j, a, b) = 2.0 * amps.t2(i, j, a, b) - amps.t2(i, j, b, a);

        for (int i = 0; i < system.n_occ; ++i)
            for (int a = 0; a < system.n_virt; ++a)
            {
                double value = dressed.fock(virt(a), i);
                for (int k = 0; k < system.n_occ; ++k)
                    for (int c = 0; c < system.n_virt; ++c)
                        value += dressed.fock(k, virt(c)) * c_t2(i, k, a, c);
                for (int k = 0; k < system.n_occ; ++k)
                    for (int c = 0; c < system.n_virt; ++c)
                        for (int d = 0; d < system.n_virt; ++d)
                            value += dressed.eri(virt(a), k, virt(c), virt(d)) *
                                     c_t2(i, k, c, d);
                for (int k = 0; k < system.n_occ; ++k)
                    for (int l = 0; l < system.n_occ; ++l)
                        for (int c = 0; c < system.n_virt; ++c)
                            value -= dressed.eri(k, l, i, virt(c)) * c_t2(k, l, a, c);
                residuals.r1(i, a) = value;
            }

        for (int i = 0; i < system.n_occ; ++i)
            for (int j = 0; j < system.n_occ; ++j)
                for (int a = 0; a < system.n_virt; ++a)
                    for (int b = 0; b < system.n_virt; ++b)
                    {
                        double value = 0.5 * dressed.eri(virt(a), virt(b), i, j);
                        for (int c = 0; c < system.n_virt; ++c)
                            value += ints.f_vv(b, c) * amps.t2(i, j, a, c);
                        for (int k = 0; k < system.n_occ; ++k)
                            value -= ints.f_oo(k, j) * amps.t2(i, k, a, b);
                        for (int c = 0; c < system.n_virt; ++c)
                            for (int d = 0; d < system.n_virt; ++d)
                                value += 0.5 * dressed.eri(virt(a), virt(b), virt(c), virt(d)) *
                                         amps.t2(i, j, c, d);
                        for (int k = 0; k < system.n_occ; ++k)
                            for (int l = 0; l < system.n_occ; ++l)
                                value += 0.5 * ints.w_oooo(k, l, i, j) * amps.t2(k, l, a, b);
                        for (int k = 0; k < system.n_occ; ++k)
                            for (int c = 0; c < system.n_virt; ++c)
                            {
                                value += ints.w_ovov(k, a, j, c) * amps.t2(i, k, c, b);
                                value -= 2.0 * ints.w_ovvo(k, a, c, i) * amps.t2(k, j, c, b);
                                value += ints.w_ovov(k, a, i, c) * amps.t2(k, j, c, b);
                                value += ints.w_ovvo(k, a, c, i) * amps.t2(j, k, c, b);
                            }
                        residuals.r2(i, j, a, b) = value;
                    }

        return residuals;
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

    [[nodiscard]] RCCSDTAmplitudes project_rccsd_warm_start_to_restricted(
        const TensorRCCSDResult &rccsd,
        const RHFReference &reference)
    {
        RCCSDTAmplitudes amps{
            .t1 = Tensor2D(reference.n_occ, reference.n_virt, 0.0),
            .t2 = Tensor4D(reference.n_occ, reference.n_occ, reference.n_virt, reference.n_virt, 0.0),
            .t3 = Tensor6D(
                reference.n_occ, reference.n_occ, reference.n_occ,
                reference.n_virt, reference.n_virt, reference.n_virt, 0.0),
        };

        for (int i = 0; i < reference.n_occ; ++i)
            for (int a = 0; a < reference.n_virt; ++a)
            {
                const double alpha = rccsd.amplitudes.t1(2 * i, 2 * a);
                const double beta = rccsd.amplitudes.t1(2 * i + 1, 2 * a + 1);
                amps.t1(i, a) = 0.5 * (alpha + beta);
            }

        for (int i = 0; i < reference.n_occ; ++i)
            for (int j = 0; j < reference.n_occ; ++j)
                for (int a = 0; a < reference.n_virt; ++a)
                    for (int b = 0; b < reference.n_virt; ++b)
                    {
                        const double ab =
                            rccsd.amplitudes.t2(2 * i, 2 * j + 1, 2 * a, 2 * b + 1);
                        const double ba =
                            rccsd.amplitudes.t2(2 * i + 1, 2 * j, 2 * a + 1, 2 * b);
                        amps.t2(i, j, a, b) = 0.5 * (ab + ba);
                    }

        return amps;
    }

    [[nodiscard]] double restricted_d1(
        const RHFReference &reference,
        int i, int a) noexcept
    {
        return reference.eps_occ(i) - reference.eps_virt(a);
    }

    [[nodiscard]] double restricted_d2(
        const RHFReference &reference,
        int i, int j,
        int a, int b) noexcept
    {
        return reference.eps_occ(i) + reference.eps_occ(j) -
               reference.eps_virt(a) - reference.eps_virt(b);
    }

    [[nodiscard]] double restricted_d3(
        const RHFReference &reference,
        int i, int j, int k,
        int a, int b, int c) noexcept
    {
        return reference.eps_occ(i) + reference.eps_occ(j) + reference.eps_occ(k) -
               reference.eps_virt(a) - reference.eps_virt(b) - reference.eps_virt(c);
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

    void add_dressed_triples_feedback_into_sd_residuals(
        const ProductionSpinOrbitalChemistsSystem &system,
        const DressedSpinOrbitalSystem &dressed,
        const RCCSDTAmplitudes &amps,
        RCCSDResiduals &residuals)
    {
        const auto virt = [&system](int a) noexcept
        {
            return system.n_occ + a;
        };

        for (int i = 0; i < system.n_occ; ++i)
            for (int a = 0; a < system.n_virt; ++a)
            {
                double corr = 0.0;
                for (int j = 0; j < system.n_occ; ++j)
                    for (int k = 0; k < system.n_occ; ++k)
                        for (int b = 0; b < system.n_virt; ++b)
                            for (int c = 0; c < system.n_virt; ++c)
                                corr += 0.5 *
                                        dressed.eri(j, k, virt(b), virt(c)) *
                                        t3_p422(amps.t3, k, i, j, c, a, b);
                residuals.r1(i, a) += corr;
            }

        for (int i = 0; i < system.n_occ; ++i)
            for (int j = 0; j < system.n_occ; ++j)
                for (int a = 0; a < system.n_virt; ++a)
                    for (int b = 0; b < system.n_virt; ++b)
                    {
                        double corr = 0.0;
                        for (int k = 0; k < system.n_occ; ++k)
                            for (int c = 0; c < system.n_virt; ++c)
                                corr += 0.5 * dressed.fock(k, virt(c)) *
                                        t3_p201(amps.t3, k, i, j, c, a, b);
                        for (int k = 0; k < system.n_occ; ++k)
                            for (int c = 0; c < system.n_virt; ++c)
                                for (int d = 0; d < system.n_virt; ++d)
                                    corr += dressed.eri(virt(b), k, virt(c), virt(d)) *
                                            t3_p201(amps.t3, k, i, j, d, a, c);
                        for (int k = 0; k < system.n_occ; ++k)
                            for (int l = 0; l < system.n_occ; ++l)
                                for (int c = 0; c < system.n_virt; ++c)
                                    corr -= dressed.eri(k, l, j, virt(c)) *
                                            t3_p201(amps.t3, l, i, k, c, a, b);
                        residuals.r2(i, j, a, b) += corr;
                    }
    }

    [[nodiscard]] DressedTriplesIntermediates build_dressed_triples_intermediates(
        const ProductionSpinOrbitalChemistsSystem &system,
        const DressedSpinOrbitalSystem &dressed,
        const DressedSinglesDoublesIntermediates &sd_ints,
        const Tensor4D &t2)
    {
        DressedTriplesIntermediates ints{
            .f_oo = Tensor2D(sd_ints.f_oo.dim1, sd_ints.f_oo.dim2, 0.0),
            .f_vv = Tensor2D(sd_ints.f_vv.dim1, sd_ints.f_vv.dim2, 0.0),
            .w_oooo = Tensor4D(
                sd_ints.w_oooo.dim1, sd_ints.w_oooo.dim2,
                sd_ints.w_oooo.dim3, sd_ints.w_oooo.dim4, 0.0),
            .w_ovvo = Tensor4D(system.n_occ, system.n_virt, system.n_virt, system.n_occ, 0.0),
            .w_ovov = Tensor4D(system.n_occ, system.n_virt, system.n_occ, system.n_virt, 0.0),
            .w_vooo = Tensor4D(system.n_virt, system.n_occ, system.n_occ, system.n_occ, 0.0),
            .w_vvvo = Tensor4D(system.n_virt, system.n_virt, system.n_virt, system.n_occ, 0.0),
            .w_vvvv = Tensor4D(system.n_virt, system.n_virt, system.n_virt, system.n_virt, 0.0),
        };
        ints.f_oo.data = sd_ints.f_oo.data;
        ints.f_vv.data = sd_ints.f_vv.data;
        ints.w_oooo.data = sd_ints.w_oooo.data;

        const auto virt = [&system](int a) noexcept
        {
            return system.n_occ + a;
        };

        Tensor4D c_t2(t2.dim1, t2.dim2, t2.dim3, t2.dim4, 0.0);
        for (int i = 0; i < t2.dim1; ++i)
            for (int j = 0; j < t2.dim2; ++j)
                for (int a = 0; a < t2.dim3; ++a)
                    for (int b = 0; b < t2.dim4; ++b)
                        c_t2(i, j, a, b) = 2.0 * t2(i, j, a, b) - t2(i, j, b, a);

        for (int a = 0; a < system.n_virt; ++a)
            for (int b = 0; b < system.n_virt; ++b)
                for (int d = 0; d < system.n_virt; ++d)
                    for (int e = 0; e < system.n_virt; ++e)
                    {
                        double value = dressed.eri(virt(a), virt(b), virt(d), virt(e));
                        for (int l = 0; l < system.n_occ; ++l)
                            for (int m = 0; m < system.n_occ; ++m)
                                value += dressed.eri(l, m, virt(d), virt(e)) * t2(l, m, a, b);
                        ints.w_vvvv(a, b, d, e) = value;
                    }

        for (int a = 0; a < system.n_virt; ++a)
            for (int l = 0; l < system.n_occ; ++l)
                for (int i = 0; i < system.n_occ; ++i)
                    for (int j = 0; j < system.n_occ; ++j)
                    {
                        double value = dressed.eri(virt(a), l, i, j);
                        for (int d = 0; d < system.n_virt; ++d)
                            value += dressed.fock(l, virt(d)) * t2(i, j, a, d);
                        for (int m = 0; m < system.n_occ; ++m)
                            for (int d = 0; d < system.n_virt; ++d)
                            {
                                value += dressed.eri(m, l, virt(d), j) * c_t2(m, i, d, a);
                                value -= 0.5 * dressed.eri(m, l, j, virt(d)) * c_t2(m, i, d, a);
                                value -= 0.5 * dressed.eri(m, l, j, virt(d)) * t2(i, m, d, a);
                                value -= dressed.eri(m, l, i, virt(d)) * t2(j, m, d, a);
                            }
                        for (int d = 0; d < system.n_virt; ++d)
                            for (int e = 0; e < system.n_virt; ++e)
                                value += dressed.eri(virt(a), l, virt(d), virt(e)) * t2(i, j, d, e);
                        ints.w_vooo(a, l, i, j) = value;
                    }

        for (int a = 0; a < system.n_virt; ++a)
            for (int b = 0; b < system.n_virt; ++b)
                for (int d = 0; d < system.n_virt; ++d)
                    for (int j = 0; j < system.n_occ; ++j)
                    {
                        double value = dressed.eri(virt(a), virt(b), virt(d), j);
                        for (int l = 0; l < system.n_occ; ++l)
                            for (int e = 0; e < system.n_virt; ++e)
                            {
                                value += dressed.eri(l, virt(a), virt(e), virt(d)) * c_t2(l, j, e, b);
                                value -= 0.5 * dressed.eri(l, virt(a), virt(d), virt(e)) * c_t2(l, j, e, b);
                                value -= 0.5 * dressed.eri(l, virt(a), virt(d), virt(e)) * t2(j, l, e, b);
                                value -= dressed.eri(l, virt(b), virt(d), virt(e)) * t2(j, l, e, a);
                            }
                        for (int l = 0; l < system.n_occ; ++l)
                            for (int m = 0; m < system.n_occ; ++m)
                                value += dressed.eri(l, m, virt(d), j) * t2(l, m, a, b);
                        ints.w_vvvo(a, b, d, j) = value;
                    }

        for (int l = 0; l < system.n_occ; ++l)
            for (int a = 0; a < system.n_virt; ++a)
                for (int d = 0; d < system.n_virt; ++d)
                    for (int i = 0; i < system.n_occ; ++i)
                    {
                        // Match PySCF RCCSDT `intermediates_t3` exactly.  The
                        // triples path uses a different dressed W_ovvo than the
                        // SD equations:
                        // W_ovvo = 2 * t1_eris[o,v,v,o]
                        //        -     t1_eris[o,v,o,v]^T_{id}
                        //        + 2 * t1_eris[o,o,v,v] * t2
                        //        -     t1_eris[o,o,v,v]^T_{de} * t2
                        double value =
                            2.0 * dressed.eri(l, virt(a), virt(d), i) -
                            dressed.eri(l, virt(a), i, virt(d));
                        for (int m = 0; m < system.n_occ; ++m)
                            for (int e = 0; e < system.n_virt; ++e)
                            {
                                const double c_t2 = 2.0 * t2(m, i, e, a) -
                                                    t2(m, i, a, e);
                                value +=
                                    2.0 * dressed.eri(m, l, virt(e), virt(d)) *
                                    c_t2;
                                value -=
                                    dressed.eri(m, l, virt(d), virt(e)) *
                                    c_t2;
                            }
                        ints.w_ovvo(l, a, d, i) = value;
                    }

        for (int l = 0; l < system.n_occ; ++l)
            for (int a = 0; a < system.n_virt; ++a)
                for (int i = 0; i < system.n_occ; ++i)
                    for (int d = 0; d < system.n_virt; ++d)
                    {
                        // Match PySCF RCCSDT `intermediates_t3` exactly:
                        // W_ovov = t1_eris[o,v,o,v] - t1_eris[o,o,v,v]^T_{de} * t2
                        double value = dressed.eri(l, virt(a), i, virt(d));
                        for (int m = 0; m < system.n_occ; ++m)
                            for (int e = 0; e < system.n_virt; ++e)
                                value -= dressed.eri(m, l, virt(d), virt(e)) *
                                         t2(i, m, e, a);
                        ints.w_ovov(l, a, i, d) = value;
                    }

        return ints;
    }

    void add_dressed_triples_feedback_into_triples_intermediates(
        const ProductionSpinOrbitalChemistsSystem &system,
        const DressedSpinOrbitalSystem &dressed,
        const Tensor6D &t3,
        DressedTriplesIntermediates &ints)
    {
        const auto virt = [&system](int a) noexcept
        {
            return system.n_occ + a;
        };

        for (int a = 0; a < system.n_virt; ++a)
            for (int l = 0; l < system.n_occ; ++l)
                for (int i = 0; i < system.n_occ; ++i)
                    for (int j = 0; j < system.n_occ; ++j)
                    {
                        double corr = 0.0;
                        for (int m = 0; m < system.n_occ; ++m)
                            for (int d = 0; d < system.n_virt; ++d)
                                for (int e = 0; e < system.n_virt; ++e)
                                    corr += dressed.eri(l, m, virt(d), virt(e)) *
                                            t3_p201(t3, m, i, j, e, a, d);
                        ints.w_vooo(a, l, i, j) += corr;
                    }

        for (int a = 0; a < system.n_virt; ++a)
            for (int b = 0; b < system.n_virt; ++b)
                for (int d = 0; d < system.n_virt; ++d)
                    for (int j = 0; j < system.n_occ; ++j)
                    {
                        double corr = 0.0;
                        for (int l = 0; l < system.n_occ; ++l)
                            for (int m = 0; m < system.n_occ; ++m)
                                for (int e = 0; e < system.n_virt; ++e)
                                    corr -= dressed.eri(l, m, virt(d), virt(e)) *
                                            t3_p201(t3, m, j, l, e, b, a);
                        ints.w_vvvo(a, b, d, j) += corr;
                    }
    }

    void build_dressed_triples_residual(
        const ProductionSpinOrbitalChemistsSystem &system,
        const DressedTriplesIntermediates &ints,
        const RCCSDTAmplitudes &amps,
        TensorTriplesWorkspace &triples)
    {
        if (!triples.allocated)
            return;

        std::fill(triples.r3.data.begin(), triples.r3.data.end(), 0.0);

        for (int i = 0; i < system.n_occ; ++i)
            for (int j = 0; j < system.n_occ; ++j)
                for (int k = 0; k < system.n_occ; ++k)
                    for (int a = 0; a < system.n_virt; ++a)
                        for (int b = 0; b < system.n_virt; ++b)
                            for (int c = 0; c < system.n_virt; ++c)
                            {
                                double value = 0.0;
                                for (int d = 0; d < system.n_virt; ++d)
                                {
                                    value += ints.w_vvvo(a, b, d, j) * amps.t2(i, k, d, c);
                                    value += 0.5 * ints.f_vv(a, d) * amps.t3(i, j, k, d, b, c);
                                }
                                for (int l = 0; l < system.n_occ; ++l)
                                {
                                    value -= ints.w_vooo(a, l, i, j) * amps.t2(l, k, b, c);
                                    value -= 0.5 * ints.f_oo(l, i) * amps.t3(l, j, k, a, b, c);
                                }
                                for (int l = 0; l < system.n_occ; ++l)
                                    for (int d = 0; d < system.n_virt; ++d)
                                    {
                                        value += 0.25 * ints.w_ovvo(l, a, d, i) *
                                                 t3_p201(amps.t3, l, j, k, d, b, c);
                                        value -= 0.5 * ints.w_ovov(l, a, i, d) *
                                                 amps.t3(j, l, k, d, b, c);
                                        value -= ints.w_ovov(l, b, i, d) *
                                                 amps.t3(j, l, k, d, a, c);
                                    }
                                for (int l = 0; l < system.n_occ; ++l)
                                    for (int m = 0; m < system.n_occ; ++m)
                                        value += 0.5 * ints.w_oooo(l, m, i, j) *
                                                 amps.t3(l, m, k, a, b, c);
                                for (int d = 0; d < system.n_virt; ++d)
                                    for (int e = 0; e < system.n_virt; ++e)
                                        value += 0.5 * ints.w_vvvv(a, b, d, e) *
                                                 amps.t3(i, j, k, d, e, c);
                                triples.r3(i, j, k, a, b, c) = value;
                            }
    }

    void apply_restricted_t3_permutation_symmetry(Tensor6D &tensor)
    {
        Tensor6D original(
            tensor.dim1, tensor.dim2, tensor.dim3,
            tensor.dim4, tensor.dim5, tensor.dim6, 0.0);
        original.data = tensor.data;

        for (int i = 0; i < tensor.dim1; ++i)
            for (int j = 0; j < tensor.dim2; ++j)
                for (int k = 0; k < tensor.dim3; ++k)
                    for (int a = 0; a < tensor.dim4; ++a)
                        for (int b = 0; b < tensor.dim5; ++b)
                            for (int c = 0; c < tensor.dim6; ++c)
                            {
                                const int occ[3] = {i, j, k};
                                const int virt[3] = {a, b, c};

                                double simultaneous_sum = 0.0;
                                for (const auto &occ_perm : kPermutations3)
                                {
                                    simultaneous_sum += original(
                                        occ[occ_perm[0]],
                                        occ[occ_perm[1]],
                                        occ[occ_perm[2]],
                                        virt[occ_perm[0]],
                                        virt[occ_perm[1]],
                                        virt[occ_perm[2]]);
                                }
                                tensor(i, j, k, a, b, c) = simultaneous_sum;
                            }
    }

    void apply_restricted_t3_p3_full(Tensor6D &tensor)
    {
        Tensor6D permuted(
            tensor.dim1, tensor.dim2, tensor.dim3,
            tensor.dim4, tensor.dim5, tensor.dim6, 0.0);
        permuted.data = tensor.data;

        for (int i = 0; i < tensor.dim1; ++i)
            for (int j = 0; j < tensor.dim2; ++j)
                for (int k = 0; k < tensor.dim3; ++k)
                    for (int a = 0; a < tensor.dim4; ++a)
                        for (int b = 0; b < tensor.dim5; ++b)
                            for (int c = 0; c < tensor.dim6; ++c)
                            {
                                const int virt[3] = {a, b, c};
                                double total_sum = 0.0;
                                for (const auto &virt_perm : kPermutations3)
                                    total_sum += permuted(
                                        i, j, k,
                                        virt[virt_perm[0]],
                                        virt[virt_perm[1]],
                                        virt[virt_perm[2]]);
                                tensor(i, j, k, a, b, c) =
                                    permuted(i, j, k, a, b, c) - total_sum / 6.0;
                            }
    }

    void purify_restricted_t3(Tensor6D &tensor)
    {
        for (int i = 0; i < tensor.dim1; ++i)
            for (int j = 0; j < tensor.dim2; ++j)
                for (int k = 0; k < tensor.dim3; ++k)
                    for (int a = 0; a < tensor.dim4; ++a)
                        for (int b = 0; b < tensor.dim5; ++b)
                            for (int c = 0; c < tensor.dim6; ++c)
                                if ((i == j && j == k) || (a == b && b == c))
                                    tensor(i, j, k, a, b, c) = 0.0;
    }

    void restore_restricted_t3_structure(Tensor6D &tensor)
    {
        apply_restricted_t3_permutation_symmetry(tensor);
        apply_restricted_t3_p3_full(tensor);
        purify_restricted_t3(tensor);
    }

    void restore_restricted_t2_from_unique(Tensor4D &t2)
    {
        for (int i = 0; i < t2.dim1; ++i)
            for (int a = 0; a < t2.dim3; ++a)
                for (int b = 0; b < t2.dim4; ++b)
                    t2(i, i, a, b) *= 0.5;

        Tensor4D original(t2.dim1, t2.dim2, t2.dim3, t2.dim4, 0.0);
        original.data = t2.data;
        for (int i = 0; i < t2.dim1; ++i)
            for (int j = 0; j < t2.dim2; ++j)
                for (int a = 0; a < t2.dim3; ++a)
                    for (int b = 0; b < t2.dim4; ++b)
                        t2(i, j, a, b) += original(j, i, b, a);
    }

    void restore_restricted_t3_from_unique(Tensor6D &t3)
    {
        for (int i = 0; i < t3.dim1; ++i)
            for (int j = 0; j < t3.dim2; ++j)
                for (int k = 0; k < t3.dim3; ++k)
                {
                    const bool all_equal = (i == j && j == k);
                    const bool two_equal =
                        !all_equal && (i == j || j == k || i == k);
                    if (!two_equal && !all_equal)
                        continue;
                    const double scale = all_equal ? (1.0 / 6.0) : 0.5;
                    for (int a = 0; a < t3.dim4; ++a)
                        for (int b = 0; b < t3.dim5; ++b)
                            for (int c = 0; c < t3.dim6; ++c)
                                t3(i, j, k, a, b, c) *= scale;
                }

        apply_restricted_t3_permutation_symmetry(t3);
        purify_restricted_t3(t3);
    }

    [[nodiscard]] std::size_t restricted_unique_rccsdt_size(
        const RCCSDTAmplitudes &amps) noexcept
    {
        const std::size_t nocc = static_cast<std::size_t>(amps.t1.dim1);
        const std::size_t nvirt = static_cast<std::size_t>(amps.t1.dim2);
        const std::size_t t1_size = nocc * nvirt;
        const std::size_t t2_size = (nocc * (nocc + 1) / 2) * nvirt * nvirt;
        const std::size_t t3_size =
            (nocc * (nocc + 1) * (nocc + 2) / 6) * nvirt * nvirt * nvirt;
        return t1_size + t2_size + t3_size;
    }

    Eigen::VectorXd pack_restricted_unique_rccsdt_amplitudes(
        const RCCSDTAmplitudes &amps)
    {
        Eigen::VectorXd packed(
            static_cast<Eigen::Index>(restricted_unique_rccsdt_size(amps)));
        Eigen::Index offset = 0;

        for (const double value : amps.t1.data)
            packed(offset++) = value;

        for (int i = 0; i < amps.t2.dim1; ++i)
            for (int j = i; j < amps.t2.dim2; ++j)
                for (int a = 0; a < amps.t2.dim3; ++a)
                    for (int b = 0; b < amps.t2.dim4; ++b)
                        packed(offset++) = amps.t2(i, j, a, b);

        for (int i = 0; i < amps.t3.dim1; ++i)
            for (int j = i; j < amps.t3.dim2; ++j)
                for (int k = j; k < amps.t3.dim3; ++k)
                    for (int a = 0; a < amps.t3.dim4; ++a)
                        for (int b = 0; b < amps.t3.dim5; ++b)
                            for (int c = 0; c < amps.t3.dim6; ++c)
                                packed(offset++) = amps.t3(i, j, k, a, b, c);

        return packed;
    }

    void unpack_restricted_unique_rccsdt_amplitudes(
        const Eigen::VectorXd &packed,
        RCCSDTAmplitudes &amps)
    {
        std::fill(amps.t1.data.begin(), amps.t1.data.end(), 0.0);
        std::fill(amps.t2.data.begin(), amps.t2.data.end(), 0.0);
        std::fill(amps.t3.data.begin(), amps.t3.data.end(), 0.0);

        Eigen::Index offset = 0;
        for (double &value : amps.t1.data)
            value = packed(offset++);

        for (int i = 0; i < amps.t2.dim1; ++i)
            for (int j = i; j < amps.t2.dim2; ++j)
                for (int a = 0; a < amps.t2.dim3; ++a)
                    for (int b = 0; b < amps.t2.dim4; ++b)
                        amps.t2(i, j, a, b) = packed(offset++);
        restore_restricted_t2_from_unique(amps.t2);

        for (int i = 0; i < amps.t3.dim1; ++i)
            for (int j = i; j < amps.t3.dim2; ++j)
                for (int k = j; k < amps.t3.dim3; ++k)
                    for (int a = 0; a < amps.t3.dim4; ++a)
                        for (int b = 0; b < amps.t3.dim5; ++b)
                            for (int c = 0; c < amps.t3.dim6; ++c)
                                amps.t3(i, j, k, a, b, c) = packed(offset++);
        restore_restricted_t3_from_unique(amps.t3);
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
        double damping,
        bool use_diis)
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
        if (use_diis && calculator._scf._use_DIIS && diis.ready())
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

    [[nodiscard]] double compute_restricted_rccsdt_correlation_energy(
        const ProductionSpinOrbitalChemistsSystem &system,
        const RCCSDTAmplitudes &amps)
    {
        double ed = 0.0;
        double ex = 0.0;
        double singles = 0.0;
        for (int i = 0; i < system.n_occ; ++i)
            for (int j = 0; j < system.n_occ; ++j)
                for (int a = 0; a < system.n_virt; ++a)
                    for (int b = 0; b < system.n_virt; ++b)
                    {
                        const int va = system.n_occ + a;
                        const int vb = system.n_occ + b;
                        const double tau =
                            amps.t2(i, j, a, b) + amps.t1(i, a) * amps.t1(j, b);
                        ed += 2.0 * tau * system.eri(i, j, va, vb);
                        ex -= tau * system.eri(i, j, vb, va);
                    }

        for (int i = 0; i < system.n_occ; ++i)
            for (int a = 0; a < system.n_virt; ++a)
                singles += system.fock(system.n_occ + a, i) * amps.t1(i, a);

        return ed + ex + 2.0 * singles;
    }

    [[nodiscard]] double update_restricted_t3_from_r3_jacobi(
        const RHFReference &reference,
        RCCSDTAmplitudes &amps,
        Tensor6D &r3,
        double damping)
    {
        double sum_sq = 0.0;
        std::size_t count = 0;
        for (int i = 0; i < amps.t3.dim1; ++i)
            for (int j = 0; j < amps.t3.dim2; ++j)
                for (int k = 0; k < amps.t3.dim3; ++k)
                    for (int a = 0; a < amps.t3.dim4; ++a)
                        for (int b = 0; b < amps.t3.dim5; ++b)
                            for (int c = 0; c < amps.t3.dim6; ++c)
                            {
                                const double denom = restricted_d3(reference, i, j, k, a, b, c);
                                if (std::abs(denom) < 1e-12)
                                    continue;
                                const double delta = damping * r3(i, j, k, a, b, c) / denom;
                                amps.t3(i, j, k, a, b, c) += delta;
                                sum_sq += delta * delta;
                                ++count;
                            }
        if (count == 0)
            return 0.0;
        return std::sqrt(sum_sq / static_cast<double>(count));
    }

    struct RestrictedRCCSDTUpdateMetrics
    {
        double sd_residual_rms = 0.0;
        double r3_residual_rms = 0.0;
        double r1_feedback_rms = 0.0;
        double r2_feedback_rms = 0.0;
        double t1_step_rms = 0.0;
        double t2_step_rms = 0.0;
        double t3_step_rms = 0.0;
        double norm_dtamps = 0.0;
    };

    [[nodiscard]] RestrictedRCCSDTUpdateMetrics update_restricted_rccsdt_amplitudes_once(
        const ProductionSpinOrbitalChemistsSystem &system,
        const RHFReference &reference,
        RCCSDTAmplitudes &amps)
    {
        RestrictedRCCSDTUpdateMetrics metrics;

        const DressedSpinOrbitalSystem dressed =
            build_dressed_spin_orbital_system(system, amps);

        RCCSDAmplitudes sd_amps{
            .t1 = Tensor2D(amps.t1.dim1, amps.t1.dim2, 0.0),
            .t2 = Tensor4D(amps.t2.dim1, amps.t2.dim2, amps.t2.dim3, amps.t2.dim4, 0.0),
        };
        sd_amps.t1.data = amps.t1.data;
        sd_amps.t2.data = amps.t2.data;

        const DressedSinglesDoublesIntermediates sd_ints =
            build_dressed_sd_intermediates(system, dressed, amps.t2);
        RCCSDResiduals residuals =
            build_dressed_sd_residuals(system, dressed, sd_ints, sd_amps);
        const RCCSDResiduals residuals_before_t3 = residuals;
        add_dressed_triples_feedback_into_sd_residuals(
            system, dressed, amps, residuals);

        Tensor2D r1_feedback(residuals.r1.dim1, residuals.r1.dim2, 0.0);
        for (std::size_t idx = 0; idx < residuals.r1.data.size(); ++idx)
            r1_feedback.data[idx] =
                residuals.r1.data[idx] - residuals_before_t3.r1.data[idx];

        Tensor4D r2_feedback(
            residuals.r2.dim1, residuals.r2.dim2,
            residuals.r2.dim3, residuals.r2.dim4, 0.0);
        for (std::size_t idx = 0; idx < residuals.r2.data.size(); ++idx)
            r2_feedback.data[idx] =
                residuals.r2.data[idx] - residuals_before_t3.r2.data[idx];

        Tensor4D full_r2_before_sym = residuals.r2;
        for (int i = 0; i < residuals.r2.dim1; ++i)
            for (int j = 0; j < residuals.r2.dim2; ++j)
                for (int a = 0; a < residuals.r2.dim3; ++a)
                    for (int b = 0; b < residuals.r2.dim4; ++b)
                        residuals.r2(i, j, a, b) += full_r2_before_sym(j, i, b, a);

        Tensor4D sym_r2_feedback = r2_feedback;
        for (int i = 0; i < sym_r2_feedback.dim1; ++i)
            for (int j = 0; j < sym_r2_feedback.dim2; ++j)
                for (int a = 0; a < sym_r2_feedback.dim3; ++a)
                    for (int b = 0; b < sym_r2_feedback.dim4; ++b)
                        sym_r2_feedback(i, j, a, b) +=
                            r2_feedback(j, i, b, a);

        metrics.r1_feedback_rms = tensor_rms(r1_feedback);
        metrics.r2_feedback_rms = tensor_rms(sym_r2_feedback);
        metrics.sd_residual_rms = rms_norm(pack_residuals(residuals));

        double t1_sum_sq = 0.0;
        std::size_t t1_count = 0;
        for (int i = 0; i < amps.t1.dim1; ++i)
            for (int a = 0; a < amps.t1.dim2; ++a)
            {
                const double denom = restricted_d1(reference, i, a);
                if (std::abs(denom) < 1e-12)
                    continue;
                const double delta = residuals.r1(i, a) / denom;
                amps.t1(i, a) += delta;
                t1_sum_sq += delta * delta;
                ++t1_count;
            }

        double t2_sum_sq = 0.0;
        std::size_t t2_count = 0;
        for (int i = 0; i < amps.t2.dim1; ++i)
            for (int j = 0; j < amps.t2.dim2; ++j)
                for (int a = 0; a < amps.t2.dim3; ++a)
                    for (int b = 0; b < amps.t2.dim4; ++b)
                    {
                        const double denom = restricted_d2(reference, i, j, a, b);
                        if (std::abs(denom) < 1e-12)
                            continue;
                        const double delta = residuals.r2(i, j, a, b) / denom;
                        amps.t2(i, j, a, b) += delta;
                        t2_sum_sq += delta * delta;
                        ++t2_count;
                    }

        metrics.t1_step_rms = t1_count == 0
                                  ? 0.0
                                  : std::sqrt(t1_sum_sq / static_cast<double>(t1_count));
        metrics.t2_step_rms = t2_count == 0
                                  ? 0.0
                                  : std::sqrt(t2_sum_sq / static_cast<double>(t2_count));

        const DressedSinglesDoublesIntermediates sd_ints_t3 =
            build_dressed_sd_intermediates(system, dressed, amps.t2);
        DressedTriplesIntermediates triples_ints =
            build_dressed_triples_intermediates(
                system, dressed, sd_ints_t3, amps.t2);
        add_dressed_triples_feedback_into_triples_intermediates(
            system, dressed, amps.t3, triples_ints);

        TensorTriplesWorkspace triples{
            .amplitudes = clone_rccsdt_amplitudes(amps),
            .r3 = Tensor6D(
                amps.t3.dim1, amps.t3.dim2, amps.t3.dim3,
                amps.t3.dim4, amps.t3.dim5, amps.t3.dim6, 0.0),
            .allocated = true,
        };
        build_dressed_triples_residual(system, triples_ints, amps, triples);
        restore_restricted_t3_structure(triples.r3);
        metrics.r3_residual_rms = triples_residual_rms(triples.r3);
        metrics.t3_step_rms =
            update_restricted_t3_from_r3_jacobi(reference, amps, triples.r3, 1.0);

        metrics.norm_dtamps = std::sqrt(
            metrics.t1_step_rms * metrics.t1_step_rms +
            metrics.t2_step_rms * metrics.t2_step_rms +
            metrics.t3_step_rms * metrics.t3_step_rms);

        return metrics;
    }

    std::expected<TensorTriplesStageMetrics, std::string> run_restricted_tensor_rccsdt_no_fallback(
        HartreeFock::Calculator &calculator,
        const TensorRCCSDTState &state,
        const ProductionSpinOrbitalChemistsSystem &system,
        const TensorRCCSDResult &rccsd)
    {
        const RHFReference &reference = state.reference.orbital_partition;
        const unsigned int max_iter =
            std::min(64u, std::max(24u, 2u * calculator._scf.get_max_cycles(calculator._shells.nbasis())));
        const double tol_energy = std::max(1e-10, calculator._scf._tol_energy);
        const double tol_normt = 1e-6;

        RCCSDTAmplitudes amps =
            project_rccsd_warm_start_to_restricted(rccsd, reference);

        TensorTriplesStageMetrics metrics;
        TensorTriplesStageMetrics best_metrics;
        bool have_best = false;
        double best_score = std::numeric_limits<double>::infinity();
        AmplitudeDIIS diis(static_cast<int>(std::max(2u, calculator._scf._DIIS_dim)));

        double previous_energy =
            compute_restricted_rccsdt_correlation_energy(system, amps);

        for (unsigned int iter = 1; iter <= max_iter; ++iter)
        {
            const Eigen::VectorXd unique_before =
                pack_restricted_unique_rccsdt_amplitudes(amps);
            const RestrictedRCCSDTUpdateMetrics update_metrics =
                update_restricted_rccsdt_amplitudes_once(system, reference, amps);
            Eigen::VectorXd unique_after =
                pack_restricted_unique_rccsdt_amplitudes(amps);
            const Eigen::VectorXd unique_step = unique_after - unique_before;

            diis.push(unique_after, unique_step);
            if (calculator._scf._use_DIIS && diis.ready())
            {
                auto diis_res = diis.extrapolate();
                if (diis_res)
                {
                    unique_after = std::move(*diis_res);
                    unpack_restricted_unique_rccsdt_amplitudes(unique_after, amps);
                }
            }

            metrics.iterations = iter;
            metrics.sd_residual_rms = update_metrics.sd_residual_rms;
            metrics.r3_rms = update_metrics.r3_residual_rms;
            metrics.r1_feedback_rms = update_metrics.r1_feedback_rms;
            metrics.r2_feedback_rms = update_metrics.r2_feedback_rms;
            metrics.t1_step_rms = update_metrics.t1_step_rms;
            metrics.t2_step_rms = update_metrics.t2_step_rms;
            metrics.t3_step_rms = update_metrics.t3_step_rms;
            metrics.estimated_correlation_energy =
                compute_restricted_rccsdt_correlation_energy(system, amps);
            metrics.energy_change =
                metrics.estimated_correlation_energy - previous_energy;
            previous_energy = metrics.estimated_correlation_energy;
            metrics.quality_score = stage_quality_score(metrics);

            if (metrics.quality_score + 1e-12 < best_score)
            {
                best_score = metrics.quality_score;
                best_metrics = metrics;
                best_metrics.best_iteration = iter;
                have_best = true;
            }

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "RCCSDT[TENSOR-R] :",
                std::format(
                    "{:3d}  E_corr={:.10f}  dE={:+.3e}  norm(d tamps)={:.3e}  rms(SD)={:.3e}  rms(R3)={:.3e}  rms(R1[T3])={:.3e}  rms(R2[T3])={:.3e}",
                    iter,
                    metrics.estimated_correlation_energy,
                    metrics.energy_change,
                    update_metrics.norm_dtamps,
                    metrics.sd_residual_rms,
                    metrics.r3_rms,
                    metrics.r1_feedback_rms,
                    metrics.r2_feedback_rms));

            if (std::abs(metrics.energy_change) < tol_energy &&
                update_metrics.norm_dtamps < tol_normt)
            {
                metrics.converged = true;
                metrics.best_iteration = iter;
                return metrics;
            }
        }

        if (have_best)
            return best_metrics;
        return metrics;
    }

    std::expected<TensorTriplesStageMetrics, std::string> run_staged_tensor_triples_iterations(
        HartreeFock::Calculator &calculator,
        const TensorRCCSDTState &state,
        const ProductionSpinOrbitalBlocks &so_blocks,
        const ProductionSpinOrbitalChemistsSystem &full_system,
        TensorTriplesWorkspace &triples,
        unsigned int max_stage_iterations,
        bool require_convergence)
    {
        if (!triples.allocated)
            return std::unexpected(
                "run_staged_tensor_triples_iterations: triples workspace is not allocated.");

        constexpr double kT3Damping = 0.35;
        constexpr double kSDDamping = 0.35;
        // The staged tensor path is meant to become the production solver for
        // larger systems, so it should not stop at a tolerance inherited from
        // the more forgiving SCF density threshold. We keep the criterion
        // modest enough for the current incomplete residual, but tight enough
        // that the stage performs real coupled refinement before handing off
        // to the moderate-case fallback.
        const double tol_stage =
            std::max(1e-7, calculator._scf._tol_density);
        const double tol_energy =
            std::max(1e-10, calculator._scf._tol_energy);
        const ProductionSpinOrbitalReference so_ref =
            build_spin_orbital_reference(state.reference);
        TensorTriplesStageMetrics metrics;
        TensorTriplesStageMetrics best_metrics;
        RCCSDTAmplitudes best_amplitudes = clone_rccsdt_amplitudes(triples.amplitudes);
        bool have_best_iterate = false;
        double best_stage_score = std::numeric_limits<double>::infinity();
        unsigned int stale_iterations = 0;
        AmplitudeDIIS diis(static_cast<int>(std::max(2u, calculator._scf._DIIS_dim)));
        AmplitudeDIIS full_diis(static_cast<int>(std::max(2u, calculator._scf._DIIS_dim)));
        double previous_energy = state.warm_start_correlation_energy;
        const unsigned int min_iterations_before_break =
            require_convergence ? 12u : 4u;
        const unsigned int stall_patience =
            require_convergence ? 8u : 2u;
        const double deterioration_factor =
            require_convergence ? 1.25 : 1.05;

        for (unsigned int iter = 1; iter <= max_stage_iterations; ++iter)
        {
            RCCSDAmplitudes sd_amps = extract_sd_amplitudes(triples);
            const DressedSpinOrbitalSystem dressed =
                build_dressed_spin_orbital_system(
                    full_system,
                    triples.amplitudes);
            const DressedSinglesDoublesIntermediates sd_ints =
                build_dressed_sd_intermediates(
                    full_system,
                    dressed,
                    sd_amps.t2);
            RCCSDResiduals residuals =
                build_dressed_sd_residuals(
                    full_system,
                    dressed,
                    sd_ints,
                    sd_amps);

            // Match the PySCF RCCSDT update ordering: form the SD residual
            // first with the current T3 correction, update T1/T2, and only
            // then build the T3 residual from the refreshed SD amplitudes.
            const RCCSDResiduals residuals_before_t3 = residuals;
            add_dressed_triples_feedback_into_sd_residuals(
                full_system,
                dressed,
                triples.amplitudes,
                residuals);
            Tensor4D r2_feedback(
                residuals.r2.dim1, residuals.r2.dim2,
                residuals.r2.dim3, residuals.r2.dim4, 0.0);
            for (std::size_t idx = 0; idx < residuals.r2.data.size(); ++idx)
                r2_feedback.data[idx] =
                    residuals.r2.data[idx] - residuals_before_t3.r2.data[idx];
            Tensor4D unsym_r2 = residuals.r2;
            for (int i = 0; i < residuals.r2.dim1; ++i)
                for (int j = 0; j < residuals.r2.dim2; ++j)
                    for (int a = 0; a < residuals.r2.dim3; ++a)
                        for (int b = 0; b < residuals.r2.dim4; ++b)
                            residuals.r2(i, j, a, b) += unsym_r2(j, i, b, a);
            Tensor2D r1_feedback(
                residuals.r1.dim1, residuals.r1.dim2, 0.0);
            for (std::size_t idx = 0; idx < residuals.r1.data.size(); ++idx)
                r1_feedback.data[idx] =
                    residuals.r1.data[idx] - residuals_before_t3.r1.data[idx];
            Tensor4D sym_r2_feedback = r2_feedback;
            for (int i = 0; i < sym_r2_feedback.dim1; ++i)
                for (int j = 0; j < sym_r2_feedback.dim2; ++j)
                    for (int a = 0; a < sym_r2_feedback.dim3; ++a)
                        for (int b = 0; b < sym_r2_feedback.dim4; ++b)
                            sym_r2_feedback(i, j, a, b) +=
                                r2_feedback(j, i, b, a);
            metrics.r1_feedback_rms = tensor_rms(r1_feedback);
            metrics.r2_feedback_rms = tensor_rms(sym_r2_feedback);
            metrics.sd_residual_rms = rms_norm(pack_residuals(residuals));
            const SDUpdateMetrics sd_update = update_sd_amplitudes_with_feedback(
                calculator, state, residuals, sd_amps, diis, kSDDamping, false);
            metrics.t1_step_rms = sd_update.t1_step_rms;
            metrics.t2_step_rms = sd_update.t2_step_rms;
            store_sd_amplitudes(sd_amps, triples);

            const DressedSpinOrbitalSystem refreshed_dressed =
                build_dressed_spin_orbital_system(
                    full_system,
                    triples.amplitudes);
            const DressedSinglesDoublesIntermediates refreshed_sd_ints =
                build_dressed_sd_intermediates(
                    full_system,
                    refreshed_dressed,
                    sd_amps.t2);
            DressedTriplesIntermediates triples_ints =
                build_dressed_triples_intermediates(
                    full_system,
                    refreshed_dressed,
                    refreshed_sd_ints,
                    sd_amps.t2);
            add_dressed_triples_feedback_into_triples_intermediates(
                full_system,
                refreshed_dressed,
                triples.amplitudes.t3,
                triples_ints);
            build_dressed_triples_residual(
                full_system,
                triples_ints,
                triples.amplitudes,
                triples);
            restore_restricted_t3_structure(triples.r3);
            metrics.r3_rms = triples_residual_rms(triples.r3);
            metrics.t3_step_rms = update_t3_from_r3_jacobi(
                state.reference, triples, kT3Damping);

            // Project T3 onto restricted subspace BEFORE pushing to DIIS so the
            // subspace vectors are consistent with what the next iteration will see.
            restore_restricted_t3_structure(triples.amplitudes.t3);

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
            // Re-project after DIIS extrapolation to restore restricted structure.
            restore_restricted_t3_structure(triples.amplitudes.t3);
            metrics.estimated_correlation_energy =
                compute_rccsdt_stage_correlation_energy(
                    so_ref, so_blocks, triples.amplitudes);
            metrics.energy_change =
                metrics.estimated_correlation_energy - previous_energy;
            previous_energy = metrics.estimated_correlation_energy;
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
                    "{:3d}  E_est={:.10f}  dE={:+.3e}  rms(SD)={:.3e}  rms(R3)={:.3e}  rms(dT3)={:.3e}  rms(R1[T3])={:.3e}  rms(dT1)={:.3e}  rms(R2[T3])={:.3e}  rms(dT2)={:.3e}",
                    iter,
                    metrics.estimated_correlation_energy,
                    metrics.energy_change,
                    metrics.sd_residual_rms,
                    metrics.r3_rms,
                    metrics.t3_step_rms,
                    metrics.r1_feedback_rms,
                    metrics.t1_step_rms,
                    metrics.r2_feedback_rms,
                    metrics.t2_step_rms));

            if (metrics.r3_rms < tol_stage &&
                metrics.sd_residual_rms < tol_stage &&
                std::abs(metrics.energy_change) < tol_energy &&
                metrics.t3_step_rms < 10.0 * tol_stage &&
                metrics.t2_step_rms < 10.0 * tol_stage &&
                metrics.t1_step_rms < 10.0 * tol_stage)
            {
                metrics.converged = true;
                break;
            }

            if (iter >= min_iterations_before_break &&
                stale_iterations >= stall_patience &&
                score > deterioration_factor * best_stage_score)
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
            metrics.energy_change = best_metrics.energy_change;
            metrics.best_iteration = best_metrics.best_iteration;
            metrics.converged =
                best_metrics.quality_score < tol_stage &&
                std::abs(best_metrics.energy_change) < tol_energy &&
                best_metrics.t3_step_rms < 10.0 * tol_stage &&
                best_metrics.t2_step_rms < 10.0 * tol_stage &&
                best_metrics.t1_step_rms < 10.0 * tol_stage;
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
        calculator._have_ccsd_reference_energy = false;
        calculator._ccsd_reference_correlation_energy = 0.0;

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
        calculator._ccsd_reference_correlation_energy = rccsd_res->correlation_energy;
        calculator._have_ccsd_reference_energy = true;
        const ProductionSpinOrbitalBlocks so_blocks =
            build_spin_orbital_blocks(state_res->reference, state_res->mo_blocks);
        auto full_system_res = build_spin_orbital_chemists_system(
            calculator,
            shell_pairs,
            state_res->reference);
        if (!full_system_res)
            return std::unexpected("run_tensor_rccsdt: " + full_system_res.error());
        const DeterminantBackstopDecision backstop =
            choose_determinant_backstop(state_res->reference);
        const unsigned int stage_iteration_limit = backstop.enabled
            ? 8u
            : std::min(
                  48u,
                  std::max(20u, 2u * calculator._scf.get_max_cycles(calculator._shells.nbasis())));
        seed_triples_from_rccsd(*rccsd_res, state_res->triples);

        if (!backstop.enabled)
        {
            auto restricted_system_res = build_restricted_spatial_system(
                calculator,
                shell_pairs,
                state_res->reference);
            if (!restricted_system_res)
                return std::unexpected("run_tensor_rccsdt: " + restricted_system_res.error());

            auto restricted_res = run_restricted_tensor_rccsdt_no_fallback(
                calculator,
                *state_res,
                *restricted_system_res,
                *rccsd_res);
            if (!restricted_res)
                return std::unexpected("run_tensor_rccsdt: " + restricted_res.error());

            if (!restricted_res->converged)
            {
                return std::unexpected(
                    std::format(
                        "run_tensor_rccsdt: no determinant backstop is available for this larger system, and the standalone restricted tensor RCCSDT iterations did not converge (best rms(R3)={:.3e}, best rms(SD)={:.3e}, best rms(R2[T3])={:.3e}) within {} steps.",
                        restricted_res->r3_rms,
                        restricted_res->sd_residual_rms,
                        restricted_res->r2_feedback_rms,
                        restricted_res->iterations));
            }

            HartreeFock::Logger::blank();
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "RCCSDT[TENSOR] :",
                std::format(
                    "Standalone restricted tensor RCCSDT converged in {} steps; using the converged tensor result directly.",
                    restricted_res->iterations));
            calculator._correlation_energy = restricted_res->estimated_correlation_energy;
            return {};
        }

        auto staged_triples_res = run_staged_tensor_triples_iterations(
            calculator,
            *state_res,
            so_blocks,
            *full_system_res,
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

        return std::unexpected(
            "run_tensor_rccsdt: unexpected control flow after determinant-backstop path.");
    }
} // namespace HartreeFock::Correlation::CC
