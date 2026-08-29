#include "post_hf/cc/tensor_backend_internal.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <format>
#include <limits>
#include <stdexcept>

#include "io/logging.h"
// run_rccgen: the arbitrary-order harness, where the generated kernels are correct.
#include "post_hf/cc/generated_arbitrary_runtime.h"
#include "post_hf/cc/generated_kernel_registry.h"
#include "post_hf/cc/rccgen.h"
#include "post_hf/cc/determinant_space.h"
#include "post_hf/cc/diis.h"
// rebind_physicist: generated kernels index physicist <pq|rs> (T1b).
#include "post_hf/cc/generated_arbitrary_runtime.h"
#include "post_hf/integrals.h"

#include "generated/cc/ccsdt_planck_generated.cpp"

namespace
{
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
    using HartreeFock::Correlation::CC::AmplitudeDIIS;
    using HartreeFock::Correlation::CC::CanonicalRHFCCReference;
    using HartreeFock::Correlation::CC::RCCSDAmplitudes;
    using HartreeFock::Correlation::CC::RCCSDTAmplitudes;
    using HartreeFock::Correlation::CC::RHFReference;
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
        Tensor2D fock; // [p,q] chemists/MO ordering
        Tensor4D eri;  // [p,r,q,s] to match PySCF eris.pppp storage
    };

    struct DressedSpinOrbitalSystem
    {
        Tensor2D fock; // [p,q]
        Tensor4D eri;  // [p,r,q,s] to match PySCF's t1_eris storage
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

    TauCache build_tau_cache(const RCCSDAmplitudes &amps);

    RCCSDIntermediates build_intermediates(
        const ProductionSpinOrbitalReference &reference,
        const ProductionSpinOrbitalBlocks &blocks,
        const RCCSDAmplitudes &amps,
        const TauCache &tau_cache);

    RCCSDResiduals build_residuals(
        const ProductionSpinOrbitalReference &reference,
        const ProductionSpinOrbitalBlocks &blocks,
        const RCCSDAmplitudes &amps,
        const TauCache &tau_cache,
        const RCCSDIntermediates &ints);

#include "generated/cc/ccsd_spinorbital_warm_start.inc"

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

        // Start from the spatial-orbital RHF partition and duplicate each
        // orbital energy into alpha/beta spin-orbital slots.  The tensor CC
        // backend then works entirely in spin-orbital indexing even though the
        // source SCF reference is restricted.
        for (int i = 0; i < so.n_occ; ++i)
            so.eps_occ(i) = base.eps_occ(spatial_index(i));
        for (int a = 0; a < so.n_virt; ++a)
            so.eps_virt(a) = base.eps_virt(spatial_index(a));

        return so;
    }

    std::expected<void, std::string> enforce_spin_orbital_vvvv_memory_cap(
        const HartreeFock::Calculator &calculator,
        const ProductionSpinOrbitalReference &so_ref,
        std::string_view context)
    {
        const double cap_gb = calculator._scf._cc_max_memory_gb;
        if (cap_gb <= 0.0)
            return {};

        const long double nvirt = static_cast<long double>(so_ref.n_virt);
        const long double estimated_bytes =
            nvirt * nvirt * nvirt * nvirt * static_cast<long double>(sizeof(double));
        const long double cap_bytes =
            cap_gb * 1024.0L * 1024.0L * 1024.0L;
        if (estimated_bytes <= cap_bytes)
            return {};

        return std::unexpected(std::format(
            "{} would allocate an estimated {:.2f} GiB spin-orbital vvvv block, exceeding cc_max_memory_gb={:.2f}.",
            context,
            static_cast<double>(estimated_bytes / (1024.0L * 1024.0L * 1024.0L)),
            cap_gb));
    }

    std::expected<ProductionSpinOrbitalBlocks, std::string> build_spin_orbital_blocks(
        const HartreeFock::Calculator &calculator,
        const CanonicalRHFCCReference &reference,
        const TensorCCBlockCache &spatial)
    {
        const auto so = build_spin_orbital_reference(reference);
        if (auto cap_res = enforce_spin_orbital_vvvv_memory_cap(
                calculator,
                so,
                "Spin-orbital tensor expansion");
            !cap_res)
        {
            return std::unexpected(cap_res.error());
        }

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

        // Expand each spatial block into antisymmetrized spin-orbital blocks.
        // The `same_spin` gates encode the RHF selection rule that a spatial
        // integral contributes only when the corresponding spin labels match,
        // and the second term in each assignment inserts the exchange piece.
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

        // This variant keeps plain chemists-notation Coulomb blocks without the
        // antisymmetrization step above.  The generated RCCSDT path and the
        // PySCF-aligned "dressed intermediate" experiments consume these raw
        // blocks directly and apply their own permutation algebra later.
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
        const TensorRCCSDTState &state,
        bool use_generated_kernels)
    {
        const ProductionSpinOrbitalReference so_ref = build_spin_orbital_reference(state.reference);
        auto so_blocks_res = build_spin_orbital_blocks(calculator, state.reference, state.mo_blocks);
        if (!so_blocks_res)
            return std::unexpected(so_blocks_res.error());
        const ProductionSpinOrbitalBlocks so_blocks = std::move(*so_blocks_res);

        RCCSDAmplitudes amps{
            .t1 = Tensor2D(so_ref.n_occ, so_ref.n_virt, 0.0),
            .t2 = Tensor4D(so_ref.n_occ, so_ref.n_occ, so_ref.n_virt, so_ref.n_virt, 0.0),
        };
        initialize_mp2_guess(so_blocks, state, amps);

        const unsigned int max_iter =
            std::max(calculator._scf.get_max_cycles(calculator._shells.nbasis()), 100u);
        const double tol_energy = calculator._scf._tol_energy;
        const double tol_residual = calculator._scf._tol_density;
        const bool use_diis = calculator._scf._use_DIIS;
        const double damping = std::clamp(calculator._scf._cc_damping, 0.0, 1.0);

        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "RCCSDT[TENSOR] :",
            std::format(
                "Stage-1 RCCSD warm start dimensions: nocc={} nvirt={} (kernels={})",
                so_ref.n_occ,
                so_ref.n_virt,
                use_generated_kernels ? "ccgen-generated" : "hand-optimized"));

        AmplitudeDIIS diis(static_cast<int>(std::max(2u, calculator._scf._DIIS_dim)));
        double energy = use_generated_kernels
                            ? compute_generated_spin_orbital_rccsd_correlation_energy(
                                  so_ref, so_blocks, amps)
                            : compute_rccsd_correlation_energy(so_ref, so_blocks, amps);
        double previous_energy = energy;

        for (unsigned int iter = 1; iter <= max_iter; ++iter)
        {
            const auto iter_start = std::chrono::steady_clock::now();

            RCCSDResiduals residuals;
            if (use_generated_kernels)
            {
                residuals = compute_generated_spin_orbital_rccsd_residuals(
                    so_ref, so_blocks, amps);
            }
            else
            {
                const TauCache tau_cache = build_tau_cache(amps);
                const RCCSDIntermediates ints = build_intermediates(
                    so_ref, so_blocks, amps, tau_cache);
                residuals = build_residuals(
                    so_ref, so_blocks, amps, tau_cache, ints);
            }
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
            energy = use_generated_kernels
                         ? compute_generated_spin_orbital_rccsd_correlation_energy(
                               so_ref, so_blocks, amps)
                         : compute_rccsd_correlation_energy(so_ref, so_blocks, amps);
            const double delta_energy = energy - previous_energy;
            previous_energy = energy;

            const double time_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - iter_start).count();

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "RCCSDT[TENSOR-RCCSD] :",
                std::format(
                    "{:3d}  E_corr={:.10f}  dE={:+.3e}  rms(res)={:.3e}  rms(step)={:.3e}  diis={}  kernel={}  t={:.3f}s",
                    iter,
                    energy,
                    delta_energy,
                    residual_rms,
                    update_rms,
                    diis.size(),
                    use_generated_kernels ? "ccgen" : "native",
                    time_sec));

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
                                    value += ints.w_vvvo(a, c, d, k) * amps.t2(i, j, d, b);
                                    value += ints.w_vvvo(b, a, d, i) * amps.t2(j, k, d, c);
                                    value += ints.w_vvvo(b, c, d, k) * amps.t2(j, i, d, a);
                                    value += ints.w_vvvo(c, a, d, i) * amps.t2(k, j, d, b);
                                    value += ints.w_vvvo(c, b, d, j) * amps.t2(k, i, d, a);

                                    value += ints.f_vv(a, d) * amps.t3(i, j, k, d, b, c);
                                    value += ints.f_vv(b, d) * amps.t3(j, i, k, d, a, c);
                                    value += ints.f_vv(c, d) * amps.t3(k, j, i, d, b, a);
                                }
                                for (int l = 0; l < system.n_occ; ++l)
                                {
                                    value -= ints.w_vooo(a, l, i, j) * amps.t2(l, k, b, c);
                                    value -= ints.w_vooo(a, l, i, k) * amps.t2(l, j, c, b);
                                    value -= ints.w_vooo(b, l, j, i) * amps.t2(l, k, a, c);
                                    value -= ints.w_vooo(b, l, j, k) * amps.t2(l, i, c, a);
                                    value -= ints.w_vooo(c, l, k, i) * amps.t2(l, j, a, b);
                                    value -= ints.w_vooo(c, l, k, j) * amps.t2(l, i, b, a);

                                    value -= ints.f_oo(l, i) * amps.t3(l, j, k, a, b, c);
                                    value -= ints.f_oo(l, j) * amps.t3(l, i, k, b, a, c);
                                    value -= ints.f_oo(l, k) * amps.t3(l, j, i, c, b, a);
                                }
                                for (int l = 0; l < system.n_occ; ++l)
                                    for (int d = 0; d < system.n_virt; ++d)
                                    {
                                        value += 0.5 * ints.w_ovvo(l, a, d, i) *
                                                 t3_p201(amps.t3, l, j, k, d, b, c);
                                        value += 0.5 * ints.w_ovvo(l, b, d, j) *
                                                 t3_p201(amps.t3, l, i, k, d, a, c);
                                        value += 0.5 * ints.w_ovvo(l, c, d, k) *
                                                 t3_p201(amps.t3, l, j, i, d, b, a);

                                        value -= ints.w_ovov(l, b, i, d) *
                                                 amps.t3(j, l, k, d, a, c);
                                        value -= ints.w_ovov(l, c, i, d) *
                                                 amps.t3(k, l, j, d, a, b);
                                        value -= 0.5 * ints.w_ovov(l, a, i, d) *
                                                 (amps.t3(j, l, k, d, b, c) +
                                                  amps.t3(k, l, j, d, c, b));

                                        value -= ints.w_ovov(l, a, j, d) *
                                                 amps.t3(i, l, k, d, b, c);
                                        value -= ints.w_ovov(l, c, j, d) *
                                                 amps.t3(k, l, i, d, b, a);
                                        value -= 0.5 * ints.w_ovov(l, b, j, d) *
                                                 (amps.t3(i, l, k, d, a, c) +
                                                  amps.t3(k, l, i, d, c, a));

                                        value -= ints.w_ovov(l, a, k, d) *
                                                 amps.t3(i, l, j, d, c, b);
                                        value -= ints.w_ovov(l, b, k, d) *
                                                 amps.t3(j, l, i, d, c, a);
                                        value -= 0.5 * ints.w_ovov(l, c, k, d) *
                                                 (amps.t3(i, l, j, d, a, b) +
                                                  amps.t3(j, l, i, d, b, a));
                                    }
                                for (int l = 0; l < system.n_occ; ++l)
                                    for (int m = 0; m < system.n_occ; ++m)
                                    {
                                        value += ints.w_oooo(l, m, i, j) *
                                                 amps.t3(l, m, k, a, b, c);
                                        value += ints.w_oooo(l, m, i, k) *
                                                 amps.t3(l, m, j, a, c, b);
                                        value += ints.w_oooo(l, m, j, k) *
                                                 amps.t3(l, m, i, b, c, a);
                                    }
                                for (int d = 0; d < system.n_virt; ++d)
                                    for (int e = 0; e < system.n_virt; ++e)
                                    {
                                        value += ints.w_vvvv(a, b, d, e) *
                                                 amps.t3(i, j, k, d, e, c);
                                        value += 0.5 * ints.w_vvvv(a, c, d, e) *
                                                 (amps.t3(i, k, j, d, e, b) +
                                                  amps.t3(k, i, j, e, d, b));
                                        value += 0.5 * ints.w_vvvv(b, c, d, e) *
                                                 (amps.t3(j, k, i, d, e, a) +
                                                  amps.t3(k, j, i, e, d, a));
                                    }
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
        const TensorRCCSDTState &state,
        const ProductionSpinOrbitalChemistsSystem &system,
        const RHFReference &reference,
        RCCSDTAmplitudes &amps,
        bool use_generated_kernels,
        const TensorCCBlockCache &physicist_blocks)
    {
        RestrictedRCCSDTUpdateMetrics metrics;
        RCCSDResiduals residuals;
        Tensor6D triples_residual(
            amps.t3.dim1, amps.t3.dim2, amps.t3.dim3,
            amps.t3.dim4, amps.t3.dim5, amps.t3.dim6, 0.0);
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
        residuals =
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

        // T2 probe (PLANCK_CC_T3_LADDER=N, N>=1 repeats): the THREE-armed ladder
        // measurement scoped in docs/CCGEN_DRESSED_LADDER_SCOPE.md.
        //
        // Why it lives here and not beside PLANCK_CC_T3_TIME: that probe sits inside
        // `if (use_generated_kernels)`, a branch the rank-3 representation fix rerouted
        // away from, so it cannot fire in any build. This one runs unconditionally on
        // the path that executes.
        //
        // Three arms, one binary, one fixture -- because the question is whether
        // dressing changes the generated kernel's SCALING EXPONENTS or only its
        // CONSTANT, and answering that needs the hand-written kernel as the known-good
        // asymptotic standard (o^3.94 v^4.18), not merely as a baseline. Comparing two
        // generated arms alone would measure dressing's speedup while saying nothing
        // about whether the scaling is now correct.
        //
        //   arm A  hand-written    build_dressed_triples_residual
        //   arm B  generated       via the arbitrary-order harness (the path that runs)
        //   arm C  arm B again under a dressed build -- selected by PLANCK_CC_DRESSING
        //          at CONFIGURE time, so B and C are the same call in two binaries.
        //
        // Diagnostic only: it evaluates into scratch and never touches
        // `triples_residual`.
        if (const char *ladder = std::getenv("PLANCK_CC_T3_LADDER");
            ladder != nullptr && ladder[0] != '\0' && ladder[0] != '0')
        {
            const int repeats = std::max(1, std::atoi(ladder));
            const int no = state.reference.orbital_partition.n_occ;
            const int nv = state.reference.orbital_partition.n_virt;

            // One amplitude source for both arms. to_tensor_nd is a pure
            // reinterpretation (same buffer, dims in order, no permutation) and is exact
            // because the layouts already agree: RCCSDTAmplitudes::t3 is (i,j,k,a,b,c)
            // allocated (n_occ x3, n_virt x3), which is rank_dims' occ-first order. The
            // (vir...,occ...) transpose recorded in CCGEN_SPIN_ADAPT_DEFAULT.md is a
            // ccgen-Python-vs-C++ concern and does not apply between these two C++ types.
            HartreeFock::Correlation::CC::ArbitraryOrderRCCAmplitudes seed;
            seed.by_rank.push_back(to_tensor_nd(amps.t1));
            seed.by_rank.push_back(to_tensor_nd(amps.t2));
            seed.by_rank.push_back(to_tensor_nd(amps.t3));

            // Match prepare_generated_arbitrary_order_state exactly: it builds the
            // denominators from `reference.orbital_partition`, not from a separately
            // supplied RHFReference. Using the wrong one is a silent scale error.
            auto arb_denoms = build_arbitrary_order_denominator_cache(
                state.reference.orbital_partition, 3);
            auto gen_kernels =
                HartreeFock::Correlation::CC::make_generated_rcc_kernels(3);

            if (!arb_denoms || !gen_kernels)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Error, "RCCSDT[T3-LADDER] :",
                    std::format("setup failed: {}",
                                !arb_denoms ? arb_denoms.error() : gen_kernels.error()));
            }
            else
            {
                HartreeFock::Correlation::CC::ArbitraryOrderTensorCCState gen_state{
                    .reference = state.reference,
                    .mo_blocks = physicist_blocks,
                    .denominators = std::move(*arb_denoms),
                    .amplitudes = std::move(seed),
                    .max_excitation_rank = 3,
                };

                // --- the agreement gate, BEFORE any timing -----------------------
                // A timing comparison across arms that are not evaluating the same
                // equation is worthless, and this codebase has produced exactly that
                // twice (the -7.56e-05 representation defect, and the 52% dressed
                // defect). If this gate fires, the timings below are not reported.
                auto gate = evaluate_generated_arbitrary_order_residuals(gen_state, *gen_kernels);
                if (!gate)
                {
                    HartreeFock::Logger::logging(
                        HartreeFock::LogLevel::Error, "RCCSDT[T3-LADDER] :",
                        std::format("generated residual failed: {}", gate.error()));
                }
                else
                {
                    auto gen_r3 = gate->tensor(3);

                    // T2.5: is the disagreement CONFINED to rank 3, or does it appear at
                    // every rank? This is the one framing the rank-3 investigation did
                    // not test -- it compared converged energies and the rank-3 residual,
                    // never the per-rank split at fixed amplitudes.
                    //
                    // `residuals` here is the hand-written r1/r2 AFTER
                    // add_dressed_triples_feedback_into_sd_residuals, i.e. with the
                    // T3->SD feedback already folded in. The generated arm returns all
                    // ranks from one evaluation. If the arms differ at rank 1 and 2 as
                    // well, the disagreement is not a rank-3 property at all and the
                    // per-rank slicing is the wrong comparison.
                    for (int rr = 1; rr <= 2; ++rr)
                    {
                        auto gen_lo = gate->tensor(rr);
                        if (!gen_lo)
                            continue;
                        const std::vector<double> &hand_lo =
                            (rr == 1) ? residuals.r1.data : residuals.r2.data;
                        if (gen_lo->size() != hand_lo.size())
                        {
                            HartreeFock::Logger::logging(
                                HartreeFock::LogLevel::Info, "RCCSDT[T3-LADDER] :",
                                std::format("T2.5 rank {}: SHAPE differs gen={} hand={}",
                                            rr, gen_lo->size(), hand_lo.size()));
                            continue;
                        }
                        double d = 0.0;
                        double m = 0.0;
                        for (std::size_t idx = 0; idx < hand_lo.size(); ++idx)
                        {
                            d = std::max(d, std::abs(gen_lo->data[idx] - hand_lo[idx]));
                            m = std::max(m, std::abs(hand_lo[idx]));
                        }
                        HartreeFock::Logger::logging(
                            HartreeFock::LogLevel::Info, "RCCSDT[T3-LADDER] :",
                            std::format("T2.5 rank {}: max|gen-hand|={:.6e} max|hand|={:.6e} "
                                        "rel={:.3e}",
                                        rr, d, m, d / std::max(m, 1e-30)));
                    }

                    DressedTriplesIntermediates gate_ints =
                        build_dressed_triples_intermediates(system, dressed, sd_ints, amps.t2);
                    add_dressed_triples_feedback_into_triples_intermediates(
                        system, dressed, amps.t3, gate_ints);
                    TensorTriplesWorkspace gate_ws{
                        .amplitudes = clone_rccsdt_amplitudes(amps),
                        .r3 = Tensor6D(
                            amps.t3.dim1, amps.t3.dim2, amps.t3.dim3,
                            amps.t3.dim4, amps.t3.dim5, amps.t3.dim6, 0.0),
                        .allocated = true,
                    };
                    build_dressed_triples_residual(system, gate_ints, amps, gate_ws);

                    // Both arms are symmetrized before comparison, because they emit
                    // DIFFERENT BUT EQUIVALENT representatives of the same permutation
                    // orbit: the generated kernels emit every index permutation
                    // explicitly, while the hand-written path keeps one canonical
                    // representative and relies on `restore` to rebuild the rest. The
                    // production path applies exactly this call to its own residual
                    // (line ~2714) before consuming it, so comparing the RAW outputs
                    // compares two conventions rather than two equations -- which is
                    // what the first version of this gate did, reporting rel=1.000.
                    //
                    // This is not loosening the gate: `restore` is idempotent on an
                    // already-symmetric tensor, so if the two arms genuinely disagreed
                    // the difference would survive it.
                    bool comparable =
                        gen_r3.has_value() &&
                        gen_r3->size() == gate_ws.r3.data.size();

                    double max_abs_diff = 0.0;
                    double max_abs_hand = 0.0;
                    if (comparable)
                    {
                        // UNRESOLVED (see docs/CCGEN_DRESSED_LADDER_SCOPE.md T2.5).
                        // Both residuals are individually CORRECT -- each converges to
                        // E_corr=-0.0791116825 on CH4, matching PySCF to 1.4e-08 -- so
                        // this is a representation mismatch, not a kernel defect. Do not
                        // "fix" it by loosening the tolerance.
                        //
                        // NEITHER arm is restored here, and that is deliberate.
                        // `restore` belongs to a wedge-packed AMPLITUDE inside its own
                        // solver -- the packing and restore are ONE COUPLED CONVENTION,
                        // established in CCGEN_RANK3_KERNEL_AND_SOLVER.md (:21-24) during
                        // the (now retired) dressed-operator work, not here. Applying it
                        // to a raw residual is a category error.
                        //
                        // T2.2 only re-measured the magnitude: restore annihilates the
                        // hand-written residual by 2.0e+05 (7.00e-03 -> 3.56e-08),
                        // because stage 2 (apply_restricted_t3_p3_full) subtracts the
                        // virt-permutation mean, which for the fully-symmetric tensor
                        // stage 1 produces is that tensor itself.
                        //
                        // An earlier revision read "restore both -> rel=3.6e-02" as the
                        // closest framing. It was comparing against a near-zero tensor;
                        // the small number was the generated arm's own magnitude. The
                        // multisets of |value| do not match (5.24e-03), so this is not an
                        // index permutation either.
                        //
                        // The generated arm is NOT restored here: the arbitrary-order
                        // harness never calls restore (grep confirms zero call sites in
                        // solver_arbitrary.cpp / generated_arbitrary_runtime.cpp),
                        // because the generated kernels emit every index permutation
                        // explicitly and are already complete. Only the hand-written arm
                        // needs completing -- its solver keeps one canonical
                        // representative per orbit and calls restore before consuming it
                        // (line ~2714). Symmetrizing the generated arm too would be
                        // double-counting, which is what the previous revision did.
                        Tensor6D sym_gen(
                            gate_ws.r3.dim1, gate_ws.r3.dim2, gate_ws.r3.dim3,
                            gate_ws.r3.dim4, gate_ws.r3.dim5, gate_ws.r3.dim6, 0.0);
                        sym_gen.data.assign(gen_r3->data, gen_r3->data + gen_r3->size());

                        Tensor6D sym_hand(
                            gate_ws.r3.dim1, gate_ws.r3.dim2, gate_ws.r3.dim3,
                            gate_ws.r3.dim4, gate_ws.r3.dim5, gate_ws.r3.dim6, 0.0);
                        sym_hand.data = gate_ws.r3.data;
                        restore_restricted_t3_structure(sym_hand);

                        // T2.1: dump BOTH arms elementwise so T2.2 can compare them as
                        // multisets of values. A scalar max cannot distinguish "wrong
                        // values" from "right values in a different index order" -- the
                        // same reason R4.2c added the generated-arm dump in rccgen.cpp,
                        // whose format this matches byte-for-byte (rank ndims / dims /
                        // one value per line at 17 digits) so the two are directly
                        // comparable.
                        //
                        // Opt-in via PLANCK_CC_T3_LADDER_DUMP=<dir>; without it the probe
                        // writes nothing. Written BEFORE the gate verdict, because the
                        // dump is what diagnoses a failing gate.
                        if (const char *dump_dir = std::getenv("PLANCK_CC_T3_LADDER_DUMP");
                            dump_dir != nullptr && dump_dir[0] != '\0')
                        {
                            const std::filesystem::path dir(dump_dir);
                            std::error_code ec;
                            std::filesystem::create_directories(dir, ec);
                            const auto emit = [&](const char *name, const Tensor6D &t) {
                                if (std::ofstream fh(dir / name); fh)
                                {
                                    fh << 3 << ' ' << 6 << '\n';
                                    fh << t.dim1 << ' ' << t.dim2 << ' ' << t.dim3 << ' '
                                       << t.dim4 << ' ' << t.dim5 << ' ' << t.dim6 << '\n';
                                    fh << std::setprecision(17);
                                    for (const double value : t.data)
                                        fh << value << '\n';
                                }
                            };
                            // Both the raw and the restored hand-written arm: T2.4 needs
                            // to decompose `restore`, and re-running to get the other one
                            // would risk comparing two different amplitude states.
                            Tensor6D raw_hand(
                                gate_ws.r3.dim1, gate_ws.r3.dim2, gate_ws.r3.dim3,
                                gate_ws.r3.dim4, gate_ws.r3.dim5, gate_ws.r3.dim6, 0.0);
                            raw_hand.data = gate_ws.r3.data;
                            emit("r3_gen_raw.txt", sym_gen);
                            emit("r3_hand_raw.txt", raw_hand);
                            emit("r3_hand_restored.txt", sym_hand);
                            HartreeFock::Logger::logging(
                                HartreeFock::LogLevel::Info, "RCCSDT[T3-LADDER] :",
                                std::format("T2.1 dumped r3_gen_raw / r3_hand_raw / "
                                            "r3_hand_restored to '{}' (n={})",
                                            dir.string(), sym_hand.data.size()));
                        }

                        for (std::size_t idx = 0; idx < sym_hand.data.size(); ++idx)
                        {
                            max_abs_diff = std::max(
                                max_abs_diff,
                                std::abs(sym_gen.data[idx] - sym_hand.data[idx]));
                            max_abs_hand =
                                std::max(max_abs_hand, std::abs(sym_hand.data[idx]));
                        }
                    }

                    // Relative, because the residual magnitude varies by orders of
                    // magnitude across the ladder; an absolute 1e-12 would be vacuous
                    // at one end and unmeetable at the other.
                    double max_abs_gen = 0.0;
                    if (comparable)
                        for (std::size_t idx = 0; idx < gen_r3->size(); ++idx)
                            max_abs_gen = std::max(max_abs_gen, std::abs(gen_r3->data[idx]));
                    HartreeFock::Logger::logging(
                        HartreeFock::LogLevel::Info, "RCCSDT[T3-LADDER] :",
                        std::format("DIAG(raw) max|gen|={:.6e} max|hand|={:.6e} "
                                    "gen_dims={} hand_n={}",
                                    max_abs_gen, max_abs_hand,
                                    gen_r3.has_value() ? gen_r3->size() : 0,
                                    gate_ws.r3.data.size()));
                    const double rel =
                        max_abs_diff / std::max(max_abs_hand, 1e-30);
                    constexpr double kAgreementTol = 1e-10;

                    if (!comparable)
                    {
                        HartreeFock::Logger::logging(
                            HartreeFock::LogLevel::Error, "RCCSDT[T3-LADDER] :",
                            "AGREEMENT GATE FAILED: residual shapes differ; the two arms "
                            "are not evaluating the same manifold. No timings reported.");
                    }
                    else if (!(rel <= kAgreementTol))
                    {
                        HartreeFock::Logger::logging(
                            HartreeFock::LogLevel::Error, "RCCSDT[T3-LADDER] :",
                            std::format(
                                "AGREEMENT GATE FAILED: max|gen-hand|={:.3e} "
                                "rel={:.3e} > {:.0e}. The arms are timing DIFFERENT "
                                "equations; no timings reported.",
                                max_abs_diff, rel, kAgreementTol));
                    }
                    else
                    {
                        // --- arms, timed only after the gate passes -------------
                        double gen_seconds = 0.0;
                        for (int r = 0; r < repeats; ++r)
                        {
                            const auto t0 = std::chrono::steady_clock::now();
                            auto scratch = evaluate_generated_arbitrary_order_residuals(
                                gen_state, *gen_kernels);
                            gen_seconds += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0).count();
                            (void)scratch;
                        }
                        gen_seconds /= repeats;

                        // Intermediates once, outside the loop: the generated kernel
                        // builds none, so charging them per-repeat would overstate the
                        // hand-written arm. Reported separately as ints=.
                        const auto ti = std::chrono::steady_clock::now();
                        DressedTriplesIntermediates time_ints =
                            build_dressed_triples_intermediates(system, dressed, sd_ints, amps.t2);
                        add_dressed_triples_feedback_into_triples_intermediates(
                            system, dressed, amps.t3, time_ints);
                        const double int_seconds = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - ti).count();

                        double hand_seconds = 0.0;
                        for (int r = 0; r < repeats; ++r)
                        {
                            const auto t0 = std::chrono::steady_clock::now();
                            build_dressed_triples_residual(system, time_ints, amps, gate_ws);
                            hand_seconds += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0).count();
                        }
                        hand_seconds /= repeats;

                        const double t3_mib =
                            static_cast<double>(amps.t3.data.size() * sizeof(double)) /
                            (1024.0 * 1024.0);

                        // t3 MiB is reported so an H1 cache transition is VISIBLE rather
                        // than inferred; the whole reachable ladder sits under 0.85 MiB
                        // (inside L2), which is why H1 stays untestable on it.
                        HartreeFock::Logger::logging(
                            HartreeFock::LogLevel::Info, "RCCSDT[T3-LADDER] :",
                            std::format(
                                "no={} nv={} o/v={:.3f} t3={:.3f}MiB reps={} "
                                "dressing={} gen={:.6f}s hand={:.6f}s ints={:.6f}s "
                                "ratio={:.1f}x gate_rel={:.2e}",
                                no, nv,
                                nv > 0 ? static_cast<double>(no) / nv : 0.0,
                                t3_mib, repeats,
                                PLANCK_CC_DRESS_OPERATORS ? "on" : "off",
                                gen_seconds, hand_seconds, int_seconds,
                                gen_seconds / std::max(hand_seconds, 1e-12),
                                rel));
                    }
                }
            }
        }

        if (use_generated_kernels)
        {
            // T1b: ccgen emits against the physicist <pq|rs> convention, but `state.mo_blocks`
            // holds chemists' (pq|rs). The arbitrary-order path has always rebound before
            // calling a generated kernel; this path did not, so the first execution of
            // `compute_ccsdt_triples_residual` read permuted integrals -- a wrong-but-plausible
            // T3 that still converged, 1.8e-4 Eh off.
            //
            // `physicist_blocks` is rebound ONCE by the caller and passed in, not cached here:
            // this function runs every iteration, and a function-local static would both repeat
            // the transform and (worse) leak one molecule's integrals into the next run in the
            // same process. The shared `state.mo_blocks` stays chemists', because the
            // hand-written branch below and `build_spin_orbital_blocks` read it.
            triples_residual = HartreeFock::Correlation::CC::compute_ccsdt_triples_residual(
                state.reference, physicist_blocks, state.denominators, amps);

            // P3 probe (PLANCK_CC_T3_TIME=N, N>=1 repeats): time the generated and hand-written
            // T3 residual evaluations from identical amplitudes and report the ratio alongside
            // o, v and the t3 working-set size. Separate from T3_DIFF because P3 wants the
            // timing on cases where the value comparison is uninteresting, and because at post-
            // accessor-fix speeds (~1e-3 s) a single shot is noise; N repeats are averaged.
            // Diagnostic only -- it re-evaluates into scratch and does not touch
            // `triples_residual`. See docs/CCGEN_KERNEL_SCALING_SCOPE.md.
            if (const char *timing = std::getenv("PLANCK_CC_T3_TIME");
                timing != nullptr && timing[0] != '\0' && timing[0] != '0')
            {
                const int repeats = std::max(1, std::atoi(timing));
                const int no = state.reference.orbital_partition.n_occ;
                const int nv = state.reference.orbital_partition.n_virt;

                double gen_seconds = 0.0;
                for (int r = 0; r < repeats; ++r)
                {
                    const auto t0 = std::chrono::steady_clock::now();
                    auto scratch = HartreeFock::Correlation::CC::compute_ccsdt_triples_residual(
                        state.reference, physicist_blocks, state.denominators, amps);
                    gen_seconds +=
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();
                    HartreeFock::Correlation::CC::Tensor6D sink = std::move(scratch);
                    (void)sink;
                }
                gen_seconds /= repeats;

                // Intermediates are built once outside the timed loop: the generated kernel
                // builds none, so charging their cost per-repeat to the hand-written side
                // would overstate it. Their one-off cost is reported separately.
                const auto ti = std::chrono::steady_clock::now();
                const DressedSinglesDoublesIntermediates time_sd =
                    build_dressed_sd_intermediates(system, dressed, amps.t2);
                DressedTriplesIntermediates time_ints =
                    build_dressed_triples_intermediates(system, dressed, time_sd, amps.t2);
                add_dressed_triples_feedback_into_triples_intermediates(
                    system, dressed, amps.t3, time_ints);
                const double int_seconds =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - ti).count();

                TensorTriplesWorkspace time_ws{
                    .amplitudes = clone_rccsdt_amplitudes(amps),
                    .r3 = Tensor6D(
                        amps.t3.dim1, amps.t3.dim2, amps.t3.dim3,
                        amps.t3.dim4, amps.t3.dim5, amps.t3.dim6, 0.0),
                    .allocated = true,
                };
                double hand_seconds = 0.0;
                for (int r = 0; r < repeats; ++r)
                {
                    const auto t0 = std::chrono::steady_clock::now();
                    build_dressed_triples_residual(system, time_ints, amps, time_ws);
                    hand_seconds +=
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();
                }
                hand_seconds /= repeats;

                const double t3_mib =
                    static_cast<double>(amps.t3.data.size() * sizeof(double)) / (1024.0 * 1024.0);

                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "RCCSDT[T3-TIME] :",
                    std::format(
                        "no={} nv={} o/v={:.3f} t3={:.3f}MiB reps={} "
                        "gen={:.6f}s hand={:.6f}s ints={:.6f}s ratio={:.1f}x",
                        no, nv,
                        nv > 0 ? static_cast<double>(no) / nv : 0.0,
                        t3_mib, repeats,
                        gen_seconds, hand_seconds, int_seconds,
                        gen_seconds / std::max(hand_seconds, 1e-12)));
            }

            // T0 probe (PLANCK_CC_T3_DIFF=1): compute the HAND-WRITTEN residual from the same
            // amplitudes and report the elementwise difference, both before and after
            // `restore_restricted_t3_structure`. One evaluation, not a ~21-minute solve, and it
            // separates "the generated residual value is wrong" from "the values agree but the
            // solver path diverges". The before/after pair additionally says whether the
            // restricted-T3 convention is the remaining gap. Diagnostic only -- off by default,
            // and it does not alter `triples_residual`.
            if (const char *probe = std::getenv("PLANCK_CC_T3_DIFF");
                probe != nullptr && probe[0] == '1')
            {
                Tensor6D raw_generated(
                    triples_residual.dim1, triples_residual.dim2, triples_residual.dim3,
                    triples_residual.dim4, triples_residual.dim5, triples_residual.dim6, 0.0);
                raw_generated.data = triples_residual.data;

                const DressedSinglesDoublesIntermediates probe_sd =
                    build_dressed_sd_intermediates(system, dressed, amps.t2);
                DressedTriplesIntermediates probe_ints =
                    build_dressed_triples_intermediates(system, dressed, probe_sd, amps.t2);
                add_dressed_triples_feedback_into_triples_intermediates(
                    system, dressed, amps.t3, probe_ints);
                TensorTriplesWorkspace probe_ws{
                    .amplitudes = clone_rccsdt_amplitudes(amps),
                    .r3 = Tensor6D(
                        amps.t3.dim1, amps.t3.dim2, amps.t3.dim3,
                        amps.t3.dim4, amps.t3.dim5, amps.t3.dim6, 0.0),
                    .allocated = true,
                };
                build_dressed_triples_residual(system, probe_ints, amps, probe_ws);

                const auto report = [](const char *label,
                                       const Tensor6D &a, const Tensor6D &b) {
                    double max_abs = 0.0, sum_sq = 0.0, ref_max = 0.0;
                    for (std::size_t i = 0; i < a.data.size(); ++i)
                    {
                        const double d = a.data[i] - b.data[i];
                        max_abs = std::max(max_abs, std::abs(d));
                        sum_sq += d * d;
                        ref_max = std::max(ref_max, std::abs(b.data[i]));
                    }
                    HartreeFock::Logger::logging(
                        HartreeFock::LogLevel::Info,
                        "RCCSDT[T3-DIFF] :",
                        std::format("{}: max|gen-hand|={:.6e} rms={:.6e} max|hand|={:.6e}",
                                    label, max_abs,
                                    std::sqrt(sum_sq / static_cast<double>(a.data.size())),
                                    ref_max));
                };

                report("raw (no restore)", raw_generated, probe_ws.r3);
                Tensor6D restored_generated(
                    raw_generated.dim1, raw_generated.dim2, raw_generated.dim3,
                    raw_generated.dim4, raw_generated.dim5, raw_generated.dim6, 0.0);
                restored_generated.data = raw_generated.data;
                restore_restricted_t3_structure(restored_generated);
                Tensor6D restored_hand(
                    probe_ws.r3.dim1, probe_ws.r3.dim2, probe_ws.r3.dim3,
                    probe_ws.r3.dim4, probe_ws.r3.dim5, probe_ws.r3.dim6, 0.0);
                restored_hand.data = probe_ws.r3.data;
                restore_restricted_t3_structure(restored_hand);
                report("after restore   ", restored_generated, restored_hand);
            }

            // `restore` is REQUIRED here, and is not optional: this solver's DIIS packs only
            // the unique wedge (i<=j<=k) and rebuilds the rest via
            // restore_restricted_t3_from_unique, which is information-preserving only if the
            // amplitudes carry full permutational symmetry. Removing this call diverges --
            // measured, with both hand-written and generated residual sources. See
            // docs/CCGEN_RANK3_KERNEL_AND_SOLVER.md.
            restore_restricted_t3_structure(triples_residual);
            metrics.r3_residual_rms = triples_residual_rms(triples_residual);
        }
        else
        {
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
            triples_residual.data = triples.r3.data;
            metrics.r3_residual_rms = triples_residual_rms(triples_residual);
        }

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
        metrics.t3_step_rms =
            update_restricted_t3_from_r3_jacobi(reference, amps, triples_residual, 1.0);

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
        const TensorRCCSDResult &rccsd,
        bool use_generated_kernels)
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

        // T1b: rebind ONCE, outside the loop. ccgen kernels index physicist <pq|rs> while
        // `state.mo_blocks` is chemists' (pq|rs); only the generated branch consumes this, and
        // the shared cache must stay chemists' for the hand-written branch. Built
        // unconditionally (a few tensor transposes) rather than lazily, to keep the loop body
        // free of a first-iteration special case.
        const TensorCCBlockCache physicist_blocks =
            HartreeFock::Correlation::CC::rebind_physicist(state.mo_blocks);

        for (unsigned int iter = 1; iter <= max_iter; ++iter)
        {
            const Eigen::VectorXd unique_before =
                pack_restricted_unique_rccsdt_amplitudes(amps);
            const RestrictedRCCSDTUpdateMetrics update_metrics =
                update_restricted_rccsdt_amplitudes_once(
                    state, system, reference, amps, use_generated_kernels,
                    physicist_blocks);
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
                    "{:3d}  E_corr={:.10f}  dE={:+.3e}  norm(d tamps)={:.3e}  rms(SD)={:.3e}  rms(R3)={:.3e}  rms(R1[T3])={:.3e}  rms(R2[T3])={:.3e}  kernel={}",
                    iter,
                    metrics.estimated_correlation_energy,
                    metrics.energy_change,
                    update_metrics.norm_dtamps,
                    metrics.sd_residual_rms,
                    metrics.r3_rms,
                    metrics.r1_feedback_rms,
                    metrics.r2_feedback_rms,
                    use_generated_kernels ? "ccgen" : "native"));

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

        const double triples_damping = calculator._scf._cc_damping;
        const double sd_damping = calculator._scf._cc_damping;
        if (triples_damping < 0.0 || triples_damping > 1.0)
            return std::unexpected("run_staged_tensor_triples_iterations: cc_damping must be between 0 and 1.");
        // The staged tensor path is meant to become the production solver for
        // larger systems, so it should not stop at a tolerance inherited from
        // the more forgiving SCF density threshold. Keep the stage criterion
        // tight enough that the SD/T3 coupling is genuinely refined before
        // handing off to the moderate-case fallback.
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
                calculator, state, residuals, sd_amps, diis, sd_damping, false);
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
                state.reference, triples, triples_damping);

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
    RCCSDTBackend choose_rccsdt_backend(
        const RHFReference &reference) noexcept
    {
        constexpr int kPrototypeMaxSpinOrbitals = 12;
        constexpr std::size_t kPrototypeMaxDeterminants = 1200;

        // TensorOptimized is the only backend that runs the ccgen-GENERATED restricted
        // triples kernel; the other two run hand-written residuals. It is therefore what a
        // build configured with -DPLANCK_CC_DRESS_OPERATORS=ON is asking for: dressing
        // rewrites the generated kernel, so selecting a hand-written backend would silently
        // ignore the option and report numbers from code the flag never touched.
        //
        // Selected by build configuration rather than by system size, because it is a
        // statement about which kernel to exercise, not about cost. Without the option the
        // size-based choice below is unchanged, so default builds do not move.
#ifdef PLANCK_CC_DRESS_OPERATORS
        if constexpr (PLANCK_CC_DRESS_OPERATORS)
            return RCCSDTBackend::TensorOptimized;
#endif

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

    std::expected<void, std::string> run_tensor_rccsdt_impl(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        bool use_generated_warm_start,
        bool use_generated_triples_kernel)
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
            use_generated_warm_start
                ? "Running the ccgen-generated RCCSD warm start before enabling T3 residuals."
                : "Running the production-path RCCSD warm start before enabling T3 residuals.");
        if (use_generated_triples_kernel)
        {
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "RCCSDT[TENSOR] :",
                "Using the ccgen-generated restricted CCSDT triples residual inside the standalone tensor solver.");
        }
        HartreeFock::Logger::blank();

        auto rccsd_res = run_tensor_rccsd_stage(
            calculator,
            *state_res,
            use_generated_warm_start);
        if (!rccsd_res)
            return std::unexpected("run_tensor_rccsdt: " + rccsd_res.error());

        state_res->warm_start_correlation_energy = rccsd_res->correlation_energy;
        state_res->warm_start_iterations = rccsd_res->iterations;
        calculator._ccsd_reference_correlation_energy = rccsd_res->correlation_energy;
        calculator._have_ccsd_reference_energy = true;
        auto so_blocks_res =
            build_spin_orbital_blocks(calculator, state_res->reference, state_res->mo_blocks);
        if (!so_blocks_res)
            return std::unexpected("run_tensor_rccsdt: " + so_blocks_res.error());
        const ProductionSpinOrbitalBlocks so_blocks = std::move(*so_blocks_res);
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
                *rccsd_res,
                use_generated_triples_kernel);
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
                        detail::format_bytes(state_res->triples.storage_bytes),
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

    std::expected<void, std::string> run_tensor_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        return run_tensor_rccsdt_impl(calculator, shell_pairs, false, false);
    }

    std::expected<void, std::string> run_tensor_optimized_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        // This backend used to call `run_tensor_rccsdt_impl(..., true, true)`: the generated
        // triples kernel inside `tensor_backend`'s solver. That combination is WRONG -- it
        // converges to a self-consistent but incorrect fixed point (CH4/STO-3G: -39.8059200873
        // against PySCF rccsdt -39.8058445240, i.e. -7.56e-05, recovering more correlation than
        // CCSDTQ, which is variationally impossible).
        //
        // The cause is a representation mismatch, not a bad kernel. `tensor_backend`'s solver
        // is built around a SYMMETRY-PACKED amplitude representation: its DIIS packs only the
        // unique wedge (i<=j for t2, i<=j<=k for t3) and rebuilds the rest on unpack via
        // `restore_restricted_t{2,3}_from_unique`, which is information-preserving only if the
        // amplitudes carry full permutational symmetry -- the property
        // `restore_restricted_t3_structure` imposes on the residual each iteration. The ccgen
        // kernels instead emit every index permutation explicitly, so they do not produce
        // residuals in that representation. `restore` and the wedge DIIS are one coupled
        // convention: removing either half diverges, and no combination of the two residual
        // sources converges to the right answer. Measured, all on CH4/STO-3G:
        //
        //   r1/r2 hand + r3 gen + restore  -> converges, -7.56e-05   (the old behavior)
        //   r1/r2 gen  + r3 gen + restore  -> converges, +8.23e-05
        //   r1/r2 gen  + r3 gen, no restore-> diverges
        //   arbitrary harness (dense pack) -> +1.49e-08   <- correct
        //
        // The arbitrary-order harness is the generated kernels' native home: it packs dense
        // tensors, needs no symmetrization step, and reproduces PySCF. So route there instead
        // of maintaining a second, subtly-incompatible solver around the same kernels.
        //
        // Full record: docs/CCGEN_RANK3_KERNEL_AND_SOLVER.md.
        constexpr int kArbitraryLowerRanks = PLANCK_CC_ARBITRARY_LOWER_RANKS;
        if constexpr (kArbitraryLowerRanks == 0)
        {
            return std::unexpected(
                "run_tensor_optimized_rccsdt: the generated rank-3 CCSDT kernel runs only in the "
                "arbitrary-order harness, which needs -DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON. "
                "Reconfigure with that option, or use the hand-written tensor backend "
                "(PLANCK_RCCSDT_BACKEND=tensor), which is PySCF-validated by ch4_rccsdt_sto3g.");
        }
        else
        {
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "RCCSDT[OPT] :",
                "Routing the ccgen-generated rank-3 CCSDT kernels through the arbitrary-order "
                "harness (the representation they are emitted for).");
            calculator._scf._cc_generated_rank = 3;
            return run_rccgen(calculator, shell_pairs);
        }
    }
} // namespace HartreeFock::Correlation::CC
