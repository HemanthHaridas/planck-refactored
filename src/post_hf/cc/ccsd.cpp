#include "post_hf/cc/ccsd.h"

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <limits>

#include "io/logging.h"
#include "post_hf/cc/determinant_space.h"

namespace
{
    using HartreeFock::Correlation::CC::MOBlockCache;
    using HartreeFock::Correlation::CC::RCCSDAmplitudes;
    using HartreeFock::Correlation::CC::RCCSDState;
    using HartreeFock::Correlation::CC::RHFReference;
    using HartreeFock::Correlation::CC::Tensor2D;
    using HartreeFock::Correlation::CC::Tensor4D;

    // The first RCCSD implementation works in spin-orbital form because the
    // algebra is easier to validate term by term. The surrounding driver still
    // starts from an RHF spatial-orbital reference, so these helpers perform the
    // explicit spin expansion once at the top of the solver.
    struct SpinOrbitalReference
    {
        int n_occ = 0;
        int n_virt = 0;
        Eigen::VectorXd eps_occ;
        Eigen::VectorXd eps_virt;
    };

    struct SpinOrbitalBlocks
    {
        Tensor4D oooo; // <ij||kl>
        Tensor4D ooov; // <ij||ka>
        Tensor4D oovv; // <ij||ab>
        Tensor4D ovov; // <ia||jb>
        Tensor4D ovvo; // <ia||bj>
        Tensor4D ovvv; // <ia||bc>
        Tensor4D vvvv; // <ab||cd>
    };

    // `tau` and `tau_tilde` appear in many CCSD intermediates. Caching them in
    // one place keeps the residual builder readable and avoids re-forming the
    // same disconnected combinations throughout the iteration.
    struct TauCache
    {
        Tensor4D tau;
        Tensor4D tau_tilde;
    };

    struct RCCSDIntermediates
    {
        Tensor2D fae;      // F_ae
        Tensor2D fmi;      // F_mi
        Tensor2D fme;      // F_me
        Tensor4D wmnij;    // W_mnij
        Tensor4D wabef;    // W_abef
        Tensor4D wmbej;    // W_mbej
    };

    struct RCCSDResiduals
    {
        Tensor2D r1;
        Tensor4D r2;
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

    SpinOrbitalReference build_spin_orbital_reference(const RHFReference &reference)
    {
        SpinOrbitalReference so;
        so.n_occ = 2 * reference.n_occ;
        so.n_virt = 2 * reference.n_virt;
        so.eps_occ = Eigen::VectorXd(so.n_occ);
        so.eps_virt = Eigen::VectorXd(so.n_virt);

        for (int i = 0; i < so.n_occ; ++i)
            so.eps_occ(i) = reference.eps_occ(spatial_index(i));
        for (int a = 0; a < so.n_virt; ++a)
            so.eps_virt(a) = reference.eps_virt(spatial_index(a));

        return so;
    }

    SpinOrbitalBlocks build_spin_orbital_blocks(
        const RHFReference &reference,
        const MOBlockCache &spatial)
    {
        const SpinOrbitalReference so = build_spin_orbital_reference(reference);
        const auto mo = [&](int p, int q, int r, int s) -> double
        {
            return spatial.full(p, q, r, s);
        };
        const auto full_occ = [](int i) -> int
        {
            return spatial_index(i);
        };
        const auto full_virt = [&reference](int a) -> int
        {
            return reference.n_occ + spatial_index(a);
        };
        SpinOrbitalBlocks blocks{
            .oooo = Tensor4D(so.n_occ, so.n_occ, so.n_occ, so.n_occ, 0.0),
            .ooov = Tensor4D(so.n_occ, so.n_occ, so.n_occ, so.n_virt, 0.0),
            .oovv = Tensor4D(so.n_occ, so.n_occ, so.n_virt, so.n_virt, 0.0),
            .ovov = Tensor4D(so.n_occ, so.n_virt, so.n_occ, so.n_virt, 0.0),
            .ovvo = Tensor4D(so.n_occ, so.n_virt, so.n_virt, so.n_occ, 0.0),
            .ovvv = Tensor4D(so.n_occ, so.n_virt, so.n_virt, so.n_virt, 0.0),
            .vvvv = Tensor4D(so.n_virt, so.n_virt, so.n_virt, so.n_virt, 0.0),
        };

        // Expand the spatial chemists' integrals into antisymmetrized
        // spin-orbital blocks explicitly. This is not the most compact possible
        // code, but it makes the spin delta structure visible to readers.
        for (int i = 0; i < so.n_occ; ++i)
            for (int j = 0; j < so.n_occ; ++j)
                for (int k = 0; k < so.n_occ; ++k)
                    for (int l = 0; l < so.n_occ; ++l)
                        blocks.oooo(i, j, k, l) =
                            (same_spin(i, k) && same_spin(j, l)
                                 ? mo(full_occ(i), full_occ(k), full_occ(j), full_occ(l))
                                 : 0.0) -
                            (same_spin(i, l) && same_spin(j, k)
                                 ? mo(full_occ(i), full_occ(l), full_occ(j), full_occ(k))
                                 : 0.0);

        for (int i = 0; i < so.n_occ; ++i)
            for (int j = 0; j < so.n_occ; ++j)
                for (int k = 0; k < so.n_occ; ++k)
                    for (int a = 0; a < so.n_virt; ++a)
                        blocks.ooov(i, j, k, a) =
                            (same_spin(i, k) && same_spin(j, a)
                                 ? mo(full_occ(i), full_occ(k), full_occ(j), full_virt(a))
                                 : 0.0) -
                            (same_spin(i, a) && same_spin(j, k)
                                 ? mo(full_occ(i), full_virt(a), full_occ(j), full_occ(k))
                                 : 0.0);

        for (int i = 0; i < so.n_occ; ++i)
            for (int j = 0; j < so.n_occ; ++j)
                for (int a = 0; a < so.n_virt; ++a)
                    for (int b = 0; b < so.n_virt; ++b)
                        blocks.oovv(i, j, a, b) =
                            (same_spin(i, a) && same_spin(j, b)
                                 ? mo(full_occ(i), full_virt(a), full_occ(j), full_virt(b))
                                 : 0.0) -
                            (same_spin(i, b) && same_spin(j, a)
                                 ? mo(full_occ(i), full_virt(b), full_occ(j), full_virt(a))
                                 : 0.0);

        for (int i = 0; i < so.n_occ; ++i)
            for (int a = 0; a < so.n_virt; ++a)
                for (int j = 0; j < so.n_occ; ++j)
                    for (int b = 0; b < so.n_virt; ++b)
                        blocks.ovov(i, a, j, b) =
                            (same_spin(i, j) && same_spin(a, b)
                                 ? mo(full_occ(i), full_occ(j), full_virt(a), full_virt(b))
                                 : 0.0) -
                            (same_spin(i, b) && same_spin(a, j)
                                 ? mo(full_occ(i), full_virt(b), full_virt(a), full_occ(j))
                                 : 0.0);

        for (int i = 0; i < so.n_occ; ++i)
            for (int a = 0; a < so.n_virt; ++a)
                for (int b = 0; b < so.n_virt; ++b)
                    for (int j = 0; j < so.n_occ; ++j)
                        blocks.ovvo(i, a, b, j) =
                            (same_spin(i, b) && same_spin(a, j)
                                 ? mo(full_occ(i), full_virt(b), full_virt(a), full_occ(j))
                                 : 0.0) -
                            (same_spin(i, j) && same_spin(a, b)
                                 ? mo(full_occ(i), full_occ(j), full_virt(a), full_virt(b))
                                 : 0.0);

        for (int i = 0; i < so.n_occ; ++i)
            for (int a = 0; a < so.n_virt; ++a)
                for (int b = 0; b < so.n_virt; ++b)
                    for (int c = 0; c < so.n_virt; ++c)
                        blocks.ovvv(i, a, b, c) =
                            (same_spin(i, b) && same_spin(a, c)
                                 ? mo(full_occ(i), full_virt(b), full_virt(a), full_virt(c))
                                 : 0.0) -
                            (same_spin(i, c) && same_spin(a, b)
                                 ? mo(full_occ(i), full_virt(c), full_virt(a), full_virt(b))
                                 : 0.0);

        for (int a = 0; a < so.n_virt; ++a)
            for (int b = 0; b < so.n_virt; ++b)
                for (int c = 0; c < so.n_virt; ++c)
                    for (int d = 0; d < so.n_virt; ++d)
                        blocks.vvvv(a, b, c, d) =
                            (same_spin(a, c) && same_spin(b, d)
                                 ? mo(full_virt(a), full_virt(c), full_virt(b), full_virt(d))
                                 : 0.0) -
                            (same_spin(a, d) && same_spin(b, c)
                                 ? mo(full_virt(a), full_virt(d), full_virt(b), full_virt(c))
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
        const SpinOrbitalReference &reference,
        const SpinOrbitalBlocks &blocks,
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
        const SpinOrbitalReference &reference,
        const SpinOrbitalBlocks &blocks,
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
        const SpinOrbitalReference &reference,
        const SpinOrbitalBlocks &blocks,
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

    double rms_norm(const Eigen::VectorXd &vec)
    {
        if (vec.size() == 0)
            return 0.0;
        return std::sqrt(vec.squaredNorm() / static_cast<double>(vec.size()));
    }

    void initialize_mp2_guess(
        const SpinOrbitalReference &reference,
        const SpinOrbitalBlocks &blocks,
        const RCCSDState &prepared,
        RCCSDAmplitudes &amps)
    {
        // Canonical RHF implies Brillouin singles vanish at first order, so the
        // natural starting point is zero T1 and MP2 doubles.
        (void)reference;
        for (int i = 0; i < amps.t2.dim1; ++i)
            for (int j = 0; j < amps.t2.dim2; ++j)
                for (int a = 0; a < amps.t2.dim3; ++a)
                    for (int b = 0; b < amps.t2.dim4; ++b)
                        amps.t2(i, j, a, b) = blocks.oovv(i, j, a, b) /
                                              prepared.denominators.d2(
                                                  spatial_index(i), spatial_index(j),
                                                  spatial_index(a), spatial_index(b));
    }
} // namespace

namespace HartreeFock::Correlation::CC
{
    std::expected<RCCSDState, std::string> prepare_rccsd(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        if (calculator._calculation != HartreeFock::CalculationType::SinglePoint)
            return std::unexpected("prepare_rccsd: RCCSD is currently available only for single-point calculations.");

        auto reference_res = build_rhf_reference(calculator);
        if (!reference_res)
            return std::unexpected(reference_res.error());

        auto block_res = build_mo_block_cache(calculator, shell_pairs, *reference_res, "RCCSD :");
        if (!block_res)
            return std::unexpected(block_res.error());

        auto denom_res = build_denominator_cache(*reference_res, false);
        if (!denom_res)
            return std::unexpected(denom_res.error());

        RCCSDAmplitudes amplitudes = make_zero_rccsd_amplitudes(*reference_res);
        RCCSDState state{
            .reference = std::move(*reference_res),
            .mo_blocks = std::move(*block_res),
            .denominators = std::move(*denom_res),
            .amplitudes = std::move(amplitudes),
        };
        return state;
    }

    std::expected<void, std::string> run_rccsd(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        auto state_res = prepare_rccsd(calculator, shell_pairs);
        if (!state_res)
            return std::unexpected(state_res.error());

        RCCSDState state = std::move(*state_res);
        const SpinOrbitalReference so_ref = build_spin_orbital_reference(state.reference);
        const SpinOrbitalBlocks so_blocks = build_spin_orbital_blocks(state.reference, state.mo_blocks);

        RCCSDAmplitudes so_amps{
            .t1 = Tensor2D(so_ref.n_occ, so_ref.n_virt, 0.0),
            .t2 = Tensor4D(so_ref.n_occ, so_ref.n_occ, so_ref.n_virt, so_ref.n_virt, 0.0),
        };
        initialize_mp2_guess(so_ref, so_blocks, state, so_amps);

        const unsigned int max_iter = calculator._scf.get_max_cycles(calculator._shells.nbasis());
        const double tol_energy = calculator._scf._tol_energy;
        const double tol_residual = calculator._scf._tol_density;
        const bool use_diis = calculator._scf._use_DIIS;
        const double damping = 0.8;

        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "RCCSD :",
            std::format("Spin-orbital solver dimensions: nocc={} nvirt={}", so_ref.n_occ, so_ref.n_virt));
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "RCCSD :",
            std::format("Convergence targets: |dE|<{:.1e}  rms(res)<{:.1e}", tol_energy, tol_residual));
        HartreeFock::Logger::blank();

        AmplitudeDIIS diis(static_cast<int>(std::max(2u, calculator._scf._DIIS_dim)));
        double energy = compute_rccsd_correlation_energy(so_ref, so_blocks, so_amps);
        double previous_energy = energy;

        for (unsigned int iter = 1; iter <= max_iter; ++iter)
        {
            const auto iter_start = std::chrono::steady_clock::now();

            // Each sweep follows the textbook CC pattern: build disconnected
            // combinations, form intermediates, evaluate residuals, then apply a
            // diagonal Jacobi-style update accelerated by DIIS.
            const TauCache tau_cache = build_tau_cache(so_amps);
            const RCCSDIntermediates ints = build_intermediates(so_ref, so_blocks, so_amps, tau_cache);
            const RCCSDResiduals residuals = build_residuals(so_ref, so_blocks, so_amps, tau_cache, ints);
            const Eigen::VectorXd residual_vec = pack_residuals(residuals);
            const double residual_rms = rms_norm(residual_vec);

            Eigen::VectorXd current = pack_amplitudes(so_amps);
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

            unpack_amplitudes(updated, so_amps);
            energy = compute_rccsd_correlation_energy(so_ref, so_blocks, so_amps);
            const double delta_energy = energy - previous_energy;
            previous_energy = energy;

            const double time_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - iter_start).count();

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "RCCSD Iter :",
                std::format(
                    "{:3d}  E_corr={:.10f}  dE={:+.3e}  rms(res)={:.3e}  rms(step)={:.3e}  diis={}  t={:.3f}s",
                    iter, energy, delta_energy, residual_rms, update_rms, diis.size(), time_sec));

            if (std::abs(delta_energy) < tol_energy && residual_rms < tol_residual)
            {
                calculator._correlation_energy = energy;
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "RCCSD :",
                    std::format("Converged in {} iterations.", iter));
                return {};
            }
        }

        return std::unexpected(
            std::format("run_rccsd: failed to converge in {} iterations.", max_iter));
    }

    std::expected<UCCSDState, std::string> prepare_uccsd(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        (void)shell_pairs;
        if (calculator._calculation != HartreeFock::CalculationType::SinglePoint)
            return std::unexpected("prepare_uccsd: UCCSD is currently available only for single-point calculations.");

        auto reference_res = build_uhf_reference(calculator);
        if (!reference_res)
            return std::unexpected(reference_res.error());

        return UCCSDState{
            .reference = std::move(*reference_res),
        };
    }

    std::expected<void, std::string> run_uccsd(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        auto state_res = prepare_uccsd(calculator, shell_pairs);
        if (!state_res)
            return std::unexpected(state_res.error());

        auto system_res = build_uhf_spin_orbital_system(
            calculator, shell_pairs, state_res->reference, "UCCSD :");
        if (!system_res)
            return std::unexpected(system_res.error());

        auto corr_res = solve_determinant_cc(
            calculator, *system_res, 2, "UCCSD :");
        if (!corr_res)
            return std::unexpected("run_uccsd: " + corr_res.error());

        calculator._correlation_energy = *corr_res;
        return {};
    }
} // namespace HartreeFock::Correlation::CC
