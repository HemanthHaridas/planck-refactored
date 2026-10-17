#ifndef HF_MP2_H
#define HF_MP2_H

#include <Eigen/Core>
#include <functional>
#include <limits>
#include <memory>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::Correlation
{
    struct RMP2T2PairTile;
    struct RMP2T2TileSource;
    struct RMP2NaturalOrbitals
    {
        Eigen::VectorXd occupations;
        Eigen::MatrixXd coefficients_mo;
        Eigen::MatrixXd coefficients_ao;
    };

    // PySCF-aligned restricted MP2 state. This is the public object callers
    // pass into RDM and gradient helpers rather than re-running MP2 implicitly.
    struct RMP2Result
    {
        double e_corr = 0.0;
        double e_corr_ss = 0.0;
        double e_corr_os = 0.0;
        bool converged = true;
        int n_iter = 0;
        int n_occ = 0;
        int n_virt = 0;
        std::vector<int> active_mo;
        std::vector<double> t2; // [i,j,a,b] row-major; empty if not requested
        std::vector<double> t2_pair_packed; // upper triangle over composite (i,a)
        // Geometry-scoped nonresident canonical-amplitude source. Compact DH
        // uses this instead of retaining t2_pair_packed in memory.
        std::shared_ptr<const RMP2T2TileSource> t2_source;
        Eigen::MatrixXd mo_coeff;
        Eigen::VectorXd mo_energy;
        Eigen::VectorXd mo_occ;

        [[nodiscard]] bool has_t2() const;
        [[nodiscard]] double t2_at(int i, int j, int a, int b) const;
        [[nodiscard]] std::size_t resident_t2_bytes() const;
    };

    // Exact packed representation of canonical RHF T2. Pair interchange gives
    // t_ij^ab=t_ji^ba, so the upper triangle over composite (i,a) indices is
    // sufficient. This is an adapter until the bounded canonical provider
    // consumes it directly.
    struct RMP2PairPackedT2
    {
        int n_occ = 0;
        int n_virt = 0;
        std::vector<double> values;

        [[nodiscard]] double at(int i, int j, int a, int b) const;
    };

    [[nodiscard]] std::expected<RMP2PairPackedT2, std::string>
    pack_rmp2_pair_t2(const RMP2Result &result);

    // A bounded, deterministic view of a rectangular composite-pair block.
    // p=i*n_virt+a labels the row pair and q=j*n_virt+b labels the column
    // pair.  The tile deliberately represents the full ordered p,q domain:
    // consumers therefore need not make any storage-specific symmetry
    // assumptions.  At most pair_tile_width^2 doubles are live per callback.
    struct RMP2T2PairTile
    {
        int n_virt = 0;
        int p0 = 0, q0 = 0;
        int rows = 0, cols = 0;
        std::vector<double> values; // [p-p0,q-q0] row-major

        [[nodiscard]] double at(int i, int j, int a, int b) const;
    };

    struct RMP2T2TileSource
    {
        virtual ~RMP2T2TileSource() = default;
        [[nodiscard]] virtual int n_occ() const = 0;
        [[nodiscard]] virtual int n_virt() const = 0;
        [[nodiscard]] virtual std::size_t resident_bytes() const = 0;
        [[nodiscard]] virtual std::expected<RMP2T2PairTile, std::string>
        read(int p0, int q0, int rows, int cols) const = 0;
    };

    [[nodiscard]] std::expected<std::shared_ptr<const RMP2T2TileSource>, std::string>
    make_rmp2_scratch_t2_source(int n_occ, int n_virt,
                                const std::vector<double> &pair_packed);

    inline bool RMP2Result::has_t2() const
    {
        const std::size_t nt=static_cast<std::size_t>(n_occ)*n_occ*n_virt*n_virt;
        const int npair=n_occ*n_virt;
        return t2.size()==nt || t2_pair_packed.size()==static_cast<std::size_t>(npair)*(npair+1)/2 ||
            (t2_source && t2_source->n_occ()==n_occ && t2_source->n_virt()==n_virt);
    }

    inline double RMP2Result::t2_at(int i, int j, int a, int b) const
    {
        if (i<0 || j<0 || a<0 || b<0 || i>=n_occ || j>=n_occ || a>=n_virt || b>=n_virt)
            return std::numeric_limits<double>::quiet_NaN();
        if (!t2.empty()) return t2[((static_cast<std::size_t>(i)*n_occ+j)*n_virt+a)*n_virt+b];
        const int npair=n_occ*n_virt;
        if (t2_pair_packed.size()==static_cast<std::size_t>(npair)*(npair+1)/2)
        {
            int p=i*n_virt+a, q=j*n_virt+b;
            if (p>q) std::swap(p,q);
            return t2_pair_packed[static_cast<std::size_t>(p)*npair-
                static_cast<std::size_t>(p)*(p-1)/2+(q-p)];
        }
        if (!t2_source) return std::numeric_limits<double>::quiet_NaN();
        auto tile=t2_source->read(i*n_virt+a,j*n_virt+b,1,1);
        return tile ? tile->values[0] : std::numeric_limits<double>::quiet_NaN();
    }

    inline std::size_t RMP2Result::resident_t2_bytes() const
    {
        return t2.size()*sizeof(double)+t2_pair_packed.size()*sizeof(double)+
            (t2_source ? t2_source->resident_bytes() : 0);
    }

    using RMP2T2PairTileConsumer = std::function<bool(const RMP2T2PairTile &)>;

    // Streams the complete canonical T2 tensor as composite-pair tiles in
    // lexicographic (p0,q0) order.  This is the stable consumer seam for a
    // later recoverable/nonresident provider; today it reads either full or
    // pair-packed RMP2Result storage through the same accessor.
    [[nodiscard]] std::expected<void, std::string>
    for_each_rmp2_t2_pair_tile(const RMP2Result &result, int pair_tile_width,
                               const RMP2T2PairTileConsumer &consumer);

    // Read one bounded composite-pair rectangle.  This is the random-tile
    // counterpart to the ordered stream above: contractions such as D' need
    // two differently oriented occupied-pair panels at once.  A future
    // recoverable owner implements this operation by reloading/recomputing
    // just this rectangle, never by exposing scalar random access to DH.
    [[nodiscard]] inline std::expected<RMP2T2PairTile, std::string>
    read_rmp2_t2_pair_tile(const RMP2Result &result, int p0, int q0,
                           int rows, int cols)
    {
        const int npair=result.n_occ*result.n_virt;
        if (!result.has_t2() || result.n_occ<=0 || result.n_virt<=0 ||
            p0<0 || q0<0 || rows<=0 || cols<=0 || p0+rows>npair || q0+cols>npair)
            return std::unexpected("RMP2 T2 tile read: invalid canonical source or rectangle.");
        if (result.t2_source)
            return result.t2_source->read(p0,q0,rows,cols);
        RMP2T2PairTile tile{result.n_virt,p0,q0,rows,cols,
            std::vector<double>(static_cast<std::size_t>(rows)*cols)};
        for (int ip=0;ip<rows;++ip)
            for (int iq=0;iq<cols;++iq)
            {
                const int p=p0+ip, q=q0+iq;
                tile.values[static_cast<std::size_t>(ip)*cols+iq]=result.t2_at(
                    p/result.n_virt,q/result.n_virt,
                    p%result.n_virt,q%result.n_virt);
            }
        return tile;
    }

    // PySCF-aligned unrestricted MP2 state.
    struct UMP2Result
    {
        double e_corr = 0.0;
        double e_corr_ss = 0.0;
        double e_corr_os = 0.0;
        bool converged = true;
        int n_iter = 0;
        int nocca = 0;
        int noccb = 0;
        int nvira = 0;
        int nvirb = 0;
        std::vector<int> active_mo_alpha;
        std::vector<int> active_mo_beta;
        std::vector<double> t2_aa;
        std::vector<double> t2_ab;
        std::vector<double> t2_bb;
        Eigen::MatrixXd mo_coeff_alpha;
        Eigen::MatrixXd mo_coeff_beta;
        Eigen::VectorXd mo_energy_alpha;
        Eigen::VectorXd mo_energy_beta;
        Eigen::VectorXd mo_occ_alpha;
        Eigen::VectorXd mo_occ_beta;
    };

    // Core MP2 kernels. These are the public entry points for running MP2 and
    // mirror the role of PySCF's kernel methods.
    std::expected<RMP2Result, std::string> rmp2_kernel(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const HartreeFock::OptionsMP2 &options);

    std::expected<UMP2Result, std::string> ump2_kernel(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const HartreeFock::OptionsMP2 &options);

    // Persist a kernel result back onto Calculator so the rest of the codebase
    // can consume the correlation energy and, when requested, cached amplitudes.
    std::expected<void, std::string> apply_rmp2_result(
        HartreeFock::Calculator &calculator,
        const RMP2Result &result,
        bool retain_amplitudes = true);

    std::expected<void, std::string> apply_ump2_result(
        HartreeFock::Calculator &calculator,
        const UMP2Result &result);

    std::expected<RMP2NaturalOrbitals, std::string> rmp2_make_natural_orbitals(
        const RMP2Result &result);

    std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
    rmp2_gamma1_intermediates(const RMP2Result &result);

    std::expected<Eigen::MatrixXd, std::string>
    rmp2_make_rdm1(const RMP2Result &result, bool ao_repr = false);

    std::expected<std::vector<double>, std::string>
    rmp2_make_rdm2(const RMP2Result &result, bool ao_repr = false);

    std::expected<
        std::pair<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>,
                  std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>,
        std::string>
    ump2_gamma1_intermediates(const UMP2Result &result);

    std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
    ump2_make_rdm1(const UMP2Result &result, bool ao_repr = false);
} // namespace HartreeFock::Correlation

#endif
