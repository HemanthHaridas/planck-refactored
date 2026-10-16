#include "dh_pt2_gradient.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include "analytic_hessian.h"
#include "dft_gradient.h"
#include "integrals/base.h"

namespace
{
    std::size_t idx(int i, int j, int a, int b, int no, int nv)
    {
        return ((static_cast<std::size_t>(i) * no + j) * nv + a) * nv + b;
    }

    std::size_t eri_idx(int p, int q, int r, int s, int nmo)
    {
        return ((static_cast<std::size_t>(p) * nmo + q) * nmo + r) * nmo + s;
    }

    // Pack only a bounded slice of the reduction dimension. Both operands
    // are owned before GEMM; no tensor-layout or permutation symmetry is
    // delegated to Eigen. Output and packing require quadratic and linear
    // (at fixed tile width) storage, respectively.
    template<class Left, class Right>
    Eigen::MatrixXd tiled_product(int rows, int cols, std::size_t reduction,
        int tile, Left left, Right right)
    {
        Eigen::MatrixXd out = Eigen::MatrixXd::Zero(rows, cols);
        const int capacity=static_cast<int>(std::min<std::size_t>(tile,reduction));
        Eigen::MatrixXd a(rows,capacity), b(capacity,cols);
        for (std::size_t begin = 0; begin < reduction; begin += tile)
        {
            const int width = static_cast<int>(std::min<std::size_t>(tile, reduction-begin));
            for (int k = 0; k < width; ++k)
            {
                for (int p = 0; p < rows; ++p) a(p,k) = left(p,begin+k);
                for (int q = 0; q < cols; ++q) b(k,q) = right(begin+k,q);
            }
            out.noalias() += a.leftCols(width) * b.topRows(width);
        }
        return out;
    }

    std::vector<double> eri_eightfold_symmetrize(const std::vector<double> &raw, int n)
    {
        std::vector<double> symmetric(raw.size(), 0.0);
        for (int mu = 0; mu < n; ++mu)
            for (int nu = 0; nu < n; ++nu)
                for (int ka = 0; ka < n; ++ka)
                    for (int ta = 0; ta < n; ++ta)
                    {
                        const auto get = [&raw, n](int p, int q, int r, int s)
                        { return raw[eri_idx(p, q, r, s, n)]; };
                        symmetric[eri_idx(mu, nu, ka, ta, n)] = 0.125 * (
                            get(mu, nu, ka, ta) + get(nu, mu, ka, ta) +
                            get(mu, nu, ta, ka) + get(nu, mu, ta, ka) +
                            get(ka, ta, mu, nu) + get(ta, ka, mu, nu) +
                            get(ka, ta, nu, mu) + get(ta, ka, nu, mu));
                    }
        return symmetric;
    }

    // Eq. (33)'s non-XC contraction at one nuclear Cartesian coordinate.
    // Keeping this algebra independent of how h/S/ERI derivatives are
    // produced is the O6 seam: the legacy dense bundle and the forthcoming
    // quartet-streaming producer must feed these identical contractions.
    struct Eq33NonXCAccumulator
    {
        const std::vector<double> &separable_ao;
        const std::vector<double> &nonseparable_ao;

        void accumulate_eri(Eigen::Index coordinate, const std::vector<double> &eri,
                            DFT::Gradient::DHEq33NonXCGradient &out) const
        {
            double separable = 0.0;
            double nonseparable = 0.0;
            for (std::size_t q = 0; q < eri.size(); ++q)
            {
                separable += separable_ao[q] * eri[q];
                nonseparable += nonseparable_ao[q] * eri[q];
            }
            out.two_electron_separable(coordinate) = separable;
            out.two_electron_nonseparable(coordinate) = nonseparable;
        }
    };

    // Preserve row-major tensor order while replacing one index. For each
    // prefix, map [old-index, suffix] and multiply C[new,old] into column
    // panels. Only the input and output tensors coexist; no permuted copy,
    // full AO-by-MO-quartet product, or repeated amplitude sum is formed.
    std::vector<double> transform_pair_axis(const std::vector<double> &input,
        std::array<std::size_t,4> &shape, int axis,
        const Eigen::Ref<const Eigen::MatrixXd> &coefficients, int column_tile)
    {
        std::size_t prefix=1, suffix=1;
        for(int k=0;k<axis;++k) prefix*=shape[k];
        for(int k=axis+1;k<4;++k) suffix*=shape[k];
        const std::size_t old=shape[axis], nao=coefficients.rows();
        std::vector<double> output(prefix*nao*suffix);
        using RowMatrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
        if(suffix==1)
        {
            // The last-index transform is one batched matrix product, not
            // one matrix-vector call for every i,a,j prefix.
            const Eigen::Map<const RowMatrix> source(input.data(),prefix,old);
            Eigen::Map<RowMatrix> target(output.data(),prefix,nao);
            for(std::size_t begin=0;begin<prefix;begin+=column_tile)
            {
                const auto width=static_cast<Eigen::Index>(std::min<std::size_t>(column_tile,prefix-begin));
                target.middleRows(begin,width).noalias() = source.middleRows(begin,width) * coefficients.transpose();
            }
            shape[axis]=nao;
            return output;
        }
        for(std::size_t p=0;p<prefix;++p)
        {
            const Eigen::Map<const RowMatrix> source(input.data()+p*old*suffix,old,suffix);
            Eigen::Map<RowMatrix> target(output.data()+p*nao*suffix,nao,suffix);
            for(std::size_t begin=0;begin<suffix;begin+=column_tile)
            {
                const auto width=static_cast<Eigen::Index>(std::min<std::size_t>(column_tile,suffix-begin));
                target.middleCols(begin,width).noalias() = coefficients * source.middleCols(begin,width);
            }
        }
        shape[axis]=nao;
        return output;
    }
}

namespace DFT::Gradient
{
    double DHTildeAmplitudePairTile::at(int i, int j, int a, int b) const
    {
        if (n_virt<=0 || i<0 || j<0 || a<0 || b<0)
            return std::numeric_limits<double>::quiet_NaN();
        const int p=i*n_virt+a, q=j*n_virt+b;
        if (p<p0 || p>=p0+rows || q<q0 || q>=q0+cols ||
            values.size()!=static_cast<std::size_t>(rows)*cols)
            return std::numeric_limits<double>::quiet_NaN();
        return values[static_cast<std::size_t>(p-p0)*cols+(q-q0)];
    }

    double DHPT2AmplitudeDensity::tilde(int i, int j, int a, int b) const
    {
        const std::size_t need=static_cast<std::size_t>(n_occ)*n_occ*n_virt*n_virt;
        const auto at=[this](int ii,int jj,int aa,int bb)
        { return ((static_cast<std::size_t>(ii)*n_occ+jj)*n_virt+aa)*n_virt+bb; };
        if (i<0 || j<0 || a<0 || b<0 || i>=n_occ || j>=n_occ || a>=n_virt || b>=n_virt)
            return std::numeric_limits<double>::quiet_NaN();
        // Do not infer the representation from vector emptiness: a corrupted
        // one-amplitude Materialized object is empty after truncation.
        if (storage==DHTildeAmplitudeStorage::Materialized)
            return t_tilde.size()==need ? t_tilde[at(i,j,a,b)]
                                        : std::numeric_limits<double>::quiet_NaN();
        if (!t_tilde.empty() || source_pt2==nullptr || !source_pt2->has_t2())
            return std::numeric_limits<double>::quiet_NaN();
        return c_pt2*2.0/static_cast<double>(1+(i==j))*
            (2.0*source_pt2->t2_at(i,j,a,b)-source_pt2->t2_at(i,j,b,a));
    }

    bool DHPT2AmplitudeDensity::has_tilde_storage() const
    {
        const std::size_t need=static_cast<std::size_t>(n_occ)*n_occ*n_virt*n_virt;
        return storage==DHTildeAmplitudeStorage::Materialized
            ? t_tilde.size()==need
            : (t_tilde.empty() && source_pt2!=nullptr && source_pt2->has_t2());
    }

    DHAmplitudeOwnershipAudit audit_dh_amplitude_ownership(
        const HartreeFock::Correlation::RMP2Result &result,
        const DHPT2AmplitudeDensity &amplitudes)
    {
        DHAmplitudeOwnershipAudit out;
        out.raw_t2_bytes=result.t2.size()*sizeof(double);
        out.packed_t2_bytes=result.t2_pair_packed.size()*sizeof(double);
        out.source_resident_bytes=result.t2_source ? result.t2_source->resident_bytes() : 0;
        out.tilde_bytes=amplitudes.t_tilde.size()*sizeof(double);
        out.source_valid=amplitudes.storage==DHTildeAmplitudeStorage::SourceView &&
            amplitudes.source_pt2==&result && result.t2_source &&
            result.t2_source->n_occ()==result.n_occ && result.t2_source->n_virt()==result.n_virt;
        return out;
    }

    std::expected<void, std::string>
    DHPT2AmplitudeDensity::for_each_pair_tile(
        int pair_tile_width, const DHTildeAmplitudePairTileConsumer &consumer) const
    {
        if (!has_tilde_storage() || pair_tile_width<=0 || !consumer)
            return std::unexpected("DH PT2 tilde tile stream: invalid source, width, or consumer.");
        const int npair=n_occ*n_virt;
        for (int p0=0;p0<npair;p0+=pair_tile_width)
            for (int q0=0;q0<npair;q0+=pair_tile_width)
            {
                const int rows=std::min(pair_tile_width,npair-p0);
                const int cols=std::min(pair_tile_width,npair-q0);
                auto tile=read_pair_tile(p0,q0,rows,cols);
                if (!tile) return std::unexpected(tile.error());
                if (!consumer(*tile)) return {};
            }
        return {};
    }

    std::expected<DHTildeAmplitudePairTile, std::string>
    DHPT2AmplitudeDensity::read_pair_tile(int p0, int q0, int rows, int cols) const
    {
        const int npair=n_occ*n_virt;
        if (!has_tilde_storage() || p0<0 || q0<0 || rows<=0 || cols<=0 ||
            p0+rows>npair || q0+cols>npair)
            return std::unexpected("DH PT2 tilde tile read: invalid source or rectangle.");
        DHTildeAmplitudePairTile tile{n_virt,p0,q0,rows,cols,
            std::vector<double>(static_cast<std::size_t>(rows)*cols)};
        for (int ip=0;ip<rows;++ip)
            for (int iq=0;iq<cols;++iq)
            {
                const int p=p0+ip, q=q0+iq;
                tile.values[static_cast<std::size_t>(ip)*cols+iq]=
                    tilde(p/n_virt,q/n_virt,p%n_virt,q%n_virt);
            }
        return tile;
    }

    std::expected<DHEq41ResponseOperator, std::string>
    make_dh_eq41_response_operator(
        double exact_exchange,
        DHResponseFn coulomb,
        DHResponseFn exchange,
        DHResponseFn xc)
    {
        if (!std::isfinite(exact_exchange) || exact_exchange < 0.0 ||
            !coulomb || !xc || (exact_exchange != 0.0 && !exchange))
            return std::unexpected("DH Eq. 41 response: invalid channel configuration.");
        return DHEq41ResponseOperator{exact_exchange, std::move(coulomb),
                                    std::move(exchange), std::move(xc)};
    }

    std::expected<DHEq41ResponseOperator, std::string>
    make_dh_eq41_direct_eri_response_operator(const DHEq41DirectERIInputs &inputs)
    {
        if (inputs.shell_pairs == nullptr || inputs.nbasis == 0 ||
            !std::isfinite(inputs.tol_eri) || inputs.tol_eri < 0.0)
            return std::unexpected("DH Eq. 41 response: invalid direct-ERI inputs.");

        const auto direct = inputs;
        DHResponseFn coulomb = [direct](const Eigen::Ref<const Eigen::MatrixXd> &density)
            -> std::expected<Eigen::MatrixXd, std::string>
        {
            if (density.rows() != static_cast<Eigen::Index>(direct.nbasis) ||
                density.cols() != static_cast<Eigen::Index>(direct.nbasis))
                return std::unexpected("DH Eq. 41 Coulomb: density dimensions do not match the AO basis.");
            return _compute_2e_j_direct(*direct.shell_pairs, density, direct.nbasis,
                direct.engine, HartreeFock::ERIKernel::Coulomb, 0.0,
                direct.tol_eri, direct.sym_ops);
        };
        DHResponseFn exchange = [direct](const Eigen::Ref<const Eigen::MatrixXd> &density)
            -> std::expected<Eigen::MatrixXd, std::string>
        {
            if (density.rows() != static_cast<Eigen::Index>(direct.nbasis) ||
                density.cols() != static_cast<Eigen::Index>(direct.nbasis))
                return std::unexpected("DH Eq. 41 exchange: density dimensions do not match the AO basis.");
            return _compute_2e_k_direct(*direct.shell_pairs, density, direct.nbasis,
                direct.engine, HartreeFock::ERIKernel::Coulomb, 0.0,
                direct.tol_eri, direct.sym_ops);
        };
        return make_dh_eq41_response_operator(inputs.exact_exchange,
                                                std::move(coulomb), std::move(exchange),
                                                inputs.xc);
    }

    std::expected<DHResponseFn, std::string>
    make_dh_eq41_xc_response_callback(const DHEq41XCInputs &inputs)
    {
        if (inputs.molecular_grid == nullptr || inputs.ao_grid == nullptr ||
            inputs.exchange_functional == nullptr || inputs.correlation_functional == nullptr ||
            inputs.ground_density.rows() == 0 ||
            inputs.ground_density.rows() != inputs.ground_density.cols() ||
            inputs.ground_density.rows() != inputs.ao_grid->nbasis())
            return std::unexpected("DH Eq. 41 XC response: invalid fixed-geometry inputs.");
        auto kernel = DFT::Driver::prepare_rks_xc_kernel(
            *inputs.molecular_grid, *inputs.ao_grid, inputs.ground_density,
            *inputs.exchange_functional, *inputs.correlation_functional);
        if (!kernel) return std::unexpected(kernel.error());
        return DHResponseFn{[kernel = std::move(*kernel)](const Eigen::Ref<const Eigen::MatrixXd> &trial_density)
            -> std::expected<Eigen::MatrixXd, std::string>
        {
            return kernel.apply(trial_density);
        }};
    }

    std::expected<Eigen::MatrixXd, std::string>
    DHEq41ResponseOperator::apply(const Eigen::Ref<const Eigen::MatrixXd> &density) const
    {
        auto channels = apply_channels(density);
        if (!channels) return std::unexpected(channels.error());
        return std::move(channels->total);
    }

    std::expected<DHEq41ResponseOperator::Channels, std::string>
    DHEq41ResponseOperator::apply_channels(const Eigen::Ref<const Eigen::MatrixXd> &density) const
    {
        if (density.rows() != density.cols())
            return std::unexpected("DH Eq. 41 response: density must be square.");
        auto j = coulomb(density);
        auto vxc = xc(density);
        if (!j || !vxc)
            return std::unexpected("DH Eq. 41 response: Coulomb or XC channel failed.");
        if (j->rows() != density.rows() || j->cols() != density.cols() ||
            vxc->rows() != density.rows() || vxc->cols() != density.cols())
            return std::unexpected("DH Eq. 41 response: channel dimensions do not match density.");
        Channels out;
        out.coulomb = 4.0 * *j;
        // `density` is the symmetric correction-density test object D' in
        // R(D').  Its adjoint against the physical closed-shell orbital
        // variation delta P=2(Cv X Co^T+Co X^T Cv^T) carries the same factor
        // four as the Coulomb channel.  The HVP itself is linear in D'; the
        // factor belongs to the closed-shell adjoint projection, not libxc.
        out.xc = 4.0 * *vxc;
        out.exchange = Eigen::MatrixXd::Zero(density.rows(), density.cols());
        if (exact_exchange != 0.0)
        {
            auto k = exchange(density);
            if (!k || k->rows() != density.rows() || k->cols() != density.cols())
                return std::unexpected("DH Eq. 41 response: exchange channel failed or has wrong dimensions.");
            // Eq. (41): -a_x [K[D'] + K[D']^T].  For symmetric D', the
            // two contractions are equal but remain explicit here.
            out.exchange.noalias() -= exact_exchange * (*k + k->transpose());
        }
        out.total = out.coulomb + out.exchange + out.xc;
        return out;
    }

    std::expected<DHEq41ResponseDensity, std::string>
    build_dh_eq41_response_density(
        const DHPT2AmplitudeDensity &dprime,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq41ResponseOperator &response)
    {
        const int nmo = dprime.n_occ + dprime.n_virt;
        if (nmo <= 0 || dprime.dprime_mo.rows() != nmo ||
            dprime.dprime_mo.cols() != nmo || mo_coeff.cols() != nmo)
            return std::unexpected("DH Eq. 41 response: inconsistent D' or MO-coefficient dimensions.");
        DHEq41ResponseDensity out;
        out.dprime_ao.noalias() = mo_coeff * dprime.dprime_mo * mo_coeff.transpose();
        auto applied = response.apply(out.dprime_ao);
        if (!applied) return std::unexpected(applied.error());
        out.response_ao = std::move(*applied);
        return out;
    }

    std::expected<DHPT2AmplitudeDensity, std::string>
    build_dh_pt2_amplitude_density(
        const HartreeFock::Correlation::RMP2Result &result,
        double c_pt2, DHTildeAmplitudeStorage storage)
    {
        const int no = result.n_occ;
        const int nv = result.n_virt;
        if (no <= 0 || nv <= 0 || !std::isfinite(c_pt2))
            return std::unexpected("DH PT2 amplitudes: invalid dimensions or PT2 scale.");
        const std::size_t need = static_cast<std::size_t>(no) * no * nv * nv;
        if (!result.has_t2())
            return std::unexpected("DH PT2 amplitudes: stored T2 dimensions are inconsistent.");

        DHPT2AmplitudeDensity out;
        out.n_occ = no;
        out.n_virt = nv;
        out.storage = storage;
        out.source_pt2 = &result;
        out.c_pt2 = c_pt2;
        if (storage == DHTildeAmplitudeStorage::Materialized)
            out.t_tilde.assign(need, 0.0);
        out.dprime_oo = Eigen::MatrixXd::Zero(no, no);
        out.dprime_vv = Eigen::MatrixXd::Zero(nv, nv);

        // Eq. (39). c_PT2 belongs to the derivative correction, so scale the
        // amplitude-linear tilde tensor here and all density products below
        // use one scaled and one unscaled amplitude.
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b)
                        if (storage == DHTildeAmplitudeStorage::Materialized)
                            out.t_tilde[idx(i, j, a, b, no, nv)] = c_pt2 * 2.0 /
                                static_cast<double>(1 + (i == j)) *
                                (2.0 * result.t2_at(i,j,a,b) - result.t2_at(i,j,b,a));

        // Eq. (37): D'_ij = -sum_k (1+delta_ik) tr(t~^(ik) t^(kj)).
        // The two virtual panels have different composite-pair orientation:
        // tilde is [a,b] at (i,k), while the unscaled partner is [b,a] at
        // (k,j).  Read both bounded panels explicitly; this is the natural
        // recoverable-source access pattern, rather than scalar t2_at calls.
        out.dprime_oo.setZero();
        for (int i=0;i<no;++i) for (int j=0;j<no;++j) for (int k=0;k<no;++k)
        {
            auto left=out.read_pair_tile(i*nv,k*nv,nv,nv);
            auto right=HartreeFock::Correlation::read_rmp2_t2_pair_tile(
                result,k*nv,j*nv,nv,nv);
            if (!left) return std::unexpected(left.error());
            if (!right) return std::unexpected(right.error());
            for (int a=0;a<nv;++a) for (int b=0;b<nv;++b)
                out.dprime_oo(i,j) -= (1+(i==k))*
                    left->values[static_cast<std::size_t>(a)*nv+b]*
                    right->values[static_cast<std::size_t>(b)*nv+a];
        }

        // Eq. (38): D'_ab = sum_{i<=j}(t~_ij t_ij^T + t~_ij^T t_ij)_ab.
        // Both operands are the (i,j) occupied-pair panel, so this keeps a
        // single bounded tilde tile and raw T2 tile live per occupied pair.
        for (int i = 0; i < no; ++i)
            for (int j = i; j < no; ++j)
            {
                auto left=out.read_pair_tile(i*nv,j*nv,nv,nv);
                auto right=HartreeFock::Correlation::read_rmp2_t2_pair_tile(
                    result,i*nv,j*nv,nv,nv);
                if (!left) return std::unexpected(left.error());
                if (!right) return std::unexpected(right.error());
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b)
                        for (int k = 0; k < nv; ++k)
                        {
                            out.dprime_vv(a,b) += left->values[static_cast<std::size_t>(a)*nv+k]*
                                right->values[static_cast<std::size_t>(b)*nv+k];
                            out.dprime_vv(a,b) += left->values[static_cast<std::size_t>(k)*nv+a]*
                                right->values[static_cast<std::size_t>(k)*nv+b];
                        }
            }

        out.dprime_mo = Eigen::MatrixXd::Zero(no + nv, no + nv);
        out.dprime_mo.topLeftCorner(no, no) = out.dprime_oo;
        out.dprime_mo.bottomRightCorner(nv, nv) = out.dprime_vv;
        return out;
    }

    std::expected<DHMolecularERIBlocks, std::string>
    build_dh_molecular_eri_blocks(const std::vector<double> &mo_eri, int no, int nv)
    {
        const int n=no+nv;
        const std::size_t n4=static_cast<std::size_t>(n)*n*n*n;
        if (no<=0 || nv<=0 || mo_eri.size()!=n4)
            return std::unexpected("DH MO-ERI blocks: invalid occupied/virtual dimensions or dense tensor extent.");
        DHMolecularERIBlocks out;
        out.n_occ=no; out.n_virt=nv;
        out.ovov.resize(static_cast<std::size_t>(no)*nv*no*nv);
        out.vvov.resize(static_cast<std::size_t>(nv)*nv*no*nv);
        out.ooov.resize(static_cast<std::size_t>(no)*no*no*nv);
        out.ovvv.resize(static_cast<std::size_t>(no)*nv*nv*nv);
        out.ovoo.resize(static_cast<std::size_t>(no)*nv*no*no);
        for(int i=0;i<no;++i) for(int a=0;a<nv;++a) for(int j=0;j<no;++j) for(int b=0;b<nv;++b)
            out.ovov[((static_cast<std::size_t>(i)*nv+a)*no+j)*nv+b]=mo_eri[eri_idx(i,no+a,j,no+b,n)];
        for(int a=0;a<nv;++a) for(int b=0;b<nv;++b) for(int i=0;i<no;++i) for(int c=0;c<nv;++c)
            out.vvov[((static_cast<std::size_t>(a)*nv+b)*no+i)*nv+c]=mo_eri[eri_idx(no+a,no+b,i,no+c,n)];
        for(int i=0;i<no;++i) for(int j=0;j<no;++j) for(int k=0;k<no;++k) for(int a=0;a<nv;++a)
            out.ooov[((static_cast<std::size_t>(i)*no+j)*no+k)*nv+a]=mo_eri[eri_idx(i,j,k,no+a,n)];
        for(int i=0;i<no;++i) for(int a=0;a<nv;++a) for(int b=0;b<nv;++b) for(int c=0;c<nv;++c)
            out.ovvv[((static_cast<std::size_t>(i)*nv+a)*nv+b)*nv+c]=mo_eri[eri_idx(i,no+a,no+b,no+c,n)];
        for(int i=0;i<no;++i) for(int a=0;a<nv;++a) for(int j=0;j<no;++j) for(int k=0;k<no;++k)
            out.ovoo[((static_cast<std::size_t>(i)*nv+a)*no+j)*no+k]=mo_eri[eri_idx(i,no+a,j,k,n)];
        return out;
    }

    std::expected<DHPairCoefficientGradient, std::string>
    build_dh_pair_coefficient_gradient(const DHPT2AmplitudeDensity &amplitudes,
        const std::vector<double> &mo_eri, int reduction_tile)
    {
        const int no=amplitudes.n_occ, nv=amplitudes.n_virt, n=no+nv;
        if (no<=0 || nv<=0 || reduction_tile<=0 ||
            !amplitudes.has_tilde_storage() ||
            mo_eri.size()!=static_cast<std::size_t>(n)*n*n*n)
            return std::unexpected("DH pair coefficient gradient: invalid dimensions or reduction tile.");
        const std::array<int,4> dimensions{no,nv,no,nv};
        DHPairCoefficientGradient out;
        out.total=Eigen::MatrixXd::Zero(n,n);
        for (int slot=0; slot<4; ++slot)
        {
            // Substitute the differentiated index q before reduction. The
            // three remaining indices r enumerate the other coefficient
            // slots, not all ijab for each orbital direction.
            std::size_t count=1;
            for (int s=0; s<4; ++s) if (s!=slot) count*=dimensions[s];
            const auto unpack=[&](std::size_t r, int q) {
                std::array<int,4> x{};
                for (int s=3; s>=0; --s)
                    if (s!=slot) { x[s]=r%dimensions[s]; r/=dimensions[s]; }
                x[slot]=q;
                return x;
            };
            out.slots[slot]=tiled_product(n,dimensions[slot],count,reduction_tile,
                [&](int p, std::size_t r) {
                    auto x=unpack(r,0);
                    x[1]+=no; x[3]+=no; x[slot]=p;
                    return mo_eri[eri_idx(x[0],x[1],x[2],x[3],n)];
                },
                [&](std::size_t r, int q) {
                    const auto x=unpack(r,q);
                    return (1+(x[0]==x[2]))*amplitudes.tilde(x[0],x[2],x[1],x[3]);
                });
            if (slot%2==0) out.total.leftCols(no)+=out.slots[slot];
            else out.total.rightCols(nv)+=out.slots[slot];
        }
        if (!out.total.allFinite())
            return std::unexpected("DH pair coefficient gradient: nonfinite contraction.");
        return out;
    }

    std::expected<DHPairCoefficientGradient, std::string>
    build_dh_pair_coefficient_gradient(const DHPT2AmplitudeDensity &amplitudes,
        const DHMolecularERIBlocks &b, int reduction_tile)
    {
        const int no=amplitudes.n_occ, nv=amplitudes.n_virt, n=no+nv;
        const std::size_t novov=static_cast<std::size_t>(no)*nv*no*nv;
        if (no<=0 || nv<=0 || reduction_tile<=0 || b.n_occ!=no || b.n_virt!=nv ||
            !amplitudes.has_tilde_storage() || b.ovov.size()!=novov ||
            b.vvov.size()!=static_cast<std::size_t>(nv)*nv*no*nv ||
            b.ooov.size()!=static_cast<std::size_t>(no)*no*no*nv ||
            b.ovvv.size()!=static_cast<std::size_t>(no)*nv*nv*nv ||
            b.ovoo.size()!=static_cast<std::size_t>(no)*nv*no*no)
            return std::unexpected("DH pair coefficient gradient: invalid typed MO-ERI sectors.");
        const auto ovov=[&](int i,int a,int j,int c)
        { return b.ovov[((static_cast<std::size_t>(i)*nv+a)*no+j)*nv+c]; };
        const auto vvov=[&](int a,int q,int j,int c)
        { return b.vvov[((static_cast<std::size_t>(a)*nv+q)*no+j)*nv+c]; };
        const auto ooov=[&](int i,int q,int j,int c)
        { return b.ooov[((static_cast<std::size_t>(i)*no+q)*no+j)*nv+c]; };
        const auto ovvv=[&](int i,int a,int q,int c)
        { return b.ovvv[((static_cast<std::size_t>(i)*nv+a)*nv+q)*nv+c]; };
        const auto ovoo=[&](int i,int a,int j,int q)
        { return b.ovoo[((static_cast<std::size_t>(i)*nv+a)*no+j)*no+q]; };
        DHPairCoefficientGradient out;
        out.total=Eigen::MatrixXd::Zero(n,n);
        out.slots[0]=Eigen::MatrixXd::Zero(n,no);
        out.slots[1]=Eigen::MatrixXd::Zero(n,nv);
        out.slots[2]=Eigen::MatrixXd::Zero(n,no);
        out.slots[3]=Eigen::MatrixXd::Zero(n,nv);
        // This is the literal four-coefficient derivative of Eq. (47).  The
        // typed production route consumes each bounded tilde pair tile once,
        // then scatters its four coefficient variations directly into their
        // named slots.  No scalar amplitude accessor or materialized tilde
        // tensor is part of the contraction.
        const auto streamed=amplitudes.for_each_pair_tile(128,[&](const auto &tile)
        {
            for (int ip=0;ip<tile.rows;++ip)
                for (int iq=0;iq<tile.cols;++iq)
                {
                    const int p=tile.p0+ip, q=tile.q0+iq;
                    const int i=p/nv, a=p%nv, j=q/nv, b=q%nv;
                    const double theta=(1+(i==j))*
                        tile.values[static_cast<std::size_t>(ip)*tile.cols+iq];
                    for (int r=0;r<n;++r)
                    {
                        out.slots[0](r,i) += theta*(r<no ? ovov(r,a,j,b) : vvov(r-no,a,j,b));
                        out.slots[1](r,a) += theta*(r<no ? ooov(i,r,j,b) : ovov(i,r-no,j,b));
                        out.slots[2](r,j) += theta*(r<no ? ovov(i,a,r,b) : ovvv(i,a,r-no,b));
                        out.slots[3](r,b) += theta*(r<no ? ovoo(i,a,j,r) : ovov(i,a,j,r-no));
                    }
                }
            return true;
        });
        if(!streamed) return std::unexpected(streamed.error());
        out.total.leftCols(no)=out.slots[0]+out.slots[2];
        out.total.rightCols(nv)=out.slots[1]+out.slots[3];
        if(!out.total.allFinite()) return std::unexpected("DH pair coefficient gradient: nonfinite typed contraction.");
        return out;
    }

    std::expected<DHEq40AmplitudeRHS, std::string>
    build_dh_eq40_amplitude_rhs(
        const HartreeFock::Correlation::RMP2Result &result,
        const DHPT2AmplitudeDensity &amplitudes,
        const std::vector<double> &mo_eri,
        double c_pt2,
        bool include_legacy_internal)
    {
        const int no = result.n_occ;
        const int nv = result.n_virt;
        const int nmo = no + nv;
        const std::size_t nt = static_cast<std::size_t>(no) * no * nv * nv;
        const std::size_t ne = static_cast<std::size_t>(nmo) * nmo * nmo * nmo;
        if (no <= 0 || nv <= 0 || !std::isfinite(c_pt2) || !result.has_t2() ||
            amplitudes.n_occ != no || amplitudes.n_virt != nv ||
            !amplitudes.has_tilde_storage() || mo_eri.size() != ne)
            return std::unexpected("DH Eq. 40 RHS: inconsistent amplitudes or MO ERI dimensions.");

        DHEq40AmplitudeRHS out;
        auto pair = build_dh_pair_coefficient_gradient(amplitudes, mo_eri);
        if (!pair) return std::unexpected(pair.error());
        out.pair_coefficients = std::move(*pair);
        out.three_external = out.pair_coefficients.total.bottomLeftCorner(nv,no) -
            out.pair_coefficients.total.topRightCorner(no,nv).transpose();
        out.internal_same_spin_exchange = Eigen::MatrixXd::Zero(nv, no);
        out.internal_opposite_spin_direct = Eigen::MatrixXd::Zero(nv, no);
        out.three_internal = Eigen::MatrixXd::Zero(nv, no);
        for (int a = 0; a < nv; ++a)
            for (int i = 0; i < no; ++i)
            {
                // The excluded internal bracket is available only for
                // standalone reference tests, never evaluated by production.
                if (!include_legacy_internal) continue;
                for (int k = 0; k < no; ++k)
                    for (int l = 0; l < no; ++l)
                        for (int b = 0; b < nv; ++b)
                        {
                            const int bp = no + b;
                            const double scaled_t = c_pt2 * result.t2_at(k,l,a,b);
                            // Eq. (22) -> Eq. (40), retained literally:
                            // same-spin exchange and opposite-spin direct.
                            out.internal_same_spin_exchange(a, i) -= scaled_t *
                                mo_eri[eri_idx(k, i, l, bp, nmo)];
                            out.internal_opposite_spin_direct(a, i) += scaled_t *
                                mo_eri[eri_idx(k, bp, l, i, nmo)];
                        }
            }
        out.three_internal = out.internal_same_spin_exchange + out.internal_opposite_spin_direct;
        out.total = out.three_external + out.three_internal;
        return out;
    }

    std::expected<DHEq40AmplitudeRHS, std::string>
    build_dh_eq40_amplitude_rhs(
        const HartreeFock::Correlation::RMP2Result &result,
        const DHPT2AmplitudeDensity &amplitudes,
        const DHMolecularERIBlocks &eri_blocks,
        double c_pt2,
        bool include_legacy_internal)
    {
        const int no=result.n_occ, nv=result.n_virt;
        const std::size_t nt=static_cast<std::size_t>(no)*no*nv*nv;
        if (no<=0 || nv<=0 || !std::isfinite(c_pt2) || !result.has_t2() ||
            amplitudes.n_occ!=no || amplitudes.n_virt!=nv || include_legacy_internal)
            return std::unexpected("DH Eq. 40 typed RHS: invalid inputs or unsupported legacy-internal bracket.");
        auto pair=build_dh_pair_coefficient_gradient(amplitudes,eri_blocks);
        if(!pair) return std::unexpected(pair.error());
        DHEq40AmplitudeRHS out;
        out.pair_coefficients=std::move(*pair);
        out.three_external=out.pair_coefficients.total.bottomLeftCorner(nv,no)-
            out.pair_coefficients.total.topRightCorner(no,nv).transpose();
        out.internal_same_spin_exchange=Eigen::MatrixXd::Zero(nv,no);
        out.internal_opposite_spin_direct=Eigen::MatrixXd::Zero(nv,no);
        out.three_internal=Eigen::MatrixXd::Zero(nv,no);
        out.total=out.three_external;
        return out;
    }

    std::expected<DHLagrangianRHS, std::string>
    build_dh_lagrangian_rhs(
        const DHEq41ResponseDensity &eq41,
        const Eigen::Ref<const Eigen::MatrixXd> &c_occ,
        const Eigen::Ref<const Eigen::MatrixXd> &c_virt,
        const DHEq40AmplitudeRHS &amplitude,
        DHRHSConvention convention)
    {
        const Eigen::Index nao = c_occ.rows();
        const Eigen::Index no = c_occ.cols();
        const Eigen::Index nv = c_virt.cols();
        if (nao == 0 || c_virt.rows() != nao || eq41.response_ao.rows() != nao ||
            eq41.response_ao.cols() != nao || amplitude.three_external.rows() != nv ||
            amplitude.three_external.cols() != no || amplitude.three_internal.rows() != nv ||
            amplitude.three_internal.cols() != no || amplitude.total.rows() != nv ||
            amplitude.total.cols() != no)
            return std::unexpected("DH Eq. 40 RHS: incompatible AO response, orbitals, or amplitude blocks.");

        DHLagrangianRHS out;
        out.response_ai.noalias() = c_virt.transpose() * eq41.response_ao * c_occ;
        out.three_external_ai = amplitude.three_external;
        out.internal_same_spin_exchange_ai = amplitude.internal_same_spin_exchange;
        out.internal_opposite_spin_direct_ai = amplitude.internal_opposite_spin_direct;
        out.three_internal_ai = amplitude.three_internal;
        out.included_internal_ai = amplitude.three_internal;
        if (convention == DHRHSConvention::LiteralEq47)
            out.included_internal_ai.setZero();
        out.amplitude_ai = out.three_external_ai + out.included_internal_ai;
        out.total_ai = out.response_ai + out.amplitude_ai;
        return out;
    }

    std::expected<DHRelaxedDifferenceDensity, std::string>
    build_dh_relaxed_difference_density(
        const DHPT2AmplitudeDensity &dprime,
        const Eigen::Ref<const Eigen::MatrixXd> &z_ai)
    {
        const int no = dprime.n_occ;
        const int nv = dprime.n_virt;
        const int nmo = no + nv;
        if (no <= 0 || nv <= 0 || dprime.dprime_mo.rows() != nmo ||
            dprime.dprime_mo.cols() != nmo || z_ai.rows() != nv || z_ai.cols() != no)
            return std::unexpected("DH Eq. 28 relaxed density: incompatible D' or Z_ai dimensions.");

        DHRelaxedDifferenceDensity out;
        out.z_ai = z_ai;
        out.raw_mo = dprime.dprime_mo;
        out.raw_mo.bottomLeftCorner(nv, no) = z_ai;

        // For a symmetric M, <Z_vo,M> = <1/2 Z_vo + 1/2 Z_ov,M>.
        // Keep this adapter separate: the raw Eq. (28) matrix is not symmetric.
        out.symmetric_mo = dprime.dprime_mo;
        out.symmetric_mo.bottomLeftCorner(nv, no) = 0.5 * z_ai;
        out.symmetric_mo.topRightCorner(no, nv) = 0.5 * z_ai.transpose();
        return out;
    }

    std::expected<DHEq27HessianAction, std::string>
    apply_dh_eq27_hessian(
        const Eigen::Ref<const Eigen::MatrixXd> &z_ai,
        const Eigen::Ref<const Eigen::MatrixXd> &c_occ,
        const Eigen::Ref<const Eigen::MatrixXd> &c_virt,
        const Eigen::Ref<const Eigen::VectorXd> &orbital_energies,
        const DHEq41ResponseOperator &response)
    {
        const Eigen::Index nao = c_occ.rows();
        const Eigen::Index no = c_occ.cols();
        const Eigen::Index nv = c_virt.cols();
        if (nao == 0 || c_virt.rows() != nao || z_ai.rows() != nv || z_ai.cols() != no ||
            orbital_energies.size() != no + nv || !z_ai.allFinite() || !c_occ.allFinite() ||
            !c_virt.allFinite() || !orbital_energies.allFinite() || !response.coulomb || !response.xc ||
            !std::isfinite(response.exact_exchange) || response.exact_exchange<0 ||
            (response.exact_exchange!=0 && !response.exchange))
            return std::unexpected("DH Eq. 27 Hessian: incompatible Z_ai, orbitals, or orbital energies.");

        DHEq27HessianAction out;
        out.raw_z_ao.noalias() = c_virt * z_ai * c_occ.transpose();
        // Eq. (27) is an orbital Hessian, not the Eq. (41) D' response.
        // A non-metric orbital rotation z_ai changes the *closed-shell
        // total density* by delta P = 2(Cv z Co^T + Co z^T Cv^T).  Feeding
        // the raw Cv z Co^T matrix into R(D') misses this symmetrization and
        // its factor of two, most visibly in the XC Hessian channel.
        const Eigen::MatrixXd delta_density = 2.0 * (out.raw_z_ao + out.raw_z_ao.transpose());
        auto coulomb = response.coulomb(delta_density);
        if (!coulomb) return std::unexpected("DH Eq. 27 Coulomb: "+coulomb.error());
        auto xc = response.xc(delta_density);
        if (!xc) return std::unexpected("DH Eq. 27 XC: "+xc.error());
        if (coulomb->rows() != nao || coulomb->cols() != nao || xc->rows() != nao || xc->cols() != nao)
            return std::unexpected("DH Eq. 27 Hessian: Coulomb or XC orbital-density response has wrong dimensions.");
        if (!coulomb->allFinite() || !xc->allFinite())
            return std::unexpected("DH Eq. 27 Hessian: nonfinite Coulomb or XC response.");
        const auto transform_ai = [&c_occ, &c_virt](const Eigen::MatrixXd &ao)
        { return c_virt.transpose() * ao * c_occ; };
        out.coulomb_ai = transform_ai(*coulomb);
        out.xc_ai = transform_ai(*xc);
        out.exchange_ai = Eigen::MatrixXd::Zero(nv, no);
        if (response.exact_exchange != 0.0)
        {
            auto exchange = response.exchange(delta_density);
            if (!exchange) return std::unexpected("DH Eq. 27 exchange: "+exchange.error());
            if (exchange->rows() != nao || exchange->cols() != nao || !exchange->allFinite())
                return std::unexpected("DH Eq. 27 Hessian: exchange orbital-density response failed.");
            out.exchange_ai.noalias() = -0.5 * response.exact_exchange * transform_ai(*exchange);
        }
        out.response_ai = out.coulomb_ai + out.exchange_ai + out.xc_ai;
        out.orbital_energy_ai = Eigen::MatrixXd::Zero(nv, no);
        for (Eigen::Index a = 0; a < nv; ++a)
            for (Eigen::Index i = 0; i < no; ++i)
                out.orbital_energy_ai(a, i) =
                    (orbital_energies(no + a) - orbital_energies(i)) * z_ai(a, i);
        out.total_ai = out.orbital_energy_ai + out.response_ai;
        if (!out.total_ai.allFinite()) return std::unexpected("DH Eq. 27 Hessian: nonfinite total action.");
        return out;
    }

    std::expected<DHEq42_43EnergyWeightedDensity, std::string>
    build_dh_eq42_43_energy_weighted_density(
        const DHRelaxedDifferenceDensity &relaxed,
        const HartreeFock::Correlation::RMP2Result &result,
        const DHPT2AmplitudeDensity &amplitudes,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const Eigen::Ref<const Eigen::VectorXd> &orbital_energies,
        const std::vector<double> &mo_eri,
        const DHEq41ResponseOperator &response,
        bool include_legacy_pair)
    {
        const int no = result.n_occ;
        const int nv = result.n_virt;
        const int nmo = no + nv;
        const std::size_t nt = static_cast<std::size_t>(no) * no * nv * nv;
        const std::size_t ne = static_cast<std::size_t>(nmo) * nmo * nmo * nmo;
        if (no <= 0 || nv <= 0 || !result.has_t2() || amplitudes.n_occ != no ||
            amplitudes.n_virt != nv || !amplitudes.has_tilde_storage() ||
            relaxed.raw_mo.rows() != nmo || relaxed.raw_mo.cols() != nmo ||
            mo_coeff.cols() != nmo || orbital_energies.size() != nmo ||
            (include_legacy_pair && mo_eri.size() != ne))
            return std::unexpected("DH Eq. 42-43 density: incompatible densities, orbitals, amplitudes, or MO ERIs.");

        // After Eq. (28), the paper explicitly symmetrizes D before forming
        // the energy-weighted density.  R(D)_ij in Eq. (42) is therefore the
        // response to D' + 1/2(Z_vo + Z_ov), not to raw D' + Z_vo.
        const Eigen::MatrixXd relaxed_ao =
            mo_coeff * relaxed.symmetric_mo * mo_coeff.transpose();
        auto response_ao = response.apply(relaxed_ao);
        if (!response_ao) return std::unexpected(response_ao.error());
        const Eigen::MatrixXd response_mo = mo_coeff.transpose() * *response_ao * mo_coeff;

        DHEq42_43EnergyWeightedDensity out;
        out.response_oo = -0.5 * response_mo.topLeftCorner(no, no);
        out.orbital_oo = Eigen::MatrixXd::Zero(no, no);
        out.amplitude_oo = Eigen::MatrixXd::Zero(no, no);
        out.orbital_vv = Eigen::MatrixXd::Zero(nv, nv);
        out.amplitude_vv = Eigen::MatrixXd::Zero(nv, nv);
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
            {
                out.orbital_oo(i,j) = -0.5 * relaxed.raw_mo(i,j) *
                    (orbital_energies(i) + orbital_energies(j));
                if (!include_legacy_pair) continue;
                for (int k = 0; k < no; ++k)
                    for (int a = 0; a < nv; ++a)
                        for (int b = 0; b < nv; ++b)
                            out.amplitude_oo(i,j) -= 0.5 *
                                amplitudes.tilde(j,k,a,b) *
                                mo_eri[eri_idx(i,no+a,k,no+b,nmo)];
            }
        for (int a = 0; a < nv; ++a)
            for (int b = 0; b < nv; ++b)
            {
                out.orbital_vv(a,b) = -0.5 * relaxed.raw_mo(no+a,no+b) *
                    (orbital_energies(no+a) + orbital_energies(no+b));
                if (!include_legacy_pair) continue;
                for (int i = 0; i < no; ++i)
                    for (int j = i; j < no; ++j)
                        for (int c = 0; c < nv; ++c)
                            out.amplitude_vv(a,b) -=
                                mo_eri[eri_idx(i,no+a,j,no+c,nmo)] *
                                amplitudes.tilde(i,j,b,c) /
                                static_cast<double>(1 + (i == j));
            }
        out.w_oo = out.response_oo + out.orbital_oo + out.amplitude_oo;
        out.w_vv = out.orbital_vv + out.amplitude_vv;
        return out;
    }

    std::expected<DHEq44_45EnergyWeightedDensity, std::string>
    build_dh_eq44_45_energy_weighted_density(
        const DHRelaxedDifferenceDensity &relaxed,
        const HartreeFock::Correlation::RMP2Result &result,
        const DHPT2AmplitudeDensity &amplitudes,
        const Eigen::Ref<const Eigen::VectorXd> &orbital_energies,
        const std::vector<double> &mo_eri,
        bool include_legacy_pair)
    {
        const int no = result.n_occ;
        const int nv = result.n_virt;
        const int nmo = no + nv;
        const std::size_t nt = static_cast<std::size_t>(no) * no * nv * nv;
        const std::size_t ne = static_cast<std::size_t>(nmo) * nmo * nmo * nmo;
        if (no <= 0 || nv <= 0 || !result.has_t2() || amplitudes.n_occ != no ||
            amplitudes.n_virt != nv || !amplitudes.has_tilde_storage() ||
            relaxed.z_ai.rows() != nv || relaxed.z_ai.cols() != no ||
            orbital_energies.size() != nmo ||
            (include_legacy_pair && mo_eri.size() != ne))
            return std::unexpected("DH Eq. 44-45 density: incompatible Z_ai, amplitudes, orbital energies, or MO ERIs.");

        DHEq44_45EnergyWeightedDensity out;
        out.w_ia = Eigen::MatrixXd::Zero(no, nv);
        out.w_ai = Eigen::MatrixXd::Zero(nv, no);
        for (int i = 0; i < no; ++i)
            for (int a = 0; a < nv; ++a)
            {
                out.w_ai(a,i) = -orbital_energies(i) * relaxed.z_ai(a,i);
                if (!include_legacy_pair) continue;
                for (int k = 0; k < no; ++k)
                    for (int j = 0; j < no; ++j)
                        for (int b = 0; b < nv; ++b)
                            out.w_ia(i,a) -= amplitudes.tilde(k,j,a,b) *
                                mo_eri[eri_idx(k,i,j,no+b,nmo)];
            }
        return out;
    }

    std::expected<DHEq47PairMetricOverlapDensity, std::string>
    build_dh_eq47_pair_metric_overlap_density(
        const HartreeFock::Correlation::RMP2Result &result,
        const DHPT2AmplitudeDensity &amplitudes,
        const std::vector<double> &mo_eri)
    {
        const int no = result.n_occ;
        const int nv = result.n_virt;
        const int nmo = no + nv;
        const std::size_t nt = static_cast<std::size_t>(no) * no * nv * nv;
        const std::size_t ne = static_cast<std::size_t>(nmo) * nmo * nmo * nmo;
        if (no <= 0 || nv <= 0 || amplitudes.n_occ != no || amplitudes.n_virt != nv ||
            !amplitudes.has_tilde_storage() || mo_eri.size() != ne)
            return std::unexpected("DH Eq. 47 pair metric overlap: incompatible amplitudes or MO ERIs.");

        auto coefficients = build_dh_pair_coefficient_gradient(amplitudes, mo_eri);
        if (!coefficients) return std::unexpected(coefficients.error());
        return build_dh_eq47_pair_metric_overlap_density(*coefficients, no, nv);
    }

    std::expected<DHEq47PairMetricOverlapDensity, std::string>
    build_dh_eq47_pair_metric_overlap_density(
        const DHPairCoefficientGradient &coefficients, int no, int nv)
    {
        const int n=no+nv;
        const auto &g=coefficients.total;
        if (no<=0 || nv<=0 || g.rows()!=n || g.cols()!=n || !g.allFinite())
            return std::unexpected("DH Eq. 47 pair metric overlap: invalid coefficient gradient.");
        // dE=G:U; insert U_oo=U_vv=-S/2, U_ov=-S_ov, U_vo=0.
        // Do not symmetrize G or infer any equality between coefficient slots.
        DHEq47PairMetricOverlapDensity out;
        out.w_oo=-0.5*g.topLeftCorner(no,no);
        out.w_vv=-0.5*g.bottomRightCorner(nv,nv);
        out.w_ia=-g.topRightCorner(no,nv);
        out.w_ai=Eigen::MatrixXd::Zero(nv,no);
        return out;
    }

    std::expected<DHOverlapDensity, std::string>
    build_dh_overlap_density_adapter(
        const DHEq42_43EnergyWeightedDensity &diagonal,
        const DHEq44_45EnergyWeightedDensity &off_diagonal)
    {
        const Eigen::Index no = diagonal.w_oo.rows();
        const Eigen::Index nv = diagonal.w_vv.rows();
        if (no <= 0 || nv <= 0 || diagonal.w_oo.cols() != no || diagonal.w_vv.cols() != nv ||
            off_diagonal.w_ia.rows() != no || off_diagonal.w_ia.cols() != nv ||
            off_diagonal.w_ai.rows() != nv || off_diagonal.w_ai.cols() != no)
            return std::unexpected("DH overlap-density adapter: incompatible Eq. 42-45 block dimensions.");

        DHOverlapDensity out;
        out.raw_mo = Eigen::MatrixXd::Zero(no + nv, no + nv);
        out.raw_mo.topLeftCorner(no, no) = diagonal.w_oo;
        out.raw_mo.topRightCorner(no, nv) = off_diagonal.w_ia;
        out.raw_mo.bottomLeftCorner(nv, no) = off_diagonal.w_ai;
        out.raw_mo.bottomRightCorner(nv, nv) = diagonal.w_vv;
        out.symmetric_mo = 0.5 * (out.raw_mo + out.raw_mo.transpose());
        return out;
    }

    namespace
    {
        std::expected<std::vector<double>, std::string>
        dh_eq47_penultimate(const DHPT2AmplitudeDensity &amplitudes,
            const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff, int column_tile)
        {
            const int no=amplitudes.n_occ, nv=amplitudes.n_virt;
            const Eigen::Index nao=mo_coeff.rows();
            if(no<=0 || nv<=0 || nao<=0 || column_tile<=0 ||
                mo_coeff.cols()!=static_cast<Eigen::Index>(no)+nv || !mo_coeff.allFinite())
                return std::unexpected("DH Eq. 47 backtransform: invalid dimensions, coefficients, or column tile.");
            std::size_t capacity=1;
            const auto bound=static_cast<std::size_t>(std::max({nao,static_cast<Eigen::Index>(no),static_cast<Eigen::Index>(nv)}));
            for(int k=0;k<4;++k)
            {
                if(capacity>static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max())/sizeof(double)/bound)
                    return std::unexpected("DH Eq. 47 backtransform: tensor extent overflow.");
                capacity*=bound;
            }
            const std::size_t nt=static_cast<std::size_t>(no)*no*nv*nv;
            if(!amplitudes.has_tilde_storage())
                return std::unexpected("DH Eq. 47 backtransform: invalid tilde amplitudes.");
            std::vector<double> tensor(nt);
            // Eq. (47)'s first transform consumes the bounded source
            // contract. A recoverable backend changes only this traversal.
            const auto streamed=amplitudes.for_each_pair_tile(128,[&](const auto &tile)
            {
                for(int ip=0;ip<tile.rows;++ip) for(int iq=0;iq<tile.cols;++iq)
                {
                    const int p=tile.p0+ip, q=tile.q0+iq;
                    const int i=p/nv, a=p%nv, j=q/nv, b=q%nv;
                    tensor[((static_cast<std::size_t>(i)*nv+a)*no+j)*nv+b]=
                        (1+(i==j))*tile.values[static_cast<std::size_t>(ip)*tile.cols+iq];
                }
                return true;
            });
            if(!streamed) return std::unexpected(streamed.error());
            std::array<std::size_t,4> shape{static_cast<std::size_t>(no),static_cast<std::size_t>(nv),
                                         static_cast<std::size_t>(no),static_cast<std::size_t>(nv)};
            tensor=transform_pair_axis(tensor,shape,3,mo_coeff.rightCols(nv),column_tile); // i,a,j,tau
            tensor=transform_pair_axis(tensor,shape,2,mo_coeff.leftCols(no),column_tile);  // i,a,kappa,tau
            tensor=transform_pair_axis(tensor,shape,1,mo_coeff.rightCols(nv),column_tile); // i,nu,kappa,tau
            return tensor;
        }
    }

    std::expected<void, std::string>
    for_each_dh_eq47_nonseparable_raw_ao_tile(
        const DHPT2AmplitudeDensity &amplitudes,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        int transform_tile, int mu_tile,
        const DHEq47RawAOTileConsumer &consumer)
    {
        const int no=amplitudes.n_occ;
        const Eigen::Index nao=mo_coeff.rows();
        if(mu_tile<=0 || !consumer)
            return std::unexpected("DH Eq. 47 tiled backtransform: invalid mu tile or consumer.");
        auto penultimate=dh_eq47_penultimate(amplitudes,mo_coeff,transform_tile);
        if(!penultimate) return std::unexpected(penultimate.error());
        const std::size_t suffix=static_cast<std::size_t>(nao)*nao*nao;
        if(penultimate->size()!=static_cast<std::size_t>(no)*suffix)
            return std::unexpected("DH Eq. 47 tiled backtransform: invalid penultimate tensor extent.");
        using RowMatrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
        const Eigen::Map<const RowMatrix> source(penultimate->data(),no,static_cast<Eigen::Index>(suffix));
        for(Eigen::Index begin=0;begin<nao;begin+=mu_tile)
        {
            const Eigen::Index width=std::min<Eigen::Index>(mu_tile,nao-begin);
            std::vector<double> tile(static_cast<std::size_t>(width)*suffix);
            Eigen::Map<RowMatrix> target(tile.data(),width,static_cast<Eigen::Index>(suffix));
            target.noalias()=mo_coeff.middleRows(begin,width).leftCols(no)*source;
            if(!std::all_of(tile.begin(),tile.end(),[](double x){return std::isfinite(x);}))
                return std::unexpected("DH Eq. 47 tiled backtransform: nonfinite tile.");
            auto consumed=consumer(static_cast<int>(begin),static_cast<int>(width),tile);
            if(!consumed) return std::unexpected(consumed.error());
        }
        return {};
    }

    std::expected<void, std::string>
    for_each_dh_eq47_nonseparable_raw_ao_block(
        const DHPT2AmplitudeDensity &amplitudes,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        int occupied_tile, int mu_tile, int nu_tile, int kappa_tile, int tau_tile,
        const DHEq47RawAOBlockConsumer &consumer,
        DHEq47BlockWorkspace *workspace)
    {
        const int no=amplitudes.n_occ, nv=amplitudes.n_virt;
        const int nao=static_cast<int>(mo_coeff.rows());
        const std::size_t nt=static_cast<std::size_t>(no)*no*nv*nv;
        if (no<=0 || nv<=0 || nao<=0 || mo_coeff.cols()!=no+nv || !mo_coeff.allFinite() ||
            occupied_tile<=0 || mu_tile<=0 || nu_tile<=0 || kappa_tile<=0 || tau_tile<=0 || !consumer ||
            !amplitudes.has_tilde_storage())
            return std::unexpected("DH Eq. 47 blocked backtransform: invalid dimensions, tiles, coefficients, or amplitudes.");
        if (workspace) *workspace = {};
        const auto record = [workspace](std::size_t live, std::size_t slab, std::size_t emitted)
        {
            if (!workspace) return;
            workspace->peak_live_elements=std::max(workspace->peak_live_elements,live);
            workspace->peak_output_slab_elements=std::max(workspace->peak_output_slab_elements,slab);
            workspace->peak_emitted_tile_elements=std::max(workspace->peak_emitted_tile_elements,emitted);
        };
        using RowMatrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
        for (int tau0=0; tau0<nao; tau0+=tau_tile)
        {
            const int ntile=std::min(tau_tile,nao-tau0);
            for (int kappa0=0; kappa0<nao; kappa0+=kappa_tile)
            {
                const int ktile=std::min(kappa_tile,nao-kappa0);
                for (int nu0=0; nu0<nao; nu0+=nu_tile)
                {
                    const int vtile=std::min(nu_tile,nao-nu0);
                    const std::size_t suffix=static_cast<std::size_t>(vtile)*ktile*ntile;
                    std::vector<double> output(static_cast<std::size_t>(nao)*suffix,0.0);
                    record(output.size(),output.size(),0);
                    Eigen::Map<RowMatrix> result(output.data(),nao,static_cast<Eigen::Index>(suffix));
                    for (int i0=0; i0<no; i0+=occupied_tile)
                    {
                        const int itile=std::min(occupied_tile,no-i0);
                        std::vector<double> tensor(static_cast<std::size_t>(itile)*nv*no*nv);
                        record(output.size()+tensor.size(),output.size(),0);
                        for (int il=0; il<itile; ++il) for (int a=0; a<nv; ++a)
                            for (int j=0; j<no; ++j) for (int b=0; b<nv; ++b)
                                tensor[((static_cast<std::size_t>(il)*nv+a)*no+j)*nv+b]=
                                    (1+((i0+il)==j))*amplitudes.tilde(i0+il,j,a,b);
                        std::array<std::size_t,4> shape{static_cast<std::size_t>(itile),static_cast<std::size_t>(nv),
                            static_cast<std::size_t>(no),static_cast<std::size_t>(nv)};
                        record(output.size()+tensor.size()+static_cast<std::size_t>(itile)*nv*no*ntile,output.size(),0);
                        tensor=transform_pair_axis(tensor,shape,3,
                            mo_coeff.middleRows(tau0,ntile).rightCols(nv),tau_tile);
                        record(output.size()+tensor.size()+static_cast<std::size_t>(itile)*nv*ktile*ntile,output.size(),0);
                        tensor=transform_pair_axis(tensor,shape,2,
                            mo_coeff.middleRows(kappa0,ktile).leftCols(no),kappa_tile);
                        record(output.size()+tensor.size()+static_cast<std::size_t>(itile)*vtile*ktile*ntile,output.size(),0);
                        tensor=transform_pair_axis(tensor,shape,1,
                            mo_coeff.middleRows(nu0,vtile).rightCols(nv),nu_tile);
                        if (tensor.size()!=static_cast<std::size_t>(itile)*suffix)
                            return std::unexpected("DH Eq. 47 blocked backtransform: invalid intermediate extent.");
                        const Eigen::Map<const RowMatrix> source(tensor.data(),itile,static_cast<Eigen::Index>(suffix));
                        result.noalias() += mo_coeff.leftCols(no).middleCols(i0,itile)*source;
                    }
                    if (!std::all_of(output.begin(),output.end(),[](double x){ return std::isfinite(x); }))
                        return std::unexpected("DH Eq. 47 blocked backtransform: nonfinite output tile.");
                    for (int mu0=0; mu0<nao; mu0+=mu_tile)
                    {
                        const int mtile=std::min(mu_tile,nao-mu0);
                        std::vector<double> tile(static_cast<std::size_t>(mtile)*suffix);
                        record(output.size()+tile.size(),output.size(),tile.size());
                        std::copy(output.begin()+static_cast<std::size_t>(mu0)*suffix,
                            output.begin()+static_cast<std::size_t>(mu0+mtile)*suffix,tile.begin());
                        auto consumed=consumer(mu0,mtile,nu0,vtile,kappa0,ktile,tau0,ntile,tile);
                        if (!consumed) return std::unexpected(consumed.error());
                    }
                }
            }
        }
        return {};
    }

    std::expected<std::vector<double>, std::string>
    build_dh_eq47_nonseparable_raw_ao(const DHPT2AmplitudeDensity &amplitudes,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff, int column_tile)
    {
        const Eigen::Index nao=mo_coeff.rows();
        const std::size_t n4=static_cast<std::size_t>(nao)*nao*nao*nao;
        std::vector<double> out(n4);
        auto streamed=for_each_dh_eq47_nonseparable_raw_ao_tile(
            amplitudes,mo_coeff,column_tile,column_tile,
            [&out,nao](int begin,int width,const std::vector<double> &tile)
            -> std::expected<void,std::string>
            {
                const std::size_t suffix=static_cast<std::size_t>(nao)*nao*nao;
                if(tile.size()!=static_cast<std::size_t>(width)*suffix)
                    return std::unexpected("DH Eq. 47 raw adapter: invalid tile extent.");
                std::copy(tile.begin(),tile.end(),out.begin()+static_cast<std::size_t>(begin)*suffix);
                return {};
            });
        if(!streamed) return std::unexpected(streamed.error());
        return out;
    }

    std::expected<DHEq46_47TwoParticleDensity, std::string>
    build_dh_eq46_47_two_particle_density(
        const DHRelaxedDifferenceDensity &relaxed,
        const DHPT2AmplitudeDensity &amplitudes,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_density_ao,
        double exact_exchange, DHGammaStorage storage)
    {
        const int no = amplitudes.n_occ;
        const int nv = amplitudes.n_virt;
        const int nmo = no + nv;
        const Eigen::Index nao = mo_coeff.rows();
        const std::size_t nt = static_cast<std::size_t>(no) * no * nv * nv;
        if (!std::isfinite(exact_exchange) || exact_exchange < 0.0)
            return std::unexpected("DH Eq. 46-47 density: invalid exact-exchange coefficient.");
        if (no <= 0 || nv <= 0 || nao <= 0 || mo_coeff.cols() != nmo ||
            !amplitudes.has_tilde_storage() || relaxed.symmetric_mo.rows() != nmo ||
            relaxed.symmetric_mo.cols() != nmo || ground_density_ao.rows() != nao ||
            ground_density_ao.cols() != nao)
            return std::unexpected("DH Eq. 46-47 density: incompatible relaxed density, amplitudes, orbitals, or ground density.");

        DHEq46_47TwoParticleDensity out;
        out.n_ao = static_cast<int>(nao);
        out.ground_density_ao = ground_density_ao;
        out.exact_exchange = exact_exchange;
        out.relaxed_difference_ao.noalias() =
            mo_coeff * relaxed.symmetric_mo * mo_coeff.transpose();
        const std::size_t n4 = static_cast<std::size_t>(nao) * nao * nao * nao;

        // Eq. (46), correction only: its 1/2 P P - 1/4 P P terms are the
        // SCF reference and deliberately do not enter the PT2 correction.
        // The separable term differentiates D:F_KS, so it must retain the
        // hybrid coefficient in F_KS=h+J[P]-a_x K[P]/2+V_XC[P]. This also
        // applies to the Z-dependent part of D. Eq. (47) is unscaled by a_x.
        const auto separable_raw = [&](int mu, int nu, int ka, int ta)
        {
            return out.relaxed_difference_ao(mu, nu) * ground_density_ao(ka, ta) -
                0.5 * exact_exchange * out.relaxed_difference_ao(mu, ka) * ground_density_ao(nu, ta);
        };
        if (storage == DHGammaStorage::ReferenceAll)
        {
            out.separable_raw_ao.assign(n4, 0.0);
            for (int mu = 0; mu < nao; ++mu)
                for (int nu = 0; nu < nao; ++nu)
                    for (int ka = 0; ka < nao; ++ka)
                        for (int ta = 0; ta < nao; ++ta)
                            out.separable_raw_ao[eri_idx(mu, nu, ka, ta, nao)] = separable_raw(mu, nu, ka, ta);
            out.separable_symmetric_ao = eri_eightfold_symmetrize(out.separable_raw_ao, out.n_ao);
        }
        // ContractionOnly retains the O(N^2) D/P/a_x factorization above.
        // O6.3 evaluates this literal raw Eq. (46) value alongside each
        // derivative quartet; no separable O(N^4) tensor is needed there.
        // Eq. (47), direct MO-to-AO backtransformation.  The AO pair order
        // is (mu nu | ka ta), so its closed-shell spatial realization is
        // C_mu,i C_nu,a C_ka,j C_ta,b t~_ab^ij, exactly as printed in the
        // paper.  In particular, this is the MP2 (ia|jb) pairing -- it is
        // not C_mu,i C_nu,j C_ka,a C_ta,b, which would contract t~ with the
        // unrelated (ij|ab) integral and violates Eq. (47).  t_tilde already
        // carries c_PT2; the explicit (1+delta_ij) is paper-defined and must
        // not be folded into the amplitude convention.
        if (storage == DHGammaStorage::ReferenceAll)
        {
            auto nonseparable=build_dh_eq47_nonseparable_raw_ao(amplitudes,mo_coeff);
            if (!nonseparable) return std::unexpected(nonseparable.error());
            out.nonseparable_raw_ao=std::move(*nonseparable);
            out.nonseparable_symmetric_ao = eri_eightfold_symmetrize(out.nonseparable_raw_ao, out.n_ao);
            out.total_raw_ao = out.separable_raw_ao;
            for (std::size_t p = 0; p < n4; ++p)
                out.total_raw_ao[p] += out.nonseparable_raw_ao[p];
            out.total_symmetric_ao = eri_eightfold_symmetrize(out.total_raw_ao, out.n_ao);
        }
        // ContractionOnly retains no final Eq. (47) AO tensor.  O7.4.3
        // passes a call-scoped tile execution context to the direct ERI
        // consumer, which immediately contracts each emitted raw tile.
        return out;
    }

    std::expected<DHEq33NonXCGradient, std::string>
    build_dh_eq33_non_xc_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const DHOverlapDensity &overlap,
        const DHEq46_47TwoParticleDensity &two_particle,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33NonXCDerivatives &derivatives,
        const DHEq47TiledExecution *eq47_tiled)
    {
        const Eigen::Index nao = mo_coeff.rows();
        const Eigen::Index nmo = mo_coeff.cols();
        const std::size_t n4 = static_cast<std::size_t>(nao) * nao * nao * nao;
        const bool has_dense_one_electron_overlap =
            !derivatives.hamiltonian_ao.empty() || !derivatives.overlap_ao.empty();
        const bool dense_one_electron_overlap = has_dense_one_electron_overlap &&
            derivatives.hamiltonian_ao.size() == derivatives.overlap_ao.size();
        const bool direct_one_electron_overlap =
            static_cast<bool>(derivatives.direct_one_electron_overlap);
        const bool dense_two_electron = !derivatives.eri_ao.empty();
        const bool direct_two_electron = static_cast<bool>(derivatives.direct_two_electron);
        const bool separable_tensor = two_particle.separable_symmetric_ao.size() == n4;
        const bool nonseparable_symmetric = two_particle.nonseparable_symmetric_ao.size() == n4;
        const bool nonseparable_raw = two_particle.nonseparable_raw_ao.size() == n4;
        const bool separable_factorized = two_particle.ground_density_ao.rows() == nao &&
            two_particle.ground_density_ao.cols() == nao && two_particle.ground_density_ao.allFinite() &&
            two_particle.relaxed_difference_ao.rows() == nao &&
            two_particle.relaxed_difference_ao.cols() == nao && two_particle.relaxed_difference_ao.allFinite() &&
            std::isfinite(two_particle.exact_exchange) && two_particle.exact_exchange >= 0.0;
        if (nao <= 0 || nmo <= 0 || relaxed.symmetric_mo.rows() != nmo ||
            relaxed.symmetric_mo.cols() != nmo || overlap.symmetric_mo.rows() != nmo ||
            overlap.symmetric_mo.cols() != nmo || two_particle.n_ao != nao ||
            (!separable_tensor && !separable_factorized) ||
            (!nonseparable_symmetric && !nonseparable_raw && !eq47_tiled) ||
            (!two_particle.total_symmetric_ao.empty() && two_particle.total_symmetric_ao.size() != n4) ||
            (!dense_one_electron_overlap && !direct_one_electron_overlap) ||
            (dense_one_electron_overlap && direct_one_electron_overlap) ||
            (has_dense_one_electron_overlap && !dense_one_electron_overlap) ||
            (!dense_two_electron && !direct_two_electron) ||
            (dense_two_electron && direct_two_electron) ||
            (dense_two_electron && (!separable_tensor || !nonseparable_symmetric)))
            return std::unexpected("DH Eq. 33 non-XC gradient: incompatible densities, orbitals, or derivative dimensions.");

        const Eigen::MatrixXd d_ao = mo_coeff * relaxed.symmetric_mo * mo_coeff.transpose();
        const Eigen::MatrixXd w_ao = mo_coeff * overlap.symmetric_mo * mo_coeff.transpose();
        std::size_t ncoord = dense_one_electron_overlap ? derivatives.hamiltonian_ao.size() :
            (dense_two_electron ? derivatives.eri_ao.size() : 0);
        std::optional<DHEq33OneElectronOverlapGradient> direct_one_electron;
        if (direct_one_electron_overlap)
        {
            auto direct = derivatives.direct_one_electron_overlap(d_ao, w_ao);
            if (!direct || direct->one_electron.size() <= 0 ||
                direct->overlap.size() != direct->one_electron.size() ||
                !direct->one_electron.allFinite() || !direct->overlap.allFinite())
                return std::unexpected("DH Eq. 33 non-XC gradient: direct h/S contraction failed or has invalid dimensions.");
            if (ncoord == 0) ncoord = static_cast<std::size_t>(direct->one_electron.size());
            if (direct->one_electron.size() != static_cast<Eigen::Index>(ncoord))
                return std::unexpected("DH Eq. 33 non-XC gradient: direct h/S coordinate count disagrees with derivative data.");
            direct_one_electron = std::move(*direct);
        }
        if (ncoord == 0 || (dense_one_electron_overlap && derivatives.overlap_ao.size() != ncoord) ||
            (dense_two_electron && derivatives.eri_ao.size() != ncoord))
            return std::unexpected("DH Eq. 33 non-XC gradient: derivative coordinate counts are inconsistent.");
        DHEq33NonXCGradient out;
        out.one_electron = Eigen::VectorXd::Zero(ncoord);
        out.overlap = Eigen::VectorXd::Zero(ncoord);
        out.two_electron_separable = Eigen::VectorXd::Zero(ncoord);
        out.two_electron_nonseparable = Eigen::VectorXd::Zero(ncoord);
        const Eq33NonXCAccumulator accumulator{
            two_particle.separable_symmetric_ao,
            two_particle.nonseparable_symmetric_ao};

        if (direct_one_electron)
        {
            out.one_electron = std::move(direct_one_electron->one_electron);
            out.overlap = std::move(direct_one_electron->overlap);
        }
        else for (std::size_t x = 0; x < ncoord; ++x)
        {
            if (derivatives.hamiltonian_ao[x].rows() != nao || derivatives.hamiltonian_ao[x].cols() != nao ||
                derivatives.overlap_ao[x].rows() != nao || derivatives.overlap_ao[x].cols() != nao)
                return std::unexpected("DH Eq. 33 non-XC gradient: dense h/S AO dimensions are inconsistent.");
            out.one_electron(static_cast<Eigen::Index>(x)) =
                (d_ao.array() * derivatives.hamiltonian_ao[x].array()).sum();
            out.overlap(static_cast<Eigen::Index>(x)) =
                (w_ao.array() * derivatives.overlap_ao[x].array()).sum();
        }

        if (direct_two_electron)
        {
            auto direct = derivatives.direct_two_electron(two_particle, eq47_tiled);
            if (!direct || direct->separable.size() != static_cast<Eigen::Index>(ncoord) ||
                direct->nonseparable.size() != static_cast<Eigen::Index>(ncoord) ||
                !direct->separable.allFinite() || !direct->nonseparable.allFinite())
                return std::unexpected("DH Eq. 33 non-XC gradient: direct ERI contraction failed or has invalid dimensions.");
            out.two_electron_separable = std::move(direct->separable);
            out.two_electron_nonseparable = std::move(direct->nonseparable);
        }
        else for (std::size_t x = 0; x < ncoord; ++x)
        {
            const auto &eri = derivatives.eri_ao[x];
            if (eri.size() != n4)
                return std::unexpected("DH Eq. 33 non-XC gradient: dense ERI derivative dimensions are inconsistent.");
            accumulator.accumulate_eri(static_cast<Eigen::Index>(x), eri, out);
        }
        out.two_electron = out.two_electron_separable + out.two_electron_nonseparable;
        out.total = out.one_electron + out.overlap + out.two_electron;
        return out;
    }

    std::expected<DHEq33XCFixedDensityGradient, std::string>
    build_dh_eq33_xc_fixed_density_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs)
    {
        if (inputs.molecule == nullptr || inputs.basis == nullptr || inputs.molecular_grid == nullptr ||
            inputs.ao_grid == nullptr || inputs.ao_hessian == nullptr ||
            inputs.exchange_functional == nullptr || inputs.correlation_functional == nullptr ||
            inputs.exchange_functional->is_lda_like() != inputs.correlation_functional->is_lda_like() ||
            inputs.exchange_functional->is_gga_like() != inputs.correlation_functional->is_gga_like())
            return std::unexpected("DH Eq. 33 XC-II: invalid fixed-density grid or functional inputs.");

        const Eigen::Index nao = mo_coeff.rows();
        const Eigen::Index nmo = mo_coeff.cols();
        const auto &mol = *inputs.molecule;
        const auto &grid = *inputs.molecular_grid;
        const auto &ao = *inputs.ao_grid;
        const auto &hess = *inputs.ao_hessian;
        if (nao == 0 || relaxed.symmetric_mo.rows() != nmo || relaxed.symmetric_mo.cols() != nmo ||
            inputs.ground_density_ao.rows() != nao || inputs.ground_density_ao.cols() != nao ||
            ao.nbasis() != nao || grid.points.rows() != ao.npoints() || grid.points.cols() != 4 ||
            hess.nbasis() != nao || hess.npoints() != ao.npoints())
            return std::unexpected("DH Eq. 33 XC-II: inconsistent density, AO grid, or Hessian dimensions.");

        const Eigen::MatrixXd d_ao = mo_coeff * relaxed.symmetric_mo * mo_coeff.transpose();
        auto ground = evaluate_density_on_grid(ao, inputs.ground_density_ao);
        auto difference = evaluate_density_on_grid(ao, d_ao);
        if (!ground || !difference)
            return std::unexpected("DH Eq. 33 XC-II: failed to evaluate ground or relaxed density on grid.");
        auto atoms_bf = atom_bf_lists(mol, *inputs.basis);
        if (!atoms_bf) return std::unexpected(atoms_bf.error());

        const Eigen::Index npts = ao.npoints();
        std::vector<double> rho(static_cast<std::size_t>(npts));
        for (Eigen::Index p = 0; p < npts; ++p)
            rho[static_cast<std::size_t>(p)] = ground->total.rho(p);

        const auto zero_correlation_if_combined = [&inputs](std::vector<double> &values)
        {
            if (inputs.exchange_functional->is_combined_exchange_correlation())
                std::fill(values.begin(), values.end(), 0.0);
        };
        std::vector<double> frr_x, frr_c, frs_x, frs_c, fss_x, fss_c, fs_x, fs_c;
        if (inputs.exchange_functional->is_lda_like())
        {
            auto x = inputs.exchange_functional->evaluate_lda_fxc(rho, static_cast<int>(npts), frr_x);
            auto c = inputs.correlation_functional->evaluate_lda_fxc(rho, static_cast<int>(npts), frr_c);
            if (!x || !c) return std::unexpected("DH Eq. 33 XC-II: LDA fxc evaluation failed.");
            zero_correlation_if_combined(frr_c);
        }
        else if (inputs.exchange_functional->is_gga_like())
        {
            std::vector<double> sigma(static_cast<std::size_t>(npts));
            {
                // Whole-grid sigma is evaluated once; release it before Libxc.
                const Eigen::VectorXd ground_sigma = ground->total.gradient_squared();
                for (Eigen::Index p = 0; p < npts; ++p)
                    sigma[static_cast<std::size_t>(p)] = ground_sigma(p);
            }
            std::vector<double> exc;
            std::vector<double> vrho_x, vrho_c, exc_c;
            auto vx = inputs.exchange_functional->evaluate_gga_exc_vxc(
                rho, sigma, static_cast<int>(npts), exc, vrho_x, fs_x);
            auto vc = inputs.correlation_functional->evaluate_gga_exc_vxc(
                rho, sigma, static_cast<int>(npts), exc_c, vrho_c, fs_c);
            auto fx = inputs.exchange_functional->evaluate_gga_fxc(
                rho, sigma, static_cast<int>(npts), frr_x, frs_x, fss_x);
            auto fc = inputs.correlation_functional->evaluate_gga_fxc(
                rho, sigma, static_cast<int>(npts), frr_c, frs_c, fss_c);
            if (!vx || !vc || !fx || !fc)
                return std::unexpected("DH Eq. 33 XC-II: GGA vxc/fxc evaluation failed.");
            if (inputs.exchange_functional->is_combined_exchange_correlation())
            {
                std::fill(frr_c.begin(), frr_c.end(), 0.0);
                std::fill(frs_c.begin(), frs_c.end(), 0.0);
                std::fill(fss_c.begin(), fss_c.end(), 0.0);
                std::fill(fs_c.begin(), fs_c.end(), 0.0);
            }
        }
        else
            return std::unexpected("DH Eq. 33 XC-II: functional is neither LDA-like nor GGA-like.");

        DHEq33XCFixedDensityGradient out;
        out.gradient = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);
        const bool gga = inputs.exchange_functional->is_gga_like();
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            const double weight = grid.points(p, 3);
            if (weight == 0.0) continue;
            const std::size_t pi = static_cast<std::size_t>(p);
            const Eigen::Vector3d grad_p{ground->total.grad_x(p), ground->total.grad_y(p), ground->total.grad_z(p)};
            const Eigen::Vector3d grad_d{difference->total.grad_x(p), difference->total.grad_y(p), difference->total.grad_z(p)};
            for (std::size_t atom = 0; atom < mol.natoms; ++atom)
                for (int q = 0; q < 3; ++q)
                {
                    const double r = drho_channel(inputs.ground_density_ao, ao, p, static_cast<int>(atom), q, *atoms_bf);
                    if (!gga)
                    {
                        out.gradient(static_cast<Eigen::Index>(atom), q) +=
                            weight * (frr_x[pi] + frr_c[pi]) * r * difference->total.rho(p);
                        continue;
                    }
                    const Eigen::Vector3d g{
                        dg_axis_spin(inputs.ground_density_ao, ao, hess, p, 0, static_cast<int>(atom), q, *atoms_bf),
                        dg_axis_spin(inputs.ground_density_ao, ao, hess, p, 1, static_cast<int>(atom), q, *atoms_bf),
                        dg_axis_spin(inputs.ground_density_ao, ao, hess, p, 2, static_cast<int>(atom), q, *atoms_bf)};
                    const double dot_pg = grad_p.dot(g);
                    const double frr = frr_x[pi] + frr_c[pi];
                    const double frs = frs_x[pi] + frs_c[pi];
                    const double fss = fss_x[pi] + fss_c[pi];
                    const double fs = fs_x[pi] + fs_c[pi];
                    out.gradient(static_cast<Eigen::Index>(atom), q) += weight * (
                        (frr * r + 2.0 * frs * dot_pg) * difference->total.rho(p) +
                        2.0 * (frs * r + 2.0 * fss * dot_pg) * grad_p.dot(grad_d) +
                        2.0 * fs * g.dot(grad_d));
                }
        }
        return out;
    }

    std::expected<DHEq33XCLDAMovingGridGradient, std::string>
    build_dh_eq33_xc_lda_moving_grid_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs)
    {
        if (inputs.molecule == nullptr || inputs.molecular_grid == nullptr || inputs.ao_grid == nullptr ||
            inputs.exchange_functional == nullptr || inputs.correlation_functional == nullptr ||
            !inputs.exchange_functional->is_lda_like() || !inputs.correlation_functional->is_lda_like())
            return std::unexpected("DH Eq. 33 LDA moving-grid XC-II: invalid LDA grid or functional inputs.");

        const Eigen::Index nao = mo_coeff.rows();
        const Eigen::Index nmo = mo_coeff.cols();
        const auto &mol = *inputs.molecule;
        const auto &grid = *inputs.molecular_grid;
        const auto &ao = *inputs.ao_grid;
        if (nao == 0 || relaxed.symmetric_mo.rows() != nmo || relaxed.symmetric_mo.cols() != nmo ||
            inputs.ground_density_ao.rows() != nao || inputs.ground_density_ao.cols() != nao ||
            ao.nbasis() != nao || grid.points.rows() != ao.npoints() || grid.points.cols() != 4 ||
            grid.owner.size() != ao.npoints() || grid.atomic_weights.size() != ao.npoints())
            return std::unexpected("DH Eq. 33 LDA moving-grid XC-II: inconsistent density or molecular grid dimensions.");

        const Eigen::MatrixXd d_ao = mo_coeff * relaxed.symmetric_mo * mo_coeff.transpose();
        const auto ground = evaluate_density_on_grid(ao, inputs.ground_density_ao);
        const auto difference = evaluate_density_on_grid(ao, d_ao);
        if (!ground || !difference)
            return std::unexpected("DH Eq. 33 LDA moving-grid XC-II: failed to evaluate density on grid.");

        const Eigen::Index npts = ao.npoints();
        std::vector<double> rho(static_cast<std::size_t>(npts));
        for (Eigen::Index p = 0; p < npts; ++p)
            rho[static_cast<std::size_t>(p)] = ground->total.rho(p);
        std::vector<double> exc_x, exc_c, vrho_x, vrho_c, frr_x, frr_c;
        const auto vx = inputs.exchange_functional->evaluate_lda_exc_vxc(
            rho, static_cast<int>(npts), exc_x, vrho_x);
        const auto vc = inputs.correlation_functional->evaluate_lda_exc_vxc(
            rho, static_cast<int>(npts), exc_c, vrho_c);
        const auto fx = inputs.exchange_functional->evaluate_lda_fxc(
            rho, static_cast<int>(npts), frr_x);
        const auto fc = inputs.correlation_functional->evaluate_lda_fxc(
            rho, static_cast<int>(npts), frr_c);
        if (!vx || !vc || !fx || !fc)
            return std::unexpected("DH Eq. 33 LDA moving-grid XC-II: LDA vxc/fxc evaluation failed.");
        if (inputs.exchange_functional->is_combined_exchange_correlation())
        {
            std::fill(vrho_c.begin(), vrho_c.end(), 0.0);
            std::fill(frr_c.begin(), frr_c.end(), 0.0);
        }

        DHEq33XCLDAMovingGridGradient out;
        out.becke_partition = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);
        out.point_translation = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            const double weight = grid.points(p, 3);
            if (weight == 0.0) continue;
            const int owner = grid.owner(p);
            if (owner < 0 || owner >= static_cast<int>(mol.natoms))
                return std::unexpected("DH Eq. 33 LDA moving-grid XC-II: grid owner is out of range.");
            const std::size_t pi = static_cast<std::size_t>(p);
            const double vrho = vrho_x[pi] + vrho_c[pi];
            const double frr = frr_x[pi] + frr_c[pi];
            const double rho_d = difference->total.rho(p);
            const double integrand = vrho * rho_d;
            const auto dpartition = becke_partition_owner_derivatives(grid, mol, p);
            if (!dpartition) return std::unexpected(dpartition.error());
            for (std::size_t atom = 0; atom < mol.natoms; ++atom)
                for (int q = 0; q < 3; ++q)
                    out.becke_partition(static_cast<Eigen::Index>(atom), q) +=
                        grid.atomic_weights(p) * (*dpartition)(static_cast<Eigen::Index>(atom), q) * integrand;
            for (int q = 0; q < 3; ++q)
            {
                const double drho_dq = q == 0 ? difference->total.grad_x(p) :
                    (q == 1 ? difference->total.grad_y(p) : difference->total.grad_z(p));
                const double drho_pq = q == 0 ? ground->total.grad_x(p) :
                    (q == 1 ? ground->total.grad_y(p) : ground->total.grad_z(p));
                out.point_translation(owner, q) += weight *
                    (frr * drho_pq * rho_d + vrho * drho_dq);
            }
        }
        out.total = out.becke_partition + out.point_translation;
        return out;
    }

    std::expected<DHEq33XCLDADifferenceAOGradient, std::string>
    build_dh_eq33_xc_lda_difference_ao_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs)
    {
        if (inputs.molecule == nullptr || inputs.basis == nullptr || inputs.molecular_grid == nullptr ||
            inputs.ao_grid == nullptr || inputs.exchange_functional == nullptr ||
            inputs.correlation_functional == nullptr || !inputs.exchange_functional->is_lda_like() ||
            !inputs.correlation_functional->is_lda_like())
            return std::unexpected("DH Eq. 33 LDA D-AO XC-II: invalid LDA grid or functional inputs.");
        const Eigen::Index nao = mo_coeff.rows();
        const Eigen::Index nmo = mo_coeff.cols();
        const auto &mol = *inputs.molecule;
        const auto &grid = *inputs.molecular_grid;
        const auto &ao = *inputs.ao_grid;
        if (nao == 0 || relaxed.symmetric_mo.rows() != nmo || relaxed.symmetric_mo.cols() != nmo ||
            inputs.ground_density_ao.rows() != nao || inputs.ground_density_ao.cols() != nao ||
            ao.nbasis() != nao || grid.points.rows() != ao.npoints() || grid.points.cols() != 4)
            return std::unexpected("DH Eq. 33 LDA D-AO XC-II: inconsistent density or AO grid dimensions.");
        const Eigen::MatrixXd d_ao = mo_coeff * relaxed.symmetric_mo * mo_coeff.transpose();
        const auto ground = evaluate_density_on_grid(ao, inputs.ground_density_ao);
        if (!ground)
            return std::unexpected("DH Eq. 33 LDA D-AO XC-II: failed to evaluate ground density on grid.");
        const auto atoms_bf = atom_bf_lists(mol, *inputs.basis);
        if (!atoms_bf) return std::unexpected(atoms_bf.error());
        const Eigen::Index npts = ao.npoints();
        std::vector<double> rho(static_cast<std::size_t>(npts)), exc_x, exc_c, vrho_x, vrho_c;
        for (Eigen::Index p = 0; p < npts; ++p)
            rho[static_cast<std::size_t>(p)] = ground->total.rho(p);
        const auto vx = inputs.exchange_functional->evaluate_lda_exc_vxc(
            rho, static_cast<int>(npts), exc_x, vrho_x);
        const auto vc = inputs.correlation_functional->evaluate_lda_exc_vxc(
            rho, static_cast<int>(npts), exc_c, vrho_c);
        if (!vx || !vc)
            return std::unexpected("DH Eq. 33 LDA D-AO XC-II: LDA vxc evaluation failed.");
        if (inputs.exchange_functional->is_combined_exchange_correlation())
            std::fill(vrho_c.begin(), vrho_c.end(), 0.0);
        DHEq33XCLDADifferenceAOGradient out;
        out.gradient = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            const double weight = grid.points(p, 3);
            if (weight == 0.0) continue;
            const double vrho = vrho_x[static_cast<std::size_t>(p)] + vrho_c[static_cast<std::size_t>(p)];
            for (std::size_t atom = 0; atom < mol.natoms; ++atom)
                for (int q = 0; q < 3; ++q)
                    out.gradient(static_cast<Eigen::Index>(atom), q) += weight * vrho *
                        drho_channel(d_ao, ao, p, static_cast<int>(atom), q, *atoms_bf);
        }
        return out;
    }

    std::expected<DHEq33XCGGAMovingGridGradient, std::string>
    build_dh_eq33_xc_gga_moving_grid_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs)
    {
        if (inputs.molecule == nullptr || inputs.molecular_grid == nullptr || inputs.ao_grid == nullptr ||
            inputs.ao_hessian == nullptr || inputs.exchange_functional == nullptr ||
            inputs.correlation_functional == nullptr || !inputs.exchange_functional->is_gga_like() ||
            !inputs.correlation_functional->is_gga_like())
            return std::unexpected("DH Eq. 33 GGA moving-grid XC-II: invalid GGA grid or functional inputs.");

        const Eigen::Index nao = mo_coeff.rows();
        const Eigen::Index nmo = mo_coeff.cols();
        const auto &mol = *inputs.molecule;
        const auto &grid = *inputs.molecular_grid;
        const auto &ao = *inputs.ao_grid;
        const auto &hess = *inputs.ao_hessian;
        if (nao == 0 || relaxed.symmetric_mo.rows() != nmo || relaxed.symmetric_mo.cols() != nmo ||
            inputs.ground_density_ao.rows() != nao || inputs.ground_density_ao.cols() != nao ||
            ao.nbasis() != nao || hess.nbasis() != nao || hess.npoints() != ao.npoints() ||
            grid.points.rows() != ao.npoints() || grid.points.cols() != 4 ||
            grid.owner.size() != ao.npoints() || grid.atomic_weights.size() != ao.npoints())
            return std::unexpected("DH Eq. 33 GGA moving-grid XC-II: inconsistent density, AO grid, or Hessian dimensions.");

        const Eigen::MatrixXd d_ao = mo_coeff * relaxed.symmetric_mo * mo_coeff.transpose();
        const auto ground = evaluate_density_on_grid(ao, inputs.ground_density_ao);
        const auto difference = evaluate_density_on_grid(ao, d_ao);
        if (!ground || !difference)
            return std::unexpected("DH Eq. 33 GGA moving-grid XC-II: failed to evaluate density on grid.");

        const Eigen::Index npts = ao.npoints();
        std::vector<double> rho(static_cast<std::size_t>(npts)), sigma(static_cast<std::size_t>(npts));
        {
            const Eigen::VectorXd ground_sigma = ground->total.gradient_squared();
            for (Eigen::Index p = 0; p < npts; ++p)
            {
                rho[static_cast<std::size_t>(p)] = ground->total.rho(p);
                sigma[static_cast<std::size_t>(p)] = ground_sigma(p);
            }
        }
        std::vector<double> exc_x, exc_c, vrho_x, vrho_c, fs_x, fs_c;
        std::vector<double> frr_x, frr_c, frs_x, frs_c, fss_x, fss_c;
        const auto vx = inputs.exchange_functional->evaluate_gga_exc_vxc(
            rho, sigma, static_cast<int>(npts), exc_x, vrho_x, fs_x);
        const auto vc = inputs.correlation_functional->evaluate_gga_exc_vxc(
            rho, sigma, static_cast<int>(npts), exc_c, vrho_c, fs_c);
        const auto fx = inputs.exchange_functional->evaluate_gga_fxc(
            rho, sigma, static_cast<int>(npts), frr_x, frs_x, fss_x);
        const auto fc = inputs.correlation_functional->evaluate_gga_fxc(
            rho, sigma, static_cast<int>(npts), frr_c, frs_c, fss_c);
        if (!vx || !vc || !fx || !fc)
            return std::unexpected("DH Eq. 33 GGA moving-grid XC-II: GGA vxc/fxc evaluation failed.");
        if (inputs.exchange_functional->is_combined_exchange_correlation())
        {
            std::fill(vrho_c.begin(), vrho_c.end(), 0.0);
            std::fill(fs_c.begin(), fs_c.end(), 0.0);
            std::fill(frr_c.begin(), frr_c.end(), 0.0);
            std::fill(frs_c.begin(), frs_c.end(), 0.0);
            std::fill(fss_c.begin(), fss_c.end(), 0.0);
        }

        const Eigen::MatrixXd p_symmetric = 0.5 * (inputs.ground_density_ao + inputs.ground_density_ao.transpose());
        const Eigen::MatrixXd d_symmetric = 0.5 * (d_ao + d_ao.transpose());
        const auto spatial_hessian_column = [&ao, &hess](const Eigen::MatrixXd &density,
                                                Eigen::Index p, int q) -> Eigen::Vector3d
        {
            const Eigen::VectorXd phi = ao.values.row(p).transpose();
            const std::array<Eigen::VectorXd, 3> grad = {
                ao.grad_x.row(p).transpose(), ao.grad_y.row(p).transpose(), ao.grad_z.row(p).transpose()};
            const std::array<Eigen::VectorXd, 3> h_column = {
                q == 0 ? hess.h_xx.row(p).transpose() : (q == 1 ? hess.h_xy.row(p).transpose() : hess.h_xz.row(p).transpose()),
                q == 0 ? hess.h_xy.row(p).transpose() : (q == 1 ? hess.h_yy.row(p).transpose() : hess.h_yz.row(p).transpose()),
                q == 0 ? hess.h_xz.row(p).transpose() : (q == 1 ? hess.h_yz.row(p).transpose() : hess.h_zz.row(p).transpose())};
            const Eigen::VectorXd density_phi = density * phi;
            const Eigen::VectorXd density_grad_q = density * grad[static_cast<std::size_t>(q)];
            Eigen::Vector3d out;
            for (int axis = 0; axis < 3; ++axis)
                out(axis) = 2.0 * (h_column[static_cast<std::size_t>(axis)].dot(density_phi) +
                                   grad[static_cast<std::size_t>(axis)].dot(density_grad_q));
            return out;
        };

        DHEq33XCGGAMovingGridGradient out;
        out.becke_partition = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);
        out.point_translation = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            const double weight = grid.points(p, 3);
            if (weight == 0.0) continue;
            const int owner = grid.owner(p);
            if (owner < 0 || owner >= static_cast<int>(mol.natoms))
                return std::unexpected("DH Eq. 33 GGA moving-grid XC-II: grid owner is out of range.");
            const std::size_t pi = static_cast<std::size_t>(p);
            const Eigen::Vector3d grad_p{ground->total.grad_x(p), ground->total.grad_y(p), ground->total.grad_z(p)};
            const Eigen::Vector3d grad_d{difference->total.grad_x(p), difference->total.grad_y(p), difference->total.grad_z(p)};
            const double vrho = vrho_x[pi] + vrho_c[pi];
            const double fs = fs_x[pi] + fs_c[pi];
            const double frr = frr_x[pi] + frr_c[pi];
            const double frs = frs_x[pi] + frs_c[pi];
            const double fss = fss_x[pi] + fss_c[pi];
            const double rho_d = difference->total.rho(p);
            const double dot_pd = grad_p.dot(grad_d);
            const auto dpartition = becke_partition_owner_derivatives(grid, mol, p);
            if (!dpartition) return std::unexpected(dpartition.error());
            const double integrand = vrho * rho_d + 2.0 * fs * dot_pd;
            for (std::size_t atom = 0; atom < mol.natoms; ++atom)
                for (int q = 0; q < 3; ++q)
                    out.becke_partition(static_cast<Eigen::Index>(atom), q) +=
                        grid.atomic_weights(p) * (*dpartition)(static_cast<Eigen::Index>(atom), q) * integrand;
            for (int q = 0; q < 3; ++q)
            {
                const Eigen::Vector3d h_p_q = spatial_hessian_column(p_symmetric, p, q);
                const Eigen::Vector3d h_d_q = spatial_hessian_column(d_symmetric, p, q);
                const double dp_sigma = 2.0 * grad_p.dot(h_p_q);
                const double derivative =
                    (frr * grad_p(q) + frs * dp_sigma) * rho_d + vrho * grad_d(q) +
                    2.0 * (frs * grad_p(q) + fss * dp_sigma) * dot_pd +
                    2.0 * fs * (h_p_q.dot(grad_d) + grad_p.dot(h_d_q));
                out.point_translation(owner, q) += weight * derivative;
            }
        }
        out.total = out.becke_partition + out.point_translation;
        return out;
    }

    std::expected<DHEq33XCGGADifferenceAOGradient, std::string>
    build_dh_eq33_xc_gga_difference_ao_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs)
    {
        if (inputs.molecule == nullptr || inputs.basis == nullptr || inputs.molecular_grid == nullptr ||
            inputs.ao_grid == nullptr || inputs.ao_hessian == nullptr || inputs.exchange_functional == nullptr ||
            inputs.correlation_functional == nullptr || !inputs.exchange_functional->is_gga_like() ||
            !inputs.correlation_functional->is_gga_like())
            return std::unexpected("DH Eq. 33 GGA D-AO XC-II: invalid GGA grid or functional inputs.");
        const Eigen::Index nao = mo_coeff.rows();
        const Eigen::Index nmo = mo_coeff.cols();
        const auto &mol = *inputs.molecule;
        const auto &grid = *inputs.molecular_grid;
        const auto &ao = *inputs.ao_grid;
        const auto &hess = *inputs.ao_hessian;
        if (nao == 0 || relaxed.symmetric_mo.rows() != nmo || relaxed.symmetric_mo.cols() != nmo ||
            inputs.ground_density_ao.rows() != nao || inputs.ground_density_ao.cols() != nao ||
            ao.nbasis() != nao || hess.nbasis() != nao || hess.npoints() != ao.npoints() ||
            grid.points.rows() != ao.npoints() || grid.points.cols() != 4)
            return std::unexpected("DH Eq. 33 GGA D-AO XC-II: inconsistent density, AO grid, or Hessian dimensions.");
        const Eigen::MatrixXd d_ao = mo_coeff * relaxed.symmetric_mo * mo_coeff.transpose();
        const auto ground = evaluate_density_on_grid(ao, inputs.ground_density_ao);
        if (!ground)
            return std::unexpected("DH Eq. 33 GGA D-AO XC-II: failed to evaluate ground density on grid.");
        const auto atoms_bf = atom_bf_lists(mol, *inputs.basis);
        if (!atoms_bf) return std::unexpected(atoms_bf.error());
        const Eigen::Index npts = ao.npoints();
        std::vector<double> rho(static_cast<std::size_t>(npts)), sigma(static_cast<std::size_t>(npts));
        {
            const Eigen::VectorXd ground_sigma = ground->total.gradient_squared();
            for (Eigen::Index p = 0; p < npts; ++p)
            {
                rho[static_cast<std::size_t>(p)] = ground->total.rho(p);
                sigma[static_cast<std::size_t>(p)] = ground_sigma(p);
            }
        }
        std::vector<double> exc_x, exc_c, vrho_x, vrho_c, fs_x, fs_c;
        const auto vx = inputs.exchange_functional->evaluate_gga_exc_vxc(
            rho, sigma, static_cast<int>(npts), exc_x, vrho_x, fs_x);
        const auto vc = inputs.correlation_functional->evaluate_gga_exc_vxc(
            rho, sigma, static_cast<int>(npts), exc_c, vrho_c, fs_c);
        if (!vx || !vc)
            return std::unexpected("DH Eq. 33 GGA D-AO XC-II: GGA vxc evaluation failed.");
        if (inputs.exchange_functional->is_combined_exchange_correlation())
        {
            std::fill(vrho_c.begin(), vrho_c.end(), 0.0);
            std::fill(fs_c.begin(), fs_c.end(), 0.0);
        }
        DHEq33XCGGADifferenceAOGradient out;
        out.gradient = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(mol.natoms), 3);
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            const double weight = grid.points(p, 3);
            if (weight == 0.0) continue;
            const std::size_t pi = static_cast<std::size_t>(p);
            const double vrho = vrho_x[pi] + vrho_c[pi];
            const double fs = fs_x[pi] + fs_c[pi];
            const Eigen::Vector3d grad_p{ground->total.grad_x(p), ground->total.grad_y(p), ground->total.grad_z(p)};
            for (std::size_t atom = 0; atom < mol.natoms; ++atom)
                for (int q = 0; q < 3; ++q)
                {
                    const Eigen::Vector3d dgrad{
                        dg_axis_spin(d_ao, ao, hess, p, 0, static_cast<int>(atom), q, *atoms_bf),
                        dg_axis_spin(d_ao, ao, hess, p, 1, static_cast<int>(atom), q, *atoms_bf),
                        dg_axis_spin(d_ao, ao, hess, p, 2, static_cast<int>(atom), q, *atoms_bf)};
                    out.gradient(static_cast<Eigen::Index>(atom), q) += weight * (
                        vrho * drho_channel(d_ao, ao, p, static_cast<int>(atom), q, *atoms_bf) +
                        2.0 * fs * grad_p.dot(dgrad));
                }
        }
        return out;
    }

    std::expected<DHEq33CompleteXCIIGradient, std::string>
    build_dh_eq33_complete_xc_ii_gradient(
        const DHRelaxedDifferenceDensity &relaxed,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33XCFixedDensityInputs &inputs)
    {
        if (inputs.exchange_functional == nullptr || inputs.correlation_functional == nullptr ||
            inputs.exchange_functional->is_lda_like() != inputs.correlation_functional->is_lda_like() ||
            inputs.exchange_functional->is_gga_like() != inputs.correlation_functional->is_gga_like())
            return std::unexpected("DH Eq. 33 complete XC-II: incompatible exchange/correlation functionals.");

        const auto p_side = build_dh_eq33_xc_fixed_density_gradient(relaxed, mo_coeff, inputs);
        if (!p_side) return std::unexpected(p_side.error());
        DHEq33CompleteXCIIGradient out;
        out.p_side_fixed = p_side->gradient;
        if (inputs.exchange_functional->is_lda_like())
        {
            const auto d_side = build_dh_eq33_xc_lda_difference_ao_gradient(relaxed, mo_coeff, inputs);
            const auto moving = build_dh_eq33_xc_lda_moving_grid_gradient(relaxed, mo_coeff, inputs);
            if (!d_side) return std::unexpected(d_side.error());
            if (!moving) return std::unexpected(moving.error());
            out.d_side_ao = d_side->gradient;
            out.becke_partition = moving->becke_partition;
            out.point_translation = moving->point_translation;
        }
        else if (inputs.exchange_functional->is_gga_like())
        {
            const auto d_side = build_dh_eq33_xc_gga_difference_ao_gradient(relaxed, mo_coeff, inputs);
            const auto moving = build_dh_eq33_xc_gga_moving_grid_gradient(relaxed, mo_coeff, inputs);
            if (!d_side) return std::unexpected(d_side.error());
            if (!moving) return std::unexpected(moving.error());
            out.d_side_ao = d_side->gradient;
            out.becke_partition = moving->becke_partition;
            out.point_translation = moving->point_translation;
        }
        else
            return std::unexpected("DH Eq. 33 complete XC-II: functional is neither LDA-like nor GGA-like.");
        if (out.p_side_fixed.rows() != out.d_side_ao.rows() ||
            out.p_side_fixed.cols() != out.d_side_ao.cols() ||
            out.p_side_fixed.rows() != out.becke_partition.rows() ||
            out.p_side_fixed.cols() != out.becke_partition.cols() ||
            out.p_side_fixed.rows() != out.point_translation.rows() ||
            out.p_side_fixed.cols() != out.point_translation.cols())
            return std::unexpected("DH Eq. 33 complete XC-II: component dimensions do not agree.");
        out.total = out.p_side_fixed + out.d_side_ao + out.becke_partition + out.point_translation;
        return out;
    }

    std::expected<DHKSFixedDensityFockDerivative, std::string>
    build_dh_ks_fixed_density_fock_derivative(
        const Eigen::Ref<const Eigen::MatrixXd> &ground_density_ao,
        double exact_exchange_coefficient,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq33NonXCDerivatives &derivatives,
        const DHEq33XCFixedDensityInputs &xc_inputs)
    {
        const Eigen::Index nao = ground_density_ao.rows();
        const std::size_t ncoord = derivatives.hamiltonian_ao.size();
        const std::size_t n4 = static_cast<std::size_t>(nao) * nao * nao * nao;
        if (nao == 0 || ground_density_ao.cols() != nao || mo_coeff.rows() != nao ||
            mo_coeff.cols() != nao || !std::isfinite(exact_exchange_coefficient) ||
            derivatives.overlap_ao.size() != ncoord || derivatives.eri_ao.size() != ncoord ||
            xc_inputs.ground_density_ao.rows() != nao || xc_inputs.ground_density_ao.cols() != nao ||
            (xc_inputs.ground_density_ao - ground_density_ao).cwiseAbs().maxCoeff() > 1e-12)
            return std::unexpected("DH fixed-density KS Fock derivative: inconsistent AO density, MO coefficients, or derivative inputs.");
        if ((ground_density_ao - ground_density_ao.transpose()).cwiseAbs().maxCoeff() > 1e-10)
            return std::unexpected("DH fixed-density KS Fock derivative: total AO density must be symmetric.");

        Eigen::FullPivLU<Eigen::MatrixXd> mo_lu(mo_coeff);
        if (!mo_lu.isInvertible())
            return std::unexpected("DH fixed-density KS Fock derivative: MO coefficient matrix is not invertible.");
        const Eigen::MatrixXd mo_inverse = mo_lu.inverse();

        DHKSFixedDensityFockDerivative out;
        out.hamiltonian.resize(ncoord);
        out.coulomb.resize(ncoord);
        out.exact_exchange.resize(ncoord);
        out.xc_geometry.resize(ncoord);
        out.total.resize(ncoord);
        for (std::size_t x = 0; x < ncoord; ++x)
        {
            if (derivatives.hamiltonian_ao[x].rows() != nao ||
                derivatives.hamiltonian_ao[x].cols() != nao ||
                derivatives.overlap_ao[x].rows() != nao ||
                derivatives.overlap_ao[x].cols() != nao || derivatives.eri_ao[x].size() != n4)
                return std::unexpected("DH fixed-density KS Fock derivative: derivative AO dimensions are inconsistent.");
            out.hamiltonian[x] = derivatives.hamiltonian_ao[x];
            out.coulomb[x] = Eigen::MatrixXd::Zero(nao, nao);
            out.exact_exchange[x] = Eigen::MatrixXd::Zero(nao, nao);
            const auto &eri_x = derivatives.eri_ao[x];
            for (Eigen::Index mu = 0; mu < nao; ++mu)
                for (Eigen::Index nu = 0; nu < nao; ++nu)
                    for (Eigen::Index ka = 0; ka < nao; ++ka)
                        for (Eigen::Index ta = 0; ta < nao; ++ta)
                        {
                            const double p = ground_density_ao(ka, ta);
                            out.coulomb[x](mu, nu) += p * eri_x[eri_idx(
                                static_cast<int>(mu), static_cast<int>(nu),
                                static_cast<int>(ka), static_cast<int>(ta), static_cast<int>(nao))];
                            out.exact_exchange[x](mu, nu) -= 0.5 * exact_exchange_coefficient * p *
                                eri_x[eri_idx(static_cast<int>(mu), static_cast<int>(ka),
                                              static_cast<int>(nu), static_cast<int>(ta), static_cast<int>(nao))];
                        }
            out.xc_geometry[x] = Eigen::MatrixXd::Zero(nao, nao);
        }

        // The complete XC-II scalar is linear in a symmetric test density D:
        // G_D^(x) = D : (V_XC[P])_geom^(x).  Probe its diagonal with E_mm
        // and each off-diagonal with 1/2(E_mn + E_nm), then recover the
        // symmetric AO matrix exactly without omitting the D-side AO or
        // moving-grid channels.
        for (Eigen::Index mu = 0; mu < nao; ++mu)
            for (Eigen::Index nu = 0; nu <= mu; ++nu)
            {
                Eigen::MatrixXd probe_ao = Eigen::MatrixXd::Zero(nao, nao);
                if (mu == nu)
                    probe_ao(mu, nu) = 1.0;
                else
                {
                    probe_ao(mu, nu) = 0.5;
                    probe_ao(nu, mu) = 0.5;
                }
                DHRelaxedDifferenceDensity probe;
                probe.symmetric_mo = mo_inverse * probe_ao * mo_inverse.transpose();
                probe.raw_mo = probe.symmetric_mo;
                if ((mo_coeff * probe.symmetric_mo * mo_coeff.transpose() - probe_ao).cwiseAbs().maxCoeff() > 1e-10)
                    return std::unexpected("DH fixed-density KS Fock derivative: AO probe transformation is not reversible.");
                const auto xc_gradient = build_dh_eq33_complete_xc_ii_gradient(probe, mo_coeff, xc_inputs);
                if (!xc_gradient)
                    return std::unexpected("DH fixed-density KS Fock derivative: complete XC geometry probe failed: " +
                                           xc_gradient.error());
                if (xc_gradient->total.rows() * xc_gradient->total.cols() != static_cast<Eigen::Index>(ncoord))
                    return std::unexpected("DH fixed-density KS Fock derivative: XC coordinate layout does not match derivative arrays.");
                for (std::size_t x = 0; x < ncoord; ++x)
                {
                    const Eigen::Index atom = static_cast<Eigen::Index>(x / 3);
                    const Eigen::Index cartesian = static_cast<Eigen::Index>(x % 3);
                    const double value = xc_gradient->total(atom, cartesian);
                    out.xc_geometry[x](mu, nu) = value;
                    out.xc_geometry[x](nu, mu) = value;
                }
            }
        for (std::size_t x = 0; x < ncoord; ++x)
            out.total[x] = out.hamiltonian[x] + out.coulomb[x] +
                out.exact_exchange[x] + out.xc_geometry[x];
        return out;
    }

    std::expected<DHStationaryProducts, std::string>
    build_dh_stationary_products(const HartreeFock::Correlation::RMP2Result &pt2,
        double c_pt2, const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const std::vector<double> &mo_eri, const DHEq41ResponseOperator &response,
        const DHMolecularERIBlocks *eq40_blocks)
    {
        if (eq40_blocks)
            return build_dh_stationary_products(pt2, c_pt2, mo_coeff, response, *eq40_blocks);
        auto dense_blocks=build_dh_molecular_eri_blocks(mo_eri,pt2.n_occ,pt2.n_virt);
        if(!dense_blocks) return std::unexpected(dense_blocks.error());
        return build_dh_stationary_products(pt2, c_pt2, mo_coeff, response, *dense_blocks,
            DHTildeAmplitudeStorage::Materialized);
    }

    std::expected<DHStationaryProducts, std::string>
    build_dh_stationary_products(const HartreeFock::Correlation::RMP2Result &pt2,
        double c_pt2, const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,
        const DHEq41ResponseOperator &response, const DHMolecularERIBlocks &eq40_blocks,
        DHTildeAmplitudeStorage storage)
    {
        DHStationaryProducts out;
        out.source_pt2 = &pt2;
        out.c_pt2 = c_pt2; out.exact_exchange = response.exact_exchange;
        out.source_mo_coeff = mo_coeff;
        auto amplitudes = build_dh_pt2_amplitude_density(pt2, c_pt2, storage);
        if (!amplitudes) return std::unexpected(amplitudes.error());
        out.amplitudes = std::move(*amplitudes);
        auto eq41 = build_dh_eq41_response_density(out.amplitudes, mo_coeff, response);
        if (!eq41) return std::unexpected(eq41.error());
        out.eq41_response = std::move(*eq41);
        auto eq40 = build_dh_eq40_amplitude_rhs(pt2, out.amplitudes, eq40_blocks, c_pt2);
        if (!eq40) return std::unexpected(eq40.error());
        out.eq40_amplitude_rhs = std::move(*eq40);
        auto rhs = build_dh_lagrangian_rhs(out.eq41_response, mo_coeff.leftCols(pt2.n_occ),
            mo_coeff.rightCols(pt2.n_virt), out.eq40_amplitude_rhs, DHRHSConvention::LiteralEq47);
        if (!rhs) return std::unexpected(rhs.error());
        out.lagrangian_rhs = std::move(*rhs);
        return out;
    }

    static std::expected<DHGradientDriverContract, std::string>
    assemble_dh_gradient_driver_contract(const DHGradientDriverInputs &inputs,
        DHStationaryProducts &&stationary, DHGammaStorage storage)
    {
        if (inputs.pt2_result == nullptr || !inputs.workspace || !std::isfinite(inputs.c_pt2))
            return std::unexpected("DH gradient driver contract: missing PT2 result or invalid PT2 coefficient.");
        const auto &pt2 = *inputs.pt2_result;
        const int nocc = pt2.n_occ;
        const int nvirt = pt2.n_virt;
        const Eigen::Index nmo = static_cast<Eigen::Index>(nocc + nvirt);
        if (nocc <= 0 || nvirt <= 0 || inputs.mo_coeff.cols() != nmo ||
            inputs.mo_coeff.rows() == 0 || inputs.orbital_energies.size() != nmo ||
            inputs.z_ai.rows() != nvirt || inputs.z_ai.cols() != nocc)
            return std::unexpected("DH gradient driver contract: inconsistent PT2, MO, or Z-vector dimensions.");
        if (inputs.xc_inputs.ground_density_ao.rows() != inputs.mo_coeff.rows() ||
            inputs.xc_inputs.ground_density_ao.cols() != inputs.mo_coeff.rows())
            return std::unexpected("DH gradient driver contract: ground AO density does not match MO coefficients.");

        DHGradientDriverContract out;
        const auto valid = [](const Eigen::MatrixXd &m, Eigen::Index rows, Eigen::Index cols)
        { return m.rows()==rows && m.cols()==cols && m.allFinite(); };
        if (stationary.source_pt2 != inputs.pt2_result ||
            stationary.c_pt2 != inputs.c_pt2 || stationary.exact_exchange != inputs.response_operator.exact_exchange ||
            !valid(stationary.source_mo_coeff,inputs.mo_coeff.rows(),nmo) ||
            !stationary.source_mo_coeff.isApprox(inputs.mo_coeff,0.0) ||
            stationary.amplitudes.n_occ!=nocc || stationary.amplitudes.n_virt!=nvirt ||
            !stationary.amplitudes.has_tilde_storage() ||
            !valid(stationary.amplitudes.dprime_oo,nocc,nocc) ||
            !valid(stationary.amplitudes.dprime_vv,nvirt,nvirt) ||
            !valid(stationary.amplitudes.dprime_mo,nmo,nmo) ||
            !valid(stationary.eq41_response.dprime_ao,inputs.mo_coeff.rows(),inputs.mo_coeff.rows()) ||
            !valid(stationary.eq41_response.response_ao,inputs.mo_coeff.rows(),inputs.mo_coeff.rows()) ||
            !valid(stationary.eq40_amplitude_rhs.three_external,nvirt,nocc) ||
            !valid(stationary.eq40_amplitude_rhs.total,nvirt,nocc) ||
            !valid(stationary.eq40_amplitude_rhs.pair_coefficients.total,nmo,nmo) ||
            !valid(stationary.lagrangian_rhs.response_ai,nvirt,nocc) ||
            !valid(stationary.lagrangian_rhs.amplitude_ai,nvirt,nocc) ||
            !valid(stationary.lagrangian_rhs.total_ai,nvirt,nocc) ||
            !valid(stationary.lagrangian_rhs.included_internal_ai,nvirt,nocc) ||
            stationary.lagrangian_rhs.included_internal_ai.norm()!=0.0 ||
            !inputs.mo_coeff.allFinite() || !inputs.orbital_energies.allFinite() || !inputs.z_ai.allFinite())
            return std::unexpected("DH gradient driver contract: stale or invalid stationary products.");
        if (!(stationary.lagrangian_rhs.total_ai-stationary.lagrangian_rhs.response_ai-
              stationary.lagrangian_rhs.amplitude_ai).isZero(1e-12) ||
            !(stationary.lagrangian_rhs.amplitude_ai-stationary.eq40_amplitude_rhs.three_external).isZero(1e-12) ||
            !(stationary.eq40_amplitude_rhs.three_external-
              stationary.eq40_amplitude_rhs.pair_coefficients.total.bottomLeftCorner(nvirt,nocc)+
              stationary.eq40_amplitude_rhs.pair_coefficients.total.topRightCorner(nocc,nvirt).transpose()).isZero(1e-12))
            return std::unexpected("DH gradient driver contract: stationary RHS convention mismatch.");
        static_cast<DHStationaryProducts &>(out) = std::move(stationary);
        auto relaxed = build_dh_relaxed_difference_density(out.amplitudes, inputs.z_ai);
        if (!relaxed) return std::unexpected(relaxed.error());
        out.relaxed_density = std::move(*relaxed);
        // The literal Eq. 47 pair-metric replacement below owns the
        // production amplitude pieces of Eqs. 42--45. Passing no MO ERI
        // data proves these builders cannot retain the old legacy-pair
        // dependency; ReferenceAll callers still request it explicitly.
        const std::vector<double> no_legacy_pair_eri;
        auto diagonal = build_dh_eq42_43_energy_weighted_density(
            out.relaxed_density, pt2, out.amplitudes, inputs.mo_coeff,
            inputs.orbital_energies, no_legacy_pair_eri, inputs.response_operator, false);
        if (!diagonal) return std::unexpected(diagonal.error());
        out.eq42_43_overlap = std::move(*diagonal);
        auto off_diagonal = build_dh_eq44_45_energy_weighted_density(
            out.relaxed_density, pt2, out.amplitudes, inputs.orbital_energies, no_legacy_pair_eri, false);
        if (!off_diagonal) return std::unexpected(off_diagonal.error());
        out.eq44_45_overlap = std::move(*off_diagonal);
        // Always use the independently validated four-coefficient pair
        // metric derivative, for production as well as contract callers.
        {
            const auto literal_pair = build_dh_eq47_pair_metric_overlap_density(
                out.eq40_amplitude_rhs.pair_coefficients, nocc, nvirt);
            if (!literal_pair) return std::unexpected(literal_pair.error());

            // Preserve the independently derived Eq. (42) response and
            // orbital-energy pieces, then replace only the pair-amplitude
            // pieces.  `w_ai` belongs to Eq. (45), not to the Eq. (47) pair
            // metric derivative, and consequently remains untouched.
            out.eq42_43_overlap.amplitude_oo = literal_pair->w_oo;
            out.eq42_43_overlap.amplitude_vv = literal_pair->w_vv;
            out.eq42_43_overlap.w_oo = out.eq42_43_overlap.response_oo +
                out.eq42_43_overlap.orbital_oo + out.eq42_43_overlap.amplitude_oo;
            out.eq42_43_overlap.w_vv = out.eq42_43_overlap.orbital_vv +
                out.eq42_43_overlap.amplitude_vv;
            out.eq44_45_overlap.w_ia = literal_pair->w_ia;
        }
        auto overlap = build_dh_overlap_density_adapter(out.eq42_43_overlap, out.eq44_45_overlap);
        if (!overlap) return std::unexpected(overlap.error());
        out.overlap_density = std::move(*overlap);
        auto pair_density = build_dh_eq46_47_two_particle_density(
            out.relaxed_density, out.amplitudes, inputs.mo_coeff, inputs.xc_inputs.ground_density_ao,
            inputs.response_operator.exact_exchange, storage);
        if (!pair_density) return std::unexpected(pair_density.error());
        out.two_particle_density = std::move(*pair_density);
        // This context is deliberately stack-bound to the synchronous Eq. 33
        // call. `out` owns the amplitudes; `inputs` owns the MO coefficients.
        // Nothing returned in DHGradientDriverContract may retain either
        // address through this O7.4.3 execution handoff.
        const DHEq47TiledExecution eq47_tiled{
            &out.amplitudes, &inputs.mo_coeff, 128, 32, 16, 16, 16, 16};
        auto non_xc = build_dh_eq33_non_xc_gradient(
            out.relaxed_density, out.overlap_density, out.two_particle_density,
            inputs.mo_coeff, inputs.workspace->non_xc_derivatives,
            storage == DHGammaStorage::ContractionOnly ? &eq47_tiled : nullptr);
        if (!non_xc) return std::unexpected(non_xc.error());
        out.non_xc_gradient = std::move(*non_xc);
        auto xc_ii = build_dh_eq33_complete_xc_ii_gradient(
            out.relaxed_density, inputs.mo_coeff, inputs.xc_inputs);
        if (!xc_ii) return std::unexpected(xc_ii.error());
        out.xc_ii_gradient = std::move(*xc_ii);
        return out;
    }

    std::expected<DHGradientDriverContract, std::string>
    build_dh_gradient_driver_contract(const DHGradientDriverInputs &inputs)
    {
        if (!inputs.pt2_result || !inputs.workspace)
            return std::unexpected("DH gradient reference contract: missing snapshot or workspace.");
        auto stationary = build_dh_stationary_products(*inputs.pt2_result, inputs.c_pt2,
            inputs.mo_coeff, inputs.workspace->mo_eri, inputs.response_operator);
        if (!stationary) return std::unexpected(stationary.error());
        return assemble_dh_gradient_driver_contract(inputs,std::move(*stationary),DHGammaStorage::ReferenceAll);
    }

    std::expected<DHGradientDriverContract, std::string>
    build_dh_gradient_driver_contract(const DHGradientDriverInputs &inputs, DHStationaryProducts &&stationary)
    {
        // Dense ERI derivatives are a retained reference/test adapter and
        // require the symmetric Eq. (46) tensor. The production driver
        // provides the O6.3 direct callback and can retain only D/P/a_x.
        const auto storage = inputs.workspace && inputs.workspace->non_xc_derivatives.direct_two_electron
            ? DHGammaStorage::ContractionOnly : DHGammaStorage::ReferenceAll;
        return assemble_dh_gradient_driver_contract(inputs,std::move(stationary),storage);
    }

    std::expected<DHEq33PT2CorrectionGradient, std::string>
    build_dh_eq33_pt2_correction_gradient(const DHGradientDriverContract &contract)
    {
        const Eigen::MatrixXd &xc = contract.xc_ii_gradient.total;
        const Eigen::VectorXd &non_xc = contract.non_xc_gradient.total;
        if (xc.cols() != 3 || xc.rows() == 0 || non_xc.size() != 3 * xc.rows())
            return std::unexpected("DH Eq. 33 correction: non-XC coordinate vector and XC-II atom rows do not agree.");
        DHEq33PT2CorrectionGradient out;
        out.non_xc = Eigen::MatrixXd::Zero(xc.rows(), 3);
        for (Eigen::Index atom = 0; atom < xc.rows(); ++atom)
            for (int q = 0; q < 3; ++q)
                out.non_xc(atom, q) = non_xc(3 * atom + q);
        out.xc_ii = xc;
        out.total = out.non_xc + out.xc_ii;
        return out;
    }
}
