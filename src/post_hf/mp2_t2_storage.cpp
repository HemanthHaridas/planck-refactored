#include "mp2.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace HartreeFock::Correlation
{
    namespace
    {
        std::size_t rmp2_t2_index(int i,int j,int a,int b,int no,int nv)
        { return ((static_cast<std::size_t>(i)*no+j)*nv+a)*nv+b; }

        std::size_t rmp2_pair_index(int p,int q,int npair)
        {
            if (p>q) std::swap(p,q);
            return static_cast<std::size_t>(p)*npair-
                static_cast<std::size_t>(p)*(p-1)/2+(q-p);
        }
    }

    double RMP2PairPackedT2::at(int i,int j,int a,int b) const
    {
        const int npair=n_occ*n_virt;
        if (i<0 || j<0 || a<0 || b<0 || i>=n_occ || j>=n_occ || a>=n_virt || b>=n_virt ||
            values.size()!=static_cast<std::size_t>(npair)*(npair+1)/2)
            return std::numeric_limits<double>::quiet_NaN();
        return values[rmp2_pair_index(i*n_virt+a,j*n_virt+b,npair)];
    }

    std::expected<RMP2PairPackedT2, std::string>
    pack_rmp2_pair_t2(const RMP2Result &result)
    {
        const int no=result.n_occ, nv=result.n_virt;
        const std::size_t nt=static_cast<std::size_t>(no)*no*nv*nv;
        if (no<=0 || nv<=0 || result.t2.size()!=nt)
            return std::unexpected("RMP2 packed T2: invalid canonical amplitude dimensions.");
        const int npair=no*nv;
        RMP2PairPackedT2 out{no,nv,std::vector<double>(static_cast<std::size_t>(npair)*(npair+1)/2)};
        for (int i=0;i<no;++i) for (int j=0;j<no;++j)
            for (int a=0;a<nv;++a) for (int b=0;b<nv;++b)
            {
                const double value=result.t2[rmp2_t2_index(i,j,a,b,no,nv)];
                const double partner=result.t2[rmp2_t2_index(j,i,b,a,no,nv)];
                if (!std::isfinite(value) || std::abs(value-partner)>1e-12)
                    return std::unexpected("RMP2 packed T2: pair-interchange symmetry is violated.");
                out.values[rmp2_pair_index(i*nv+a,j*nv+b,npair)]=value;
            }
        return out;
    }

    double RMP2T2PairTile::at(int i,int j,int a,int b) const
    {
        if (n_virt<=0 || i<0 || j<0 || a<0 || b<0)
            return std::numeric_limits<double>::quiet_NaN();
        const int p=i*n_virt+a, q=j*n_virt+b;
        if (p<p0 || p>=p0+rows || q<q0 || q>=q0+cols ||
            values.size()!=static_cast<std::size_t>(rows)*cols)
            return std::numeric_limits<double>::quiet_NaN();
        return values[static_cast<std::size_t>(p-p0)*cols+(q-q0)];
    }

    std::expected<void, std::string>
    for_each_rmp2_t2_pair_tile(const RMP2Result &result, int pair_tile_width,
                               const RMP2T2PairTileConsumer &consumer)
    {
        if (!result.has_t2() || result.n_occ<=0 || result.n_virt<=0)
            return std::unexpected("RMP2 T2 tile stream: canonical T2 amplitudes are unavailable.");
        if (pair_tile_width<=0)
            return std::unexpected("RMP2 T2 tile stream: pair tile width must be positive.");
        if (!consumer)
            return std::unexpected("RMP2 T2 tile stream: consumer is empty.");
        const int npair=result.n_occ*result.n_virt;
        for (int p0=0;p0<npair;p0+=pair_tile_width)
            for (int q0=0;q0<npair;q0+=pair_tile_width)
            {
                const int rows=std::min(pair_tile_width,npair-p0);
                const int cols=std::min(pair_tile_width,npair-q0);
                auto tile=read_rmp2_t2_pair_tile(result,p0,q0,rows,cols);
                if (!tile) return std::unexpected(tile.error());
                if (!consumer(*tile)) return {};
            }
        return {};
    }

}
