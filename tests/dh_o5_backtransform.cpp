#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include "dft/dh_pt2_gradient.h"

namespace
{
    using namespace DFT::Gradient;
    std::size_t four(int p,int q,int r,int s,int n)
    { return ((static_cast<std::size_t>(p)*n+q)*n+r)*n+s; }
    std::size_t amp(int i,int j,int a,int b,int o,int v)
    { return ((static_cast<std::size_t>(i)*o+j)*v+a)*v+b; }

    // Retain the eighth-order expression only in this independent oracle.
    std::vector<double> literal(const DHPT2AmplitudeDensity &d,const Eigen::MatrixXd &c)
    {
        const int n=c.rows(),o=d.n_occ,v=d.n_virt;
        std::vector<double> raw(static_cast<std::size_t>(n)*n*n*n);
        for(int mu=0;mu<n;++mu) for(int nu=0;nu<n;++nu)
            for(int ka=0;ka<n;++ka) for(int ta=0;ta<n;++ta)
            {
                double value=0;
                for(int i=0;i<o;++i) for(int j=0;j<o;++j)
                    for(int a=0;a<v;++a) for(int b=0;b<v;++b)
                        value+=c(mu,i)*c(nu,o+a)*c(ka,j)*c(ta,o+b)*
                            (1+(i==j))*d.t_tilde[amp(i,j,a,b,o,v)];
                raw[four(mu,nu,ka,ta,n)]=value;
            }
        return raw;
    }

    std::vector<double> symmetric(const std::vector<double> &raw,int n)
    {
        std::vector<double> out(raw.size());
        for(int p=0;p<n;++p) for(int q=0;q<n;++q)
            for(int r=0;r<n;++r) for(int s=0;s<n;++s)
                for(int flip1=0;flip1<2;++flip1) for(int flip2=0;flip2<2;++flip2)
                    for(int swap=0;swap<2;++swap)
                    {
                        std::array<int,4> x{p,q,r,s};
                        if(flip1) std::swap(x[0],x[1]);
                        if(flip2) std::swap(x[2],x[3]);
                        if(swap) { std::swap(x[0],x[2]); std::swap(x[1],x[3]); }
                        out[four(p,q,r,s,n)]+=raw[four(x[0],x[1],x[2],x[3],n)]/8;
                    }
        return out;
    }

    double error(const std::vector<double> &a,const std::vector<double> &b)
    {
        if(a.size()!=b.size()) return std::numeric_limits<double>::infinity();
        double result=0;
        for(std::size_t k=0;k<a.size();++k)
        {
            if(!std::isfinite(a[k]) || !std::isfinite(b[k])) return std::numeric_limits<double>::infinity();
            result=std::max(result,std::abs(a[k]-b[k]));
        }
        return result;
    }

    double dot(const std::vector<double> &a,const std::vector<double> &b)
    { return std::inner_product(a.begin(),a.end(),b.begin(),0.0); }

    DHPT2AmplitudeDensity amplitudes(int o,int v,double scale)
    {
        DHPT2AmplitudeDensity d; d.n_occ=o; d.n_virt=v;
        d.t_tilde.resize(static_cast<std::size_t>(o)*o*v*v);
        for(std::size_t k=0;k<d.t_tilde.size();++k) d.t_tilde[k]=scale*0.03*std::sin(0.63*(k+1));
        return d;
    }

    Eigen::MatrixXd coefficients(int n,int m)
    {
        Eigen::MatrixXd c(n,m);
        for(int p=0;p<n;++p) for(int q=0;q<m;++q)
            c(p,q)=0.3*std::cos(0.31*(p*m+q+1))+(p==q?0.5:0);
        return c;
    }

    bool fixture(int n,int o,int v,double scale,int basis_kind=0)
    {
        auto d=amplitudes(o,v,scale);
        auto c=coefficients(n,o+v);
        if(basis_kind==1) c.setIdentity();
        if(basis_kind==2)
        {
            c.setZero();
            for(int q=0;q<c.cols();++q) c((q+1)%n,q)=q%2?-1:1;
        }
        const auto ref=literal(d,c),sym=symmetric(ref,n);
        double worst=0;
        for(int tile : {1,3,7,128,4096})
        {
            const auto raw=build_dh_eq47_nonseparable_raw_ao(d,c,tile);
            if(!raw) { std::fprintf(stderr,"%s\n",raw.error().c_str()); return false; }
            worst=std::max(worst,error(*raw,ref));
        }
        // O7.4.1: concatenate bounded final-mu tiles and compare to the
        // independently literal raw tensor. The producer must not allocate
        // that final tensor to satisfy this consumer.
        for(int mu_tile : {1,2,5,128})
        {
            std::vector<double> tiled(ref.size(),0.0);
            const auto streamed=for_each_dh_eq47_nonseparable_raw_ao_tile(
                d,c,3,mu_tile,[&](int begin,int width,const std::vector<double> &tile)
                -> std::expected<void,std::string>
                {
                    const std::size_t suffix=static_cast<std::size_t>(n)*n*n;
                    if(tile.size()!=static_cast<std::size_t>(width)*suffix)
                        return std::unexpected("O7.4 tile extent mismatch.");
                    std::copy(tile.begin(),tile.end(),tiled.begin()+static_cast<std::size_t>(begin)*suffix);
                    return {};
                });
            if(!streamed) { std::fprintf(stderr,"%s\n",streamed.error().c_str()); return false; }
            worst=std::max(worst,error(tiled,ref));
        }
        // O7.4.4: tile every AO output index and the occupied reduction.
        // Reconstruct only in this dense oracle; the producer itself has no
        // [i,nu,kappa,tau] or final [mu,nu,kappa,tau] workspace.
        for(int width : {1,2,5})
        {
            std::vector<double> blocked(ref.size(),0.0);
            DHEq47BlockWorkspace workspace;
            const auto streamed=for_each_dh_eq47_nonseparable_raw_ao_block(
                d,c,2,width,width,width,width,
                [&](int mu0,int mc,int nu0,int nc,int ka0,int kc,int ta0,int tc,const std::vector<double> &tile)
                -> std::expected<void,std::string>
                {
                    const std::size_t suffix=static_cast<std::size_t>(nc)*kc*tc;
                    if(tile.size()!=static_cast<std::size_t>(mc)*suffix)
                        return std::unexpected("O7.4 block extent mismatch.");
                    for(int lm=0;lm<mc;++lm) for(int ln=0;ln<nc;++ln)
                        for(int lk=0;lk<kc;++lk) for(int lt=0;lt<tc;++lt)
                            blocked[(((static_cast<std::size_t>(mu0+lm)*n+(nu0+ln))*n+(ka0+lk))*n+(ta0+lt))]=
                                tile[(((static_cast<std::size_t>(lm)*nc+ln)*kc+lk)*tc+lt)];
                    return {};
                },&workspace);
            if(!streamed) { std::fprintf(stderr,"%s\n",streamed.error().c_str()); return false; }
            worst=std::max(worst,error(blocked,ref));
            std::printf("O7.4.4 N=%d block=%d peak_live=%zu slab=%zu emitted=%zu doubles\n",
                n,width,workspace.peak_live_elements,workspace.peak_output_slab_elements,
                workspace.peak_emitted_tile_elements);
        }
        // O7.4.2: consume raw tiles directly against a symmetric derivative
        // tensor. This is the scalar operation later used by the live
        // derivative primitive, but deliberately has no final AO Gamma
        // reconstruction.
        for(int axis=0;axis<3;++axis)
        {
            std::vector<double> probe(ref.size());
            for(std::size_t p=0;p<probe.size();++p) probe[p]=std::sin(0.17*(axis+1)*(p+1));
            const auto derivative=symmetric(probe,n);
            double tiled_contraction=0.0;
            const auto streamed=for_each_dh_eq47_nonseparable_raw_ao_tile(
                d,c,3,2,[&](int begin,int width,const std::vector<double> &tile)
                -> std::expected<void,std::string>
                {
                    const std::size_t suffix=static_cast<std::size_t>(n)*n*n;
                    for(int local_mu=0;local_mu<width;++local_mu)
                        for(std::size_t q=0;q<suffix;++q)
                            tiled_contraction+=tile[static_cast<std::size_t>(local_mu)*suffix+q]*
                                derivative[(static_cast<std::size_t>(begin+local_mu))*suffix+q];
                    return {};
                });
            if(!streamed || std::abs(tiled_contraction-dot(ref,derivative))>2e-11)
                return false;
        }
        DHRelaxedDifferenceDensity relaxed;
        relaxed.symmetric_mo=Eigen::MatrixXd::Zero(o+v,o+v);
        DHOverlapDensity w; w.symmetric_mo=relaxed.symmetric_mo;
        const Eigen::MatrixXd p=Eigen::MatrixXd::Zero(n,n);
        const auto full=build_dh_eq46_47_two_particle_density(relaxed,d,c,p,0.53);
        const auto compact=build_dh_eq46_47_two_particle_density(relaxed,d,c,p,0.0,DHGammaStorage::ContractionOnly);
        if(!full || !compact) return false;
        worst=std::max({worst,error(full->nonseparable_raw_ao,ref),error(full->nonseparable_symmetric_ao,sym),
            error(full->total_raw_ao,ref),error(full->total_symmetric_ao,sym)});
        bool ok=worst<2e-12 && compact->separable_raw_ao.capacity()==0 &&
            compact->separable_symmetric_ao.capacity()==0 && compact->ground_density_ao==p &&
            compact->nonseparable_raw_ao.capacity()==0 && compact->nonseparable_symmetric_ao.capacity()==0 &&
            compact->total_raw_ao.empty();
        // Check arbitrary raw-tensor pairing, then symmetric derivative
        // pairing through the actual Eq. 33 consumer (zero other channels).
        DHEq33NonXCDerivatives dx;
        for(int axis=0;axis<3;++axis)
        {
            std::vector<double> probe(ref.size());
            for(std::size_t k=0;k<probe.size();++k) probe[k]=std::cos(0.13*(axis+1)*(k+1));
            ok &= std::abs(dot(ref,probe)-dot(full->nonseparable_raw_ao,probe))<2e-11;
            dx.hamiltonian_ao.push_back(p); dx.overlap_ao.push_back(p);
            dx.eri_ao.push_back(symmetric(probe,n));
            ok &= std::abs(dot(ref,dx.eri_ao.back())-dot(sym,dx.eri_ao.back()))<2e-11;
        }
        const auto gradient=build_dh_eq33_non_xc_gradient(relaxed,w,*full,c,dx);
        if(!gradient) return false;
        for(int axis=0;axis<3;++axis)
            ok &= std::abs(gradient->two_electron_nonseparable(axis)-dot(ref,dx.eri_ao[axis]))<2e-11;
        ok &= !build_dh_eq47_nonseparable_raw_ao(d,c,0);
        ok &= !build_dh_eq47_nonseparable_raw_ao(d,c,-1);
        auto bad=d; bad.t_tilde.pop_back(); ok &= !build_dh_eq47_nonseparable_raw_ao(bad,c);
        bad=d; bad.n_occ=0; ok &= !build_dh_eq47_nonseparable_raw_ao(bad,c);
        bad=d; bad.n_virt=-1; ok &= !build_dh_eq47_nonseparable_raw_ao(bad,c);
        bad=d; bad.t_tilde[0]=std::numeric_limits<double>::quiet_NaN();
        ok &= !build_dh_eq47_nonseparable_raw_ao(bad,c);
        auto invalid=c; invalid(0,0)=std::numeric_limits<double>::infinity();
        ok &= !build_dh_eq47_nonseparable_raw_ao(d,invalid);
        ok &= !build_dh_eq47_nonseparable_raw_ao(d,Eigen::MatrixXd::Zero(n,o+v+1));
        ok &= !build_dh_eq47_nonseparable_raw_ao(d,Eigen::MatrixXd::Zero(0,o+v));
        std::printf("O5 N=%d o=%d v=%d scale=%g basis=%d max_raw_sym_error=%.12e %s\n",
            n,o,v,scale,basis_kind,worst,ok?"PASS":"FAIL");
        return ok;
    }

    bool scaling()
    {
        using Clock=std::chrono::steady_clock;
        bool ok=true;
        double previous_literal=0,previous_staged=0;
        for(int n : {4,8,12})
        {
            const int o=n/2,v=n-o;
            const auto d=amplitudes(o,v,0.27);
            const auto c=coefficients(n,n);
            // Warm-up and median of three samples. The fast path is batched
            // eight times per sample for timer resolution. Never assert a
            // hardware-dependent speedup; print the measured scaling ledger.
            const auto warm=build_dh_eq47_nonseparable_raw_ao(d,c);
            if(!warm) return false;
            std::array<double,3> old_times{},new_times{};
            double checksum=0;
            for(int sample=0;sample<3;++sample)
            {
                const auto start=Clock::now();
                const auto ref=literal(d,c);
                const auto middle=Clock::now();
                for(int repeat=0;repeat<8;++repeat)
                {
                    const auto staged=build_dh_eq47_nonseparable_raw_ao(d,c);
                    if(!staged) return false;
                    checksum+=(*staged)[repeat%staged->size()];
                }
                const auto end=Clock::now();
                old_times[sample]=std::chrono::duration<double>(middle-start).count();
                new_times[sample]=std::chrono::duration<double>(end-middle).count()/8;
                ok &= error(*warm,ref)<2e-11;
            }
            std::sort(old_times.begin(),old_times.end()); std::sort(new_times.begin(),new_times.end());
            const double N=n,O=o,V=v;
            const double old_terms=N*N*N*N*O*O*V*V;
            const double staged_macs=N*O*O*V*V+N*N*O*O*V+N*N*N*O*V+N*N*N*N*O;
            std::printf("O5 scaling N=%d literal_terms=%.0f staged_macs=%.0f literal_s=%.9g staged_s=%.9g speedup=%.6g literal_growth=%.6g staged_growth=%.6g checksum=%.12g\n",
                n,old_terms,staged_macs,old_times[1],new_times[1],old_times[1]/new_times[1],
                previous_literal?old_times[1]/previous_literal:0,previous_staged?new_times[1]/previous_staged:0,checksum);
            previous_literal=old_times[1]; previous_staged=new_times[1];
        }
        return ok;
    }

    // O7.7: Measure the workspace owned by the actual four-index block
    // producer, rather than the deliberately dense oracle used to validate
    // its values.  With fixed AO and occupied tiles, the producer holds one
    // [AO, nu, kappa, tau] output slab and at most an old/new pair of the
    // current occupied reduction panel.  This is the exact conservative
    // accounting recorded by DHEq47BlockWorkspace below; no final AO^4
    // tensor belongs to the producer.
    bool bounded_workspace_scaling()
    {
        constexpr int occupied_tile = 2;
        bool ok = true;
        for (int n : {4, 8, 12, 16})
        {
            const int o = n / 2, v = n - o;
            const auto d = amplitudes(o, v, 0.27);
            const auto c = coefficients(n, n);
            const auto reference = build_dh_eq47_nonseparable_raw_ao(d, c);
            if (!reference) return false;
            for (int block : {1, 2, 3})
            {
                std::vector<double> actual(reference->size(), 0.0);
                DHEq47BlockWorkspace workspace;
                const auto streamed = for_each_dh_eq47_nonseparable_raw_ao_block(
                    d, c, occupied_tile, block, block, block, block,
                    [&](int mu0, int mc, int nu0, int nc, int ka0, int kc, int ta0, int tc,
                        const std::vector<double> &tile) -> std::expected<void, std::string>
                    {
                        const std::size_t expected = static_cast<std::size_t>(mc) * nc * kc * tc;
                        if (tile.size() != expected)
                            return std::unexpected("O7.7 Eq. (47) block extent mismatch.");
                        for (int lm = 0; lm < mc; ++lm)
                            for (int ln = 0; ln < nc; ++ln)
                                for (int lk = 0; lk < kc; ++lk)
                                    for (int lt = 0; lt < tc; ++lt)
                                    {
                                        const std::size_t output =
                                            ((static_cast<std::size_t>(mu0 + lm) * n + nu0 + ln) * n + ka0 + lk) * n + ta0 + lt;
                                        const std::size_t input =
                                            ((static_cast<std::size_t>(lm) * nc + ln) * kc + lk) * tc + lt;
                                        actual[output] = tile[input];
                                    }
                        return {};
                    }, &workspace);
                const std::size_t output_slab = static_cast<std::size_t>(n) * block * block * block;
                const std::size_t initial_panel = static_cast<std::size_t>(occupied_tile) * v * o * v;
                const std::size_t tau_panel = static_cast<std::size_t>(occupied_tile) * v * o * block;
                const std::size_t kappa_panel = static_cast<std::size_t>(occupied_tile) * v * block * block;
                const std::size_t nu_panel = static_cast<std::size_t>(occupied_tile) * block * block * block;
                const std::size_t emitted_tile = static_cast<std::size_t>(block) * block * block * block;
                const std::size_t exact_bound = output_slab + std::max({
                    initial_panel + tau_panel, tau_panel + kappa_panel,
                    kappa_panel + nu_panel, emitted_tile});
                const double max_error = error(actual, *reference);
                const bool bounded = workspace.peak_live_elements <= exact_bound &&
                    workspace.peak_output_slab_elements <= output_slab &&
                    workspace.peak_emitted_tile_elements <= emitted_tile &&
                    workspace.peak_live_elements < reference->size();
                ok &= streamed && max_error < 2e-11 && bounded;
                std::printf("O7.7 Eq47 N=%d block=%d reference_bytes=%zu producer_peak_bytes=%zu "
                    "bound_bytes=%zu max_error=%.3e %s\n",
                    n, block, reference->size() * sizeof(double),
                    workspace.peak_live_elements * sizeof(double), exact_bound * sizeof(double),
                    max_error, streamed && max_error < 2e-11 && bounded ? "PASS" : "FAIL");
            }
        }
        return ok;
    }
}

bool test_dh_o5_backtransform()
{
    bool ok=true;
    for(auto shape : {std::array{2,1,1},std::array{5,2,1},std::array{5,1,3},std::array{5,3,2},std::array{3,2,3},std::array{7,2,3}})
        for(double scale : {0.0,0.27,1.0,-0.4}) ok &= fixture(shape[0],shape[1],shape[2],scale);
    ok &= fixture(5,2,3,0.27,1);
    ok &= fixture(5,2,3,0.27,2);
    ok &= scaling();
    ok &= bounded_workspace_scaling();
    return ok;
}
