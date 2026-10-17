#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include "dft/dh_pt2_gradient.h"

namespace
{
    using namespace DFT::Gradient;
    std::size_t ti(int i,int j,int a,int b,int o,int v)
    { return ((static_cast<std::size_t>(i)*o+j)*v+a)*v+b; }
    std::size_t ei(int p,int q,int r,int s,int n)
    { return ((static_cast<std::size_t>(p)*n+q)*n+r)*n+s; }

    bool close(const Eigen::MatrixXd &a,const Eigen::MatrixXd &b,const char *label)
    {
        if (a.rows()!=b.rows() || a.cols()!=b.cols()) return false;
        const double error=(a-b).cwiseAbs().maxCoeff();
        const bool pass=std::isfinite(error) && error<2e-12*std::max(1.0,b.cwiseAbs().maxCoeff());
        if (!pass) std::fprintf(stderr,"O4 %s error %.12g\n",label,error);
        return pass;
    }

    // Independent scatter oracle: visit every amplitude and differentiate
    // each coefficient, with no reduction packing or GEMM.
    DHPairCoefficientGradient scatter(const DHPT2AmplitudeDensity &d,const std::vector<double> &e)
    {
        const int o=d.n_occ,v=d.n_virt,n=o+v;
        DHPairCoefficientGradient g;
        for (int s=0;s<4;++s) g.slots[s]=Eigen::MatrixXd::Zero(n,s%2?v:o);
        for (int i=0;i<o;++i) for (int j=0;j<o;++j)
            for (int a=0;a<v;++a) for (int b=0;b<v;++b)
            {
                const double t=(1+(i==j))*d.t_tilde[ti(i,j,a,b,o,v)];
                for (int p=0;p<n;++p)
                {
                    g.slots[0](p,i)+=t*e[ei(p,o+a,j,o+b,n)];
                    g.slots[1](p,a)+=t*e[ei(i,p,j,o+b,n)];
                    g.slots[2](p,j)+=t*e[ei(i,o+a,p,o+b,n)];
                    g.slots[3](p,b)+=t*e[ei(i,o+a,j,p,n)];
                }
            }
        g.total=Eigen::MatrixXd::Zero(n,n);
        g.total.leftCols(o)=g.slots[0]+g.slots[2];
        g.total.rightCols(v)=g.slots[1]+g.slots[3];
        return g;
    }

    // Reevaluate the scalar with four independent coefficient matrices.
    // This is deliberately a tiny, unoptimized eight-index FD oracle.
    long double scalar(const DHPT2AmplitudeDensity &d,const std::vector<double> &e,
        const std::array<Eigen::MatrixXd,4> &c)
    {
        const int o=d.n_occ,v=d.n_virt,n=o+v;
        long double value=0;
        for (int i=0;i<o;++i) for (int j=0;j<o;++j)
            for (int a=0;a<v;++a) for (int b=0;b<v;++b)
                for (int p=0;p<n;++p) for (int q=0;q<n;++q)
                    for (int r=0;r<n;++r) for (int s=0;s<n;++s)
                        value+=static_cast<long double>((1+(i==j))*d.t_tilde[ti(i,j,a,b,o,v)])*
                            e[ei(p,q,r,s,n)]*c[0](p,i)*c[1](q,o+a)*c[2](r,j)*c[3](s,o+b);
        return value;
    }

    bool fixture(int o,int v,double scale)
    {
        const int n=o+v;
        HartreeFock::Correlation::RMP2Result pt2;
        pt2.n_occ=o; pt2.n_virt=v;
        pt2.t2.resize(static_cast<std::size_t>(o)*o*v*v);
        std::vector<double> e(static_cast<std::size_t>(n)*n*n*n);
        // Intentionally no ERI, pair, or amplitude permutation symmetry.
        for (std::size_t k=0;k<pt2.t2.size();++k) pt2.t2[k]=0.07*std::sin(0.73*(k+1));
        for (std::size_t k=0;k<e.size();++k) e[k]=0.11*std::cos(0.37*(k+2));
        auto d=build_dh_pt2_amplitude_density(pt2,scale);
        if (!d) return false;
        bool ok=true;
        DHPT2AmplitudeDensity ref;
        ref.n_occ=o; ref.n_virt=v; ref.t_tilde.resize(pt2.t2.size());
        for (int i=0;i<o;++i) for (int j=0;j<o;++j)
            for (int a=0;a<v;++a) for (int b=0;b<v;++b)
            {
                const auto k=ti(i,j,a,b,o,v);
                ref.t_tilde[k]=scale*2.0/(1+(i==j))*(2*pt2.t2[k]-pt2.t2[ti(i,j,b,a,o,v)]);
                ok &= std::abs(ref.t_tilde[k]-d->t_tilde[k])<1e-14;
            }
        Eigen::MatrixXd oo=Eigen::MatrixXd::Zero(o,o),vv=Eigen::MatrixXd::Zero(v,v);
        for (int i=0;i<o;++i) for (int j=0;j<o;++j) for (int k=0;k<o;++k)
            for (int a=0;a<v;++a) for (int b=0;b<v;++b)
                oo(i,j)-=(1+(i==k))*ref.t_tilde[ti(i,k,a,b,o,v)]*pt2.t2[ti(k,j,b,a,o,v)];
        for (int i=0;i<o;++i) for (int j=i;j<o;++j)
            for (int a=0;a<v;++a) for (int b=0;b<v;++b) for (int c=0;c<v;++c)
                vv(a,b)+=ref.t_tilde[ti(i,j,a,c,o,v)]*pt2.t2[ti(i,j,b,c,o,v)]+
                    ref.t_tilde[ti(i,j,c,a,o,v)]*pt2.t2[ti(i,j,c,b,o,v)];
        ok &= close(d->dprime_oo,oo,"Dprime oo");
        ok &= close(d->dprime_vv,vv,"Dprime vv");
        Eigen::MatrixXd dm=Eigen::MatrixXd::Zero(n,n);
        dm.topLeftCorner(o,o)=oo; dm.bottomRightCorner(v,v)=vv;
        ok &= close(d->dprime_mo,dm,"Dprime full");
        const auto gref=scatter(ref,e);
        for (int tile : {1,3,7,128,4096})
        {
            const auto g=build_dh_pair_coefficient_gradient(*d,e,tile);
            if (!g) return false;
            for (int s=0;s<4;++s) ok &= close(g->slots[s],gref.slots[s],"coefficient slot");
            ok &= close(g->total,gref.total,"coefficient total");
        }
        const auto rhs=build_dh_eq40_amplitude_rhs(pt2,*d,e,scale);
        if (!rhs) return false;
        // Retired scan-all-directions expression lives only in this oracle.
        Eigen::MatrixXd rhs_ref=Eigen::MatrixXd::Zero(v,o);
        for (int a=0;a<v;++a) for (int i=0;i<o;++i)
            for (int k=0;k<o;++k) for (int l=0;l<o;++l)
                for (int c=0;c<v;++c) for (int b=0;b<v;++b)
                {
                    const double t=(1+(k==l))*ref.t_tilde[ti(k,l,c,b,o,v)];
                    if(k==i) rhs_ref(a,i)+=t*e[ei(o+a,o+c,l,o+b,n)];
                    if(c==a) rhs_ref(a,i)-=t*e[ei(k,i,l,o+b,n)];
                    if(l==i) rhs_ref(a,i)+=t*e[ei(k,o+c,o+a,o+b,n)];
                    if(b==a) rhs_ref(a,i)-=t*e[ei(k,o+c,l,i,n)];
                }
        ok &= close(rhs->three_external,rhs_ref,"Eq40");
        ok &= close(rhs->total,rhs_ref,"no legacy internal");
        ok &= rhs->three_internal.norm()==0.0;
        const auto w=build_dh_eq47_pair_metric_overlap_density(rhs->pair_coefficients,o,v);
        const auto standalone=build_dh_eq47_pair_metric_overlap_density(pt2,*d,e);
        if (!w || !standalone) return false;
        ok &= close(w->w_oo,-0.5*gref.total.topLeftCorner(o,o),"W oo");
        ok &= close(w->w_vv,-0.5*gref.total.bottomRightCorner(v,v),"W vv");
        ok &= close(w->w_ia,-gref.total.topRightCorner(o,v),"W ov");
        ok &= w->w_ai.norm()==0.0;
        ok &= close(w->w_oo,standalone->w_oo,"reused W oo");
        ok &= close(w->w_vv,standalone->w_vv,"reused W vv");
        ok &= close(w->w_ia,standalone->w_ia,"reused W ov");
        DHEq42_43EnergyWeightedDensity diagonal;
        diagonal.w_oo=w->w_oo; diagonal.w_vv=w->w_vv;
        DHEq44_45EnergyWeightedDensity off;
        off.w_ia=w->w_ia; off.w_ai=w->w_ai;
        const auto overlap=build_dh_overlap_density_adapter(diagonal,off);
        if (!overlap) return false;
        Eigen::MatrixXd s(n,n);
        for (int p=0;p<n;++p) for(int q=0;q<n;++q) s(p,q)=0.2*std::cos(p+q+1);
        Eigen::MatrixXd u=-0.5*s;
        u.topRightCorner(o,v)=-s.topRightCorner(o,v); u.bottomLeftCorner(v,o).setZero();
        const double metric=(gref.total.array()*u.array()).sum();
        ok &= std::abs((overlap->raw_mo.array()*s.array()).sum()-metric)<2e-12;
        ok &= std::abs((overlap->symmetric_mo.array()*s.array()).sum()-metric)<2e-12;
        ok &= std::abs((d->dprime_mo.array()*s.array()).sum()-(dm.array()*s.array()).sum())<2e-12;

        // Independently FD each coefficient slot and the full metric path.
        if (n<=5 && scale!=0)
        {
            const Eigen::MatrixXd identity=Eigen::MatrixXd::Identity(n,n);
            std::array<Eigen::MatrixXd,4> c{identity,identity,identity,identity};
            for (int slot=0;slot<5;++slot)
            {
                double analytic=metric;
                if (slot<4)
                {
                    const Eigen::MatrixXd direction=slot%2?Eigen::MatrixXd(u.rightCols(v)):Eigen::MatrixXd(u.leftCols(o));
                    analytic=(gref.slots[slot].array()*direction.array()).sum();
                }
                std::array<double,2> estimates{};
                int step_index=0;
                for (double h : {1e-3,1e-4})
                {
                    auto plus=c,minus=c;
                    for(int k=0;k<4;++k) if(slot==4 || k==slot)
                    { plus[k]+=h*u; minus[k]-=h*u; }
                    const double fd=static_cast<double>((scalar(ref,e,plus)-scalar(ref,e,minus))/(2*h));
                    estimates[step_index++]=fd;
                    // One moving slot is linear. All four moving slots
                    // give a quartic polynomial, so central FD has only
                    // the cubic coefficient's h^2 truncation error.
                    if (slot<4) ok &= std::abs(fd-analytic)<1e-10;
                }
                const double extrapolated=(100*estimates[1]-estimates[0])/99;
                const double error=std::abs(extrapolated-analytic);
                if (!std::isfinite(error) || error>=1e-10)
                {
                    std::fprintf(stderr,"O4 scalar FD slot=%d actual=%.15g analytic=%.15g error=%.3e\n",
                        slot,extrapolated,analytic,error);
                    ok=false;
                }
            }
        }
        ok &= !build_dh_pair_coefficient_gradient(*d,e,0);
        ok &= !build_dh_pair_coefficient_gradient(*d,e,-1);
        auto bad=*d; bad.t_tilde.pop_back();
        ok &= !build_dh_pair_coefficient_gradient(bad,e);
        auto invalid=rhs->pair_coefficients;
        invalid.total(0,0)=std::numeric_limits<double>::quiet_NaN();
        ok &= !build_dh_eq47_pair_metric_overlap_density(invalid,o,v);
        e.pop_back(); ok &= !build_dh_pair_coefficient_gradient(*d,e);
        std::printf("O4 o=%d v=%d scale=%g slots/Dprime/RHS/raw-W/metric/tile checks %s\n",o,v,scale,ok?"PASS":"FAIL");
        return ok;
    }
}

bool test_dh_o4_contractions()
{
    bool ok=true;
    for (auto shape : {std::array{1,1},std::array{1,3},std::array{3,1},std::array{2,3},std::array{3,2},std::array{6,5}})
        for(double scale : {0.0,0.27,1.0,-0.4}) ok &= fixture(shape[0],shape[1],scale);
    return ok;
}
