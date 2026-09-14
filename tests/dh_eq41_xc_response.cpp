#include <cmath>
#include <cstdio>
#include <limits>

#include "dft/analytic_hessian.h"
#include "dft/dh_pt2_gradient.h"
#include "dft/ks_matrix.h"
#include "dft/xc_grid.h"

namespace
{
    bool near(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b, double tol)
    { return a.rows() == b.rows() && a.cols() == b.cols() && (a - b).norm() < tol; }

    std::expected<DFT::XC::Functional, std::string> functional(const char *name)
    {
        auto id = DFT::XC::functional_id(name);
        if (!id) return std::unexpected(id.error());
        return DFT::XC::Functional::create(*id, DFT::XC::Spin::Unpolarized);
    }

    std::expected<Eigen::MatrixXd, std::string> vxc(
        const DFT::MolecularGrid &grid, const DFT::AOGridEvaluation &ao,
        const Eigen::MatrixXd &density, const DFT::XC::Functional &x,
        const DFT::XC::Functional &c)
    {
        auto values = DFT::evaluate_xc_on_grid(grid, ao, density, x, c);
        if (!values) return std::unexpected(values.error());
        auto matrix = DFT::assemble_xc_matrix(grid, ao, *values);
        if (!matrix) return std::unexpected(matrix.error());
        return matrix->alpha;
    }

    bool check_o2_cache()
    {
        bool ok=true;
        for (const int points:{2,17,64})
        {
            DFT::MolecularGrid grid;
            grid.points=Eigen::MatrixXd::Zero(points,4);
            DFT::AOGridEvaluation ao;
            ao.values.resize(points,2); ao.grad_x.resize(points,2);
            ao.grad_y.resize(points,2); ao.grad_z.resize(points,2);
            for (int p=0;p<points;++p)
            {
                grid.points(p,3)=.3+.01*p;
                for (int m=0;m<2;++m)
                {
                    ao.values(p,m)=.8+.1*std::cos(.7*(p+1)*(m+1));
                    ao.grad_x(p,m)=.07*std::sin(.3*(p+1)*(m+1));
                    ao.grad_y(p,m)=.09*std::cos(.4*(p+1)*(m+1));
                    ao.grad_z(p,m)=.03*std::sin(.6*(p+1)*(m+1));
                }
            }
            Eigen::MatrixXd ground(2,2); ground << .9,.2,.2,.7;
            Eigen::MatrixXd trial(2,2); trial << .3,-.1,-.1,.25;
            for (const auto *name:{"lda_x","gga_x_pbe","hyb_gga_xc_b2plyp"})
            {
                auto x=functional(name);
                auto c=functional(std::string(name)=="lda_x"?"lda_c_vwn":"gga_c_pbe");
                if (!x || !c) return false;
                auto kernel=DFT::Driver::prepare_rks_xc_kernel(grid,ao,ground,*x,*c);
                if (!kernel) { std::fprintf(stderr,"O2 cache: %s\n",kernel.error().c_str()); return false; }
                double max_error=0,fd_error=0;
                for (double scale:{0.,1.,-.4,2.})
                {
                    Eigen::MatrixXd q=scale*trial;
                    auto cached=kernel->apply(q);
                    auto reference=DFT::Driver::compute_analytic_xc_hessian_vector_product(grid,ao,ground,q,*x,*c);
                    if (!cached || !reference) return false;
                    max_error=std::max(max_error,(*cached-*reference).norm());
                    ok &= near(*cached,*reference,1e-12);
                    for (double h:{1e-3,1e-4})
                    {
                        auto plus=vxc(grid,ao,ground+h*q,*x,*c),minus=vxc(grid,ao,ground-h*q,*x,*c);
                        if (!plus || !minus) return false;
                        const Eigen::MatrixXd difference=(*plus-*minus)/(2*h);
                        fd_error=std::max(fd_error,(*cached-difference).norm());
                        ok &= near(*cached,difference,2e-7);
                    }
                }
                auto expected=kernel->apply(trial);
                auto snapshot_copy=*kernel;
                auto moved=std::move(*kernel);
                ok &= !kernel->apply(trial).has_value();
                // Destroy/mutate every caller-owned dependency. The cache
                // must still represent the original snapshot. A newly built
                // cache must instead reflect the changed geometry/density.
                auto changed_grid=grid; changed_grid.points.col(3)*=1.4;
                auto changed_ao=ao; changed_ao.values*=.93;
                auto changed=DFT::Driver::prepare_rks_xc_kernel(changed_grid,changed_ao,1.1*ground,*x,*c);
                if (!expected || !changed) return false;
                auto changed_action=changed->apply(trial);
                auto changed_reference=DFT::Driver::compute_analytic_xc_hessian_vector_product(
                    changed_grid,changed_ao,1.1*ground,trial,*x,*c);
                if (!changed_action || !changed_reference) return false;
                ok &= near(*changed_action,*changed_reference,1e-12) && !near(*changed_action,*expected,1e-8);
                // Callback made from local dependencies must survive their scope.
                auto detached=[&]() -> std::expected<DFT::Gradient::DHResponseFn,std::string>
                {
                    auto local_grid=grid; auto local_ao=ao; auto local_density=ground;
                    auto lx=functional(name);
                    auto lc=functional(std::string(name)=="lda_x"?"lda_c_vwn":"gga_c_pbe");
                    if (!lx || !lc) return std::unexpected("local functional");
                    auto fn=DFT::Gradient::make_dh_eq41_xc_response_callback(
                        {&local_grid,&local_ao,local_density,&*lx,&*lc});
                    local_grid.points.setZero(); local_ao.values.setZero(); local_density.setZero();
                    return fn;
                }();
                auto again=moved.apply(trial),shared=snapshot_copy.apply(trial);
                if (!detached || !again || !shared) return false;
                auto detached_value=(*detached)(trial);
                ok &= detached_value && near(*detached_value,*expected,1e-12) &&
                      near(*again,*expected,1e-14) && near(*shared,*expected,1e-14);
                Eigen::MatrixXd bad=trial; bad(0,0)=std::numeric_limits<double>::quiet_NaN();
                ok &= !moved.apply(bad).has_value();
                ok &= !moved.apply(Eigen::MatrixXd::Zero(3,3)).has_value();
                std::printf("O2 XC cache %s G=%d retained_bytes=%zu cached/reference=%.6e FD=%.6e\n",
                    name,points,moved.storage_bytes(),max_error,fd_error);
            }
        }
        return ok;
    }
}

int main()
{
    // A non-degenerate fixed geometry/AO grid.  Both densities are arbitrary
    // symmetric AO matrices; no SCF or nuclear displacement enters this test.
    DFT::MolecularGrid grid;
    grid.points.resize(2, 4);
    grid.points << 0.0, 0.0, 0.0, 0.7,
                   0.3, -0.2, 0.1, 1.1;
    DFT::AOGridEvaluation ao;
    ao.values.resize(2, 2); ao.values << 1.1, -0.4, 0.6, 0.9;
    ao.grad_x = Eigen::MatrixXd::Zero(2, 2);
    ao.grad_y = Eigen::MatrixXd::Zero(2, 2);
    ao.grad_z = Eigen::MatrixXd::Zero(2, 2);
    Eigen::Matrix2d p; p << 0.9, 0.2, 0.2, 0.7;
    Eigen::Matrix2d d; d << 0.3, -0.1, -0.1, 0.25;

    auto x = functional("lda_x");
    if (!x) { std::fprintf(stderr, "functional setup failed: %s\n", x.error().c_str()); return 1; }
    // The same LDA functional in both slots deliberately exercises the exact
    // additive x+c convention used by the production KS potential and HVP.
    auto callback = DFT::Gradient::make_dh_eq41_xc_response_callback(
        {&grid, &ao, p, &*x, &*x});
    if (!callback) { std::fprintf(stderr, "Eq. 41 XC callback setup failed: %s\n", callback.error().c_str()); return 1; }
    auto analytic = (*callback)(d);
    if (!analytic) { std::fprintf(stderr, "Eq. 41 XC callback failed: %s\n", analytic.error().c_str()); return 1; }

    bool ok = check_o2_cache();
    // Independent fixed-geometry central difference of Planck's ordinary
    // XC-potential builder.  This is intentionally not the HVP code path.
    for (double h : {1e-3, 3e-4, 1e-4})
    {
        auto plus = vxc(grid, ao, p + h*d, *x, *x);
        auto minus = vxc(grid, ao, p - h*d, *x, *x);
        if (!plus || !minus)
        {
            std::fprintf(stderr, "XC potential FD h=%.1e failed: %s\n", h,
                (!plus ? plus.error() : minus.error()).c_str());
            ok = false;
            continue;
        }
        const Eigen::MatrixXd fd = (*plus - *minus) / (2.0*h);
        std::printf("XC callback FD h=%.1e norm_error=%.6e\n", h, (*analytic-fd).norm());
        ok &= near(*analytic, fd, 2e-7);
    }
    // The callback is the physical dV_XC/dP. The closed-shell Eq. (41)
    // adjoint applies its factor four outside that callback (doc Section 5).
    // Independently compare the composed result with four times the FD too.
    const auto zero = [](const Eigen::Ref<const Eigen::MatrixXd> &q)
        -> std::expected<Eigen::MatrixXd, std::string>
        { return Eigen::MatrixXd::Zero(q.rows(), q.cols()); };
    auto eq41 = DFT::Gradient::make_dh_eq41_response_operator(0.0, zero, {}, *callback);
    if (!eq41) ok = false;
    else
    {
        auto total = eq41->apply(d);
        ok &= total && near(*total, 4.0 * *analytic, 1e-13);
        auto plus = vxc(grid, ao, p + 1e-4*d, *x, *x);
        auto minus = vxc(grid, ao, p - 1e-4*d, *x, *x);
        if (total && plus && minus)
        {
            const Eigen::MatrixXd expected = 4.0 * (*plus-*minus)/(2e-4);
            std::printf("Eq. 41 XC adjoint FD norm_error=%.6e\n", (*total-expected).norm());
            ok &= near(*total, expected, 8e-7);
        }
        else ok = false;
    }
    if (!ok) std::fprintf(stderr, "Eq. 41 XC callback fixed-geometry FD failure\n");
    return ok ? 0 : 1;
}
