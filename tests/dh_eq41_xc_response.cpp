#include <cmath>
#include <cstdio>

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

    bool ok = true;
    // Independent fixed-geometry central difference of Planck's ordinary
    // XC-potential builder.  This is intentionally not the HVP code path.
    for (double h : {1e-3, 3e-4, 1e-4})
    {
        auto plus = vxc(grid, ao, p + h*d, *x, *x);
        auto minus = vxc(grid, ao, p - h*d, *x, *x);
        if (!plus || !minus) { ok = false; continue; }
        const Eigen::MatrixXd fd = (*plus - *minus) / (2.0*h);
        ok &= near(*analytic, fd, 2e-7);
    }
    // The callback must compose directly with the Eq. (41) operator.
    const auto zero = [](const Eigen::Ref<const Eigen::MatrixXd> &q)
        -> std::expected<Eigen::MatrixXd, std::string>
        { return Eigen::MatrixXd::Zero(q.rows(), q.cols()); };
    auto eq41 = DFT::Gradient::make_dh_eq41_response_operator(0.0, zero, {}, *callback);
    if (!eq41) ok = false;
    else
    {
        auto total = eq41->apply(d);
        ok &= total && near(*total, *analytic, 1e-13);
    }
    if (!ok) std::fprintf(stderr, "Eq. 41 XC callback fixed-geometry FD failure\n");
    return ok ? 0 : 1;
}
