#include "dft/dh_probe_gmres.h"
#include <iostream>
#include <limits>
#include <stdexcept>

int main()
{
    using namespace DFT::Driver::Probe;
    int failures = 0;
    const auto check = [&](bool ok, const char *message)
    { if (!ok) { ++failures; std::cerr << "FAIL " << message << '\n'; } };
    Eigen::MatrixXd a(4, 4);
    a << 2.0, 0.3, -0.1, 0.0,
         0.3, -1.7, 0.2, 0.1,
         -0.1, 0.2, 3.1, -0.4,
         0.0, 0.1, -0.4, 1.2;
    Eigen::VectorXd known(4);
    known << 0.2, -0.4, 0.7, 0.1;
    const Action action = [&](const Eigen::VectorXd &x) -> std::expected<Eigen::VectorXd, std::string>
    { return Eigen::VectorXd(a * x); };
    auto result = gmres(action, a * known, a.diagonal());
    check(result && result->converged && (result->x - known).norm() < 1e-10,
        "indefinite Hessian solve");
    a(0, 3) = 0.7; // deliberately nonsymmetric: no hidden SPD assumption
    result = gmres(action, a * known, a.diagonal());
    check(result && result->converged && (result->x - known).norm() < 1e-10,
        "nonsymmetric action solve");
    a = Eigen::MatrixXd::Identity(4, 4);
    a(0, 1) = 0.05;
    a(1, 2) = -0.07;
    a(2, 3) = 0.03;
    result = gmres(action, a * known, a.diagonal(), 1e-12, 2, 100);
    check(result && result->converged && result->iterations > 2 && (result->x - known).norm() < 1e-10,
        "restart path");
    result = gmres(action, Eigen::VectorXd::Zero(4), a.diagonal());
    check(result && result->converged && result->iterations == 0, "zero RHS");
    result = gmres(action, a * known, a.diagonal(), 1e-14, 1, 1);
    check(result && !result->converged && result->iterations == 1, "iteration limit is failure");
    const Action zero = [](const Eigen::VectorXd &x) -> std::expected<Eigen::VectorXd, std::string>
    { return Eigen::VectorXd::Zero(x.size()).eval(); };
    result = gmres(zero, known, a.diagonal());
    // A zero Arnoldi matrix may make Eigen's projected QR produce a
    // nonfinite iterate, rejected as an error before the breakdown return.
    // Both error and explicit nonconvergence satisfy the failure contract;
    // neither may be accepted as a converged solution.
    check(!result || !result->converged, "singular breakdown is not success");
    std::cout << "O3 singular action outcome: "
              << (result ? "nonconverged" : result.error()) << '\n';
    const Action wrong_size = [](const Eigen::VectorXd &) -> std::expected<Eigen::VectorXd, std::string>
    { return Eigen::VectorXd::Zero(1).eval(); };
    check(!gmres(wrong_size, known, a.diagonal()), "wrong-sized action rejected");
    const Action error = [](const Eigen::VectorXd &) -> std::expected<Eigen::VectorXd, std::string>
    { return std::unexpected("injected XC failure"); };
    result = gmres(error, known, a.diagonal());
    check(!result && result.error() == "injected XC failure", "callback error propagated");
    const Action nan = [](const Eigen::VectorXd &x) -> std::expected<Eigen::VectorXd, std::string>
    { return Eigen::VectorXd::Constant(x.size(), std::numeric_limits<double>::quiet_NaN()).eval(); };
    check(!gmres(nan, known, a.diagonal()), "nonfinite action rejected");
    check(!gmres(action, known, a.diagonal(), -1.0), "invalid tolerance rejected");
    const Action throws = [](const Eigen::VectorXd &) -> std::expected<Eigen::VectorXd, std::string>
    { throw std::runtime_error("injected action exception"); };
    result=gmres(throws,known,a.diagonal());
    check(!result && result.error().find("injected action exception")!=std::string::npos,
        "action exceptions propagated");
    // Larger than restart, with no dense matrix even in the oracle. The
    // off-diagonal term forces multiple restarted cycles at tight tolerance.
    for (int n : {128,256})
    {
        int calls=0;
        const Action banded=[&](const Eigen::VectorXd &x)->std::expected<Eigen::VectorXd,std::string>
        {
            ++calls;
            Eigen::VectorXd y=2*x;
            y.head(x.size()-1)+=.35*x.tail(x.size()-1);
            y.tail(x.size()-1)-=.2*x.head(x.size()-1);
            return y;
        };
        const Eigen::VectorXd exact=Eigen::VectorXd::LinSpaced(n,-.7,.9);
        const auto b=banded(exact); calls=0;
        auto large=gmres(banded,*b,Eigen::VectorXd::Constant(n,2),1e-13,4,256);
        const std::size_t expected_bytes=sizeof(double)*(n*(2*4+1)+(4+1)*4);
        check(large && large->converged && large->restarts>0 &&
              (large->x-exact).cwiseAbs().maxCoeff()<1e-12 &&
              large->action_count==calls && calls==1+2*large->iterations &&
              large->krylov_matrix_bytes==expected_bytes && expected_bytes<sizeof(double)*n*n,
              "large restarted solve, tighter tolerance, action count and bounded storage");
        if (large) std::cout<<"O3 GMRES n="<<n<<" iterations="<<large->iterations
            <<" actions="<<calls<<" restarts="<<large->restarts
            <<" matrix_bytes="<<large->krylov_matrix_bytes
            <<" residual="<<large->true_residuals.back()<<'\n';
    }
    if (!failures) std::cout << "PASS DH probe GMRES controls and failure paths\n";
    return failures ? 1 : 0;
}
