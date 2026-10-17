#include "dft/uks_level_shift.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <limits>
#include <source_location>

namespace
{
    bool ok=true;
    void check(bool passed, std::source_location at=std::source_location::current())
    {
        if (!passed) { std::cerr << "FAIL UKS level shift line " << at.line() << '\n'; ok=false; }
    }
}

int main()
{
    using namespace DFT::Driver;
    constexpr int n=4;
    constexpr double lambda=0.3;
    Eigen::Matrix4d a;
    a << 1.2,0.2,-0.1,0.3, 0.1,1.5,0.3,0.2,
         0.2,-0.1,1.3,0.1, 0.1,0.2,-0.2,1.4;
    const Eigen::Matrix4d s=a.transpose()*a;
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> overlap_solver(s);
    check(overlap_solver.info()==Eigen::Success);
    const Eigen::Matrix4d x=overlap_solver.operatorInverseSqrt();
    Eigen::Matrix4d rotation=Eigen::Matrix4d::Identity();
    rotation(0,0)=0.8; rotation(0,2)=-0.6;
    rotation(2,0)=0.6; rotation(2,2)=0.8;
    const Eigen::Matrix4d c=x*rotation;
    Eigen::Vector4d ea,eb;
    ea << -1.2,-0.7,0.2,0.8;
    eb << -1.0,0.1,0.5,0.9;
    Eigen::Vector4d returned_a,returned_b;
    for (int spin=0;spin<2;++spin)
    {
        int no=spin ? 1 : 2;
        const Eigen::Vector4d eps=spin ? eb : ea;
        const Eigen::Matrix4d p=c.leftCols(no)*c.leftCols(no).transpose();
        const Eigen::Matrix4d f=s*c*eps.asDiagonal()*c.transpose()*s;
        const auto shift=build_uks_level_shift_matrix(s,p,lambda);
        check(shift.has_value());
        if (!shift) return 1;
        Eigen::Matrix4d expected=Eigen::Matrix4d::Zero();
        expected.bottomRightCorner(n-no,n-no).diagonal().setConstant(lambda);
        check((c.transpose()*(*shift)*c-expected).norm()<1e-12);
        check(((*shift)*c.leftCols(no)).norm()<1e-12);
        check(std::abs((p.array()*shift->array()).sum())<1e-12);
        // The overlap-metric commutator is unchanged by the shift.
        check(((*shift)*p*s-s*p*(*shift)).norm()<1e-12);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> shifted_solver(x.transpose()*(f+*shift)*x);
        const Eigen::Matrix4d shifted_c=x*shifted_solver.eigenvectors();
        Eigen::Vector4d shifted_eps=eps;
        shifted_eps.tail(n-no).array()+=lambda;
        check((shifted_solver.eigenvalues()-shifted_eps).norm()<1e-12);
        check(!validate_uks_unshifted_orbitals(f,s,shifted_c,shifted_solver.eigenvalues()));
        check(validate_uks_unshifted_orbitals(f+*shift,s,shifted_c,shifted_solver.eigenvalues()).has_value());

        // Finalization diagonalizes physical F; not the DIIS extrapolate or
        // shifted F. The returned spectrum is the input to UMP2 denominators.
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> final_solver(x.transpose()*f*x);
        const Eigen::Matrix4d final_c=x*final_solver.eigenvectors();
        check(validate_uks_unshifted_orbitals(f,s,final_c,final_solver.eigenvalues()).has_value());
        check((final_solver.eigenvalues()-eps).norm()<1e-12);
        check((final_c.leftCols(no)*final_c.leftCols(no).transpose()-p).norm()<1e-12);
        (spin ? returned_b : returned_a)=final_solver.eigenvalues();

        const auto zero=build_uks_level_shift_matrix(s,p,0);
        check(zero && zero->norm()==0);

        // Off the fixed point, merely subtracting lambda from eigenvalues
        // cannot repair shifted coefficients. This must fail the handoff.
        Eigen::Matrix4d perturbation=Eigen::Matrix4d::Zero();
        perturbation(0,n-1)=perturbation(n-1,0)=0.04;
        const Eigen::Matrix4d changing_f=f+s*c*perturbation*c.transpose()*s;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> off_shell(x.transpose()*(changing_f+*shift)*x);
        Eigen::Vector4d subtracted=off_shell.eigenvalues();
        subtracted.tail(n-no).array()-=lambda;
        check(!validate_uks_unshifted_orbitals(changing_f,s,x*off_shell.eigenvectors(),subtracted));
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> actual(x.transpose()*changing_f*x);
        check(validate_uks_unshifted_orbitals(changing_f,s,x*actual.eigenvectors(),actual.eigenvalues()).has_value());
    }
    const double physical_delta=ea(0)+eb(0)-ea(2)-eb(1);
    const double returned_delta=returned_a(0)+returned_b(0)-returned_a(2)-returned_b(1);
    check(std::abs(physical_delta-returned_delta)<1e-12);
    check(std::abs(0.17*0.17/physical_delta-0.17*0.17/returned_delta)<1e-12);
    check(std::abs(0.17*0.17/(physical_delta-2*lambda)-0.17*0.17/returned_delta)>1e-4);

    const auto empty=build_uks_level_shift_matrix(s,Eigen::Matrix4d::Zero(),lambda);
    const auto full=build_uks_level_shift_matrix(s,c*c.transpose(),lambda);
    check(empty && (*empty-lambda*s).norm()<1e-12);
    check(full && full->norm()<1e-12);
    check(!build_uks_level_shift_matrix(s,Eigen::Matrix4d::Zero(),-0.3));
    check(!build_uks_level_shift_matrix(s,Eigen::Matrix4d::Zero(),std::numeric_limits<double>::quiet_NaN()));
    check(!build_uks_level_shift_matrix(s,Eigen::Matrix3d::Zero(),lambda));
    Eigen::Matrix4d invalid=s; invalid(0,1)+=0.1;
    check(!build_uks_level_shift_matrix(invalid,Eigen::Matrix4d::Zero(),lambda));
    invalid.setZero(); invalid(0,1)=0.1;
    check(!build_uks_level_shift_matrix(s,invalid,lambda));
    invalid.setConstant(std::numeric_limits<double>::infinity());
    check(!build_uks_level_shift_matrix(s,invalid,lambda));
    check(!validate_uks_unshifted_orbitals(s,s,c,ea,-1));
    check(!validate_uks_unshifted_orbitals(s,s,2*c,ea));
    if (ok) std::cout << "PASS UKS level shift: metric projector, unshifted handoff and PT2 denominators\n";
    return ok ? 0 : 1;
}
