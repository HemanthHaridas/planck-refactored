#include "uks_level_shift.h"
#include <algorithm>
#include <cmath>

namespace DFT::Driver
{
    std::expected<Eigen::MatrixXd, std::string> build_uks_level_shift_matrix(
        const Eigen::Ref<const Eigen::MatrixXd> &s,
        const Eigen::Ref<const Eigen::MatrixXd> &p, double shift)
    {
        if (!std::isfinite(shift) || shift<0 || s.rows()==0 || s.rows()!=s.cols() ||
            p.rows()!=s.rows() || p.cols()!=s.cols() || !s.allFinite() || !p.allFinite() ||
            (s-s.transpose()).cwiseAbs().maxCoeff()>1e-10 ||
            (p-p.transpose()).cwiseAbs().maxCoeff()>1e-10)
            return std::unexpected("UKS level shift: invalid overlap, spin density or nonnegative finite shift");
        Eigen::MatrixXd out=shift*(s-s*p*s);
        if (!out.allFinite()) return std::unexpected("UKS level shift: nonfinite projector");
        return out;
    }

    std::expected<void, std::string> validate_uks_unshifted_orbitals(
        const Eigen::Ref<const Eigen::MatrixXd> &f,
        const Eigen::Ref<const Eigen::MatrixXd> &s,
        const Eigen::Ref<const Eigen::MatrixXd> &c,
        const Eigen::Ref<const Eigen::VectorXd> &eps, double tolerance)
    {
        const Eigen::Index n=s.rows();
        if (n==0 || s.cols()!=n || f.rows()!=n || f.cols()!=n ||
            c.rows()!=n || c.cols()!=n || eps.size()!=n || !f.allFinite() ||
            !s.allFinite() || !c.allFinite() || !eps.allFinite() ||
            !std::isfinite(tolerance) || tolerance<=0)
            return std::unexpected("UKS unshifted handoff: invalid orbital dimensions or values");
        const Eigen::MatrixXd fc=f*c, sce=s*c*eps.asDiagonal();
        const Eigen::MatrixXd metric=c.transpose()*s*c;
        if (!fc.allFinite() || !sce.allFinite() || !metric.allFinite())
            return std::unexpected("UKS unshifted handoff: nonfinite eigenpair or metric product");
        const double scale=std::max({1.0,fc.cwiseAbs().maxCoeff(),sce.cwiseAbs().maxCoeff()});
        const Eigen::MatrixXd residual=fc-sce;
        if (!residual.allFinite() || residual.cwiseAbs().maxCoeff()>tolerance*scale ||
            (metric-Eigen::MatrixXd::Identity(n,n)).cwiseAbs().maxCoeff()>tolerance)
            return std::unexpected("UKS unshifted handoff: orbitals/energies are not canonical for the physical Fock");
        return {};
    }
}
