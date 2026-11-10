#pragma once

#include "dh_pt2_gradient.h"
#include "response_gmres.h"
#include <chrono>
#include <sstream>
#include <utility>

namespace DFT::Gradient
{
    struct DHZVectorOptions
    {
        double tolerance = 1e-12;
        int restart = 24;
        int max_iterations = 256;
    };
    struct DHZVectorStatistics
    {
        double residual_max_abs = 0.0, elapsed_seconds = 0.0;
        int iterations = 0, action_count = 0, restarts = 0;
        std::size_t krylov_matrix_bytes = 0;
        std::vector<double> true_residuals;
    };
    struct DHZVectorResult : DHZVectorStatistics { Eigen::MatrixXd z_ai; };

    // The one Eq.27 solver for RKS and UKS: A z = b by matrix-free GMRES,
    // preconditioned by the orbital-energy gaps, then a fresh final action.
    // The caller owns the packing and the action. A must be self-adjoint
    // (the physical KS orbital Hessian is), so A z = b and A^T z = b agree.
    // Never builds A and never falls back to a dense solve. An empty pair
    // space (a spin with no ov pairs on both sides) returns an empty z.
    [[nodiscard]] inline std::expected<Eigen::VectorXd,std::string> solve_packed_zvector(
        const DFT::Response::Action &action,const Eigen::VectorXd &b,const Eigen::VectorXd &diagonal,
        const DHZVectorOptions &options,DHZVectorStatistics &stats,const std::string &label)
    {
        const auto start=std::chrono::steady_clock::now();
        if (!action || diagonal.size()!=b.size() || !b.allFinite() ||
            !std::isfinite(options.tolerance) || options.tolerance<=0 || options.restart<=0 || options.max_iterations<=0)
            return std::unexpected(label+": invalid right-hand side, action or solver settings.");
        if (!diagonal.allFinite()) return std::unexpected(label+": nonfinite orbital gaps.");
        if (!b.size()) return Eigen::VectorXd();
        const Eigen::Index n=b.size();
        const DFT::Response::Action counted=[&](const Eigen::VectorXd &x)->std::expected<Eigen::VectorXd,std::string>
        {
            ++stats.action_count;
            if (x.size()!=n || !x.allFinite()) return std::unexpected(label+": invalid action trial.");
            try
            {
                auto value=action(x);
                if (!value) return std::unexpected(value.error());
                if (value->size()!=n || !value->allFinite())
                    return std::unexpected(label+": invalid Eq.27 action result.");
                return value;
            }
            catch(const std::exception &e) { return std::unexpected(label+" action: "+e.what()); }
            catch(...) { return std::unexpected(label+" action: unknown exception."); }
        };
        auto result=DFT::Response::gmres(counted,b,diagonal,options.tolerance,options.restart,options.max_iterations);
        if (!result) return std::unexpected(label+": "+result.error());
        if (!result->converged)
        {
            std::ostringstream error;
            error<<label<<" GMRES did not converge: iterations="<<result->iterations
                 <<" actions="<<stats.action_count<<" residual="<<result->true_residuals.back()
                 <<"; no dense fallback.";
            return std::unexpected(error.str());
        }
        Eigen::VectorXd z=std::move(result->x);
        stats.iterations=result->iterations; stats.restarts=result->restarts;
        stats.krylov_matrix_bytes=result->krylov_matrix_bytes;
        stats.true_residuals=std::move(result->true_residuals);
        // Independently refreshed, unpreconditioned residual after the solve.
        // No preconditioner floor or denominator shift enters this action.
        auto final=counted(z);
        if (!final) return std::unexpected(final.error());
        stats.residual_max_abs=(*final-b).cwiseAbs().maxCoeff();
        const double limit=options.tolerance*std::max(1.0,b.cwiseAbs().maxCoeff());
        if (!std::isfinite(limit) || !std::isfinite(stats.residual_max_abs) || stats.residual_max_abs>limit)
            return std::unexpected(label+": fresh final residual exceeds requested tolerance.");
        stats.elapsed_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        return z;
    }

    // RKS packing a*no+i around the shared solver. Borrows all inputs only
    // for this call; no lambda/reference escapes.
    [[nodiscard]] inline std::expected<DHZVectorResult,std::string> solve_dh_zvector(
        const Eigen::Ref<const Eigen::MatrixXd> &rhs_ai,
        const Eigen::Ref<const Eigen::MatrixXd> &c_occ,
        const Eigen::Ref<const Eigen::MatrixXd> &c_virt,
        const Eigen::Ref<const Eigen::VectorXd> &eps,
        const DHEq41ResponseOperator &response,const DHZVectorOptions &options = {})
    {
        const Eigen::Index no=c_occ.cols(),nv=c_virt.cols(),nao=c_occ.rows();
        if (no<=0 || nv<=0 || nao<=0 || no>std::numeric_limits<int>::max()/nv ||
            c_virt.rows()!=nao || rhs_ai.rows()!=nv || rhs_ai.cols()!=no || eps.size()!=no+nv ||
            !c_occ.allFinite() || !c_virt.allFinite() || !rhs_ai.allFinite() || !eps.allFinite() ||
            !response.coulomb || !response.xc || !std::isfinite(response.exact_exchange) ||
            response.exact_exchange<0 || (response.exact_exchange!=0 && !response.exchange) ||
            !std::isfinite(options.tolerance) || options.tolerance<=0 || options.restart<=0 || options.max_iterations<=0)
            return std::unexpected("DH Z-vector: invalid dimensions, finite values, callbacks or solver settings.");
        const Eigen::Index n=no*nv;
        const auto pack=[no,nv](const Eigen::MatrixXd &m)
        {
            Eigen::VectorXd v(no*nv);
            for (Eigen::Index a=0;a<nv;++a) for (Eigen::Index i=0;i<no;++i) v(a*no+i)=m(a,i);
            return v;
        };
        const auto unpack=[no,nv](const Eigen::VectorXd &v)
        {
            Eigen::MatrixXd m(nv,no);
            for (Eigen::Index a=0;a<nv;++a) for (Eigen::Index i=0;i<no;++i) m(a,i)=v(a*no+i);
            return m;
        };
        const DFT::Response::Action action=[&](const Eigen::VectorXd &x)->std::expected<Eigen::VectorXd,std::string>
        {
            auto value=apply_dh_eq27_hessian(unpack(x),c_occ,c_virt,eps,response);
            if (!value) return std::unexpected(value.error());
            if (value->total_ai.rows()!=nv || value->total_ai.cols()!=no)
                return std::unexpected("DH Z-vector: invalid Eq.27 action result.");
            return pack(value->total_ai);
        };
        Eigen::VectorXd diagonal(n);
        for (Eigen::Index a=0;a<nv;++a) for (Eigen::Index i=0;i<no;++i)
            diagonal(a*no+i)=eps(no+a)-eps(i);
        DHZVectorResult out;
        auto z=solve_packed_zvector(action,-pack(rhs_ai),diagonal,options,out,"DH Z-vector");
        if (!z) return std::unexpected(z.error());
        out.z_ai=unpack(*z);
        return out;
    }
}
