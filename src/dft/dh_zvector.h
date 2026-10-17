#pragma once

#include "dh_pt2_gradient.h"
#include "response_gmres.h"
#include <chrono>
#include <sstream>
#include <utility>

namespace DFT::Gradient
{
    enum class DHZVectorBackend { MatrixFreeGMRES, DenseReference };
    struct DHZVectorOptions
    {
        DHZVectorBackend backend = DHZVectorBackend::MatrixFreeGMRES;
        double tolerance = 1e-12;
        int restart = 24;
        int max_iterations = 256;
    };
    struct DHZVectorStatistics
    {
        DHZVectorBackend backend = DHZVectorBackend::MatrixFreeGMRES;
        double residual_max_abs = 0.0, elapsed_seconds = 0.0;
        int iterations = 0, action_count = 0, restarts = 0;
        std::size_t krylov_matrix_bytes = 0, dense_matrix_bytes = 0;
        std::vector<double> true_residuals;
    };
    struct DHZVectorResult : DHZVectorStatistics { Eigen::MatrixXd z_ai; };

    // Synchronous, checked Eq.27 solve AZ=-L. Borrows all inputs only for
    // this call; no lambda/reference escapes. Uses the SAME physical action
    // for GMRES, dense reference columns and final residual. MatrixFreeGMRES
    // never constructs columns or a dense orbital Hessian and never falls
    // back to DenseReference. The abs-gap floor is a preconditioner only.
    [[nodiscard]] inline std::expected<DHZVectorResult,std::string> solve_dh_zvector(
        const Eigen::Ref<const Eigen::MatrixXd> &rhs_ai,
        const Eigen::Ref<const Eigen::MatrixXd> &c_occ,
        const Eigen::Ref<const Eigen::MatrixXd> &c_virt,
        const Eigen::Ref<const Eigen::VectorXd> &eps,
        const DHEq41ResponseOperator &response,const DHZVectorOptions &options = {})
    {
        const auto start=std::chrono::steady_clock::now();
        const Eigen::Index no=c_occ.cols(),nv=c_virt.cols(),nao=c_occ.rows();
        if (no<=0 || nv<=0 || nao<=0 || no>std::numeric_limits<int>::max()/nv ||
            c_virt.rows()!=nao || rhs_ai.rows()!=nv || rhs_ai.cols()!=no || eps.size()!=no+nv ||
            !c_occ.allFinite() || !c_virt.allFinite() || !rhs_ai.allFinite() || !eps.allFinite() ||
            !response.coulomb || !response.xc || !std::isfinite(response.exact_exchange) ||
            response.exact_exchange<0 || (response.exact_exchange!=0 && !response.exchange) ||
            !std::isfinite(options.tolerance) || options.tolerance<=0 || options.restart<=0 || options.max_iterations<=0)
            return std::unexpected("DH Z-vector: invalid dimensions, finite values, callbacks or solver settings.");
        const Eigen::Index n=no*nv;
        DHZVectorResult out; out.backend=options.backend;
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
            ++out.action_count;
            if (x.size()!=n || !x.allFinite()) return std::unexpected("DH Z-vector: invalid action trial.");
            try
            {
                auto value=apply_dh_eq27_hessian(unpack(x),c_occ,c_virt,eps,response);
                if (!value) return std::unexpected(value.error());
                if (value->total_ai.rows()!=nv || value->total_ai.cols()!=no || !value->total_ai.allFinite())
                    return std::unexpected("DH Z-vector: invalid Eq.27 action result.");
                return pack(value->total_ai);
            }
            catch(const std::exception &e) { return std::unexpected(std::string("DH Z-vector action: ")+e.what()); }
            catch(...) { return std::unexpected("DH Z-vector action: unknown exception."); }
        };
        const Eigen::VectorXd b=-pack(rhs_ai);
        Eigen::VectorXd diagonal(n),z;
        for (Eigen::Index a=0;a<nv;++a) for (Eigen::Index i=0;i<no;++i)
            diagonal(a*no+i)=eps(no+a)-eps(i);
        if (!diagonal.allFinite()) return std::unexpected("DH Z-vector: nonfinite orbital gaps.");
        if (options.backend==DHZVectorBackend::MatrixFreeGMRES)
        {
            auto result=DFT::Response::gmres(action,b,diagonal,options.tolerance,options.restart,options.max_iterations);
            if (!result) return std::unexpected("DH Z-vector: "+result.error());
            if (!result->converged)
            {
                std::ostringstream error;
                error<<"DH Z-vector GMRES did not converge: iterations="<<result->iterations
                     <<" actions="<<out.action_count<<" residual="<<result->true_residuals.back()
                     <<"; no dense fallback.";
                return std::unexpected(error.str());
            }
            z=std::move(result->x);
            out.iterations=result->iterations; out.restarts=result->restarts;
            out.krylov_matrix_bytes=result->krylov_matrix_bytes;
            out.true_residuals=std::move(result->true_residuals);
        }
        else if (options.backend==DHZVectorBackend::DenseReference)
        {
            // Explicit oracle only: this allocation is unreachable in the
            // normal backend. No matrix-free result is used as its RHS/seed.
            Eigen::MatrixXd hessian(n,n);
            out.dense_matrix_bytes=sizeof(double)*static_cast<std::size_t>(n)*n;
            for (Eigen::Index col=0;col<n;++col)
            {
                Eigen::VectorXd unit=Eigen::VectorXd::Zero(n); unit(col)=1;
                auto value=action(unit);
                if (!value) return std::unexpected(value.error());
                hessian.col(col)=*value;
            }
            const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(hessian);
            if (qr.rank()!=n) return std::unexpected("DH Z-vector dense reference: rank-deficient Hessian.");
            z=qr.solve(b);
        }
        else return std::unexpected("DH Z-vector: unknown backend.");
        // Independently refreshed, unpreconditioned residual after the solve.
        // No preconditioner floor or denominator shift enters this action.
        auto final=action(z);
        if (!final) return std::unexpected(final.error());
        out.residual_max_abs=(*final-b).cwiseAbs().maxCoeff();
        const double limit=options.tolerance*std::max(1.0,b.cwiseAbs().maxCoeff());
        if (!std::isfinite(limit) || !std::isfinite(out.residual_max_abs) || out.residual_max_abs>limit)
            return std::unexpected("DH Z-vector: fresh final residual exceeds requested tolerance.");
        out.z_ai=unpack(z);
        out.elapsed_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        return out;
    }
}
