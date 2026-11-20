#include "orbital_hessian.h"

#include <limits>

#include "scf/scf.h"

namespace HartreeFock::Correlation
{
    namespace
    {
        using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

        // (e_a - e_i), packed a-major: one spin block.
        Eigen::VectorXd gap_diagonal(const Eigen::VectorXd &eps, int n_occ, int n_virt)
        {
            Eigen::VectorXd d(n_virt * n_occ);
            for (int a = 0; a < n_virt; ++a)
                for (int i = 0; i < n_occ; ++i)
                    d(a * n_occ + i) = eps(n_occ + a) - eps(i);
            return d;
        }

        // Symmetric AO density of a packed rotation block: C_v x C_oᵀ + transpose.
        Eigen::MatrixXd rotation_density(const double *x, const Eigen::MatrixXd &C_occ,
                                         const Eigen::MatrixXd &C_virt)
        {
            const Eigen::MatrixXd xm =
                Eigen::Map<const RowMajor>(x, C_virt.cols(), C_occ.cols());
            const Eigen::MatrixXd D = C_virt * xm * C_occ.transpose();
            return D + D.transpose();
        }

        // C_vᵀ G C_o, packed a-major.
        Eigen::VectorXd pack(const Eigen::MatrixXd &G, const Eigen::MatrixXd &C_occ,
                             const Eigen::MatrixXd &C_virt)
        {
            const RowMajor v = C_virt.transpose() * G * C_occ;
            return Eigen::Map<const Eigen::VectorXd>(v.data(), v.size());
        }
    } // namespace

    OrbitalHessian OrbitalHessian::rhf(const HartreeFock::Calculator &calculator,
                                       const std::vector<HartreeFock::ShellPair> &shell_pairs,
                                       const Eigen::MatrixXd &C, const Eigen::VectorXd &eps, int n_occ)
    {
        OrbitalHessian h(calculator, shell_pairs);
        const int n_virt = static_cast<int>(C.cols()) - n_occ;
        h.restricted_ = true;
        h.Ca_occ_ = C.leftCols(n_occ);
        h.Ca_virt_ = C.middleCols(n_occ, n_virt);
        h.diagonal_ = gap_diagonal(eps, n_occ, n_virt);
        return h;
    }

    OrbitalHessian OrbitalHessian::uhf(const HartreeFock::Calculator &calculator,
                                       const std::vector<HartreeFock::ShellPair> &shell_pairs,
                                       const Eigen::MatrixXd &Ca, const Eigen::MatrixXd &Cb,
                                       const Eigen::VectorXd &eps_a, const Eigen::VectorXd &eps_b,
                                       int n_occ_a, int n_occ_b)
    {
        OrbitalHessian h(calculator, shell_pairs);
        const int nva = static_cast<int>(Ca.cols()) - n_occ_a;
        const int nvb = static_cast<int>(Cb.cols()) - n_occ_b;
        h.restricted_ = false;
        h.Ca_occ_ = Ca.leftCols(n_occ_a);
        h.Ca_virt_ = Ca.middleCols(n_occ_a, nva);
        h.Cb_occ_ = Cb.leftCols(n_occ_b);
        h.Cb_virt_ = Cb.middleCols(n_occ_b, nvb);
        const Eigen::VectorXd da = gap_diagonal(eps_a, n_occ_a, nva);
        const Eigen::VectorXd db = gap_diagonal(eps_b, n_occ_b, nvb);
        h.diagonal_.resize(da.size() + db.size());
        h.diagonal_ << da, db;
        return h;
    }

    Eigen::VectorXd OrbitalHessian::operator()(const Eigen::VectorXd &x) const
    {
        const Eigen::VectorXd nan =
            Eigen::VectorXd::Constant(x.size(), std::numeric_limits<double>::quiet_NaN());
        Eigen::VectorXd y = diagonal_.cwiseProduct(x);
        if (restricted_)
        {
            auto G = SCF::two_electron_fock(calculator_, shell_pairs_,
                                            rotation_density(x.data(), Ca_occ_, Ca_virt_));
            if (!G)
                return nan;
            y += pack(2.0 * *G, Ca_occ_, Ca_virt_);
            return y;
        }
        const Eigen::Index nova = Ca_virt_.cols() * Ca_occ_.cols();
        auto G = SCF::two_electron_fock_uhf(calculator_, shell_pairs_,
                                            rotation_density(x.data(), Ca_occ_, Ca_virt_),
                                            rotation_density(x.data() + nova, Cb_occ_, Cb_virt_));
        if (!G)
            return nan;
        y.head(nova) += pack(G->first, Ca_occ_, Ca_virt_);
        y.tail(y.size() - nova) += pack(G->second, Cb_occ_, Cb_virt_);
        return y;
    }

    Eigen::MatrixXd OrbitalHessian::materialize() const
    {
        const Eigen::Index n = size();
        Eigen::MatrixXd A(n, n);
        for (Eigen::Index col = 0; col < n; ++col)
            A.col(col) = (*this)(Eigen::VectorXd::Unit(n, col));
        return A;
    }
} // namespace HartreeFock::Correlation
