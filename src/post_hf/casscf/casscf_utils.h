#ifndef HF_POSTHF_CASSCF_UTILS_H
#define HF_POSTHF_CASSCF_UTILS_H

#include <Eigen/Core>

namespace HartreeFock::Correlation::CASSCF
{
    inline Eigen::MatrixXd as_single_column_matrix(const Eigen::VectorXd &vec)
    {
        Eigen::MatrixXd mat(vec.size(), 1);
        mat.col(0) = vec;
        return mat;
    }

    inline Eigen::VectorXd single_weight(double weight)
    {
        Eigen::VectorXd weights(1);
        weights(0) = weight;
        return weights;
    }
} // namespace HartreeFock::Correlation::CASSCF

#endif // HF_POSTHF_CASSCF_UTILS_H
