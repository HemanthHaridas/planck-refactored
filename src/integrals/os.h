#ifndef HF_OS_H
#define HF_OS_H

#include <tuple>
#include <utility>
#include <vector>
#include <Eigen/Core>

#include "shellpair.h"
#include "base/types.h"

namespace HartreeFock
{
    namespace ObaraSaika
    {
        double _os_1d(const double gamma, const double distPA, const double distPB, const int lA, const int lB);
        std::tuple<double, double> _compute_3d_overlap_kinetic(const HartreeFock::ShellPair &shell_pair);
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> _compute_1e(const std::vector<HartreeFock::ShellPair> &shell_pairs, const std::size_t nbasis);
        Eigen::MatrixXd _compute_nuclear_attraction(const std::vector<HartreeFock::ShellPair> &shell_pairs, const std::size_t nbasis, const HartreeFock::Molecule &molecule);
    }
}

#endif // !HF_OS_H
