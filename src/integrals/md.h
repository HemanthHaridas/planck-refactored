#ifndef HF_OS_H
#define HF_OS_H

#include <vector>
#include <Eigen/Core>

#include "shellpair.h"
#include "shellquartet.h"

Eigen::MatrixXd _overlap_1d(const std::vector <HartreeFock::ShellPair> &shell_pairs);

#endif // !HF_OS_H
