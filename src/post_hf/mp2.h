#ifndef HF_MP2_H
#define HF_MP2_H

#include <Eigen/Core>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::Correlation
{
    struct RMP2NaturalOrbitals
    {
        Eigen::VectorXd occupations;
        Eigen::MatrixXd coefficients_mo;
        Eigen::MatrixXd coefficients_ao;
    };

    // PySCF-aligned restricted MP2 state. This is the public object callers
    // pass into RDM and gradient helpers rather than re-running MP2 implicitly.
    struct RMP2Result
    {
        double e_corr = 0.0;
        double e_corr_ss = 0.0;
        double e_corr_os = 0.0;
        bool converged = true;
        int n_iter = 0;
        int n_occ = 0;
        int n_virt = 0;
        std::vector<int> active_mo;
        std::vector<double> t2; // [i,j,a,b] row-major; empty if not requested
        Eigen::MatrixXd mo_coeff;
        Eigen::VectorXd mo_energy;
        Eigen::VectorXd mo_occ;
    };

    // PySCF-aligned unrestricted MP2 state.
    struct UMP2Result
    {
        double e_corr = 0.0;
        double e_corr_ss = 0.0;
        double e_corr_os = 0.0;
        bool converged = true;
        int n_iter = 0;
        int nocca = 0;
        int noccb = 0;
        int nvira = 0;
        int nvirb = 0;
        std::vector<int> active_mo_alpha;
        std::vector<int> active_mo_beta;
        std::vector<double> t2_aa;
        std::vector<double> t2_ab;
        std::vector<double> t2_bb;
        Eigen::MatrixXd mo_coeff_alpha;
        Eigen::MatrixXd mo_coeff_beta;
        Eigen::VectorXd mo_energy_alpha;
        Eigen::VectorXd mo_energy_beta;
        Eigen::VectorXd mo_occ_alpha;
        Eigen::VectorXd mo_occ_beta;
    };

    // Core MP2 kernels. These are the public entry points for running MP2 and
    // mirror the role of PySCF's kernel methods.
    std::expected<RMP2Result, std::string> rmp2_kernel(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const HartreeFock::OptionsMP2 &options);

    std::expected<UMP2Result, std::string> ump2_kernel(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const HartreeFock::OptionsMP2 &options);

    // Persist a kernel result back onto Calculator so the rest of the codebase
    // can consume the correlation energy and cached amplitudes.
    std::expected<void, std::string> apply_rmp2_result(
        HartreeFock::Calculator &calculator,
        const RMP2Result &result);

    std::expected<void, std::string> apply_ump2_result(
        HartreeFock::Calculator &calculator,
        const UMP2Result &result);

    std::expected<RMP2NaturalOrbitals, std::string> rmp2_make_natural_orbitals(
        const RMP2Result &result);

    std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
    rmp2_gamma1_intermediates(const RMP2Result &result);

    std::expected<Eigen::MatrixXd, std::string>
    rmp2_make_rdm1(const RMP2Result &result, bool ao_repr = false);

    std::expected<std::vector<double>, std::string>
    rmp2_make_rdm2(const RMP2Result &result, bool ao_repr = false);

    std::expected<
        std::pair<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>,
                  std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>,
        std::string>
    ump2_gamma1_intermediates(const UMP2Result &result);

    std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
    ump2_make_rdm1(const UMP2Result &result, bool ao_repr = false);
} // namespace HartreeFock::Correlation

#endif
