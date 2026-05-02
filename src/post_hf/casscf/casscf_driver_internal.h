#ifndef HF_POSTHF_CASSCF_DRIVER_INTERNAL_H
#define HF_POSTHF_CASSCF_DRIVER_INTERNAL_H

#include <string>
#include <vector>

#include <Eigen/Core>

#include "post_hf/casscf/ci.h"
#include "post_hf/casscf/orbital.h"
#include "post_hf/casscf/response.h"

namespace HartreeFock::Correlation::CASSCF
{
    struct StateSpecificData
    {
        double ci_energy = 0.0;
        double weight = 0.0;
        Eigen::VectorXd ci_vector;
        Eigen::MatrixXd gamma;
        std::vector<double> Gamma_vec;
        Eigen::MatrixXd F_A_mo;
        Eigen::MatrixXd Q;
        Eigen::MatrixXd g_orb;
        ResponseRHSMode response_rhs_mode = ResponseRHSMode::CommutatorOnlyApproximate;
        Eigen::VectorXd response_rhs;
        Eigen::VectorXd c1_response;
        Eigen::VectorXd response_residual;
        std::vector<double> Gamma1_vec;
        Eigen::MatrixXd Q1;
        Eigen::MatrixXd g_ci;
    };

    struct McscfState
    {
        CIDeterminantSpace ci_space;
        Eigen::MatrixXd h_eff;
        std::vector<double> ga;
        Eigen::MatrixXd F_I_mo;
        Eigen::MatrixXd F_A_mo;
        Eigen::MatrixXd gamma;
        Eigen::MatrixXd g_orb;
        std::vector<double> Gamma_vec;
        ActiveIntegralCache active_integrals;
        Eigen::VectorXd H_CI_diag;
        Eigen::VectorXd ci_energies;
        Eigen::MatrixXd ci_vecs;
        std::vector<StateSpecificData> roots;
        std::vector<std::pair<int, int>> dets;
        bool ci_used_direct_sigma = false;
        double E_cas = 0.0;
        double gnorm = 0.0;
        double weighted_root_gnorm = 0.0;
        double max_root_gnorm = 0.0;
    };

    struct RootReference
    {
        Eigen::VectorXd energies;
        Eigen::MatrixXd vecs;
        bool valid = false;
    };

    struct MacroDiagnostics
    {
        double max_response_residual = 0.0;
        int max_response_iterations = 0;
        double max_response_regularization = 0.0;
        bool ci_response_fallback_used = false;
        bool sa_coupled_step_inexact = false;
        bool numeric_newton_attempted = false;
        bool numeric_newton_failed = false;
        bool step_accepted = false;
        double accepted_step_norm = 0.0;
        double predicted_delta = 0.0;
        double max_root_predicted_delta_deviation = 0.0;
        double actual_delta = 0.0;
        double max_root_delta = 0.0;
        std::string accepted_candidate_label = "none";
        double accepted_sa_gnorm = 0.0;
        double accepted_weighted_root_gnorm = 0.0;
        double accepted_max_root_gnorm = 0.0;
    };

    struct WeightedQuadraticModelPrediction
    {
        double weighted_delta = 0.0;
        double max_root_deviation = 0.0;
    };

    struct WeightedRootProbeSignal
    {
        Eigen::VectorXd weighted_abs;
        Eigen::VectorXd weighted_signed;
    };

    struct RootResolvedOrbitalStepSet
    {
        Eigen::MatrixXd weighted;
        std::vector<Eigen::MatrixXd> per_root;
    };

    struct RootResolvedCoupledStepSet
    {
        RootResolvedOrbitalStepSet orbital_steps;
        double max_orbital_residual = 0.0;
        double max_ci_residual = 0.0;
        int max_iterations = 0;
        bool converged = true;
    };

    struct RootResolvedGradientScreen
    {
        double weighted = 0.0;
        double max_root = 0.0;
    };

    struct CandidateStep
    {
        Eigen::MatrixXd step;
        std::string label;
    };

    struct CandidateSelection
    {
        bool accepted = false;
        McscfState state;
        Eigen::MatrixXd coefficients;
        Eigen::MatrixXd step;
        double energy = 0.0;
        double sa_gnorm = 0.0;
        double merit = 0.0;
        std::string label = "none";
        double weighted_root_gnorm = 0.0;
        double max_root_gnorm = 0.0;
        double predicted_delta = 0.0;
        double max_root_predicted_delta_deviation = 0.0;
    };

    void reorder_ci_roots(
        Eigen::VectorXd &E,
        Eigen::MatrixXd &V,
        const RootReference *root_ref,
        const std::string &tag,
        bool log_tracking);

    double compute_max_root_delta(
        const RootReference &previous,
        const Eigen::VectorXd &current);

    void accumulate_weighted_tensor(
        std::vector<double> &destination,
        const std::vector<double> &source,
        double weight);

    void accumulate_weighted_matrix(
        Eigen::MatrixXd &destination,
        const Eigen::MatrixXd &source,
        double weight);

    Eigen::MatrixXd build_root_ci_matrix(
        const std::vector<StateSpecificData> &roots);

    Eigen::VectorXd build_root_energy_vector(
        const std::vector<StateSpecificData> &roots);

    Eigen::MatrixXd build_weighted_root_orbital_gradient(
        const std::vector<StateSpecificData> &roots,
        int nbasis);

    RootResolvedGradientScreen build_root_resolved_gradient_screen(
        const std::vector<StateSpecificData> &roots);

    bool sa_gradient_converged(
        double sa_gnorm,
        double tol);

    bool sa_gradient_progress_flat(
        double sa_gnorm,
        double prev_sa_gnorm);

    WeightedQuadraticModelPrediction build_weighted_root_quadratic_model_prediction(
        const std::vector<StateSpecificData> &roots,
        const Eigen::MatrixXd &F_I_mo,
        const std::vector<RotPair> &pairs,
        const Eigen::MatrixXd &step);

    WeightedRootProbeSignal build_weighted_root_probe_signal(
        const std::vector<StateSpecificData> &roots,
        const std::vector<RotPair> &pairs);

    RootResolvedOrbitalStepSet build_root_resolved_orbital_step_set(
        const std::vector<StateSpecificData> &roots,
        const Eigen::MatrixXd &F_I_mo,
        int nbasis,
        int n_core,
        int n_act,
        int n_virt,
        double level_shift,
        double max_rot,
        const std::vector<int> &mo_irreps,
        bool use_sym);

    void append_candidate_step(
        std::vector<CandidateStep> &candidates,
        Eigen::MatrixXd step,
        const std::string &label);

    void append_root_candidate_steps(
        std::vector<CandidateStep> &candidates,
        const std::vector<Eigen::MatrixXd> &root_steps,
        const std::string &base_label,
        bool cap_steps,
        double max_rot = 0.20);
} // namespace HartreeFock::Correlation::CASSCF

#endif // HF_POSTHF_CASSCF_DRIVER_INTERNAL_H
