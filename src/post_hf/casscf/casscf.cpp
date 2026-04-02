#include "post_hf/casscf/casscf.h"

#include "io/logging.h"
#include "post_hf/casscf.h"
#include "post_hf/casscf/ci.h"
#include "post_hf/casscf/orbital.h"
#include "post_hf/casscf/rdm.h"
#include "post_hf/casscf/response.h"
#include "post_hf/casscf/strings.h"
#include "post_hf/integrals.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

namespace
{

using HartreeFock::Correlation::CASSCF::CIDeterminantSpace;
using HartreeFock::Correlation::CASSCF::CISigmaApplier;
using HartreeFock::Correlation::CASSCF::CISolveResult;
using HartreeFock::Correlation::CASSCF::ResponseMode;
using HartreeFock::Correlation::CASSCF::RotPair;
using HartreeFock::Correlation::CASSCFInternal::ActiveIntegralCache;
using HartreeFock::Correlation::CASSCFInternal::CIResponseResult;
using HartreeFock::Correlation::CASSCFInternal::CIString;
using HartreeFock::Correlation::CASSCFInternal::NaturalOrbitalData;
using HartreeFock::Correlation::CASSCFInternal::RASParams;
using HartreeFock::Correlation::CASSCFInternal::SymmetryContext;
using HartreeFock::Correlation::CASSCFInternal::compute_root_overlap;
using HartreeFock::Correlation::CASSCFInternal::diagonalize_natural_orbitals;
using HartreeFock::Correlation::CASSCFInternal::kMaxPackedSpatialOrbitals;
using HartreeFock::Correlation::CASSCFInternal::kMaxSeparateSpinOrbitals;
using HartreeFock::Correlation::CASSCFInternal::match_roots_by_max_overlap;

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
    bool response_fallback_used = false;
    bool numeric_newton_attempted = false;
    bool numeric_newton_failed = false;
    bool step_accepted = false;
    double accepted_step_norm = 0.0;
    double predicted_delta = 0.0;
    double actual_delta = 0.0;
    double max_root_delta = 0.0;
};

void reorder_ci_roots(
    Eigen::VectorXd& E,
    Eigen::MatrixXd& V,
    const RootReference* root_ref,
    const std::string& tag,
    bool log_tracking)
{
    using HartreeFock::Logger::logging;
    using HartreeFock::LogLevel;

    if (root_ref == nullptr || !root_ref->valid)
        return;
    if (root_ref->vecs.rows() != V.rows() || root_ref->vecs.cols() == 0 || V.cols() == 0)
        return;

    const int nmatch = std::min<int>(root_ref->vecs.cols(), V.cols());
    const Eigen::MatrixXd overlaps =
        compute_root_overlap(root_ref->vecs.leftCols(nmatch), V.leftCols(nmatch));
    if (overlaps.size() == 0)
        return;

    const std::vector<int> assignment = match_roots_by_max_overlap(overlaps);
    Eigen::VectorXd E_reordered = E;
    Eigen::MatrixXd V_reordered = V;
    std::vector<bool> used_new(static_cast<std::size_t>(V.cols()), false);
    int swaps = 0;
    double min_overlap = 1.0;

    for (int i = 0; i < nmatch; ++i)
    {
        const int j = assignment[i];
        if (j < 0 || j >= V.cols()) continue;
        E_reordered(i) = E(j);
        V_reordered.col(i) = V.col(j);
        used_new[static_cast<std::size_t>(j)] = true;
        min_overlap = std::min(min_overlap, std::abs(overlaps(i, j)));
        if (i != j) ++swaps;
    }

    int next_slot = nmatch;
    for (int j = 0; j < V.cols() && next_slot < V.cols(); ++j)
    {
        if (used_new[static_cast<std::size_t>(j)]) continue;
        E_reordered(next_slot) = E(j);
        V_reordered.col(next_slot) = V.col(j);
        ++next_slot;
    }

    E = std::move(E_reordered);
    V = std::move(V_reordered);

    if (log_tracking && nmatch > 0)
    {
        if (swaps > 0)
            logging(LogLevel::Info, tag + " :",
                    std::format("Root tracking reordered {:d} CI roots (min |overlap| = {:.3f}).",
                                swaps, min_overlap));
        if (min_overlap < 0.7)
            logging(LogLevel::Warning, tag + " :",
                    std::format("Root tracking minimum |overlap| dropped to {:.3f}; state identities may be unstable.",
                                min_overlap));
    }
}

double compute_max_root_delta(const RootReference& previous, const Eigen::VectorXd& current)
{
    if (!previous.valid || previous.energies.size() == 0 || current.size() == 0)
        return 0.0;
    const int n = std::min<int>(previous.energies.size(), current.size());
    double delta = 0.0;
    for (int i = 0; i < n; ++i)
        delta = std::max(delta, std::abs(current(i) - previous.energies(i)));
    return delta;
}

Eigen::MatrixXd as_single_column_matrix(const Eigen::VectorXd& vec)
{
    Eigen::MatrixXd mat(vec.size(), 1);
    mat.col(0) = vec;
    return mat;
}

Eigen::VectorXd single_weight(double weight)
{
    Eigen::VectorXd weights(1);
    weights(0) = weight;
    return weights;
}

void accumulate_weighted_tensor(
    std::vector<double>& destination,
    const std::vector<double>& source,
    double weight)
{
    if (destination.empty())
        destination.assign(source.size(), 0.0);
    for (std::size_t i = 0; i < source.size(); ++i)
        destination[i] += weight * source[i];
}

Eigen::MatrixXd build_root_ci_matrix(const std::vector<StateSpecificData>& roots)
{
    if (roots.empty() || roots.front().ci_vector.size() == 0)
        return Eigen::MatrixXd();

    Eigen::MatrixXd matrix(roots.front().ci_vector.size(), static_cast<int>(roots.size()));
    for (int r = 0; r < static_cast<int>(roots.size()); ++r)
        matrix.col(r) = roots[static_cast<std::size_t>(r)].ci_vector;
    return matrix;
}

Eigen::VectorXd build_root_energy_vector(const std::vector<StateSpecificData>& roots)
{
    Eigen::VectorXd energies(static_cast<int>(roots.size()));
    for (int r = 0; r < static_cast<int>(roots.size()); ++r)
        energies(r) = roots[static_cast<std::size_t>(r)].ci_energy;
    return energies;
}

} // namespace

namespace HartreeFock::Correlation::CASSCF
{

std::expected<void, std::string> run_mcscf_loop(
    HartreeFock::Calculator&                   calc,
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    const std::string&                         tag,
    const RASParams&                           ras)
{
    using HartreeFock::Logger::logging;
    using HartreeFock::LogLevel;

    if (!calc._info._is_converged)
        return std::unexpected(tag + ": requires a converged RHF reference.");
    if (calc._scf._scf != HartreeFock::SCFType::RHF)
        return std::unexpected(tag + ": only RHF reference supported.");

    const auto& as = calc._active_space;
    if (as.nactele <= 0) return std::unexpected(tag + ": nactele must be > 0.");
    if (as.nactorb <= 0) return std::unexpected(tag + ": nactorb must be > 0.");
    if (as.nactele > 2 * as.nactorb)
        return std::unexpected(tag + ": nactele > 2*nactorb is impossible.");

    const int nbasis = static_cast<int>(calc._shells.nbasis());
    const int n_total_elec =
        static_cast<int>(calc._molecule.atomic_numbers.cast<int>().sum()) - calc._molecule.charge;
    if ((n_total_elec - as.nactele) % 2 != 0)
        return std::unexpected(tag + ": (n_elec - nactele) must be even for RHF-based CASSCF.");

    const int n_core = (n_total_elec - as.nactele) / 2;
    const int n_act = as.nactorb;
    const int n_virt = nbasis - n_core - n_act;
    if (n_act > kMaxSeparateSpinOrbitals)
        return std::unexpected(
            std::format("{}: nactorb={} exceeds the 63-orbital limit of the CI bitstring encoding.",
                        tag, n_act));
    if (n_act > kMaxPackedSpatialOrbitals)
        return std::unexpected(
            std::format("{}: nactorb={} exceeds the packed alpha/beta determinant limit ({}).",
                        tag, n_act, kMaxPackedSpatialOrbitals));
    if (n_core < 0) return std::unexpected(tag + ": nactele > total electrons.");
    if (n_virt < 0) return std::unexpected(tag + ": n_core + nactorb > nbasis.");
    if (ras.active && ras.nras1 + ras.nras2 + ras.nras3 != n_act)
        return std::unexpected(tag + ": nras1 + nras2 + nras3 must equal nactorb.");

    const int multiplicity = static_cast<int>(calc._molecule.multiplicity);
    const int n_alpha_act = (as.nactele + (multiplicity - 1)) / 2;
    const int n_beta_act = as.nactele - n_alpha_act;
    if (n_alpha_act < 0 || n_beta_act < 0 || n_alpha_act > n_act || n_beta_act > n_act)
        return std::unexpected(tag + ": invalid active-space electron count.");

    auto nchoose = [](int n, int k) -> long long
    {
        if (k > n || k < 0) return 0;
        if (k == 0 || k == n) return 1;
        long long r = 1;
        for (int i = 0; i < k; ++i)
            r = r * (n - i) / (i + 1);
        return r;
    };
    const long long ci_dim_est = nchoose(n_act, n_alpha_act) * nchoose(n_act, n_beta_act);
    if (ci_dim_est > static_cast<long long>(as.ci_max_dim))
        return std::unexpected(std::format("{}: CI dim ({}) exceeds ci_max_dim ({}).",
                                           tag, ci_dim_est, as.ci_max_dim));

    const int nroots = as.nroots;
    Eigen::VectorXd weights(nroots);
    if (static_cast<int>(as.weights.size()) == nroots)
        for (int k = 0; k < nroots; ++k) weights(k) = as.weights[k];
    else
        weights.setConstant(1.0 / nroots);
    weights /= weights.sum();

    const bool have_sym = !calc._sao_irrep_names.empty()
                       && static_cast<int>(calc._sao_irrep_names.size()) <= 8;
    const bool point_group_is_abelian_for_labels =
        point_group_has_only_1d_irreps(calc._molecule._point_group);
    std::optional<SymmetryContext> sym_ctx;
    std::vector<int> irr_act;
    std::vector<int> all_mo_irr;
    if (have_sym && point_group_is_abelian_for_labels && !calc._info._scf.alpha.mo_symmetry.empty())
    {
        sym_ctx = build_symmetry_context(calc);
        if (!sym_ctx)
            return std::unexpected(tag + ": failed to build an Abelian irrep product table for CI screening.");

        all_mo_irr = map_mo_irreps(calc._info._scf.alpha.mo_symmetry, sym_ctx->names);
        if (std::find(all_mo_irr.begin(), all_mo_irr.end(), -1) != all_mo_irr.end())
            return std::unexpected(tag + ": encountered an MO irrep label missing from the Abelian product table.");

        irr_act.resize(n_act);
        for (int t = 0; t < n_act; ++t)
            irr_act[t] = (n_core + t < static_cast<int>(all_mo_irr.size())) ? all_mo_irr[n_core + t] : -1;
    }
    const bool use_sym = sym_ctx.has_value() && !irr_act.empty();
    const auto target_irr_opt = use_sym
        ? resolve_target_irrep(as.target_irrep, *sym_ctx)
        : std::optional<int>(0);
    if (!target_irr_opt)
        return std::unexpected(std::format("{}: target_irrep '{}' is not present in the Abelian symmetry metadata.",
                                           tag, as.target_irrep));
    const int target_irr = *target_irr_opt;

    Eigen::MatrixXd C = (calc._cas_mo_coefficients.rows() == nbasis &&
                         calc._cas_mo_coefficients.cols() == nbasis)
                      ? calc._cas_mo_coefficients
                      : calc._info._scf.alpha.mo_coefficients;
    if (C.rows() != nbasis || C.cols() != nbasis)
        return std::unexpected(tag + ": MO coefficient matrix has wrong size.");

    std::vector<double> eri_local;
    const std::vector<double>& eri = HartreeFock::Correlation::ensure_eri(
        calc, shell_pairs, eri_local, tag + " :");

    std::vector<CIString> a_strs;
    std::vector<CIString> b_strs;
    build_spin_strings_unfiltered(n_act, n_alpha_act, n_beta_act, a_strs, b_strs);

    const unsigned int nmicro = std::max(1u, as.mcscf_micro_per_macro);
    const ResponseMode configured_response_mode = ResponseMode::DiagonalResponse;
    const bool use_numeric_newton_debug = as.mcscf_debug_numeric_newton;
    const int numeric_newton_pair_limit = 64;
    const int ci_dense_threshold = 500;

    logging(LogLevel::Info, tag + " :",
            std::format("Active space: ({:d}e, {:d}o)  n_core={:d}  n_virt={:d}  CI dim ≤ {:d}",
                        as.nactele, n_act, n_core, n_virt, ci_dim_est));
    logging(LogLevel::Info, tag + " :",
            std::format("Algorithm: approximate macro/micro scaffold with {}  nmicro={:d}",
                        response_mode_name(configured_response_mode), nmicro));
    if (have_sym && !point_group_is_abelian_for_labels)
        logging(LogLevel::Warning, tag + " :",
                std::format("Disabling CI symmetry screening for {} because MO labels come from an Abelian subgroup.",
                            calc._molecule._point_group));
    if (use_sym)
        logging(LogLevel::Info, tag + " :",
                std::format("Target irrep: {}",
                            as.target_irrep.empty() ? sym_ctx->names[target_irr] : as.target_irrep));
    if (nroots > 1)
        logging(LogLevel::Info, tag + " :",
                std::format("State-averaged over {:d} roots", nroots));
    if (use_numeric_newton_debug)
        logging(LogLevel::Info, tag + " :",
                "Numeric Newton debug fallback is enabled for small pair spaces.");
    HartreeFock::Logger::blank();
    HartreeFock::Logger::casscf_header();

    auto evaluate =
        [&](const Eigen::MatrixXd& C_trial,
            const RootReference* root_ref = nullptr,
            bool log_root_tracking = false) -> std::expected<McscfState, std::string>
    {
        McscfState st;
        st.F_I_mo = build_inactive_fock_mo(C_trial, calc._hcore, eri, n_core, nbasis);
        st.h_eff = st.F_I_mo.block(n_core, n_core, n_act, n_act);
        const Eigen::MatrixXd C_act = C_trial.middleCols(n_core, n_act);
        st.ga = HartreeFock::Correlation::transform_eri_internal(eri, nbasis, C_act);
        st.active_integrals = build_active_integral_cache(eri, C_trial, n_core, n_act, nbasis);

        st.ci_space = build_ci_space(
            a_strs, b_strs, ras, st.h_eff, st.ga, n_act,
            irr_act, use_sym ? &*sym_ctx : nullptr, target_irr, ci_dense_threshold);
        if (st.ci_space.dets.empty())
            return std::unexpected(tag + ": no CI determinants of target symmetry.");

        CISolveResult ci_result = solve_ci(
            st.ci_space, a_strs, b_strs, st.h_eff, st.ga, n_act,
            std::min(nroots, static_cast<int>(st.ci_space.dets.size())));
        if (ci_result.energies.size() < nroots)
            return std::unexpected(
                std::format("{}: CI returned {:d} roots (wanted {:d}).",
                            tag, static_cast<int>(ci_result.energies.size()), nroots));

        reorder_ci_roots(ci_result.energies, ci_result.vectors, root_ref, tag, log_root_tracking);
        st.dets = st.ci_space.dets;
        st.H_CI_diag = std::move(ci_result.diagonal);
        st.ci_used_direct_sigma = ci_result.used_direct_sigma;
        st.ci_energies = std::move(ci_result.energies);
        st.ci_vecs = std::move(ci_result.vectors);

        const int nr_used = std::min(nroots, static_cast<int>(st.ci_vecs.cols()));
        st.roots.clear();
        st.roots.reserve(static_cast<std::size_t>(nr_used));
        st.F_A_mo = Eigen::MatrixXd::Zero(nbasis, nbasis);
        st.gamma = Eigen::MatrixXd::Zero(n_act, n_act);
        st.Gamma_vec.clear();
        st.g_orb = Eigen::MatrixXd::Zero(nbasis, nbasis);

        for (int r = 0; r < nr_used; ++r)
        {
            StateSpecificData root;
            root.ci_energy = st.ci_energies(r);
            root.weight = weights(r);
            root.ci_vector = st.ci_vecs.col(r);

            const Eigen::MatrixXd ci_vec = as_single_column_matrix(root.ci_vector);
            root.gamma = compute_1rdm(
                ci_vec, single_weight(1.0), a_strs, b_strs, st.dets, n_act);
            root.Gamma_vec = compute_2rdm(
                ci_vec, single_weight(1.0), a_strs, b_strs, st.dets, n_act);
            root.F_A_mo = build_active_fock_mo(C_trial, root.gamma, eri, n_core, n_act, nbasis);
            root.Q = compute_Q_matrix(st.active_integrals, root.Gamma_vec);
            root.g_orb = compute_orbital_gradient(
                st.F_I_mo, root.F_A_mo, root.Q, root.gamma,
                n_core, n_act, n_virt, all_mo_irr, use_sym);

            st.gamma.noalias() += root.weight * root.gamma;
            accumulate_weighted_tensor(st.Gamma_vec, root.Gamma_vec, root.weight);
            st.F_A_mo.noalias() += root.weight * root.F_A_mo;
            st.g_orb.noalias() += root.weight * root.g_orb;
            st.roots.push_back(std::move(root));
        }

        const Eigen::MatrixXd h_mo = C_trial.transpose() * calc._hcore * C_trial;
        const double E_core = compute_core_energy(h_mo, st.F_I_mo, n_core);
        double E_act = 0.0;
        for (const auto& root : st.roots)
            E_act += root.weight * root.ci_energy;
        st.E_cas = calc._nuclear_repulsion + E_core + E_act;
        st.gnorm = st.g_orb.cwiseAbs().maxCoeff();
        return st;
    };

    std::vector<RotPair> opt_pairs;
    for (const auto& pair : non_redundant_pairs(n_core, n_act, n_virt))
    {
        if (use_sym && !all_mo_irr.empty())
        {
            const int ip = (pair.p < static_cast<int>(all_mo_irr.size())) ? all_mo_irr[pair.p] : -1;
            const int iq = (pair.q < static_cast<int>(all_mo_irr.size())) ? all_mo_irr[pair.q] : -1;
            if (ip >= 0 && iq >= 0 && ip != iq) continue;
        }
        opt_pairs.push_back(pair);
    }

    auto pack_pairs = [&](const Eigen::MatrixXd& M)
    {
        Eigen::VectorXd v = Eigen::VectorXd::Zero(static_cast<int>(opt_pairs.size()));
        for (int k = 0; k < static_cast<int>(opt_pairs.size()); ++k)
            v(k) = M(opt_pairs[k].p, opt_pairs[k].q);
        return v;
    };

    auto unpack_pairs = [&](const Eigen::VectorXd& v)
    {
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nbasis, nbasis);
        for (int k = 0; k < static_cast<int>(opt_pairs.size()); ++k)
        {
            M(opt_pairs[k].p, opt_pairs[k].q) = v(k);
            M(opt_pairs[k].q, opt_pairs[k].p) = -v(k);
        }
        return M;
    };

    auto pack_hessian_diagonal = [&](const McscfState& st)
    {
        const Eigen::MatrixXd F_sum = st.F_I_mo + st.F_A_mo;
        Eigen::VectorXd diag = Eigen::VectorXd::Zero(static_cast<int>(opt_pairs.size()));
        for (int k = 0; k < static_cast<int>(opt_pairs.size()); ++k)
            diag(k) = hess_diag(F_sum, opt_pairs[k].p, opt_pairs[k].q);
        return diag;
    };

    auto build_numeric_newton_step =
        [&](const McscfState& st_cur,
            const Eigen::MatrixXd& C_cur,
            double lm_shift,
            MacroDiagnostics& diag)
    {
        const int npairs = static_cast<int>(opt_pairs.size());
        Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(nbasis, nbasis);
        if (npairs == 0 || npairs > numeric_newton_pair_limit)
            return zero;

        const Eigen::VectorXd g0 = pack_pairs(st_cur.g_orb);
        if (g0.cwiseAbs().maxCoeff() < 1e-10)
            return zero;

        diag.numeric_newton_attempted = true;
        constexpr double fd_step = 5e-4;
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(npairs, npairs);
        const RootReference local_ref{st_cur.ci_energies, st_cur.ci_vecs, true};

        for (int k = 0; k < npairs; ++k)
        {
            Eigen::VectorXd ek = Eigen::VectorXd::Zero(npairs);
            ek(k) = fd_step;
            auto plus_res = evaluate(apply_orbital_rotation(C_cur, unpack_pairs(ek), calc._overlap), &local_ref, false);
            auto minus_res = evaluate(apply_orbital_rotation(C_cur, unpack_pairs(-ek), calc._overlap), &local_ref, false);
            if (!plus_res || !minus_res)
            {
                diag.numeric_newton_failed = true;
                return zero;
            }

            H.col(k) = (pack_pairs(plus_res->g_orb) - pack_pairs(minus_res->g_orb)) / (2.0 * fd_step);
            if (!H.col(k).allFinite())
            {
                diag.numeric_newton_failed = true;
                return zero;
            }
        }

        H = 0.5 * (H + H.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H);
        if (eig.info() != Eigen::Success)
        {
            diag.numeric_newton_failed = true;
            return zero;
        }

        Eigen::VectorXd evals = eig.eigenvalues();
        const double floor = std::max(1e-4, lm_shift);
        for (int i = 0; i < evals.size(); ++i)
            evals(i) = std::max(evals(i), floor);

        Eigen::VectorXd step = -eig.eigenvectors()
                             * evals.cwiseInverse().asDiagonal()
                             * eig.eigenvectors().transpose()
                             * g0;
        if (!step.allFinite())
        {
            diag.numeric_newton_failed = true;
            return zero;
        }

        Eigen::MatrixXd kappa = unpack_pairs(step);
        const double max_elem = kappa.cwiseAbs().maxCoeff();
        if (max_elem > 0.20)
            kappa *= 0.20 / max_elem;

        const double trust_radius = 0.80;
        const double frob = kappa.norm();
        if (frob > trust_radius)
            kappa *= trust_radius / frob;
        return kappa;
    };

    auto cap_orbital_step = [&](Eigen::MatrixXd kappa)
    {
        const double max_elem = kappa.cwiseAbs().maxCoeff();
        if (max_elem > 0.20)
            kappa *= 0.20 / max_elem;

        const double trust_radius = 0.80;
        const double frob = kappa.norm();
        if (frob > trust_radius)
            kappa *= trust_radius / frob;
        return kappa;
    };

    auto build_gradient_fallback_step = [&](const Eigen::MatrixXd& G_trial)
    {
        Eigen::MatrixXd kappa = Eigen::MatrixXd::Zero(nbasis, nbasis);
        for (const auto& pair : opt_pairs)
        {
            const double step = -G_trial(pair.p, pair.q);
            kappa(pair.p, pair.q) = step;
            kappa(pair.q, pair.p) = -step;
        }
        return cap_orbital_step(std::move(kappa));
    };

    auto build_single_pair_probe_step = [&](int pair_index, double signed_magnitude)
    {
        Eigen::MatrixXd kappa = Eigen::MatrixXd::Zero(nbasis, nbasis);
        if (pair_index < 0 || pair_index >= static_cast<int>(opt_pairs.size()))
            return kappa;

        const auto& pair = opt_pairs[static_cast<std::size_t>(pair_index)];
        kappa(pair.p, pair.q) = signed_magnitude;
        kappa(pair.q, pair.p) = -signed_magnitude;
        return kappa;
    };

    double E_prev = 0.0;
    double prev_reported_gnorm = std::numeric_limits<double>::infinity();
    bool converged = false;
    double level_shift = 0.2;
    int rejected_streak = 0;
    int stagnation_streak = 0;
    RootReference root_reference;

    for (unsigned int macro = 1; macro <= as.mcscf_max_iter; ++macro)
    {
        const RootReference previous_root_reference = root_reference;
        auto res = evaluate(C, root_reference.valid ? &root_reference : nullptr, root_reference.valid);
        if (!res) return std::unexpected(res.error());
        auto st_current = std::move(*res);

        MacroDiagnostics diag;
        diag.max_root_delta = compute_max_root_delta(
            previous_root_reference, build_root_energy_vector(st_current.roots));
        root_reference = {
            build_root_energy_vector(st_current.roots),
            build_root_ci_matrix(st_current.roots),
            true};

        const bool e_conv = macro > 1 && std::abs(st_current.E_cas - E_prev) < as.tol_mcscf_energy;
        const bool g_conv = st_current.gnorm < as.tol_mcscf_grad;
        const bool no_orb_rot = (st_current.gnorm == 0.0);
        if ((e_conv && g_conv) || (g_conv && no_orb_rot))
        {
            converged = true;
            break;
        }

        Eigen::MatrixXd G_curr = st_current.g_orb;
        Eigen::MatrixXd kappa_total = Eigen::MatrixXd::Zero(nbasis, nbasis);
        Eigen::MatrixXd kappa_first = Eigen::MatrixXd::Zero(nbasis, nbasis);
        Eigen::MatrixXd kappa_newton = Eigen::MatrixXd::Zero(nbasis, nbasis);
        const bool use_numeric_newton_fallback =
            use_numeric_newton_debug || static_cast<int>(opt_pairs.size()) <= numeric_newton_pair_limit;
        if (use_numeric_newton_fallback)
        {
            kappa_newton = build_numeric_newton_step(st_current, C, level_shift, diag);
            if (diag.numeric_newton_attempted && diag.numeric_newton_failed)
                logging(LogLevel::Warning, tag + " :",
                        "Finite-difference Newton fallback produced an inconsistent column and was discarded.");
        }

        for (unsigned int micro = 0; micro < nmicro; ++micro)
        {
            Eigen::MatrixXd kappa = augmented_hessian_step(
                G_curr, st_current.F_I_mo, st_current.F_A_mo,
                n_core, n_act, n_virt,
                level_shift, 0.20, all_mo_irr, use_sym);
            if (micro == 0)
                kappa_first = kappa;

            const Eigen::MatrixXd dh = delta_h_eff(kappa, st_current.F_I_mo, n_core, n_act);
            const int nr_used = static_cast<int>(st_current.roots.size());
            std::vector<double> Gamma1;
            const CISigmaApplier ci_apply = [&](const Eigen::VectorXd& vec, Eigen::VectorXd& sigma_vec)
            {
                apply_ci_hamiltonian(
                    st_current.ci_space, a_strs, b_strs,
                    st_current.h_eff, st_current.ga, n_act,
                    vec, sigma_vec);
            };

            for (int r = 0; r < nr_used; ++r)
            {
                const auto& root = st_current.roots[static_cast<std::size_t>(r)];
                const Eigen::VectorXd& c0r = root.ci_vector;
                const Eigen::VectorXd sigma =
                    ci_sigma_1body(dh, c0r, a_strs, b_strs, st_current.dets, n_act);

                CIResponseResult response =
                    solve_ci_response_davidson(ci_apply, c0r, root.ci_energy,
                                               st_current.H_CI_diag, sigma, 1e-8, 64, 1e-4);
                if (!response.converged)
                {
                    response = solve_ci_response_single_step(
                        ci_apply, c0r, root.ci_energy, st_current.H_CI_diag, sigma, 1e-4);
                    diag.response_fallback_used = true;
                }

                const Eigen::MatrixXd c1_vec = as_single_column_matrix(response.c1);
                const Eigen::MatrixXd c0_vec = as_single_column_matrix(c0r);
                const auto Gamma1_r = compute_2rdm_bilinear(
                    c1_vec, c0_vec, single_weight(root.weight),
                    a_strs, b_strs, st_current.dets, n_act);
                const auto Gamma1_rt = compute_2rdm_bilinear(
                    c0_vec, c1_vec, single_weight(root.weight),
                    a_strs, b_strs, st_current.dets, n_act);
                accumulate_weighted_tensor(Gamma1, Gamma1_r, 1.0);
                accumulate_weighted_tensor(Gamma1, Gamma1_rt, 1.0);

                diag.max_response_residual =
                    std::max(diag.max_response_residual, response.residual_norm);
                diag.max_response_iterations =
                    std::max(diag.max_response_iterations, response.iterations);
                diag.max_response_regularization =
                    std::max(diag.max_response_regularization, response.max_denominator_regularization);
            }

            const Eigen::MatrixXd Q1 = compute_Q_matrix(st_current.active_integrals, Gamma1);
            Eigen::MatrixXd G_CI = Eigen::MatrixXd::Zero(nbasis, nbasis);
            for (int p = 0; p < nbasis; ++p)
            for (int t = 0; t < n_act; ++t)
            {
                const int q = n_core + t;
                G_CI(p, q) += 2.0 * Q1(p, t);
                G_CI(q, p) -= 2.0 * Q1(p, t);
            }
            G_CI.topLeftCorner(n_core, n_core).setZero();
            G_CI.block(n_core, n_core, n_act, n_act).setZero();
            G_CI.bottomRightCorner(n_virt, n_virt).setZero();

            G_curr = fep1_gradient_update(
                G_curr, kappa, st_current.F_I_mo, st_current.F_A_mo, n_core, n_act, n_virt);
            G_curr += G_CI;
            kappa_total += kappa;
        }

        const double max_k = kappa_total.cwiseAbs().maxCoeff();
        if (max_k > 0.20) kappa_total *= 0.20 / max_k;
        const Eigen::MatrixXd kappa_grad = build_gradient_fallback_step(st_current.g_orb);

        const Eigen::VectorXd g_flat = pack_pairs(st_current.g_orb);
        const Eigen::VectorXd h_flat = pack_hessian_diagonal(st_current);

        bool accepted = false;
        McscfState accepted_state = st_current;
        double best_E = st_current.E_cas;
        double best_g = st_current.gnorm;
        const double merit_weight = 0.10;
        double best_merit = best_E + merit_weight * best_g * best_g;
        Eigen::MatrixXd C_best = C;
        Eigen::MatrixXd best_step = Eigen::MatrixXd::Zero(nbasis, nbasis);
        std::vector<Eigen::MatrixXd> step_candidates;
        if (kappa_newton.cwiseAbs().maxCoeff() > 1e-12)
            step_candidates.push_back(kappa_newton);
        const bool use_gradient_stagnation_fallback = stagnation_streak >= 2;
        if (use_gradient_stagnation_fallback)
        {
            // When the approximate AH/response model keeps accepting vanishingly
            // small steps without reducing the true orbital gradient, fall back
            // to direct orbital-gradient probes and let the fully reevaluated
            // CASSCF energy decide which sign is actually productive.
            if (kappa_grad.cwiseAbs().maxCoeff() > 1e-12)
            {
                step_candidates.push_back(cap_orbital_step(4.0 * kappa_grad));
                step_candidates.push_back(cap_orbital_step(2.0 * kappa_grad));
                step_candidates.push_back(kappa_grad);
                step_candidates.push_back(cap_orbital_step(0.5 * kappa_grad));
                step_candidates.push_back(cap_orbital_step(-4.0 * kappa_grad));
                step_candidates.push_back(cap_orbital_step(-2.0 * kappa_grad));
                step_candidates.push_back(-kappa_grad);
                step_candidates.push_back(cap_orbital_step(-0.5 * kappa_grad));
            }

            // Large virtual spaces can make the full preconditioned gradient
            // step too entangled: a few productive rotations get mixed with many
            // weak directions and the energy screen rejects the whole update.
            // Probe the dominant pair directions individually so the exact
            // CASSCF energy can pick the useful rotations.
            if (g_flat.size() > 0)
            {
                std::vector<int> ranked_pairs(static_cast<std::size_t>(g_flat.size()));
                std::iota(ranked_pairs.begin(), ranked_pairs.end(), 0);
                std::partial_sort(
                    ranked_pairs.begin(),
                    ranked_pairs.begin() + std::min<std::size_t>(4, ranked_pairs.size()),
                    ranked_pairs.end(),
                    [&](int lhs, int rhs)
                    {
                        return std::abs(g_flat(lhs)) > std::abs(g_flat(rhs));
                    });

                for (std::size_t i = 0; i < std::min<std::size_t>(4, ranked_pairs.size()); ++i)
                {
                    const int k = ranked_pairs[i];
                    if (std::abs(g_flat(k)) < 1e-6)
                        break;

                    const double signed_probe = (g_flat(k) >= 0.0) ? -0.20 : 0.20;
                    step_candidates.push_back(build_single_pair_probe_step(k, signed_probe));
                    step_candidates.push_back(build_single_pair_probe_step(k, -signed_probe));
                }
            }
        }
        else
        {
            if (kappa_first.cwiseAbs().maxCoeff() > 1e-12) step_candidates.push_back(kappa_first);
            if (kappa_total.cwiseAbs().maxCoeff() > 1e-12) step_candidates.push_back(kappa_total);
            if (kappa_grad.cwiseAbs().maxCoeff() > 1e-12) step_candidates.push_back(kappa_grad);
            if (kappa_newton.cwiseAbs().maxCoeff() > 1e-12 && kappa_total.cwiseAbs().maxCoeff() > 1e-12)
                step_candidates.push_back(0.5 * (kappa_newton + kappa_total));
            if (kappa_first.cwiseAbs().maxCoeff() > 1e-12 && kappa_total.cwiseAbs().maxCoeff() > 1e-12)
                step_candidates.push_back(0.5 * (kappa_first + kappa_total));
            if (kappa_total.cwiseAbs().maxCoeff() > 1e-12 && kappa_grad.cwiseAbs().maxCoeff() > 1e-12)
                step_candidates.push_back(0.5 * (kappa_total + kappa_grad));
            if (kappa_first.cwiseAbs().maxCoeff() > 1e-12 && kappa_grad.cwiseAbs().maxCoeff() > 1e-12)
                step_candidates.push_back(0.5 * (kappa_first + kappa_grad));
        }

        for (const Eigen::MatrixXd& step_base : step_candidates)
        {
            for (double scale : {1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625})
            {
                Eigen::MatrixXd kappa_try = scale * step_base;
                if (kappa_try.cwiseAbs().maxCoeff() < 1e-12) continue;

                auto trial_res = evaluate(apply_orbital_rotation(C, kappa_try, calc._overlap), &root_reference, false);
                if (!trial_res) continue;

                const auto& trial = *trial_res;
                const double trial_merit = trial.E_cas + merit_weight * trial.gnorm * trial.gnorm;
                const bool merit_improved = trial_merit < best_merit - 1e-10;
                const double flat_energy_window = std::max(1000.0 * as.tol_mcscf_energy, 1e-6);
                const bool gradient_reduced = trial.gnorm < best_g - 1e-12;
                const double gradient_worsen_window =
                    std::max(0.05 * std::max(best_g, 1e-8), 1e-6);
                const bool energy_improved = trial.E_cas < best_E - 1e-10;
                const bool energy_improved_without_hurting_gradient =
                    energy_improved && trial.gnorm <= best_g + gradient_worsen_window;
                const bool stationary_but_better_grad =
                    std::abs(trial.E_cas - best_E) <= flat_energy_window && gradient_reduced;
                if (!energy_improved_without_hurting_gradient &&
                    !merit_improved &&
                    !stationary_but_better_grad) continue;

                accepted = true;
                best_E = trial.E_cas;
                best_g = trial.gnorm;
                best_merit = trial_merit;
                accepted_state = trial;
                C_best = apply_orbital_rotation(C, kappa_try, calc._overlap);
                best_step = kappa_try;
                diag.predicted_delta = quadratic_model_delta(g_flat, h_flat, pack_pairs(best_step));
            }
        }

        if (accepted)
        {
            diag.step_accepted = true;
            diag.accepted_step_norm = best_step.norm();
            diag.actual_delta = best_E - st_current.E_cas;
            C = C_best;
            st_current = std::move(accepted_state);
            root_reference = {
                build_root_energy_vector(st_current.roots),
                build_root_ci_matrix(st_current.roots),
                true};
            level_shift = std::max(1e-3, level_shift * 0.7);
            rejected_streak = 0;
        }
        else
        {
            level_shift = std::min(20.0, level_shift * 2.0);
            ++rejected_streak;
            logging(LogLevel::Warning, tag + " :",
                    "No orbital step candidate improved the merit function; increasing damping and retrying next macroiteration.");
        }

        const double reported_gnorm = st_current.g_orb.cwiseAbs().maxCoeff();
        st_current.gnorm = reported_gnorm;
        const double dE = st_current.E_cas - E_prev;
        E_prev = st_current.E_cas;

        const bool small_energy_change = macro > 1
            && std::abs(dE) < std::max(10.0 * as.tol_mcscf_energy, 1e-8);
        const bool little_gradient_progress = std::isfinite(prev_reported_gnorm)
            && std::abs(reported_gnorm - prev_reported_gnorm)
               < std::max(0.05 * std::max(prev_reported_gnorm, 1e-8), 1e-8);
        const bool accepted_micro_step_plateau =
            diag.step_accepted && diag.accepted_step_norm < 5e-5;
        if ((!diag.step_accepted && rejected_streak >= 2) || (small_energy_change && little_gradient_progress))
            ++stagnation_streak;
        else
            stagnation_streak = 0;
        prev_reported_gnorm = reported_gnorm;

        if (stagnation_streak >= 2)
        {
            level_shift = std::min(50.0, level_shift * 1.5);
            logging(LogLevel::Warning, tag + " :",
                    std::format("Detected stagnation over {:d} macroiterations; increasing damping to {:.3f}.",
                                stagnation_streak, level_shift));
        }

        HartreeFock::Logger::casscf_iteration(
            macro, st_current.E_cas, dE, reported_gnorm, reported_gnorm, diag.accepted_step_norm,
            level_shift, 0.0);

        logging(LogLevel::Info, tag + " :",
                std::format(
                    "Macro {:3d}  mode={:<12}  ci_solver={}\n"
                    "             accepted={:<3}  max_root_dE={:.2e}  step_norm={:.2e}\n"
                    "             predicted_dE={:.2e}  actual_dE={:.2e}  response_resid={:.2e}\n"
                    "             response_iter={:3d}  level_shift={:.2e}",
                    macro,
                    response_mode_name(configured_response_mode),
                    st_current.ci_used_direct_sigma ? "direct-davidson" : "dense",
                    diag.step_accepted ? "yes" : "no",
                    diag.max_root_delta,
                    diag.accepted_step_norm,
                    diag.predicted_delta,
                    diag.actual_delta,
                    diag.max_response_residual,
                    diag.max_response_iterations,
                    level_shift));
        if (diag.response_fallback_used)
            logging(LogLevel::Warning, tag + " :",
                    "CI response Davidson solve did not fully converge for at least one root; using single-step fallback.");

        if (stagnation_streak >= 2 && small_energy_change && accepted_micro_step_plateau)
        {
            logging(LogLevel::Warning, tag + " :",
                    "Treating the stationary orbital plateau as converged: the CASSCF energy and accepted orbital step are flat, while the approximate orbital gradient is no longer improving.");
            converged = true;
            break;
        }

        const bool e_conv_post = macro > 1 && std::abs(dE) < as.tol_mcscf_energy;
        const bool g_conv_post = reported_gnorm < as.tol_mcscf_grad;
        const bool no_orb_rot_post = (reported_gnorm == 0.0);
        if ((e_conv_post && g_conv_post) || (g_conv_post && no_orb_rot_post))
        {
            converged = true;
            break;
        }
    }

    if (!converged)
        return std::unexpected(
            std::format("{}: did not converge in {:d} iterations.", tag, as.mcscf_max_iter));

    HartreeFock::Logger::blank();
    logging(LogLevel::Info, tag + " :", "Converged.");

    auto final_res = evaluate(C, root_reference.valid ? &root_reference : nullptr, false);
    if (!final_res) return std::unexpected(final_res.error());
    const auto& fst = *final_res;

    const NaturalOrbitalData natural_orbitals = diagonalize_natural_orbitals(fst.gamma);
    calc._cas_nat_occ = natural_orbitals.occupations;
    calc._cas_mo_coefficients = C;
    calc._total_energy = fst.E_cas;

    return {};
}

} // namespace HartreeFock::Correlation::CASSCF

namespace HartreeFock::Correlation
{

std::expected<void, std::string> run_casscf(
    HartreeFock::Calculator&                   calc,
    const std::vector<HartreeFock::ShellPair>& shell_pairs)
{
    RASParams ras;
    return CASSCF::run_mcscf_loop(calc, shell_pairs, "CASSCF", ras);
}

std::expected<void, std::string> run_rasscf(
    HartreeFock::Calculator&                   calc,
    const std::vector<HartreeFock::ShellPair>& shell_pairs)
{
    const auto& as = calc._active_space;
    RASParams ras;
    ras.nras1 = as.nras1;
    ras.nras2 = as.nras2;
    ras.nras3 = as.nras3;
    ras.max_holes = as.max_holes;
    ras.max_elec = as.max_elec;
    ras.active = true;
    return CASSCF::run_mcscf_loop(calc, shell_pairs, "RASSCF", ras);
}

} // namespace HartreeFock::Correlation
