#include <format>

#include "integrals.h"
#include "integrals/os.h"
#include "io/logging.h"

namespace HartreeFock::Correlation
{

// ── ensure_eri ────────────────────────────────────────────────────────────────

const std::vector<double>& ensure_eri(
    HartreeFock::Calculator&                   calc,
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    std::vector<double>&                       eri_local,
    const std::string&                         tag)
{
    if (!calc._eri.empty())
        return calc._eri;

    const std::size_t nb = calc._shells.nbasis();
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, tag,
        std::format("Building AO ERI tensor ({:.1f} MB)", nb * nb * nb * nb * 8.0 / 1e6));
    eri_local = HartreeFock::ObaraSaika::_compute_2e(shell_pairs, nb);
    return eri_local;
}


// ── transform_eri ─────────────────────────────────────────────────────────────

std::vector<double> transform_eri(
    const std::vector<double>& eri,
    std::size_t                nb,
    const Eigen::MatrixXd&     C1,
    const Eigen::MatrixXd&     C2,
    const Eigen::MatrixXd&     C3,
    const Eigen::MatrixXd&     C4)
{
    const std::size_t n1 = static_cast<std::size_t>(C1.cols());
    const std::size_t n2 = static_cast<std::size_t>(C2.cols());
    const std::size_t n3 = static_cast<std::size_t>(C3.cols());
    const std::size_t n4 = static_cast<std::size_t>(C4.cols());

    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb2;

    // T1[i,ν,λ,σ] = Σ_μ C1(μ,i) * eri[μνλσ]   shape: n1 × nb × nb × nb
    std::vector<double> T1(n1 * nb * nb * nb, 0.0);
    for (std::size_t i = 0; i < n1; ++i)
        for (std::size_t nu = 0; nu < nb; ++nu)
            for (std::size_t lam = 0; lam < nb; ++lam)
                for (std::size_t sig = 0; sig < nb; ++sig)
                    for (std::size_t mu = 0; mu < nb; ++mu)
                        T1[i*nb3 + nu*nb2 + lam*nb + sig] +=
                            C1(mu, i) * eri[mu*nb3 + nu*nb2 + lam*nb + sig];

    // T2[i,a,λ,σ] = Σ_ν C2(ν,a) * T1[i,ν,λ,σ]   shape: n1 × n2 × nb × nb
    std::vector<double> T2(n1 * n2 * nb * nb, 0.0);
    for (std::size_t i = 0; i < n1; ++i)
        for (std::size_t a = 0; a < n2; ++a)
            for (std::size_t lam = 0; lam < nb; ++lam)
                for (std::size_t sig = 0; sig < nb; ++sig)
                    for (std::size_t nu = 0; nu < nb; ++nu)
                        T2[i*n2*nb*nb + a*nb*nb + lam*nb + sig] +=
                            C2(nu, a) * T1[i*nb3 + nu*nb2 + lam*nb + sig];

    T1.clear();
    T1.shrink_to_fit();

    // T3[i,a,j,σ] = Σ_λ C3(λ,j) * T2[i,a,λ,σ]   shape: n1 × n2 × n3 × nb
    std::vector<double> T3(n1 * n2 * n3 * nb, 0.0);
    for (std::size_t i = 0; i < n1; ++i)
        for (std::size_t a = 0; a < n2; ++a)
            for (std::size_t j = 0; j < n3; ++j)
                for (std::size_t sig = 0; sig < nb; ++sig)
                    for (std::size_t lam = 0; lam < nb; ++lam)
                        T3[i*n2*n3*nb + a*n3*nb + j*nb + sig] +=
                            C3(lam, j) * T2[i*n2*nb*nb + a*nb*nb + lam*nb + sig];

    T2.clear();
    T2.shrink_to_fit();

    // out[i,a,j,b] = Σ_σ C4(σ,b) * T3[i,a,j,σ]   shape: n1 × n2 × n3 × n4
    std::vector<double> out(n1 * n2 * n3 * n4, 0.0);
    for (std::size_t i = 0; i < n1; ++i)
        for (std::size_t a = 0; a < n2; ++a)
            for (std::size_t j = 0; j < n3; ++j)
                for (std::size_t b = 0; b < n4; ++b)
                    for (std::size_t sig = 0; sig < nb; ++sig)
                        out[i*n2*n3*n4 + a*n3*n4 + j*n4 + b] +=
                            C4(sig, b) * T3[i*n2*n3*nb + a*n3*nb + j*nb + sig];

    return out;
}


// ── transform_eri_internal ────────────────────────────────────────────────────

std::vector<double> transform_eri_internal(
    const std::vector<double>& eri,
    std::size_t                nb,
    const Eigen::MatrixXd&     C_int)
{
    return transform_eri(eri, nb, C_int, C_int, C_int, C_int);
}

} // namespace HartreeFock::Correlation
