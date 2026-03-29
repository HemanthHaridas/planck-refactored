#ifndef HF_LOGGING_H
#define HF_LOGGING_H

#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <mutex>

#include "base/types.h"
#include "base/tables.h"

namespace HartreeFock
{
    enum LogLevel
    {
        Info,
        Warning,
        Error,
        Matrix,
//        Cycle
    };

    namespace Logger
    {
        inline std::mutex log_mutex;
        inline thread_local int silence_depth = 0;

        constexpr const char *info_prefix   = "[Planck][INF]    ";
        constexpr const char *error_prefix  = "[Planck][ERR]    ";
        constexpr const char *matrix_prefix = "[Planck][MAT]    ";
        constexpr const char *warn_prefix   = "[Planck][WARN]   ";
//        constexpr const char *scf_prefix    = "[Planck][SCF]    ";

        inline bool is_silenced() noexcept
        {
            return silence_depth > 0;
        }

        class ScopedSilence
        {
        public:
            ScopedSilence() noexcept
            {
                ++silence_depth;
            }

            ~ScopedSilence()
            {
                --silence_depth;
            }

            ScopedSilence(const ScopedSilence&) = delete;
            ScopedSilence& operator=(const ScopedSilence&) = delete;
        };
    
        template<typename... Args>
        static void logging(LogLevel level, const std::string& label, Args&&... message)
        {
            if (is_silenced())
                return;

            std::lock_guard<std::mutex> lock(log_mutex);

            // Prefix selection
            const char* prefix = nullptr;
            switch(level)
            {
                case Info:      prefix = info_prefix;   break;
                case Warning:   prefix = warn_prefix;   break;
                case Error:     prefix = error_prefix;  break;
                case Matrix:    prefix = matrix_prefix; break;
//                case Cycle:       prefix = scf_prefix;    break;
            }
            
            // Timestamp
            auto now = std::chrono::system_clock::now();
            std::time_t now_time = std::chrono::system_clock::to_time_t(now);

            std::tm local_tm{};
        #ifdef _WIN32
            localtime_s(&local_tm, &now_time);
        #else
            local_tm = *std::localtime(&now_time);
        #endif

            std::ostream& out_stream =
                (level == Info || level == Error) ? std::cout : std::cerr;

            // Print header
            out_stream << "["
                       << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S")
                       << "] "
                       << std::setw(20) << std::left << prefix
                       << std::setw(30) << std::left << label;

            // Variadic message printing
            if constexpr (sizeof...(Args) > 0)
                (out_stream << ... << message);

            out_stream << '\n';
        }
    
        // SCF Information
        inline void scf_header()
        {
            if (is_silenced())
                return;

            std::lock_guard<std::mutex> lock(log_mutex);

//            logging(Cycle, "Self-Consistent Field Iterations","");
            std::cout << std::string(110, '-') << "\n"
                      << std::setw(6)  << "Iter"
                      << std::setw(20) << "Energy"
                      << std::setw(15) << "DeltaE"
                      << std::setw(15) << "RMS(D)"
                      << std::setw(15) << "Max(D)"
                      << std::setw(15) << "DIIS Err"
                      << std::setw(12) << "Damp"
                      << std::setw(12) << "Time(s)"
                      << "\n"
                      << std::string(110, '-') << "\n";
        }
    
        inline void scf_iteration(std::size_t iter, double energy, double deltaE, double rmsD, double maxD, double diis_error, double damping, double time_sec)
        {
            if (is_silenced())
                return;

            std::lock_guard<std::mutex> lock(log_mutex);

            std::cout << std::setw(6)  << iter
                      << std::setw(20) << std::setprecision(10) << energy
                      << std::setw(15) << deltaE
                      << std::setw(15) << std::setprecision(3) << std::scientific << rmsD
                      << std::setw(15) << std::setprecision(3) << std::scientific << maxD
                      << std::setw(15) << std::setprecision(3) << std::scientific << diis_error
                      << std::setw(12) << std::fixed << std::setprecision(3) << damping
                      << std::setw(12) << std::fixed << std::setprecision(3) << time_sec
                      << "\n";
        }

        inline void casscf_header()
        {
            if (is_silenced())
                return;

            std::lock_guard<std::mutex> lock(log_mutex);

            std::cout << std::string(110, '-') << "\n"
                      << std::setw(6)  << "Iter"
                      << std::setw(20) << "Energy"
                      << std::setw(15) << "DeltaE"
                      << std::setw(15) << "Grad"
                      << std::setw(15) << "MaxGrad"
                      << std::setw(15) << "Step Err"
                      << std::setw(12) << "Damp"
                      << std::setw(12) << "Time(s)"
                      << "\n"
                      << std::string(110, '-') << "\n";
        }

        inline void casscf_iteration(std::size_t iter, double energy, double deltaE, double grad, double max_grad, double step_error, double damping, double time_sec)
        {
            if (is_silenced())
                return;

            std::lock_guard<std::mutex> lock(log_mutex);

            std::cout << std::setw(6)  << iter
                      << std::setw(20) << std::setprecision(10) << energy
                      << std::setw(15) << deltaE
                      << std::setw(15) << std::setprecision(3) << std::scientific << grad
                      << std::setw(15) << std::setprecision(3) << std::scientific << max_grad
                      << std::setw(15) << std::setprecision(3) << std::scientific << step_error
                      << std::setw(12) << std::fixed << std::setprecision(3) << damping
                      << std::setw(12) << std::fixed << std::setprecision(3) << time_sec
                      << "\n";
        }
    
        inline void mo_header(bool with_symmetry = false)
        {
            if (is_silenced())
                return;

            std::lock_guard<std::mutex> lock(log_mutex);
            const int W = with_symmetry ? 43 : 31;
            std::cout << std::string(W, '-') << "\n"
                      << std::setw(6)  << std::right << "MO";
            if (with_symmetry)
                std::cout << std::setw(12) << std::right << "Symmetry";
            std::cout << std::setw(25) << std::right << "Energy (Eh)"
                      << "\n"
                      << std::string(W, '-') << "\n";
        }

        inline void mo_energies(const Eigen::VectorXd& mo_energies,
                                 const std::size_t n_electrons,
                                 const std::vector<std::string>& symm = {})
        {
            if (is_silenced())
                return;

            std::lock_guard<std::mutex> lock(log_mutex);

            const std::size_t n_occ = n_electrons / 2;
            const std::size_t homo = n_occ - 1;
            const std::size_t lumo = n_occ;

            for (std::size_t i = 0; i < static_cast<std::size_t>(mo_energies.size()); i++)
            {
                std::string label = "";
                if (i == homo) label = "  <-- HOMO";
                if (i == lumo) label = "  <-- LUMO";

                std::cout << std::setw(6)  << std::right << (i + 1);
                if (!symm.empty() && i < symm.size())
                    std::cout << std::setw(12) << std::right << symm[i];
                std::cout << std::setw(25) << std::setprecision(6) << std::right << mo_energies(i)
                          << label
                          << "\n";
            }
        }

        inline void mo_energies_uhf(const Eigen::VectorXd& eps,
                                     const std::size_t n_occ,
                                     const std::vector<std::string>& symm = {})
        {
            if (is_silenced())
                return;

            std::lock_guard<std::mutex> lock(log_mutex);
            const std::size_t homo = (n_occ > 0) ? n_occ - 1 : 0;
            const std::size_t lumo = n_occ;
            for (std::size_t i = 0; i < static_cast<std::size_t>(eps.size()); ++i)
            {
                std::string label = "";
                if (i == homo && n_occ > 0) label = "  <-- HOMO";
                if (i == lumo)              label = "  <-- LUMO";
                std::cout << std::setw(6)  << std::right << (i + 1);
                if (!symm.empty() && i < symm.size())
                    std::cout << std::setw(12) << std::right << symm[i];
                std::cout << std::setw(25) << std::setprecision(6) << std::right << eps(i)
                          << label << "\n";
            }
        }

        inline void converged_energy(double energy_hartree, double nuclear_repulsion)
        {
            if (is_silenced())
                return;

            std::lock_guard<std::mutex> lock(log_mutex);

            const double electronic = energy_hartree - nuclear_repulsion;

            constexpr int LW = 32;  // label column
            constexpr int VW = 20;  // value column

            std::cout << std::string(LW + VW * 3, '-') << "\n"
                      << std::setw(LW) << std::left  << "  Quantity"
                      << std::setw(VW) << std::right << "Hartree"
                      << std::setw(VW) << std::right << "eV"
                      << std::setw(VW) << std::right << "kcal/mol"
                      << "\n"
                      << std::string(LW + VW * 3, '-') << "\n"
                      << std::fixed << std::setprecision(10)
                      << std::setw(LW) << std::left  << "  Electronic Energy"
                      << std::setw(VW) << std::right << electronic
                      << std::setw(VW) << std::right << electronic * HARTREE_TO_EV
                      << std::setw(VW) << std::right << electronic * HARTREE_TO_KCALMOL
                      << "\n"
                      << std::setw(LW) << std::left  << "  Nuclear Repulsion"
                      << std::setw(VW) << std::right << nuclear_repulsion
                      << std::setw(VW) << std::right << nuclear_repulsion * HARTREE_TO_EV
                      << std::setw(VW) << std::right << nuclear_repulsion * HARTREE_TO_KCALMOL
                      << "\n"
                      << std::string(LW + VW * 3, '-') << "\n"
                      << std::setw(LW) << std::left  << "  Total Energy"
                      << std::setw(VW) << std::right << energy_hartree
                      << std::setw(VW) << std::right << energy_hartree * HARTREE_TO_EV
                      << std::setw(VW) << std::right << energy_hartree * HARTREE_TO_KCALMOL
                      << "\n"
                      << std::string(LW + VW * 3, '-') << "\n";
        }

        inline void correlation_energy(const double E_scf, const double E_corr)
        {
            if (is_silenced())
                return;

            std::lock_guard<std::mutex> lock(log_mutex);
            const double E_total = E_scf + E_corr;
            constexpr int LW = 32;
            constexpr int VW = 20;
            std::cout << std::setw(LW) << std::left  << "  Correlation Energy"
                      << std::fixed << std::setprecision(10)
                      << std::setw(VW) << std::right << E_corr
                      << std::setw(VW) << std::right << E_corr * HARTREE_TO_EV
                      << std::setw(VW) << std::right << E_corr * HARTREE_TO_KCALMOL
                      << "\n"
                      << std::string(LW + VW * 3, '-') << "\n"
                      << std::setw(LW) << std::left  << "  Total MP2 Energy"
                      << std::setw(VW) << std::right << E_total
                      << std::setw(VW) << std::right << E_total * HARTREE_TO_EV
                      << std::setw(VW) << std::right << E_total * HARTREE_TO_KCALMOL
                      << "\n"
                      << std::string(LW + VW * 3, '-') << "\n";
        }

        inline void mp2_natural_orbitals(const Eigen::VectorXd& occupations,
                                         const Eigen::MatrixXd& coefficients_mo,
                                         double coeff_threshold = 1e-2)
        {
            if (is_silenced())
                return;

            std::lock_guard<std::mutex> lock(log_mutex);

            std::cout << "\n  MP2 Natural Orbital Occupancies :\n";
            for (Eigen::Index i = 0; i < occupations.size(); ++i)
                std::cout << std::format("    NO {:3d}     {:.6f}\n",
                                         static_cast<int>(i + 1), occupations(i));

            std::cout << "\n  MP2 Natural Orbitals (canonical MO expansion) :\n";
            for (Eigen::Index no = 0; no < coefficients_mo.cols(); ++no)
            {
                std::cout << std::format("    NO {:3d} =", static_cast<int>(no + 1));
                bool printed = false;
                for (Eigen::Index mo = 0; mo < coefficients_mo.rows(); ++mo)
                {
                    const double coeff = coefficients_mo(mo, no);
                    if (std::abs(coeff) < coeff_threshold)
                        continue;

                    std::cout << std::format(" {:+.6f}*MO{}", coeff, static_cast<int>(mo + 1));
                    printed = true;
                }
                if (!printed)
                    std::cout << " <all coefficients below threshold>";
                std::cout << "\n";
            }
        }

        inline void blank()
        {
            if (is_silenced())
                return;

            std::cout << '\n';
        }

        inline void scf_footer()
        {
            if (is_silenced())
                return;

            std::lock_guard<std::mutex> lock(log_mutex);
            std::cout << std::string(110, '-') << "\n";
        }

        // ── CASSCF / RASSCF printers ────────────────────────────────────────

        inline void casscf_summary(double E_rhf, double E_cas,
                                    const Eigen::VectorXd& nat_occ,
                                    int nroots, int nactorb)
        {
            if (is_silenced())
                return;

            std::lock_guard<std::mutex> lock(log_mutex);
            constexpr int LW = 32;
            constexpr int VW = 20;
            std::cout << std::string(LW + VW * 3, '-') << "\n"
                      << "  CASSCF Natural Occupations :\n";
            // nat_occ is sorted descending (reversed eigenvalue order)
            const int norb = static_cast<int>(nat_occ.size());
            for (int k = 0; k < norb; ++k)
                std::cout << std::format("    MO {:3d}     {:.6f}\n",
                                         nactorb - norb + k + 1, nat_occ(k));

            const double E_corr = E_cas - E_rhf;
            std::cout << std::string(LW + VW * 3, '-') << "\n"
                      << std::setw(LW) << std::left  << "  CASSCF Correlation Energy"
                      << std::fixed << std::setprecision(10)
                      << std::setw(VW) << std::right << E_corr
                      << std::setw(VW) << std::right << E_corr * HARTREE_TO_EV
                      << std::setw(VW) << std::right << E_corr * HARTREE_TO_KCALMOL
                      << "\n"
                      << std::string(LW + VW * 3, '-') << "\n"
                      << std::setw(LW) << std::left  << "  CASSCF Total Energy"
                      << std::setw(VW) << std::right << E_cas
                      << std::setw(VW) << std::right << E_cas * HARTREE_TO_EV
                      << std::setw(VW) << std::right << E_cas * HARTREE_TO_KCALMOL
                      << "\n"
                      << std::string(LW + VW * 3, '-') << "\n";

            if (nroots > 1)
                std::cout << "  (State-averaged over " << nroots << " roots)\n";
        }
    }
}

// Generic template declaration
template <typename EnumType>
const inline std::string map_enum(EnumType value);

// Specialization for Verbosity
template <>
const inline std::string map_enum<HartreeFock::Verbosity>(HartreeFock::Verbosity v)
{
    switch (v)
    {
    case HartreeFock::Verbosity::Silent:
        return "Silent";
    case HartreeFock::Verbosity::Minimal:
        return "Minimal";
    case HartreeFock::Verbosity::Normal:
        return "Normal";
    case HartreeFock::Verbosity::Verbose:
        return "Verbose";
    case HartreeFock::Verbosity::Debug:
        return "Debug";
    }
    return "Unknown";
}

// Specialization for Integral Method
template <>
const inline std::string map_enum<HartreeFock::IntegralMethod>(HartreeFock::IntegralMethod m)
{
    switch (m)
    {
    case HartreeFock::IntegralMethod::ObaraSaika:
        return "Obara-Saika";
    case HartreeFock::IntegralMethod::RysQuadrature:
        return "Rys Quadrature";
    case HartreeFock::IntegralMethod::Auto:
        return "Auto";
    }
    return "Unknown";
}

// Specialization for Calculation Type
template <>
const inline std::string map_enum<HartreeFock::CalculationType>(HartreeFock::CalculationType c)
{
    switch (c)
    {
    case HartreeFock::CalculationType::SinglePoint:
        return "Single Point Calculation";
    case HartreeFock::CalculationType::Gradient:
        return "Analytic Gradient";
    case HartreeFock::CalculationType::GeomOpt:
        return "Geometry Optimization";
    case HartreeFock::CalculationType::Frequency:
        return "Frequency Calculation";
    case HartreeFock::CalculationType::GeomOptFrequency:
        return "Geometry Optimization + Frequency";
    case HartreeFock::CalculationType::ImaginaryFollow:
        return "Imaginary Mode Follow";
    }
    return "Unknown";
}

// Specialization for SCF Type
template <>
const inline std::string map_enum<HartreeFock::SCFType>(HartreeFock::SCFType s)
{
    switch (s)
    {
        case HartreeFock::SCFType::RHF:
            return "RHF";
        case HartreeFock::SCFType::UHF:
            return "UHF";
    }
    return "Unknown";
}

// Specialization for SCF Mode
template <>
const inline std::string map_enum<HartreeFock::SCFMode>(HartreeFock::SCFMode s)
{
    switch (s)
    {
        case HartreeFock::SCFMode::Conventional:
            return "Conventional";
        case HartreeFock::SCFMode::Direct:
            return "Direct";
        case HartreeFock::SCFMode::Auto:
            return "Auto";
    }
    return "Unknown";
}

template <>
const inline std::string map_enum<HartreeFock::DFTGridQuality>(HartreeFock::DFTGridQuality g)
{
    switch (g)
    {
        case HartreeFock::DFTGridQuality::Coarse:
            return "Coarse";
        case HartreeFock::DFTGridQuality::Normal:
            return "Normal";
        case HartreeFock::DFTGridQuality::Fine:
            return "Fine";
        case HartreeFock::DFTGridQuality::UltraFine:
            return "UltraFine";
    }
    return "Unknown";
}

template <>
const inline std::string map_enum<HartreeFock::XCExchangeFunctional>(HartreeFock::XCExchangeFunctional x)
{
    switch (x)
    {
        case HartreeFock::XCExchangeFunctional::Custom:
            return "Custom";
        case HartreeFock::XCExchangeFunctional::Slater:
            return "Slater";
        case HartreeFock::XCExchangeFunctional::B88:
            return "B88";
        case HartreeFock::XCExchangeFunctional::PW91:
            return "PW91";
        case HartreeFock::XCExchangeFunctional::PBE:
            return "PBE";
    }
    return "Unknown";
}

template <>
const inline std::string map_enum<HartreeFock::XCCorrelationFunctional>(HartreeFock::XCCorrelationFunctional c)
{
    switch (c)
    {
        case HartreeFock::XCCorrelationFunctional::Custom:
            return "Custom";
        case HartreeFock::XCCorrelationFunctional::VWN5:
            return "VWN5";
        case HartreeFock::XCCorrelationFunctional::LYP:
            return "LYP";
        case HartreeFock::XCCorrelationFunctional::P86:
            return "P86";
        case HartreeFock::XCCorrelationFunctional::PW91:
            return "PW91";
        case HartreeFock::XCCorrelationFunctional::PBE:
            return "PBE";
    }
    return "Unknown";
}

#endif // !HF_LOGGING_H
