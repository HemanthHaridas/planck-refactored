#ifndef HF_LOGGING_H
#define HF_LOGGING_H

#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <mutex>

#include "base/types.h"

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
        namespace
        {
            std::mutex log_mutex;

            constexpr const char *info_prefix   = "[Planck][INF]    ";
            constexpr const char *error_prefix  = "[Planck][ERR]    ";
            constexpr const char *matrix_prefix = "[Planck][MAT]    ";
            constexpr const char *warn_prefix   = "[Planck][WARN]   ";
//            constexpr const char *scf_prefix    = "[Planck][SCF]    ";
        }
    
        template<typename... Args>
        static void logging(LogLevel level, const std::string& label, Args&&... message)
        {
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
            (out_stream << ... << message);

            out_stream << '\n';
        }
    
        // SCF Information
        inline void scf_header()
        {
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
            std::lock_guard<std::mutex> lock(log_mutex);

            std::cout << std::setw(6)  << iter
                      << std::setw(20) << std::scientific << energy
                      << std::setw(15) << deltaE
                      << std::setw(15) << rmsD
                      << std::setw(15) << maxD
                      << std::setw(15) << diis_error
                      << std::setw(12) << std::fixed << std::setprecision(3) << damping
                      << std::setw(12) << std::fixed << std::setprecision(3) << time_sec
                      << "\n";
        }
    
        inline void mo_header()
        {
            std::lock_guard<std::mutex> lock(log_mutex);
            constexpr int W = 31;   // total width of the table
            std::cout << std::string(W, '-') << "\n"
                      << std::setw(6)  << std::right << "MO"
                      << std::setw(25) << std::right << "Energy (Eh)"
                      << "\n"
                      << std::string(W, '-') << "\n";
        }

        inline void mo_energies(const Eigen::VectorXd& mo_energies, const std::size_t n_electrons)
        {
            std::lock_guard<std::mutex> lock(log_mutex);

            const std::size_t n_occ = n_electrons / 2;
            const std::size_t homo = n_occ - 1;
            const std::size_t lumo = n_occ;

            for (std::size_t i = 0; i < static_cast<std::size_t>(mo_energies.size()); i++)
            {
                std::string label = "";
                if (i == homo) label = "  <-- HOMO";
                if (i == lumo) label = "  <-- LUMO";

                std::cout << std::setw(6)  << std::right << (i + 1)
                          << std::setw(25) << std::right << mo_energies(i)
                          << label
                          << "\n";
            }
        }

        inline void blank()
        {
            std::cout << '\n';
        }

        inline void scf_footer()
        {
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cout << std::string(110, '-') << "\n";
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
    case HartreeFock::IntegralMethod::McMurchieDavidson:
        return "McMurchie-Davidson";
    case HartreeFock::IntegralMethod::Huzinaga:
        return "Huzinaga";
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
    case HartreeFock::CalculationType::GeomOpt:
        return "Geometry Optimization";
    case HartreeFock::CalculationType::Frequency:
        return "Frequency Calculation";
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
    }
    return "Unknown";
}

#endif // !HF_LOGGING_H
