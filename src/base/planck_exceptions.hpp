#pragma once  // Single-include guard (portable and less error-prone than traditional #ifndef)

#include <iostream>
#include <string>
#include <exception>

namespace Planck::Exceptions {

    // Describes broad categories of recoverable errors
    enum class ErrorCategory {
        IO,    // Input/output failures
        SCF,   // SCF convergence issues
        GEOM,  // Geometry parsing/validation failures
        OPT    // Optimization failures
    };

    // Base exception carrying a message and error category
    class PlanckException : public std::exception {
    private:
        std::string _error_message;
        ErrorCategory _error_category;

    public:
        explicit PlanckException(std::string message, ErrorCategory category)
            : _error_message(std::move(message)), _error_category(category) {}

        const char* what() const noexcept override { return _error_message.c_str(); }
        ErrorCategory category() const noexcept { return _error_category; }
    };

    // Thrown when input/output fails (e.g. file not found)
    class IOError : public PlanckException {
    public:
        explicit IOError(const std::string message) noexcept
            : PlanckException("IO Error: " + message, ErrorCategory::IO) {}
    };

    // Thrown on SCF convergence errors
    class SCFError : public PlanckException {
    public:
        explicit SCFError(const std::string& message) noexcept
            : PlanckException("SCF Error: " + message, ErrorCategory::SCF) {}
    };

    // Converts ErrorCategory to printable label
    inline const char* to_string(ErrorCategory category) {
        switch (category) {
            case ErrorCategory::IO:   return "IO";
            case ErrorCategory::SCF:  return "SCF";
            case ErrorCategory::GEOM: return "GEOM";
            case ErrorCategory::OPT:  return "OPT";
            default:                  return "Unknown";
        }
    }
}

namespace Planck::Exceptions::Diagnostics {

    // Controls verbosity of logged output
    enum class LogLevel {
        Minimal,   // Only the error message
        Standard,  // Category + message
        Debug      // Full context: file, line, function
    };

    // Logs exceptions based on configured verbosity level
    class Logger {
    private:
        LogLevel _level;

    public:
        explicit Logger(LogLevel level) : _level(level) {}

        void log(const Planck::Exceptions::PlanckException& ex,
                 const char* file = "",
                 int line = 0,
                 const char* function = "") const
        {
            switch (_level) {
                case LogLevel::Minimal:
                    std::cerr << ex.what() << '\n';
                    break;

                case LogLevel::Standard:
                    std::cerr << "[PlanckException] Category: "
                              << Planck::Exceptions::to_string(ex.category())
                              << " | Message: " << ex.what() << '\n';
                    break;

                case LogLevel::Debug:
                    std::cerr << "[PlanckException] Category: "
                              << Planck::Exceptions::to_string(ex.category())
                              << " | Message: " << ex.what()
                              << " | Location: " << file << ":" << line
                              << " (" << function << ")" << std::endl;
                    break;
            }
        }
    };
}
