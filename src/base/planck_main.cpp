#include <iostream>
#include <string>
#include <memory>
#include <stdexcept>
#include <filesystem>

#include "planck_input.hpp"

namespace Planck::Main
{
    /**
     * @brief Display program usage information
     * @param program_name Name of the executable
     */
    void print_usage(const std::string &program_name)
    {
        std::cerr << "Usage: " << program_name << " <input_file>\n"
                  << "\nDescription:\n"
                  << "  Planck quantum chemistry calculation program\n"
                  << "\nArguments:\n"
                  << "  input_file    Path to the input file containing calculation parameters\n"
                  << "\nExample:\n"
                  << "  " << program_name << " water_optimization.inp\n"
                  << std::endl;
    }

    /**
     * @brief Validate input file accessibility and format
     * @param filepath Path to the input file
     * @throws std::runtime_error if file is not accessible or invalid
     */
    void validate_input_file(const std::string &filepath)
    {
        namespace fs = std::filesystem;

        // Check if file exists
        if (!fs::exists(filepath))
        {
            throw std::runtime_error("Input file '" + filepath + "' does not exist");
        }

        // Check if it's a regular file (not a directory)
        if (!fs::is_regular_file(filepath))
        {
            throw std::runtime_error("'" + filepath + "' is not a regular file");
        }

        // Check file permissions (readable)
        std::error_code ec;
        auto perms = fs::status(filepath, ec).permissions();
        if (ec || (perms & fs::perms::owner_read) == fs::perms::none)
        {
            throw std::runtime_error("Input file '" + filepath + "' is not readable");
        }

        // Basic file size validation (not empty, not too large)
        auto file_size = fs::file_size(filepath, ec);
        if (ec)
        {
            throw std::runtime_error("Cannot determine size of input file '" + filepath + "'");
        }

        if (file_size == 0)
        {
            throw std::runtime_error("Input file '" + filepath + "' is empty");
        }

        // Reasonable upper limit for input files (100 MB)
        constexpr std::uintmax_t MAX_INPUT_SIZE = 100 * 1024 * 1024;
        if (file_size > MAX_INPUT_SIZE)
        {
            throw std::runtime_error("Input file '" + filepath + "' is too large (>" +
                                     std::to_string(MAX_INPUT_SIZE / (1024 * 1024)) + " MB)");
        }
    };

    /**
     * @brief Process the quantum chemistry calculation
     * @param input_filepath Path to the validated input file
     * @return Exit code (0 for success, non-zero for failure)
     */
    int process_calculation(const std::string &input_filepath)
    {
        try
        {
            // Create input reader with proper RAII
            auto input_reader = std::make_unique<Planck::IO::InputReader>(input_filepath);

            // TODO: Add actual calculation processing here
            // Example workflow:
            // 1. Parse input parameters
            // 2. Initialize molecular geometry
            // 3. Set up basis sets and integrals
            // 4. Perform SCF calculation
            // 5. Post-processing (properties, analysis)
            // 6. Output results

            std::cout << "Successfully processed input file: " << input_filepath << std::endl;
            std::cout << "Calculation completed normally." << std::endl;

            return 0; // Success
        }
        catch (const Planck::Exceptions::IOException &e)
        {
            std::cerr << "I/O Error: " << e.what() << std::endl;
            return 2; // I/O error code
        }
        catch (const std::runtime_error &e)
        {
            std::cerr << "Runtime Error: " << e.what() << std::endl;
            return 3; // Runtime error code
        }
        catch (const std::invalid_argument &e)
        {
            std::cerr << "Invalid Argument: " << e.what() << std::endl;
            return 4; // Invalid argument error code
        }
        catch (const std::exception &e)
        {
            std::cerr << "Unexpected Error: " << e.what() << std::endl;
            return 5; // Generic error code
        }
        catch (...)
        {
            std::cerr << "Unknown Error: An unexpected exception occurred during calculation" << std::endl;
            return 6; // Unknown error code
        }
    }
}

/**
 * @brief Main entry point for the Planck quantum chemistry program
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return Exit code (0 for success, non-zero for various error conditions)
 *
 * Exit codes:
 * 0 - Success
 * 1 - Invalid command line arguments
 * 2 - I/O error (file not found, permission denied, etc.)
 * 3 - Runtime error during calculation
 * 4 - Invalid input parameters
 * 5 - Unexpected error
 * 6 - Unknown error
 */
int main(int argc, const char *argv[])
{
    // Improve C++ iostream performance
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    try
    {
        // Validate command line arguments
        if (argc < 2)
        {
            std::cerr << "Error: No input file specified\n"
                      << std::endl;
            Planck::Main::print_usage(argv[0]);
            return 1; // Invalid arguments
        }

        if (argc > 2)
        {
            std::cerr << "Error: Too many arguments specified\n"
                      << std::endl;
            Planck::Main::print_usage(argv[0]);
            return 1; // Invalid arguments
        }

        // Handle help requests
        std::string arg1(argv[1]);
        if (arg1 == "-h" || arg1 == "--help" || arg1 == "/?" || arg1 == "help")
        {
            Planck::Main::print_usage(argv[0]);
            return 0; // Success for help request
        }

        // Extract and validate input file path
        std::string input_filepath(argv[1]);

        // Validate input file before processing
        Planck::Main::validate_input_file(input_filepath);

        // Process the calculation
        return Planck::Main::process_calculation(input_filepath);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 5; // Fatal error code
    }
    catch (...)
    {
        std::cerr << "Fatal Error: Unknown exception in main function" << std::endl;
        return 6; // Unknown fatal error
    }
}