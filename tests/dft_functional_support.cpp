#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "dft/base/wrapper.h"

namespace
{
    bool g_ok = true;

    void require(bool condition, const std::string &message)
    {
        if (!condition)
        {
            std::cerr << message << '\n';
            g_ok = false;
        }
    }

    void require_near(double actual, double expected, double tol, const std::string &message)
    {
        if (std::abs(actual - expected) > tol)
        {
            std::ostringstream oss;
            oss << message << ": expected " << expected << ", got " << actual
                << " (tol " << tol << ")";
            std::cerr << oss.str() << '\n';
            g_ok = false;
        }
    }

    DFT::XC::Functional require_functional(const std::string &name)
    {
        auto id = DFT::XC::functional_id(name);
        if (!id)
        {
            std::cerr << "functional_id(" << name << ") failed: " << id.error() << '\n';
            g_ok = false;
            return DFT::XC::Functional::create(1, DFT::XC::Spin::Unpolarized).value();
        }

        auto functional = DFT::XC::Functional::create(*id, DFT::XC::Spin::Unpolarized);
        if (!functional)
        {
            std::cerr << "Functional::create(" << name << ") failed: " << functional.error() << '\n';
            g_ok = false;
            return DFT::XC::Functional::create(1, DFT::XC::Spin::Unpolarized).value();
        }

        return std::move(*functional);
    }

    void test_hse06_metadata()
    {
        auto functional = require_functional("hse06");
        const auto cam = functional.cam_coefficients();

        require(functional.is_range_separated(), "HSE06 should be detected as range-separated");
        require(!functional.is_double_hybrid(), "HSE06 should not be detected as a double hybrid");
        require_near(cam.alpha, 0.0, 1e-12, "HSE06 long-range exact exchange coefficient mismatch");
        require_near(cam.beta, 0.25, 1e-12, "HSE06 short-range exact exchange coefficient mismatch");
        require_near(cam.omega, 0.11, 1e-12, "HSE06 omega mismatch");
        require_near(
            functional.fock_exchange_coefficient(),
            0.0,
            1e-12,
            "HSE06 full-range exact exchange coefficient mismatch");
        require_near(
            functional.short_range_exchange_coefficient(),
            0.25,
            1e-12,
            "HSE06 short-range exchange coefficient mismatch");
    }

    void test_b2plyp_metadata()
    {
        auto functional = require_functional("b2plyp");

        require(!functional.is_range_separated(), "B2PLYP should not be detected as range-separated");
        require(functional.is_double_hybrid(), "B2PLYP should be detected as a double hybrid");
        require_near(
            functional.fock_exchange_coefficient(),
            0.53,
            1e-12,
            "B2PLYP Fock exchange coefficient mismatch");
        require_near(
            functional.perturbative_correlation_coefficient(),
            0.27,
            1e-12,
            "B2PLYP PT2 correlation coefficient mismatch");
    }

    void test_wb2plyp_metadata()
    {
        auto functional = require_functional("wb2plyp");
        const auto cam = functional.cam_coefficients();

        require(functional.is_range_separated(), "wB2PLYP should be detected as range-separated");
        require(functional.is_double_hybrid(), "wB2PLYP should be detected as a double hybrid");
        require_near(cam.alpha, 1.0, 1e-12, "wB2PLYP long-range exact exchange coefficient mismatch");
        require_near(cam.beta, -0.47, 1e-12, "wB2PLYP short-range exact exchange coefficient mismatch");
        require_near(cam.omega, 0.30, 1e-12, "wB2PLYP omega mismatch");
        require_near(
            functional.perturbative_correlation_coefficient(),
            0.27,
            1e-12,
            "wB2PLYP PT2 correlation coefficient mismatch");
        require_near(
            functional.fock_exchange_coefficient(),
            1.0,
            1e-12,
            "wB2PLYP full-range exact exchange coefficient mismatch");
        require_near(
            functional.short_range_exchange_coefficient(),
            -0.47,
            1e-12,
            "wB2PLYP short-range exact exchange coefficient mismatch");
    }
} // namespace

int main()
{
    test_hse06_metadata();
    test_b2plyp_metadata();
    test_wb2plyp_metadata();
    return g_ok ? 0 : 1;
}
