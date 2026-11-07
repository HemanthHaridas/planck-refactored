#include "dft/dispersion.h"

#include <algorithm>
#include <format>
#include <memory>
#include <vector>

#include "dft/base/wrapper.h"

#ifdef PLANCK_WITH_DFTD3
#include "s-dftd3.h"
#endif

namespace DFT::Dispersion
{
    namespace
    {
        // Text after `marker` in a libxc name, underscores dropped
        // ("hyb_gga_xc_cam_b3lyp" -> "camb3lyp"); empty when absent.
        // simple-dftd3 drops '-' but not '_' when it looks a name up.
        std::string libxc_suffix(const std::string &name, const std::string &marker)
        {
            const auto pos = name.find(marker);
            if (pos == std::string::npos)
                return {};
            std::string suffix = name.substr(pos + marker.size());
            std::erase(suffix, '_');
            return suffix;
        }
    } // namespace

    std::expected<std::string, std::string> parameter_set_name(const HartreeFock::OptionsDFT &dft)
    {
        if (!dft._dispersion_functional.empty())
            return dft._dispersion_functional;

        auto x = DFT::XC::functional_name(dft._exchange_id);
        if (!x)
            return std::unexpected(x.error());
        // A combined exchange-correlation functional names itself (B3LYP,
        // PBE0 = pbeh, HSE06, B2PLYP, CAM-B3LYP).
        if (auto xc = libxc_suffix(*x, "_xc_"); !xc.empty())
            return xc;

        auto c = DFT::XC::functional_name(dft._correlation_id);
        if (!c)
            return std::unexpected(c.error());
        const std::string xs = libxc_suffix(*x, "_x_");
        const std::string cs = libxc_suffix(*c, "_c_");
        if (!xs.empty() && xs == cs)
            return xs; // pbe + pbe, pw91 + pw91
        if (xs == "b88" && cs == "lyp")
            return "blyp";
        if (xs == "b88" && cs == "p86")
            return "bp86";
        return std::unexpected(std::format(
            "D3 dispersion: no parameter set is known for exchange '{}' + correlation '{}'; "
            "name one with dispersion_functional",
            *x, *c));
    }

    std::expected<Result, std::string> compute(const HartreeFock::Calculator &calculator)
    {
        const auto &molecule = calculator._molecule;
        const Eigen::Index natoms = molecule._standard.rows();
        Result result{.energy = 0.0, .gradient = Eigen::MatrixXd::Zero(natoms, 3)};
        const std::string &variant = calculator._dft._dispersion;
        if (variant == "none")
            return result;

#ifndef PLANCK_WITH_DFTD3
        return std::unexpected("dispersion " + variant +
                               " needs a build with PLANCK_WITH_DFTD3=ON (simple-dftd3; requires a Fortran compiler)");
#else
        auto name = parameter_set_name(calculator._dft);
        if (!name)
            return std::unexpected(name.error());

        // Ghost atoms only carry basis functions; leaving them out is the
        // counterpoise-consistent choice and gives them a zero gradient row.
        std::vector<Eigen::Index> real;
        std::vector<int> numbers;
        std::vector<double> xyz; // row-major natoms x 3, Bohr
        for (Eigen::Index a = 0; a < natoms; ++a)
        {
            if (static_cast<std::size_t>(a) < molecule.is_ghost.size() && molecule.is_ghost[a])
                continue;
            real.push_back(a);
            numbers.push_back(molecule.atomic_numbers[a]);
            for (int k = 0; k < 3; ++k)
                xyz.push_back(molecule._standard(a, k));
        }
        if (real.empty())
            return result;

        const bool zero_damping = variant.starts_with("d3zero");
        const bool atm = variant.ends_with("-atm");

        dftd3_error err = dftd3_new_error();
        auto message = [&](const std::string &what) -> std::string
        {
            char buffer[512] = {};
            int size = sizeof(buffer);
            dftd3_get_error(err, buffer, &size);
            return std::format("D3 dispersion ({}, '{}'): {} failed: {}", variant, *name, what, buffer);
        };
        dftd3_structure mol = nullptr;
        dftd3_model model = nullptr;
        dftd3_param param = nullptr;
        // ponytail: one cleanup at the end of a straight-line sequence, no RAII wrappers.
        auto run = [&]() -> std::expected<void, std::string>
        {
            mol = dftd3_new_structure(err, static_cast<int>(real.size()), numbers.data(), xyz.data(),
                                      nullptr, nullptr);
            if (dftd3_check_error(err))
                return std::unexpected(message("structure setup"));
            model = dftd3_new_d3_model(err, mol);
            if (dftd3_check_error(err))
                return std::unexpected(message("model setup"));
            param = zero_damping ? dftd3_load_zero_damping(err, name->data(), atm)
                                 : dftd3_load_rational_damping(err, name->data(), atm);
            if (dftd3_check_error(err))
                return std::unexpected(message("parameter lookup"));
            std::vector<double> gradient(3 * real.size());
            double sigma[9];
            dftd3_get_dispersion(err, mol, model, param, &result.energy, gradient.data(), sigma);
            if (dftd3_check_error(err))
                return std::unexpected(message("evaluation"));
            for (std::size_t i = 0; i < real.size(); ++i)
                for (int k = 0; k < 3; ++k)
                    result.gradient(real[i], k) = gradient[3 * i + k];
            return {};
        };
        auto status = run();
        if (param)
            dftd3_delete_param(&param);
        if (model)
            dftd3_delete_model(&model);
        if (mol)
            dftd3_delete_structure(&mol);
        dftd3_delete_error(&err);
        if (!status)
            return std::unexpected(status.error());
        return result;
#endif
    }
} // namespace DFT::Dispersion
