#ifndef DFT_LIBXC_WRAPPER_H
#define DFT_LIBXC_WRAPPER_H

#include <algorithm>
#include <cctype>
#include <cstring>
#include <expected>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

extern "C"
{
#include <xc.h>
}

namespace DFT
{
    namespace XC
    {
        enum class Spin
        {
            Unpolarized = XC_UNPOLARIZED,
            Polarized = XC_POLARIZED
        };

        class Functional
        {
        public:
            Functional(int functional_id, Spin spin)
                : spin_(spin)
            {
                std::memset(&func_, 0, sizeof(func_));
                if (xc_func_init(&func_, functional_id, static_cast<int>(spin)) != 0)
                    throw std::runtime_error(
                        "xc_func_init failed for functional id " + std::to_string(functional_id));
                initialized_ = true;
            }

            ~Functional()
            {
                if (initialized_)
                    xc_func_end(&func_);
            }

            Functional(const Functional &) = delete;
            Functional &operator=(const Functional &) = delete;

            Functional(Functional &&other) noexcept : func_(other.func_), initialized_(other.initialized_)
            {
                other.initialized_ = false;
                std::memset(&other.func_, 0, sizeof(other.func_));
            }

            Functional &operator=(Functional &&other) noexcept
            {
                if (this != &other)
                {
                    if (initialized_)
                        xc_func_end(&func_);
                    func_ = other.func_;
                    initialized_ = other.initialized_;
                    other.initialized_ = false;
                    std::memset(&other.func_, 0, sizeof(other.func_));
                }
                return *this;
            }

            const xc_func_type *get() const noexcept
            {
                return &func_;
            }

            xc_func_type *get() noexcept
            {
                return &func_;
            }

            int family() const noexcept
            {
                return func_.info ? func_.info->family : XC_FAMILY_UNKNOWN;
            }

            Spin spin() const noexcept
            {
                return spin_;
            }

            int spin_components() const noexcept
            {
                return (spin_ == Spin::Polarized) ? 2 : 1;
            }

            int sigma_components() const noexcept
            {
                return (spin_ == Spin::Polarized) ? 3 : 1;
            }

            int kind() const noexcept
            {
                return func_.info ? func_.info->kind : XC_EXCHANGE_CORRELATION;
            }

            std::string name() const
            {
                return (func_.info && func_.info->name) ? func_.info->name : "unknown";
            }

            double exact_exchange_coefficient() const
            {
                return xc_hyb_exx_coef(&func_);
            }

            bool is_lda_like() const noexcept
            {
                return family() == XC_FAMILY_LDA;
            }

            bool is_gga_like() const noexcept
            {
                return family() == XC_FAMILY_GGA;
            }

            bool is_meta_gga_like() const noexcept
            {
                return family() == XC_FAMILY_MGGA;
            }

            bool is_supported_semilocal() const noexcept
            {
                return is_lda_like() || is_gga_like();
            }

            std::expected<void, std::string> evaluate_lda_exc_vxc(
                const std::vector<double> &rho,
                int npoints,
                std::vector<double> &exc,
                std::vector<double> &vrho) const
            {
                if (!is_lda_like())
                    return std::unexpected("evaluate_lda_exc_vxc requires an LDA functional");
                if (npoints <= 0)
                    return std::unexpected("evaluate_lda_exc_vxc requires at least one grid point");
                if (rho.size() != static_cast<std::size_t>(npoints * spin_components()))
                    return std::unexpected("evaluate_lda_exc_vxc received an invalid rho array size");

                exc.resize(static_cast<std::size_t>(npoints));
                vrho.resize(static_cast<std::size_t>(npoints * spin_components()));
                xc_lda_exc_vxc(
                    &func_,
                    npoints,
                    const_cast<double *>(rho.data()),
                    exc.data(),
                    vrho.data());
                return {};
            }

            std::expected<void, std::string> evaluate_gga_exc_vxc(
                const std::vector<double> &rho,
                const std::vector<double> &sigma,
                int npoints,
                std::vector<double> &exc,
                std::vector<double> &vrho,
                std::vector<double> &vsigma) const
            {
                if (!is_gga_like())
                    return std::unexpected("evaluate_gga_exc_vxc requires a GGA-like functional");
                if (npoints <= 0)
                    return std::unexpected("evaluate_gga_exc_vxc requires at least one grid point");
                if (rho.size() != static_cast<std::size_t>(npoints * spin_components()))
                    return std::unexpected("evaluate_gga_exc_vxc received an invalid rho array size");
                if (sigma.size() != static_cast<std::size_t>(npoints * sigma_components()))
                    return std::unexpected("evaluate_gga_exc_vxc received an invalid sigma array size");

                exc.resize(static_cast<std::size_t>(npoints));
                vrho.resize(static_cast<std::size_t>(npoints * spin_components()));
                vsigma.resize(static_cast<std::size_t>(npoints * sigma_components()));
                xc_gga_exc_vxc(
                    &func_,
                    npoints,
                    const_cast<double *>(rho.data()),
                    const_cast<double *>(sigma.data()),
                    exc.data(),
                    vrho.data(),
                    vsigma.data());
                return {};
            }

        private:
            xc_func_type func_{};
            bool initialized_ = false;
            Spin spin_ = Spin::Unpolarized;
        };

        inline std::string version_string()
        {
            return xc_version_string();
        }

        inline int functional_id(std::string_view name)
        {
            if (name.empty())
                throw std::invalid_argument("libxc functional name must not be empty");

            const bool numeric = std::all_of(name.begin(), name.end(),
                                             [](unsigned char c)
                                             { return std::isdigit(c) != 0; });
            if (numeric)
                return std::stoi(std::string(name));

            const int id = xc_functional_get_number(std::string(name).c_str());
            if (id <= 0)
                throw std::invalid_argument("Unknown libxc functional: " + std::string(name));
            return id;
        }

        inline std::string functional_name(int functional_id)
        {
            const char *name = xc_functional_get_name(functional_id);
            if (!name)
                throw std::invalid_argument(
                    "Unknown libxc functional id: " + std::to_string(functional_id));
            return name;
        }

        inline std::string reference()
        {
            return xc_reference();
        }

    } // namespace XC

} // namespace DFT

#endif // DFT_LIBXC_WRAPPER_H
