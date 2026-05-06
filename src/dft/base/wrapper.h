#ifndef DFT_LIBXC_WRAPPER_H
#define DFT_LIBXC_WRAPPER_H

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstring>
#include <expected>
#include <string>
#include <string_view>
#include <system_error>
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

        struct CAMCoefficients
        {
            double omega = 0.0;
            double alpha = 0.0;
            double beta = 0.0;
        };

        class Functional
        {
        public:
            static std::expected<Functional, std::string> create(int functional_id, Spin spin)
            {
                Functional functional(spin);
                if (xc_func_init(&functional.func_, functional_id, static_cast<int>(spin)) != 0)
                {
                    return std::unexpected(
                        "xc_func_init failed for functional id " + std::to_string(functional_id));
                }
                functional.initialized_ = true;
                return functional;
            }

            ~Functional()
            {
                if (initialized_)
                    xc_func_end(&func_);
            }

            Functional(const Functional &) = delete;
            Functional &operator=(const Functional &) = delete;

            Functional(Functional &&other) noexcept
                : func_(other.func_), initialized_(other.initialized_), spin_(other.spin_)
            {
                other.initialized_ = false;
                other.spin_ = Spin::Unpolarized;
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
                    spin_ = other.spin_;
                    other.initialized_ = false;
                    other.spin_ = Spin::Unpolarized;
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

            int hybrid_type() const noexcept
            {
                return xc_hyb_type(&func_);
            }

            bool is_hybrid() const noexcept
            {
                return hybrid_type() != XC_HYB_SEMILOCAL;
            }

            bool is_global_hybrid() const noexcept
            {
                return hybrid_type() == XC_HYB_HYBRID;
            }

            bool is_range_separated() const noexcept
            {
                const int type = hybrid_type();
                if (type == XC_HYB_CAM || type == XC_HYB_CAMY || type == XC_HYB_CAMG)
                    return true;

                if (func_.hyb_type == nullptr)
                    return false;

                for (int term = 0; term < func_.hyb_number_terms; ++term)
                {
                    switch (func_.hyb_type[term])
                    {
                    case XC_HYB_ERF_SR:
                    case XC_HYB_YUKAWA_SR:
                    case XC_HYB_GAUSSIAN_SR:
                        return true;
                    default:
                        break;
                    }
                }
                return false;
            }

            bool has_pt2_term() const noexcept
            {
                if (func_.hyb_type == nullptr || func_.hyb_coeff == nullptr)
                    return false;

                for (int term = 0; term < func_.hyb_number_terms; ++term)
                {
                    if (func_.hyb_type[term] == XC_HYB_PT2)
                        return true;
                }
                return false;
            }

            bool is_double_hybrid() const noexcept
            {
                return hybrid_type() == XC_HYB_DOUBLE_HYBRID || has_pt2_term();
            }

            CAMCoefficients cam_coefficients() const noexcept
            {
                CAMCoefficients coefficients;
                if (func_.hyb_type == nullptr || func_.hyb_coeff == nullptr || func_.hyb_omega == nullptr)
                    return coefficients;

                for (int term = 0; term < func_.hyb_number_terms; ++term)
                {
                    switch (func_.hyb_type[term])
                    {
                    case XC_HYB_FOCK:
                        coefficients.alpha += func_.hyb_coeff[term];
                        break;
                    case XC_HYB_ERF_SR:
                    case XC_HYB_YUKAWA_SR:
                    case XC_HYB_GAUSSIAN_SR:
                        coefficients.beta += func_.hyb_coeff[term];
                        if (std::abs(func_.hyb_omega[term]) > 0.0)
                            coefficients.omega = func_.hyb_omega[term];
                        break;
                    default:
                        break;
                    }
                }

                return coefficients;
            }

            double perturbative_correlation_coefficient() const noexcept
            {
                if (func_.hyb_type == nullptr || func_.hyb_coeff == nullptr)
                    return 0.0;

                for (int term = 0; term < func_.hyb_number_terms; ++term)
                {
                    if (func_.hyb_type[term] == XC_HYB_PT2)
                        return func_.hyb_coeff[term];
                }
                return 0.0;
            }

            double fock_exchange_coefficient() const noexcept
            {
                if (is_global_hybrid())
                    return exact_exchange_coefficient();

                if (func_.hyb_type == nullptr || func_.hyb_coeff == nullptr)
                    return 0.0;

                double coefficient = 0.0;
                for (int term = 0; term < func_.hyb_number_terms; ++term)
                {
                    if (func_.hyb_type[term] == XC_HYB_FOCK)
                        coefficient += func_.hyb_coeff[term];
                }
                return coefficient;
            }

            double short_range_exchange_coefficient() const noexcept
            {
                if (!is_range_separated())
                    return 0.0;

                return cam_coefficients().beta;
            }

            double range_separation_omega() const noexcept
            {
                if (!is_range_separated())
                    return 0.0;

                return cam_coefficients().omega;
            }

            bool is_combined_exchange_correlation() const noexcept
            {
                return kind() == XC_EXCHANGE_CORRELATION;
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
            explicit Functional(Spin spin) : spin_(spin)
            {
                std::memset(&func_, 0, sizeof(func_));
            }

            xc_func_type func_{};
            bool initialized_ = false;
            Spin spin_ = Spin::Unpolarized;
        };

        inline std::string version_string()
        {
            return xc_version_string();
        }

        inline std::expected<int, std::string> functional_id(std::string_view name)
        {
            if (name.empty())
                return std::unexpected("libxc functional name must not be empty");

            const bool numeric = std::all_of(name.begin(), name.end(),
                                             [](unsigned char c)
                                             { return std::isdigit(c) != 0; });
            if (numeric)
            {
                int parsed = 0;
                const char *begin = name.data();
                const char *end = begin + name.size();
                const auto [ptr, ec] = std::from_chars(begin, end, parsed);
                if (ec != std::errc{} || ptr != end || parsed <= 0)
                {
                    return std::unexpected("Invalid numeric libxc functional id: " +
                                           std::string(name));
                }
                return parsed;
            }

            std::string normalized(name);
            std::transform(
                normalized.begin(),
                normalized.end(),
                normalized.begin(),
                [](unsigned char c)
                {
                    if (c == '-' || std::isspace(c))
                        return '_';
                    return static_cast<char>(std::tolower(c));
                });

            std::vector<std::string> candidates = {normalized};
            const auto starts_with = [&normalized](std::string_view prefix)
            {
                return normalized.rfind(prefix, 0) == 0;
            };

            if (!starts_with("lda_") &&
                !starts_with("gga_") &&
                !starts_with("hyb_") &&
                !starts_with("mgga_") &&
                !starts_with("hyb_mgga_"))
            {
                candidates.push_back("hyb_gga_xc_" + normalized);
                candidates.push_back("gga_xc_" + normalized);
                candidates.push_back("hyb_gga_x_" + normalized);
                candidates.push_back("gga_x_" + normalized);
                candidates.push_back("gga_c_" + normalized);
                candidates.push_back("hyb_mgga_xc_" + normalized);
                candidates.push_back("mgga_xc_" + normalized);
            }

            for (const auto &candidate : candidates)
            {
                const int id = xc_functional_get_number(candidate.c_str());
                if (id > 0)
                    return id;
            }

            return std::unexpected("Unknown libxc functional: " + std::string(name));
        }

        inline std::expected<std::string, std::string> functional_name(int functional_id)
        {
            const char *name = xc_functional_get_name(functional_id);
            if (!name)
            {
                return std::unexpected(
                    "Unknown libxc functional id: " + std::to_string(functional_id));
            }
            return name;
        }

        inline std::string reference()
        {
            return xc_reference();
        }

    } // namespace XC

} // namespace DFT

#endif // DFT_LIBXC_WRAPPER_H
