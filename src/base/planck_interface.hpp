#pragma once

#include <unordered_map>
#include <string>
#include <vector>

#include "planck_geometry.hpp"

namespace Planck::Interface
{

    namespace Defaults
    {
        const std::double_t ANGTOBOHR = 1.8897259886;
        const std::uint64_t MAXSCF = 120;
        const std::uint64_t MAXITER = 120;
        const std::uint64_t DIIS_DIM = 10;
        const std::double_t TOLSCF = 1.0E-14;
        const std::double_t TOLERI = 1.0E-14;

        const std::string DEFAULT_BASIS = "sto-3g";
        const std::string DEFAULT_THEORY = "rhf";
        const std::string DEFAULT_CALC = "energy";
        const std::string DEFAULT_COORD = "ang";

        const bool USE_DIIS = true;
        const bool USE_SYMM = true;

        // this is the maximum supported boysindex
        // change this value only if you regenerate
        // the lookup table for the boysfunction
        const std::uint64_t MAXM = 60;
    }

    class BaseInterface
    {
    protected:
        std::unordered_map<std::string, std::string> _options;

    public:
        explicit BaseInterface(const std::unordered_map<std::string, std::string> &options) : _options(options) {}

        template <typename T>
        T get_value(const std::string &key, const T &default_value) const
        {
            auto iterator_ = _options.find(key);
            if (iterator_ == _options.end())
                return default_value;

            std::istringstream buffer_(iterator_->second);
            T value;
            if (!(buffer_ >> value))
                throw std::invalid_argument("Failed to convert " + value + " to target type");
            return value;
        }

        template <>
        bool get_value(const std::string &key, const bool &default_value) const
        {
            auto iterator_ = _options.find(key);
            if (iterator_ == _options.end())
                return default_value;

            const std::string value = iterator_->second;
            // std::ranges::transform(value, value.begin(), ::toupper);
            if (value == "TRUE" || value == "1")
                return true;
            if (value == "FALSE" || value == "0")
                return false;
            throw std::invalid_argument("Invalid Boolean argument for " + key);
        }

        virtual void set_parameters_from_input();
    };

    class SetupInterface : public BaseInterface
    {
    private:
        std::string _calc_type;
        std::string _theory;
        std::string _basis;
        std::string _coor_type;

        bool _use_diis;
        bool _use_symm;

        bool _is_initialized_by_user = false;

    public:
        SetupInterface() = default;
        explicit SetupInterface(const std::vector<std::string> &keys, const std::vector<std::string> &values) : BaseInterface(build_map(keys, values)) {}

        void set_parameters_from_input() override
        {
            _calc_type    = get_value<std::string>("CALC_TYPE", Defaults::DEFAULT_CALC);
            _theory       = get_value<std::string>("THEORY", Defaults::DEFAULT_THEORY);
            _basis        = get_value<std::string>("BASIS", Defaults::DEFAULT_BASIS);

            _use_diis     = get_value<bool>("USE_DIIS", Defaults::USE_DIIS);
            _use_symm     = get_value<bool>("USE_SYMM", Defaults::USE_SYMM);
            
            _is_initialized_by_user = true;
        }

    private:
        static std::unordered_map<std::string, std::string> build_map(const std::vector<std::string> &keys, const std::vector<std::string> &values)
        {
            if (keys.size() != values.size())
                throw std::invalid_argument("Mismatched number of keys and values");

            std::unordered_map<std::string, std::string> result;
            for (size_t i = 0; i < keys.size(); ++i)
                result[keys[i]] = values[i];
            return result;
        }
    };

    class ControlInterface : public BaseInterface
    {
    private:
        std::uint64_t _max_scf;
        std::uint64_t _max_iter;
        std::uint64_t _diis_dim;

        std::double_t _tol_scf;
        std::double_t _tol_eri;

        bool _is_initialized_by_user = false;

    public:
        ControlInterface() = default;
        explicit ControlInterface(const std::vector<std::string> &keys, const std::vector<std::string> &values) : BaseInterface(build_map(keys, values)) {}

        void set_parameters_from_input() override
        {
            _max_scf  = get_value<std::uint64_t>("MAX_SCF", Defaults::MAXSCF);
            _max_iter = get_value<std::uint64_t>("MAX_ITER", Defaults::MAXITER);
            _diis_dim = get_value<std::uint64_t>("DIIS_DIM", Defaults::DIIS_DIM);

            _tol_scf  = get_value<std::double_t>("TOL_SCF", Defaults::TOLSCF);
            _tol_eri  = get_value<std::double_t>("TOL_ERI", Defaults::TOLERI);

            _is_initialized_by_user = true;
        }

    private:
        static std::unordered_map<std::string, std::string> build_map(const std::vector<std::string> &keys, const std::vector<std::string> &values)
        {
            if (keys.size() != values.size())
                throw std::invalid_argument("Mismatched number of keys and values");

            std::unordered_map<std::string, std::string> result;
            for (size_t i = 0; i < keys.size(); ++i)
                result[keys[i]] = values[i];
            return result;
        }
    };

    class GeometryInterface : public BaseInterface
    {
    private:
        std::uint64_t _multiplicity;
        std::int64_t _charge;
        
        Planck::Geometry::Molecule _molecule;

    public:
        GeometryInterface() = default;
        explicit GeometryInterface(const std::vector<std::string> &keys, const std::vector<std::string> &values, std::vector<std::string> &atoms, std::vector<Eigen::Vector3f> &coords) : BaseInterface(build_map(keys, values)), _molecule(atoms, coords) {
            _multiplicity = get_value<std::uint64_t>("MULTI", 1);
            _charge       = get_value<std::int64_t>("CHARGE", 0);
        }

    private:
        static std::unordered_map<std::string, std::string> build_map(const std::vector<std::string> &keys, const std::vector<std::string> &values)
        {
            if (keys.size() != values.size())
                throw std::invalid_argument("Mismatched number of keys and values");

            std::unordered_map<std::string, std::string> result;
            for (size_t i = 0; i < keys.size(); ++i)
                result[keys[i]] = values[i];
            return result;
        }
    };
};