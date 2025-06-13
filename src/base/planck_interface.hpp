#pragma once

#include <unordered_map>
#include <string>
#include <vector>

namespace Planck::Interface
{

    class BaseInterface
    {
    protected:
        std::unordered_map<std::string, std::string> _options;

    public:
        explicit BaseInterface(const std::unordered_map<std::string, std::string> &options) : _options(options) {}

        template <typename T>
        T get_value(const std::string &key) const
        {
            auto iterator_ = _options.find(key);
            if (iterator_ == _options.end())
                throw std::out_of_range("Key " + key + " was not found in the options");

            std::istringstream buffer_(iterator_->second);
            T value;
            if (!(buffer_ >> value))
                throw std::invalid_argument("Failed to convert " + value + " to target type") 
            return value;
        }

        template <>
        bool get_value(const std::string &key) const
        {
            auto iterator_ = _options.find(key);
            if (iterator_ == _options.end())
                throw std::out_of_range("Key " + key + " was not found in the options");

            const std::string value = iterator_->second;
            if (value == "TRUE" || value == "True" || value == "true" || value == "1")
                return true;
            if (value == "FALSE" || value == "False" || value == "false" || value == "0")
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

    public:
        explicit SetupInterface(const std::vector<std::string> &keys, const std::vector<std::string> &values) : BaseInterface(build_map(keys, values)) {}

        void set_parameters_from_input() override
        {
            _calc_type = get_value<std::string>("CALC_TYPE");
            _theory    = get_value<std::string>("THEORY");
            _basis     = get_value<std::string>("BASIS");
            _use_diis  = get_value<bool>("USE_DIIS");
            _use_symm  = get_value<bool>("USE_SYMM");
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
    public:
        explicit ControlInterface(const std::vector<std::string> &keys, const std::vector<std::string> &values) : BaseInterface(build_map(keys, values)) {}

        void set_parameters_from_input() override
        {
            _calc_type = get_value<std::string>("CALC_TYPE");
            _theory    = get_value<std::string>("THEORY");
            _basis     = get_value<std::string>("BASIS");
            _use_diis  = get_value<bool>("USE_DIIS");
            _use_symm  = get_value<bool>("USE_SYMM");
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