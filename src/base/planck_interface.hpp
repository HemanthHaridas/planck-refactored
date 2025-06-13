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
            {
                throw std::out_of_range("Key " + key + " was not found in the options");
            }

            std::istringstream buffer_(iterator_->second);
            T value;
            if (!(buffer_ >> value))
            {
                throw std::invalid_argument("Failed to convert " + value + " to target type")
            }
            return value;
        }

        template <>
        bool get_value(const std::string &key) const
        {
            auto iterator_ = _options.find(key);
            if (iterator_ == _options.end())
            {
                throw std::out_of_range("Key " + key + " was not found in the options");
            }

            const std::string value = iterator_->second;
            if (value == "TRUE" || value == "True" || value == "true" || value == "1")
            {
                return true;
            }
            if (value == "FALSE" || value == "False" || value == "false" || value == "0")
            {
                return false;
            }
            throw std::invalid_argument("Invalid Boolean argument for " + key);
        }
    };

    class InputInterface : public BaseInterface
    {
    public:
        explicit InputInterface(const std::vector<std::string> &keys, const std::vector<std::string> &values) : BaseInterface(build_map(keys, values)) {}

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

    class CalculationInterface : public BaseInterface
    {
    public:
        explicit CalculationInterface(const std::vector<std::string> &keys, const std::vector<std::string> &values) : BaseInterface(build_map(keys, values)) {}

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