#pragma once

#include <string>
#include <fstream>
#include <tuple>
#include <memory>
#include <vector>
#include <sstream>
#include <Eigen/Core>

#include "planck_interface.hpp"
#include "planck_exceptions.hpp"
namespace Planck::IO
{

    class InputReader
    {
    private:
        std::string _input;
        std::shared_ptr<std::ifstream> _file;

    public:
        Planck::Interface::ControlInterface _control;
        Planck::Interface::SetupInterface _setup;
        Planck::Interface::GeometryInterface _geom;

        explicit InputReader(const std::string &input) : _input(input), _file(std::make_shared<std::ifstream>(_input)), _control(build_control_interface()), _setup(build_setup_interface()), _geom(build_geometry_interface())
        {
            if (!_file || !_file->is_open())
                throw Planck::Exceptions::IOException("Could not open input file: " + _input);
        }

    private:
        std::string toLower(const std::string &parsedString)
        {
            std::string lowerString = parsedString; // Create a copy to preserve the original string
            std::transform(lowerString.begin(), lowerString.end(), lowerString.begin(), ::tolower);
            return lowerString; // Return the transformed string
        }

        bool stringToBool(const std::string &parsedString)
        {
            std::string upperStr = parsedString;
            std::transform(upperStr.begin(), upperStr.end(), upperStr.begin(), ::toupper); // Convert to uppercase

            if (upperStr == "ON")
                return true; // "ON" maps to true
            else if (upperStr == "OFF")
                return false; // "OFF" maps to false
            else
                throw std::invalid_argument("Invalid string for boolean conversion."); // Handle unexpected input
        }

        Planck::Interface::ControlInterface build_control_interface()
        {
            auto [keys, values] = tokenize_input("CONTROL", _file);
            return Planck::Interface::ControlInterface(keys, values);
        }

        Planck::Interface::SetupInterface build_setup_interface()
        {
            auto [keys, values] = tokenize_input("SETUP", _file);
            return Planck::Interface::SetupInterface(keys, values);
        }

        Planck::Interface::GeometryInterface build_geometry_interface()
        {
            auto [keys, values] = tokenize_input("MOL_SPEC", _file);
            auto [atoms, coords] = tokenize_coords(_file);
            return Planck::Interface::GeometryInterface(keys, values, atoms, coords);
        }

        std::tuple<std::vector<std::string>, std::vector<std::string>> tokenize_input(const std::string &header, std::shared_ptr<std::ifstream> file_pointer)
        {
            if (!file_pointer || !file_pointer->is_open())
                throw Planck::Exceptions::IOException("Input stream is unavailable");

            std::vector<std::string> keys;
            std::vector<std::string> values;

            std::string line_;
            while (std::getline(*file_pointer, line_))
            {
                if (line_.starts_with("[") && line_.ends_with("]"))
                {
                    std::string header_ = line_.substr(1, line_.length() - 2);
                    if (header == header_)
                    {
                        std::string data_;
                        while (std::getline(*file_pointer, data_))
                        {
                            if (data_.find("END_" + header_) != std::string::npos)
                                return {keys, values};

                            std::string _key, _value;
                            std::stringstream _data_buffer(data_);
                            _data_buffer >> _key >> _value;

                            if (!_key.empty() && !_value.empty())
                            {
                                keys.emplace_back(_key);
                                values.emplace_back(_value);
                            }
                        }
                    }
                }
            }

            throw Planck::Exceptions::IOException("Section [" + header + "] not found or incomplete");
        }

        std::tuple<std::vector<std::string>, std::vector<Eigen::Vector3f>> tokenize_coords(std::shared_ptr<std::ifstream> file_pointer)
        {
            std::string line_;
            std::vector<std::string> atoms;
            std::vector<Eigen::Vector3f> coords;

            while (getline(*file_pointer, line_))
            {
                if (line_.starts_with("[") && line_.ends_with("]") && (line_.substr(1, line_.length() - 2) == "COORDS"))
                {
                    std::string coords_, _atom;
                    Eigen::Vector3f _coords;
                    while (getline(*file_pointer, coords_))
                    {
                        if (coords_.find("END_COORDS") != std::string::npos)
                            return {atoms, coords};

                        std::stringstream _coord_buffer(coords_);
                        _coord_buffer >> _atom >> _coords[0] >> _coords[1] >> _coords[2];

                        atoms.push_back(_atom);
                        coords.push_back(_coords);
                    }
                }
            }

            throw Planck::Exceptions::IOException("Section [COORDS] not found or incomplete");
        }
    };
    
};