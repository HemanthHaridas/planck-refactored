#include <cctype>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

#include "io.h"
#include "lookup/elements.h"

namespace HartreeFock::IO
{
    using SectionMap = std::unordered_map <std::string, std::vector<std::string>>;

    // Helper functions
    static void trim(std::string &s)
    {
        const auto first = s.find_first_not_of(" \t");
        if (first == std::string::npos)
        {
            s.clear();
            return;
        }
        
        const auto last = s.find_last_not_of(" \t");
        s = s.substr(first, last - first + 1);
    }

    static std::string toLower(const std::string &parsedString)
    {
        std::string lowerString = parsedString;
        std::transform(lowerString.begin(), lowerString.end(), lowerString.begin(), [](unsigned char c){ return std::tolower(c); });
        return lowerString;
    }


    static bool toBool(const std::string& parsedString)
    {
        std::string key = toLower(parsedString);
        
        if (key == ".true." || key == "true" || key == "1" || key == "yes")
        {
            return true;
        }
        
        if (key == ".false." || key == "false" || key == "0" || key == "no")
        {
            return false;
        }
        
        throw std::invalid_argument("Invalid boolean value: " + parsedString);
    }

    // split the input file into sections
    std::expected <SectionMap, std::string> _split_into_sections(std::istream &input)
    {
        SectionMap sections;
        std::string line;
        std::string current;
        bool in_section = false;
        
        while (std::getline(input, line))
        {
            trim(line);
            
            if (line.empty() || line.front() == '#')
                continue;
            
            if (line.front() == '%')
            {
                // remove leading '%' and trailing whitespace
                std::string tag = line.substr(1);
                trim(tag);
                
                // end tag?
                if (tag.rfind("end_", 0) == 0) // starts with "end_"
                {
                    std::string end_name = tag.substr(4);
                    trim(end_name);
                    
                    if (!in_section)
                        return std::unexpected("end without active section: " + end_name);
                    
                    if (end_name != current)
                        return std::unexpected("Mismatched end section. Expected end " + current + ", got end " + end_name);
                    
                    current.clear();
                    in_section = false;
                    continue;
                }
                
                // start tag
                if (in_section)
                    return std::unexpected("Nested section " + tag + " inside " + current);
                
                current = tag.substr(6);
                in_section = true;
                sections[current]; // create entry
                continue;
            }
            
            if (in_section)
            {
                sections[current].push_back(line);
            }
        }
        
        if (in_section)
            return std::unexpected("Unterminated section: " + current);
        
        if (sections.empty())
            return std::unexpected("No sections found in input");
        
        return sections;
    }

    template<typename T>
    T map_string_enum(const std::string& value);

    template<>
    HartreeFock::CalculationType map_string_enum<HartreeFock::CalculationType>(const std::string& value)
    {
        static const std::unordered_map<std::string, HartreeFock::CalculationType> _table =
        {
            // All possible combinations
            {"energy",  HartreeFock::CalculationType::SinglePoint},
            {"geomopt", HartreeFock::CalculationType::GeomOpt},
            {"freq",    HartreeFock::CalculationType::Frequency},
            
            {"sp",          HartreeFock::CalculationType::SinglePoint},
            {"opt",         HartreeFock::CalculationType::GeomOpt},
            {"frequency",   HartreeFock::CalculationType::Frequency}
        };
        
        auto _value = toLower(value);   // First convert to lowercase
        auto it = _table.find(_value);
        if (it != _table.end())
            return it->second;
        
        throw std::invalid_argument("Invalid Calculation Type : " + value);
    }

    template<>
    HartreeFock::Verbosity map_string_enum<HartreeFock::Verbosity>(const std::string& value)
    {
        static const std::unordered_map<std::string, HartreeFock::Verbosity> _table =
        {
            // All possible combinations
            {"silent",  HartreeFock::Verbosity::Silent},
            {"minimal", HartreeFock::Verbosity::Minimal},
            {"normal",  HartreeFock::Verbosity::Normal},
            {"verbose", HartreeFock::Verbosity::Verbose},
            {"debug",   HartreeFock::Verbosity::Debug}
        };
        
        auto _value = toLower(value);   // First convert to lowercase
        auto it = _table.find(_value);
        if (it != _table.end())
            return it->second;
        
        throw std::invalid_argument("Invalid Verbosity : " + value);
    }

    template<>
    HartreeFock::BasisType map_string_enum<HartreeFock::BasisType>(const std::string& value)
    {
        static const std::unordered_map<std::string, HartreeFock::BasisType> _table =
        {
            {"cartesian",   HartreeFock::BasisType::Cartesian},
            {"spherical",   HartreeFock::BasisType::Spherical}
        };
        
        auto _value = toLower(value);   // First convert to lowercase
        auto it = _table.find(_value);
        if (it != _table.end())
            return it->second;
        
        throw std::invalid_argument("Invalid BasisType : " + value);
    }

    std::expected <void, std::string> _parse_control(const std::vector <std::string> &lines, HartreeFock::CalculationType &calculation, HartreeFock::OptionsBasis &basis, HartreeFock::OptionsOutput &output)
    {
        // (key, value) pairs
        const std::unordered_map <std::string, std::function <void(const std::string &)>> _control_map =
        {
            {"basis",       [&basis](const std::string &value)      {basis._basis_name = toLower(value);}},
            {"basis_type",  [&basis](const std::string &value)      {basis._basis = map_string_enum <HartreeFock::BasisType>(value);}},
            {"calculation", [&calculation](const std::string &value){calculation = map_string_enum <HartreeFock::CalculationType>(value);}},
            {"verbosity",   [&output](const std::string &value)     {output._verbosity = map_string_enum <HartreeFock::Verbosity>(value);}},
            {"basis_path",  [&basis](const std::string &value)      {basis._basis_path = value;}}
        };
        
        for (const std::string line : lines)
        {
            std::istringstream _iss(line);
            std::string key, value;
            
            // Check if there exists a (key, value) pair
            if (!(_iss >> key >> value))
            {
                return std::unexpected("Missing value for control keyword: " + key);
            }
            
            // Find the (key, value) pair
            if (auto it = _control_map.find(key); it != _control_map.end())
            {
                try
                {
                    it->second(value);
                }
                catch (const std::exception &e)
                {
                    return std::unexpected(std::string("Error parsing control '") + key + "': " + e.what());
                }
            }
            else
            {
                return std::unexpected("Unknown control keyword: " + key);
            }
        }
        
        return {};
    }

    template<>
    HartreeFock::SCFType map_string_enum<HartreeFock::SCFType>(const std::string& value)
    {
        static const std::unordered_map<std::string, HartreeFock::SCFType> _table =
        {
            // All possible combinations
            {"rhf", HartreeFock::SCFType::RHF},
            {"uhf", HartreeFock::SCFType::UHF}
        };
        
        auto _value = toLower(value);   // First convert to lowercase
        auto it = _table.find(_value);
        if (it != _table.end())
            return it->second;
        
        throw std::invalid_argument("Invalid SCFType : " + value);
    }

    template<>
    HartreeFock::PostHF map_string_enum<HartreeFock::PostHF>(const std::string& value)
    {
        static const std::unordered_map<std::string, HartreeFock::PostHF> _table =
        {
            // All possible combinations
            {"rmp2",    HartreeFock::PostHF::RMP2},
            {"ump2",    HartreeFock::PostHF::UMP2},
            {"casscf",  HartreeFock::PostHF::CASSCF},
            {"rasscf",  HartreeFock::PostHF::RASSCF},
        };
        
        auto _value = toLower(value);   // First convert to lowercase
        auto it = _table.find(_value);
        if (it != _table.end())
            return it->second;
        
        throw std::invalid_argument("Invalid Correlation : " + value);
    }


    template<>
    HartreeFock::IntegralMethod map_string_enum<HartreeFock::IntegralMethod>(const std::string& value)
    {
        static const std::unordered_map<std::string, HartreeFock::IntegralMethod> _table =
        {
            // All possible combinations
            {"hermite",     HartreeFock::IntegralMethod::McMurchieDavidson},
            {"huzinaga",    HartreeFock::IntegralMethod::Huzinaga},
            {"obara-saika", HartreeFock::IntegralMethod::ObaraSaika},
            
            {"md",  HartreeFock::IntegralMethod::McMurchieDavidson},
            {"tho", HartreeFock::IntegralMethod::Huzinaga},
            {"os",  HartreeFock::IntegralMethod::ObaraSaika},
        };
        
        auto _value = toLower(value);   // First convert to lowercase
        auto it = _table.find(_value);
        if (it != _table.end())
            return it->second;
        
        throw std::invalid_argument("Invalid Correlation : " + value);
    }

    std::expected <void, std::string> _parse_scf(const std::vector <std::string> &lines, HartreeFock::OptionsSCF &scf, HartreeFock::PostHF &correlation, HartreeFock::OptionsIntegral &integral)
    {
        // (key, value) pairs
        const std::unordered_map <std::string, std::function <void(const std::string &)>> _scf_map =
        {
            {"scf_type",    [&scf](const std::string &value)        {scf._scf           = map_string_enum <HartreeFock::SCFType>(value);}},
            {"use_diis",    [&scf](const std::string &value)        {scf._use_DIIS      = toBool(value);}},
            {"diis_dim",    [&scf](const std::string &value)        {scf._DIIS_dim      = std::stoi(value);}},
            {"max_cycles",  [&scf](const std::string &value)        {scf._max_cycles    = std::stoi(value);}},
            {"tol_energy",  [&scf](const std::string &value)        {scf._tol_energy    = std::stod(value);}},
            {"tol_density", [&scf](const std::string &value)        {scf._tol_density   = std::stod(value);}},
            {"threshold",   [&scf](const std::string &value)        {scf._threshold     = std::stoi(value);}},
            
            {"correlation", [&correlation](const std::string &value){correlation        = map_string_enum <HartreeFock::PostHF>(value);}},
            {"engine",      [&integral](const std::string &value)   {integral._engine   = map_string_enum <HartreeFock::IntegralMethod>(value);}},
            {"tol_eri",     [&integral](const std::string &value)   {integral._tol_eri  = std::stod(value);}}
        };
        
        for (const std::string line : lines)
        {
            std::istringstream _iss(line);
            std::string key, value;
            
            if (!(_iss >> key >> value))
            {
                return std::unexpected("Missing value for scf keyword: " + key);
            }
            
            // Find the (key, value) pair
            if (auto it = _scf_map.find(key); it != _scf_map.end())
            {
                try
                {
                    it->second(value);
                }
                catch (const std::exception &e)
                {
                    return std::unexpected(std::string("Error parsing scf '") + key + "': " + e.what());
                }
            }
            else
            {
                return std::unexpected("Unknown scf keyword: " + key);
            }
        }
        
        return {};
    }

    template<>
    HartreeFock::Units map_string_enum<HartreeFock::Units>(const std::string& value)
    {
        static const std::unordered_map<std::string, HartreeFock::Units> _table =
        {
            // All possible combinations
            {"angstrom",    HartreeFock::Units::Angstrom},
            {"bohr",        HartreeFock::Units::Bohr}
        };
        
        auto _value = toLower(value);   // First convert to lowercase
        auto it = _table.find(_value);
        if (it != _table.end())
            return it->second;
        
        throw std::invalid_argument("Invalid Units : " + value);
    }

    template<>
    HartreeFock::CoordType map_string_enum<HartreeFock::CoordType>(const std::string& value)
    {
        static const std::unordered_map<std::string, HartreeFock::CoordType> _table =
        {
            // All possible combinations
            {"cartesian",   HartreeFock::CoordType::Cartesian},
            {"zmatrix",     HartreeFock::CoordType::ZMatrix},
            {"internal",    HartreeFock::CoordType::ZMatrix}
        };
        
        auto _value = toLower(value);   // First convert to lowercase
        auto it = _table.find(_value);
        if (it != _table.end())
            return it->second;
        
        throw std::invalid_argument("Invalid Units : " + value);
    }

    std::expected <void, std::string> _parse_geom(const std::vector <std::string> &lines, HartreeFock::OptionsGeometry &geom)
    {
        // (key, value) pairs
        const std::unordered_map <std::string, std::function <void(const std::string &)>> _geom_map =
        {
            {"coord_type",  [&geom](const std::string &value){geom._type        = map_string_enum <HartreeFock::CoordType>(value);}},
            {"coord_units", [&geom](const std::string &value){geom._units       = map_string_enum <HartreeFock::Units>(value);}},
            {"use_symm",    [&geom](const std::string &value){geom._use_symm    = toBool(value);}}
        };
        
        for (const std::string line : lines)
        {
            std::istringstream _iss(line);
            std::string key, value;
            
            if (!(_iss >> key >> value))
            {
                return std::unexpected("Missing value for geom keyword: " + key);
            }
            
            // Find the (key, value) pair
            if (auto it = _geom_map.find(key); it != _geom_map.end())
            {
                try
                {
                    it->second(value);
                }
                catch (const std::exception &e)
                {
                    return std::unexpected(std::string("Error parsing geom '") + key + "': " + e.what());
                }
            }
            else
            {
                return std::unexpected("Unknown geom keyword: " + key);
            }
        }
        
        return {};
    }

    std::expected <void, std::string> _parse_coords(const std::vector <std::string>&lines, HartreeFock::Molecule &molecule)
    {
        // Parse first two lines separately
        std::istringstream _iss0(lines[0]);
        if (!(_iss0 >> molecule.natoms))
        {
            return std::unexpected("Malformed header line: " + lines[0]);
        }
        
        std::istringstream _iss1(lines[1]);
        if (!(_iss1 >> molecule.charge >> molecule.multiplicity))
        {
            return std::unexpected("Malformed charge/multiplicity line: " + lines[1]);
        }
        
        // Check if input is consistent
        if (molecule.natoms != (lines.size() - 2))
        {
            return std::unexpected("Not enough coordinate lines: expected " + std::to_string(molecule.natoms) + ", got " + std::to_string(lines.size() - 2));
        }
        
        molecule.atomic_numbers.resize(molecule.natoms);
        molecule.atomic_masses.resize(molecule.natoms);
        
        molecule.coordinates.resize(molecule.natoms, 3);
        molecule.standard.resize(molecule.natoms, 3);
        
        molecule._coordinates.resize(molecule.natoms, 3);
        molecule._standard.resize(molecule.natoms, 3);
        
        // Temporary variables
        std::string _atom;
        double _x, _y, _z;
        
        for (std::size_t i = 0; i < molecule.natoms; i++)
        {
            std::istringstream _iss(lines[i + 2]);
            
            // Check if valid
            if (!(_iss >> _atom >> _x >> _y >> _z))
            {
                return std::unexpected("Invalid coordinate line: " + lines[i + 2]);
            }
            
            // Check if element is supported
            try
            {
                const ElementData _element  = element_from_symbol(_atom);
                molecule.atomic_numbers[i]  = _element.Z;
                molecule.atomic_masses[i]   = _element.mass;
            }
            catch (const std::exception &e)
            {
                return std::unexpected(std::string("Unknown atomic symbol: ") + _atom);
            }
            
            molecule.coordinates(i, 0) = _x;
            molecule.coordinates(i, 1) = _y;
            molecule.coordinates(i, 2) = _z;
        }
        
        // Now set other variables
        molecule.nelectrons = std::accumulate(molecule.atomic_numbers.begin(), molecule.atomic_numbers.end(), 0) + molecule.charge;
        
        return {};
    }

    std::expected <void, std::string> parse_input(std::ifstream &input, HartreeFock::Calculator &calculator)
    {
        if (!input)
        {
            return std::unexpected("Failed to open the input file");
        }
        
        // Split into [control, scf, geom, coord] sections
        auto _sections_map = _split_into_sections(input);
        
        // Propogate error if splitting fails
        if (!_sections_map)
        {
            return std::unexpected(_sections_map.error());
        }
        
        // Extract sections
        const auto &_sections = _sections_map.value();
        
        // control
        if (auto it = _sections.find("control"); it != _sections.end())
        {
            if (auto res = _parse_control(it->second, calculator._calculation, calculator._basis, calculator._output); !res)
                return std::unexpected(res.error());
        }
        else
        {
            return std::unexpected("Missing control section");
        }

        // scf
        if (auto it = _sections.find("scf"); it != _sections.end())
        {
            if (auto res = _parse_scf(it->second, calculator._scf, calculator._correlation, calculator._integral); !res)
                return std::unexpected(res.error());
        }
        else
        {
            return std::unexpected("Missing scf section");
        }
        
        // geom
        if (auto it = _sections.find("geom"); it != _sections.end())
        {
            if (auto res = _parse_geom(it->second, calculator._geometry); !res)
                return std::unexpected(res.error());
        }
        else
        {
            return std::unexpected("Missing geom section");
        }
        
        // coords
        if (auto it = _sections.find("coords"); it != _sections.end())
        {
            if (auto res = _parse_coords(it->second, calculator._molecule); !res)
                return std::unexpected(res.error());
        }
        else
        {
            return std::unexpected("Missing coords section");
        }
        
        return {};
    }
}
