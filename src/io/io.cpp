#include <cctype>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <cmath>

#include <Eigen/Geometry>

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
            {"energy",      HartreeFock::CalculationType::SinglePoint},
            {"geomopt",     HartreeFock::CalculationType::GeomOpt},
            {"freq",        HartreeFock::CalculationType::Frequency},
            {"gradient",    HartreeFock::CalculationType::Gradient},

            {"sp",          HartreeFock::CalculationType::SinglePoint},
            {"opt",         HartreeFock::CalculationType::GeomOpt},
            {"frequency",   HartreeFock::CalculationType::Frequency},
            {"grad",        HartreeFock::CalculationType::Gradient}
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

    template<>
    HartreeFock::OptCoords map_string_enum<HartreeFock::OptCoords>(const std::string& value)
    {
        static const std::unordered_map<std::string, HartreeFock::OptCoords> _table =
        {
            {"cartesian", HartreeFock::OptCoords::Cartesian},
            {"internal",  HartreeFock::OptCoords::Internal},
            {"ic",        HartreeFock::OptCoords::Internal},
            {"gic",       HartreeFock::OptCoords::Internal}
        };

        auto _value = toLower(value);
        auto it = _table.find(_value);
        if (it != _table.end())
            return it->second;

        throw std::invalid_argument("Invalid OptCoords : " + value);
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
    HartreeFock::SCFGuess map_string_enum<HartreeFock::SCFGuess>(const std::string& value)
    {
        static const std::unordered_map<std::string, HartreeFock::SCFGuess> _table =
        {
            {"hcore",   HartreeFock::SCFGuess::HCore},
            {"read",    HartreeFock::SCFGuess::ReadDensity},  // backward compat alias
            {"density", HartreeFock::SCFGuess::ReadDensity},
            {"full",    HartreeFock::SCFGuess::ReadFull},
        };

        auto _value = toLower(value);
        auto it = _table.find(_value);
        if (it != _table.end())
            return it->second;

        throw std::invalid_argument("Invalid SCFGuess: " + value);
    }

    template<>
    HartreeFock::SCFMode map_string_enum<HartreeFock::SCFMode>(const std::string& value)
    {
        static const std::unordered_map<std::string, HartreeFock::SCFMode> _table =
        {
            {"conventional", HartreeFock::SCFMode::Conventional},
            {"direct",       HartreeFock::SCFMode::Direct},
            {"auto",         HartreeFock::SCFMode::Auto}
        };

        auto _value = toLower(value);
        auto it = _table.find(_value);
        if (it != _table.end())
            return it->second;

        throw std::invalid_argument("Invalid SCFMode: " + value);
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
            {"obara-saika", HartreeFock::IntegralMethod::ObaraSaika},
            {"os",          HartreeFock::IntegralMethod::ObaraSaika},
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
            {"tol_eri",        [&integral](const std::string &value)   {integral._tol_eri       = std::stod(value);}},
            {"guess",          [&scf](const std::string &value)        {scf._guess              = map_string_enum<HartreeFock::SCFGuess>(value);}},
            {"save_checkpoint",[&scf](const std::string &value)        {scf._save_checkpoint    = toBool(value);}},
            {"level_shift",    [&scf](const std::string &value)        {scf._level_shift        = std::stod(value);}},
            {"diis_restart",   [&scf](const std::string &value)        {scf._diis_restart_factor = std::stod(value);}},
            {"scf_mode",       [&scf](const std::string &value)        {scf._mode                = map_string_enum<HartreeFock::SCFMode>(value);}}
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

    std::expected <void, std::string> _parse_geom(const std::vector <std::string> &lines, HartreeFock::OptionsGeometry &geom, HartreeFock::OptCoords &opt_coords)
    {
        // (key, value) pairs
        const std::unordered_map <std::string, std::function <void(const std::string &)>> _geom_map =
        {
            {"coord_type",  [&geom](const std::string &value){geom._type        = map_string_enum <HartreeFock::CoordType>(value);}},
            {"coord_units", [&geom](const std::string &value){geom._units       = map_string_enum <HartreeFock::Units>(value);}},
            {"use_symm",    [&geom](const std::string &value){geom._use_symm    = toBool(value);}},
            {"opt_coords",  [&opt_coords](const std::string &value){opt_coords  = map_string_enum <HartreeFock::OptCoords>(value);}}
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

    // ── Constraint section parser ─────────────────────────────────────────────
    //
    // Accepted line formats (all indices are 1-based):
    //   b  i  j           — fix bond between atoms i and j
    //   a  i  j  k        — fix angle at atoms i-j-k (j is the vertex)
    //   d  i  j  k  l     — fix dihedral i-j-k-l
    //   f  i              — freeze atom i (all three Cartesian DOFs)
    //
    // Lines starting with '#' and blank lines are ignored.
    std::expected<void, std::string>
    _parse_constraints(const std::vector<std::string>& lines,
                       std::vector<HartreeFock::GeomConstraint>& constraints)
    {
        for (const auto& raw : lines)
        {
            // Strip leading/trailing whitespace and skip blank/comment lines
            std::string line = raw;
            if (auto pos = line.find('#'); pos != std::string::npos)
                line = line.substr(0, pos);
            // trim
            std::size_t s = line.find_first_not_of(" \t\r\n");
            if (s == std::string::npos) continue;
            line = line.substr(s);
            std::size_t e = line.find_last_not_of(" \t\r\n");
            if (e != std::string::npos) line = line.substr(0, e + 1);
            if (line.empty()) continue;

            std::istringstream iss(line);
            std::string key;
            iss >> key;
            if (key.empty()) continue;
            char kind = static_cast<char>(std::tolower(static_cast<unsigned char>(key[0])));

            HartreeFock::GeomConstraint con;
            con.atoms = {-1, -1, -1, -1};

            if (kind == 'b')
            {
                con.type = HartreeFock::GeomConstraint::Type::Bond;
                if (!(iss >> con.atoms[0] >> con.atoms[1]))
                    return std::unexpected("Constraint 'b' requires two atom indices: " + line);
            }
            else if (kind == 'a')
            {
                con.type = HartreeFock::GeomConstraint::Type::Angle;
                if (!(iss >> con.atoms[0] >> con.atoms[1] >> con.atoms[2]))
                    return std::unexpected("Constraint 'a' requires three atom indices: " + line);
            }
            else if (kind == 'd')
            {
                con.type = HartreeFock::GeomConstraint::Type::Dihedral;
                if (!(iss >> con.atoms[0] >> con.atoms[1] >> con.atoms[2] >> con.atoms[3]))
                    return std::unexpected("Constraint 'd' requires four atom indices: " + line);
            }
            else if (kind == 'f')
            {
                con.type = HartreeFock::GeomConstraint::Type::FrozenAtom;
                if (!(iss >> con.atoms[0]))
                    return std::unexpected("Constraint 'f' requires one atom index: " + line);
            }
            else
            {
                return std::unexpected("Unknown constraint type '" + key + "': " + line);
            }

            constraints.push_back(con);
        }
        return {};
    }

    // Convert a Z-matrix row to Cartesian given already-placed atoms.
    // Uses the standard NeRF / Natural Extension Reference Frame algorithm.
    static Eigen::Vector3d zmat_to_cart(
        const Eigen::MatrixXd &xyz,   // Nx3, rows already filled
        int ia, double r,              // bond to atom ia, length r
        int ib, double theta,          // angle at ia, referencing ib (radians)
        int ic, double phi)            // dihedral about ia-ib axis, referencing ic (radians)
    {
        Eigen::Vector3d A = xyz.row(ia);
        Eigen::Vector3d B = xyz.row(ib);
        Eigen::Vector3d C = xyz.row(ic);

        // Unit vectors
        Eigen::Vector3d bc = (A - B).normalized();
        Eigen::Vector3d n  = (B - C).cross(bc).normalized();
        Eigen::Vector3d m  = n.cross(bc);

        // New atom position in local frame
        double cos_t = std::cos(theta);
        double sin_t = std::sin(theta);
        double cos_p = std::cos(phi);
        double sin_p = std::sin(phi);

        return A + r * (-cos_t * bc + sin_t * (cos_p * m + sin_p * n));
    }

    std::expected<void, std::string> _parse_zmatrix_coords(
        const std::vector<std::string> &lines, HartreeFock::Molecule &molecule)
    {
        // lines[0] = natoms, lines[1] = charge multiplicity, lines[2..] = Z-matrix rows
        std::istringstream iss0(lines[0]);
        if (!(iss0 >> molecule.natoms))
            return std::unexpected("Malformed header line: " + lines[0]);

        std::istringstream iss1(lines[1]);
        if (!(iss1 >> molecule.charge >> molecule.multiplicity))
            return std::unexpected("Malformed charge/multiplicity line: " + lines[1]);

        if (molecule.natoms != (lines.size() - 2))
            return std::unexpected("Not enough Z-matrix lines: expected " +
                std::to_string(molecule.natoms) + ", got " + std::to_string(lines.size() - 2));

        molecule.atomic_numbers.resize(molecule.natoms);
        molecule.atomic_masses.resize(molecule.natoms);
        molecule.coordinates.resize(molecule.natoms, 3);
        molecule.standard.resize(molecule.natoms, 3);
        molecule._coordinates.resize(molecule.natoms, 3);
        molecule._standard.resize(molecule.natoms, 3);
        molecule.coordinates.setZero();

        const double deg2rad = M_PI / 180.0;

        for (std::size_t i = 0; i < molecule.natoms; i++)
        {
            std::istringstream iss(lines[i + 2]);
            std::string sym;
            if (!(iss >> sym))
                return std::unexpected("Invalid Z-matrix line: " + lines[i + 2]);

            try
            {
                const ElementData el   = element_from_symbol(sym);
                molecule.atomic_numbers[i] = el.Z;
                molecule.atomic_masses[i]  = el.mass;
            }
            catch (...)
            {
                return std::unexpected("Unknown atomic symbol: " + sym);
            }

            if (i == 0)
            {
                // Atom 1: origin
                molecule.coordinates.row(i).setZero();
            }
            else if (i == 1)
            {
                // Atom 2: along +z from atom 1
                int ia; double r;
                if (!(iss >> ia >> r))
                    return std::unexpected("Atom 2 Z-matrix line needs: sym i1 r");
                ia--;
                molecule.coordinates(i, 0) = 0.0;
                molecule.coordinates(i, 1) = 0.0;
                molecule.coordinates(i, 2) = r;
            }
            else if (i == 2)
            {
                // Atom 3: in xz-plane
                int ia, ib; double r, theta;
                if (!(iss >> ia >> r >> ib >> theta))
                    return std::unexpected("Atom 3 Z-matrix line needs: sym i1 r i2 angle");
                ia--; ib--;
                theta *= deg2rad;

                Eigen::Vector3d A = molecule.coordinates.row(ia);
                Eigen::Vector3d B = molecule.coordinates.row(ib);
                Eigen::Vector3d AB = (B - A).normalized();

                // Pick a perpendicular in the xz-plane
                Eigen::Vector3d perp(-AB(2), 0.0, AB(0));
                if (perp.norm() < 1e-10)
                    perp = Eigen::Vector3d(0.0, 1.0, 0.0);
                perp.normalize();

                Eigen::Vector3d pos = A + r * (std::cos(M_PI - theta) * AB.normalized() +
                                               std::sin(M_PI - theta) * perp);
                molecule.coordinates.row(i) = pos.transpose();
            }
            else
            {
                // General atom: bond, angle, dihedral
                int ia, ib, ic; double r, theta, phi;
                if (!(iss >> ia >> r >> ib >> theta >> ic >> phi))
                    return std::unexpected("Z-matrix line " + std::to_string(i+1) +
                        " needs: sym i1 r i2 angle i3 dihedral");
                ia--; ib--; ic--;
                theta *= deg2rad;
                phi   *= deg2rad;

                Eigen::Vector3d pos = zmat_to_cart(molecule.coordinates, ia, r, ib, theta, ic, phi);
                molecule.coordinates.row(i) = pos.transpose();
            }
        }

        molecule.nelectrons = std::accumulate(molecule.atomic_numbers.begin(),
                                              molecule.atomic_numbers.end(), 0) - molecule.charge;
        return {};
    }

    std::expected <void, std::string> _parse_coords(const std::vector <std::string>&lines, HartreeFock::Molecule &molecule,
                                                    HartreeFock::CoordType coord_type)
    {
        if (coord_type == HartreeFock::CoordType::ZMatrix)
            return _parse_zmatrix_coords(lines, molecule);

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
        molecule.nelectrons = std::accumulate(molecule.atomic_numbers.begin(), molecule.atomic_numbers.end(), 0) - molecule.charge;

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
            if (auto res = _parse_geom(it->second, calculator._geometry, calculator._opt_coords); !res)
                return std::unexpected(res.error());
        }
        else
        {
            return std::unexpected("Missing geom section");
        }
        
        // coords
        if (auto it = _sections.find("coords"); it != _sections.end())
        {
            if (auto res = _parse_coords(it->second, calculator._molecule, calculator._geometry._type); !res)
                return std::unexpected(res.error());
        }
        else
        {
            return std::unexpected("Missing coords section");
        }

        // constraints (optional)
        if (auto it = _sections.find("constraints"); it != _sections.end())
        {
            if (auto res = _parse_constraints(it->second, calculator._constraints); !res)
                return std::unexpected(res.error());
        }

        return {};
    }
}
