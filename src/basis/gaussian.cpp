#include <fstream>
#include <iostream>

#include "base/tables.h"
#include "basis.h"
#include "lookup/elements.h"

struct GbsPrimitive
{
    double exponent;
    double coefficient;
};

struct GbsShell
{
    std::string label; // "S", "P", "D", ...
    std::vector<GbsPrimitive> primitives;
};

static bool starts_with_alpha(const std::string &line)
{
    // Iterate over each characters
    for (char c : line)
    {
        // Check if it is not a blanck space
        if (!std::isspace(static_cast<unsigned char>(c)))
        {
            // Check if the character is an alphabet [a-z, A-Z]
            return std::isalpha(static_cast<unsigned char>(c));
        }
    }
    return false;
}

static bool is_shell_label(const std::string &s)
{
    return s == "S" || s == "P" || s == "D" ||
           s == "F" || s == "G" || s == "H" ||
           s == "SP";
}

// Replace D+ with E+
static inline void normalize_fortran_exponents(std::string &line)
{
    for (char &c : line)
        if (c == 'D' || c == 'd')
            c = 'E';
}

using BasisSet = std::unordered_map<std::string, std::vector<GbsShell>>;

static BasisSet read_gbs(std::ifstream &input)
{
    BasisSet basis;
    std::string line;
    std::string current_element;

    while (std::getline(input, line))
    {
        // Skip comments and empty lines
        if (line.empty())
        {
            continue;
        }

        if (line.starts_with("!"))
        {
            continue;
        }

        // If end of current block reached, continue
        if (line == "****")
        {
            current_element.clear();
            continue;
        }

        std::istringstream header(line);
        std::string symbol;
        int charge;

        {
            // Check if header has two elements [Element name, Charge] and only two elements
            if ((header >> symbol >> charge) && header.eof())
            {
                auto _ = element_from_symbol(symbol);
                current_element = symbol;
                basis.try_emplace(symbol); // Place element
                continue;
            }
        }

        // Check if next line starts with a letter (usually shell header)
        if (!starts_with_alpha(line))
        {
            throw std::runtime_error("Expected shell header, got: " + line);
        }

        // Check if an element has been identified
        if (current_element.empty())
        {
            throw std::runtime_error("Shell before element header");
        }

        std::istringstream iss(line);
        std::string label;
        std::size_t nprim;
        double scale = 1.0;

        iss >> label >> nprim >> scale;

        if (!iss || !is_shell_label(label))
        {
            throw std::runtime_error("Malformed shell line: " + line);
        }

        // Special case in Gaussian94 basis sets
        if (label == "SP")
        {
            // Constrcut separate s and p-type shells
            GbsShell s{"S"}, p{"P"};

            for (std::size_t i = 0; i < nprim; ++i)
            {
                std::getline(input, line);
                normalize_fortran_exponents(line);

                std::istringstream prim(line);
                double expn, cs, cp;
                prim >> expn >> cs >> cp;

                s.primitives.push_back({expn, cs * scale});
                p.primitives.push_back({expn, cp * scale});
            }

            // Place the shells
            basis[current_element].push_back(std::move(s));
            basis[current_element].push_back(std::move(p));
        }
        // Regular case
        else
        {
            GbsShell shell{label};
            for (std::size_t i = 0; i < nprim; ++i)
            {
                std::getline(input, line);
                normalize_fortran_exponents(line);

                std::istringstream prim(line);
                double expn, cs;
                prim >> expn >> cs;

                shell.primitives.push_back({expn, cs * scale});
            }
            basis[current_element].push_back(std::move(shell));
        }
    }
    return basis;
}

// Returns (2n-1)!! with the convention (-1)!! = 1.
static int double_factorial(int n)
{
    if (n <= 0)
        return 1;
    int result = 1;
    while (n > 0)
    {
        result *= n;
        n -= 2;
    }
    return result;
}

HartreeFock::Basis HartreeFock::BasisFunctions::read_gbs_basis(const std::string file_name, const HartreeFock::Molecule &molecule, const HartreeFock::BasisType &basis_type)
{
    if (!(basis_type == HartreeFock::BasisType::Cartesian))
    {
        throw std::runtime_error("Spherical Harmonics are not supported. Only Cartesian basis functions are currently supported");
    }

    // Create input file stream
    std::ifstream file(file_name);
    if (!file)
    {
        throw std::runtime_error("Cannot open basis file: " + file_name);
    }

    // Parse the complete basis set
    BasisSet gbs = read_gbs(file);

    // Initialize Basis object
    HartreeFock::Basis basis;

    // Count total shells across all atoms and reserve upfront.
    // Without this, push_back on _shells may reallocate, invalidating all Shell*
    // pointers already stored in _basis_functions (ContractedView::_shell).
    {
        std::size_t total_shells = 0;
        for (std::size_t i = 0; i < molecule.natoms; i++)
        {
            const std::string sym = std::string(element_from_z(molecule.atomic_numbers[i]).symbol);
            auto it = gbs.find(sym);
            if (it != gbs.end())
                total_shells += it->second.size();
        }
        basis._shells.reserve(total_shells);
    }

    for (std::size_t i = 0; i < molecule.natoms; i++)
    {
        std::string element = std::string(
            element_from_z(molecule.atomic_numbers[i]).symbol);

        // Find element in basis set
        auto it = gbs.find(element);

        if (it == gbs.end())
        {
            throw std::runtime_error("Unsupported Element");
        }

        // Now create Cartesian shells
        for (const GbsShell &gbs_shell : it->second)
        {
            HartreeFock::Shell shell;
            shell._center = molecule._standard.row(i).transpose();
            shell._shell = _map_shell_to_L(gbs_shell.label);
            shell._atom_index = i;

            // Get size of primitives
            const std::size_t nprim = gbs_shell.primitives.size();
            shell._primitives.resize(nprim);
            shell._coefficients.resize(nprim);

            // Place primitives and coefficients
            for (std::size_t i = 0; i < nprim; ++i)
            {
                shell._primitives[i] = gbs_shell.primitives[i].exponent;
                shell._coefficients[i] = gbs_shell.primitives[i].coefficient;
            }

            // Get angular momentum of the shell
            unsigned int L = static_cast<unsigned int>(shell._shell);

            shell._normalizations = primitive_normalization(L, shell._primitives);
            const double Nc = HartreeFock::BasisFunctions::contracted_normalization(L, shell._primitives, shell._coefficients, shell._normalizations);

            // Scale all coefficients by normalization factor
            shell._coefficients = shell._coefficients * Nc;

            basis._shells.push_back(std::move(shell));
            const HartreeFock::Shell *shell_ptr = &basis._shells.back();

            for (auto am : HartreeFock::BasisFunctions::_cartesian_shell_order(L))
            {
                const std::size_t idx = basis._basis_functions.size();
                const int df = double_factorial(2 * am[0] - 1) * double_factorial(2 * am[1] - 1) * double_factorial(2 * am[2] - 1);
                basis._basis_functions.emplace_back();
                auto &basis_function = basis._basis_functions.back();
                basis_function._shell = shell_ptr;
                basis_function._index = idx;
                basis_function._component_norm = 1.0 / std::sqrt(static_cast<double>(df));
                basis_function._cartesian = am;
            }
        }
    }
    return basis;
}


