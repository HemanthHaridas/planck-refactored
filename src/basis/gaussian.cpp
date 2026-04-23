#include <algorithm>
#include <expected>
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

static std::expected<BasisSet, std::string> read_gbs(std::ifstream &input)
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
                auto element = element_from_symbol(symbol);
                if (!element)
                    return std::unexpected(element.error());
                current_element = symbol;
                basis.try_emplace(symbol); // Place element
                continue;
            }
        }

        // Check if next line starts with a letter (usually shell header)
        if (!starts_with_alpha(line))
        {
            return std::unexpected("Expected shell header, got: " + line);
        }

        // Check if an element has been identified
        if (current_element.empty())
        {
            return std::unexpected("Shell before element header");
        }

        std::istringstream iss(line);
        std::string label;
        std::size_t nprim;
        double scale = 1.0;

        iss >> label >> nprim >> scale;

        if (!iss || !is_shell_label(label))
        {
            return std::unexpected("Malformed shell line: " + line);
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

double HartreeFock::BasisFunctions::component_norm(int df)
{
    return 1.0 / std::sqrt(static_cast<double>(df));
}

std::expected<HartreeFock::Basis, std::string> HartreeFock::BasisFunctions::read_gbs_basis(const std::string file_name, const HartreeFock::Molecule &molecule, const HartreeFock::BasisType &basis_type)
{
    if (!(basis_type == HartreeFock::BasisType::Cartesian))
    {
        return std::unexpected("Spherical Harmonics are not supported. Only Cartesian basis functions are currently supported");
    }

    // Try the path as given first; if that fails, retry with the filename
    // component lowercased (supports both case-preserving names like
    // "cc-pVDZ" and the all-lowercase convention used by built-in bases).
    auto make_lowercase_path = [](const std::string &path) -> std::string
    {
        const auto sep = path.rfind('/');
        if (sep == std::string::npos)
        {
            std::string lower = path;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c)
                           { return std::tolower(c); });
            return lower;
        }
        std::string name = path.substr(sep + 1);
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c)
                       { return std::tolower(c); });
        return path.substr(0, sep + 1) + name;
    };

    std::ifstream file(file_name);
    if (!file)
    {
        const std::string lower_path = make_lowercase_path(file_name);
        if (lower_path != file_name)
            file.open(lower_path);
    }
    if (!file)
    {
        return std::unexpected("Cannot open basis file: " + file_name);
    }

    // Parse the complete basis set
    auto gbs_res = read_gbs(file);
    if (!gbs_res)
        return std::unexpected(gbs_res.error());
    BasisSet gbs = std::move(*gbs_res);

    // Initialize Basis object
    HartreeFock::Basis basis;

    for (std::size_t i = 0; i < molecule.natoms; i++)
    {
        auto element_data = element_from_z(molecule.atomic_numbers[i]);
        if (!element_data)
            return std::unexpected(element_data.error());
        std::string element = std::string(element_data->symbol);

        // Find element in basis set
        auto it = gbs.find(element);

        if (it == gbs.end())
        {
            return std::unexpected("Unsupported Element: " + element);
        }

        // Now create Cartesian shells
        for (const GbsShell &gbs_shell : it->second)
        {
            HartreeFock::Shell shell;
            shell._center = molecule._standard.row(i).transpose();
            auto shell_type = _map_shell_to_L(gbs_shell.label);
            if (!shell_type)
                return std::unexpected(shell_type.error());
            shell._shell = *shell_type;
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
            auto normalization = HartreeFock::BasisFunctions::contracted_normalization(
                L, shell._primitives, shell._coefficients, shell._normalizations);
            if (!normalization)
                return std::unexpected(normalization.error());
            const double Nc = *normalization;

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
                basis_function._component_norm = HartreeFock::BasisFunctions::component_norm(df);
                basis_function._cartesian = am;
            }
        }
    }
    return basis;
}
