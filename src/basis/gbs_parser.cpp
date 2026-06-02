#include "gbs_parser.h"

#include <cctype>
#include <sstream>

#include "lookup/elements.h"

namespace
{
    bool starts_with_alpha(const std::string &line)
    {
        for (char c : line)
        {
            if (!std::isspace(static_cast<unsigned char>(c)))
                return std::isalpha(static_cast<unsigned char>(c));
        }
        return false;
    }

    bool is_shell_label(const std::string &s)
    {
        return s == "S" || s == "P" || s == "D" ||
               s == "F" || s == "G" || s == "H" ||
               s == "I" || s == "SP";
    }

    // Replace Fortran-style D-exponents ("1.0D+01") with E-exponents ("1.0E+01")
    // so that std::istringstream parses them.
    void normalize_fortran_exponents(std::string &line)
    {
        for (char &c : line)
            if (c == 'D' || c == 'd')
                c = 'E';
    }
} // namespace

namespace HartreeFock::BasisFunctions::detail
{
    std::expected<GbsBasisSet, std::string> read_gbs(std::ifstream &input)
    {
        GbsBasisSet basis;
        std::string line;
        std::string current_element;

        while (std::getline(input, line))
        {
            if (line.empty() || line.starts_with("!"))
                continue;

            // End of current element block
            if (line == "****")
            {
                current_element.clear();
                continue;
            }

            std::istringstream header(line);
            std::string symbol;
            int charge;
            if ((header >> symbol >> charge) && header.eof())
            {
                auto element = element_from_symbol(symbol);
                if (!element)
                    return std::unexpected(element.error());
                current_element = symbol;
                basis.try_emplace(symbol);
                continue;
            }

            if (!starts_with_alpha(line))
                return std::unexpected("Expected shell header, got: " + line);

            if (current_element.empty())
                return std::unexpected("Shell before element header");

            std::istringstream iss(line);
            std::string label;
            std::size_t nprim;
            double scale = 1.0;
            iss >> label >> nprim >> scale;

            if (!iss || !is_shell_label(label))
                return std::unexpected("Malformed shell line: " + line);

            // Gaussian94 "SP" — one primitive list, two coefficient columns;
            // expand into separate S and P shells.
            if (label == "SP")
            {
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
                basis[current_element].push_back(std::move(s));
                basis[current_element].push_back(std::move(p));
            }
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
} // namespace HartreeFock::BasisFunctions::detail
