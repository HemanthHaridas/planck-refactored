#include "rifit.h"

#include <algorithm>
#include <fstream>

#include "basis.h"      // primitive_normalization, contracted_normalization, _map_shell_to_L
#include "gbs_parser.h" // shared .gbs reader
#include "lookup/elements.h"

namespace HartreeFock::BasisFunctions
{
    std::expected<AuxBasis, std::string> read_ri_basis(
        const std::string &file_name,
        const Molecule &molecule)
    {
        // Mirror the orbital loader's case-insensitive open path so users can
        // write `ri_basis cc-pVDZ-RI` regardless of how the file was named on
        // disk.
        auto make_lowercase_path = [](const std::string &path) {
            const auto sep = path.find_last_of("/\\");
            if (sep == std::string::npos)
            {
                std::string lower = path;
                std::transform(lower.begin(), lower.end(), lower.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                return lower;
            }
            std::string name = path.substr(sep + 1);
            std::transform(name.begin(), name.end(), name.begin(),
                           [](unsigned char c) { return std::tolower(c); });
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
            return std::unexpected("Cannot open RI auxiliary basis file: " + file_name);

        auto gbs_res = detail::read_gbs(file);
        if (!gbs_res)
            return std::unexpected(gbs_res.error());
        const auto &gbs = *gbs_res;

        AuxBasis aux;
        aux.cartesian = true;

        for (std::size_t i = 0; i < molecule.natoms; ++i)
        {
            auto element_data = element_from_z(molecule.atomic_numbers[i]);
            if (!element_data)
                return std::unexpected(element_data.error());
            const std::string element(element_data->symbol);

            auto it = gbs.find(element);
            if (it == gbs.end())
                return std::unexpected(
                    "RI auxiliary basis missing element: " + element);

            for (const detail::GbsShell &gbs_shell : it->second)
            {
                Shell shell;
                shell._center = molecule._standard.row(i).transpose();
                auto shell_type = _map_shell_to_L(gbs_shell.label);
                if (!shell_type)
                    return std::unexpected(shell_type.error());
                shell._shell = *shell_type;
                shell._atom_index = i;

                const std::size_t nprim = gbs_shell.primitives.size();
                shell._primitives.resize(nprim);
                shell._coefficients.resize(nprim);
                for (std::size_t k = 0; k < nprim; ++k)
                {
                    shell._primitives[k] = gbs_shell.primitives[k].exponent;
                    shell._coefficients[k] = gbs_shell.primitives[k].coefficient;
                }

                const unsigned int L = static_cast<unsigned int>(shell._shell);

                // Same normalization convention as the orbital basis: primitive
                // norms held separately, contracted norm pre-folded into the
                // contraction coefficients. The 3-center integral engine relies
                // on this contract (see the Norm Factors gotcha in docs).
                shell._normalizations = primitive_normalization(L, shell._primitives);
                auto contracted = contracted_normalization(
                    L, shell._primitives, shell._coefficients, shell._normalizations);
                if (!contracted)
                    return std::unexpected(contracted.error());
                shell._coefficients = shell._coefficients * (*contracted);

                // Cartesian function count for this shell: (L+1)(L+2)/2.
                const std::size_t n_cart =
                    (static_cast<std::size_t>(L) + 1) *
                    (static_cast<std::size_t>(L) + 2) / 2;
                aux.offsets.push_back(aux.nfunctions);
                aux.nfunctions += n_cart;
                aux.shells.push_back(std::move(shell));
            }
        }

        return aux;
    }
} // namespace HartreeFock::BasisFunctions
