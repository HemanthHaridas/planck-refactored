#ifndef HF_BASIS_GBS_PARSER_H
#define HF_BASIS_GBS_PARSER_H

// Private header — Gaussian94 (.gbs) parser shared between the orbital basis
// loader (gaussian.cpp) and the RI auxiliary basis loader (rifit.cpp).
// The on-disk format is identical for orbital and aux basis sets, so the
// element→shells→primitives reader is the only thing that needs to be shared.
// Everything downstream (Shell construction, normalization, AO indexing,
// spherical transform) differs between the two roles and stays in the
// respective .cpp files.

#include <expected>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/types.h"

namespace HartreeFock::BasisFunctions::detail
{
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

    using GbsBasisSet = std::unordered_map<std::string, std::vector<GbsShell>>;

    // One ECP term c · r^(n-2) · exp(-ζ r²); n kept as written in the file.
    struct GbsEcpTerm
    {
        int n;
        double exponent;
        double coefficient;
    };

    // Gaussian94 ECP block: channels[0] is the local (L = l_max) potential,
    // channels[1 + l] the semilocal l - l_max projector channel, l < l_max.
    struct GbsEcp
    {
        int n_core;
        std::vector<std::vector<GbsEcpTerm>> channels;
    };

    struct GbsFile
    {
        GbsBasisSet shells;
        std::unordered_map<std::string, GbsEcp> ecps;
    };

    // Parse a Gaussian94 (.gbs) stream into element-keyed shell descriptors
    // and any ECP blocks. Handles the SP "fused" shell type by splitting into
    // separate S and P shells. Returns a human-readable error on failure.
    std::expected<GbsFile, std::string> read_gbs(std::ifstream &input);

    // The ECP operator a parsed block places on atom `atom_index` at `center` (Bohr).
    HartreeFock::ECPPotential make_ecp_potential(const GbsEcp &ecp, const Eigen::Vector3d &center,
                                                 std::size_t atom_index);
} // namespace HartreeFock::BasisFunctions::detail

#endif // HF_BASIS_GBS_PARSER_H
