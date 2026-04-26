#ifndef HF_IO_H
#define HF_IO_H

#include <expected>
#include <fstream>
#include <istream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/types.h"

namespace HartreeFock
{
    namespace IO
    {
        using SectionMap = std::unordered_map<std::string, std::vector<std::string>>;

        std::expected<SectionMap, std::string> _split_into_sections(std::istream &input);

        std::expected<void, std::string> _parse_control(const std::vector<std::string> &lines, HartreeFock::CalculationType &calculation, HartreeFock::OptionsBasis &basis, HartreeFock::OptionsOutput &output);
        std::expected<void, std::string> _parse_scf(const std::vector<std::string> &lines, HartreeFock::OptionsSCF &scf, HartreeFock::PostHF &correlation, HartreeFock::OptionsIntegral &integral, HartreeFock::OptionsActiveSpace &active_space);
        std::expected<void, std::string> _parse_dft(const std::vector<std::string> &lines, HartreeFock::OptionsDFT &dft);
        std::expected<void, std::string> _parse_pcm(const std::vector<std::string> &lines, HartreeFock::OptionsSolvation &solvation);
        std::expected<void, std::string> _parse_geom(const std::vector<std::string> &lines, HartreeFock::OptionsGeometry &geom, HartreeFock::OptCoords &opt_coords, double &imag_follow_step);
        std::expected<void, std::string> _parse_coords(const std::vector<std::string> &lines, HartreeFock::Molecule &molecule, HartreeFock::CoordType coord_type);
        std::expected<void, std::string> _parse_constraints(const std::vector<std::string> &lines, std::vector<HartreeFock::GeomConstraint> &constraints);

        // Public API
        std::expected<void, std::string> parse_input(std::ifstream &input, HartreeFock::Calculator &calculator);
    } // namespace IO
} // namespace HartreeFock

#endif // !HF_IO_H
