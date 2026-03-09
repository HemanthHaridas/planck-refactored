#ifndef HF_IO_H
#define HF_IO_H

#include <istream>
#include <string>
#include <unordered_map>
#include <vector>
#include <expected>
#include <sstream>

#include "base/types.h"

namespace HartreeFock
{
    namespace IO
    {
        using SectionMap = std::unordered_map<std::string, std::vector<std::string>>;

        std::expected <SectionMap, std::string> _split_into_sections(std::istream &input);
    
        std::expected <void, std::string> _parse_control(const std::vector <std::string> &lines, HartreeFock::CalculationType &calculation, HartreeFock::OptionsBasis &basis, HartreeFock::OptionsOutput &output);
        std::expected <void, std::string> _parse_scf(const std::vector <std::string> &lines, HartreeFock::OptionsSCF &scf, HartreeFock::PostHF &correlation, HartreeFock::OptionsIntegral &integral);
        std::expected <void, std::string> _parse_geom(const std::vector <std::string> &lines, HartreeFock::OptionsGeometry &geom);
        std::expected <void, std::string> _parse_coords(const std::vector <std::string> &lines, HartreeFock::Molecule &molecule);
    
        // Public API
        std::expected <void, std::string> parse_input(std::ifstream &input, HartreeFock::Calculator &calculator);
    }
}

#endif // !HF_IO_H
