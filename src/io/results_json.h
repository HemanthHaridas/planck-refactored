#pragma once

// Machine-readable results serializer — the stable contract for the Python
// front end (python/planck.py). Keep the schema here in sync with that file,
// NOT with the human log output, which is free to change.
//
// Hand-written (no JSON dependency): the payload is a handful of scalars plus
// two N×3 arrays. Both binaries (hartree-fock, planck-dft) write the same shape.

#include <expected>
#include <format>
#include <fstream>
#include <string>

#include <Eigen/Dense>

#include "base/types.h"

namespace HartreeFock::IO
{
    inline std::expected<void, std::string>
    dump_results_json(const HartreeFock::Calculator &calc, const std::string &path)
    {
        std::ofstream out(path);
        if (!out)
            return std::unexpected("Failed to open JSON output file: " + path);

        const auto &mol = calc._molecule;
        auto num = [](double v) { return std::format("{:.12g}", v); };
        auto mat_rows = [&](const Eigen::MatrixXd &m) -> std::string
        {
            std::string s = "[";
            for (Eigen::Index i = 0; i < m.rows(); ++i)
            {
                s += (i ? ",[" : "[");
                for (Eigen::Index j = 0; j < m.cols(); ++j)
                    s += (j ? "," : "") + num(m(i, j));
                s += "]";
            }
            return s + "]";
        };
        auto vec = [&](const Eigen::VectorXd &v) -> std::string
        {
            std::string s = "[";
            for (Eigen::Index i = 0; i < v.size(); ++i)
                s += (i ? "," : "") + num(v(i));
            return s + "]";
        };

        out << "{\n";
        out << "  \"natoms\": " << mol.natoms << ",\n";
        out << "  \"charge\": " << mol.charge << ",\n";
        out << "  \"multiplicity\": " << mol.multiplicity << ",\n";
        out << "  \"atomic_numbers\": [";
        for (Eigen::Index a = 0; a < mol.atomic_numbers.size(); ++a)
            out << (a ? "," : "") << mol.atomic_numbers(a);
        out << "],\n";
        // Geometry in Bohr — the frame gradients/energies are consistent with.
        out << "  \"coordinates_bohr\": " << mat_rows(mol._standard) << ",\n";
        out << "  \"electronic_energy\": " << num(calc._total_energy - calc._nuclear_repulsion) << ",\n";
        out << "  \"nuclear_repulsion\": " << num(calc._nuclear_repulsion) << ",\n";
        out << "  \"scf_total_energy\": " << num(calc._total_energy) << ",\n";
        out << "  \"total_energy\": " << num(calc.current_total_energy()) << ",\n";
        out << "  \"has_correlation\": " << (calc._have_correlated_total_energy ? "true" : "false");
        // Gradient (Ha/Bohr), natoms×3 — only present when a gradient ran.
        if (calc._gradient.size() != 0)
            out << ",\n  \"gradient\": " << mat_rows(calc._gradient);
        // Multipole moments (au), only when the multipole report ran. Dipole is
        // a length-3 vector; quadrupole the traceless 3×3 Cartesian tensor.
        if (calc._have_multipole)
        {
            const auto &m = calc._multipole;
            out << ",\n  \"dipole_au\": " << vec(m.total_dipole);
            out << ",\n  \"quadrupole_au\": " << mat_rows(m.total_quadrupole);
        }
        out << "\n}\n";

        if (!out)
            return std::unexpected("Failed while writing JSON output file: " + path);
        return {};
    }
}
