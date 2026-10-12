#include "dft/ao_grid.h"
#include "dft/base/grid.h"
#include "dft/ks_matrix.h"
#include "dft/xc_grid.h"
#include "base/basis.h"
#include "basis/basis.h"

#include <Eigen/Dense>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using Matrix = Eigen::MatrixXd;

    bool passed = true;

    void check(const bool value,
               const std::string &message,
               const std::source_location where = std::source_location::current())
    {
        if (!value)
        {
            passed = false;
            std::cerr << "FAIL fixed-grid XC ledger line " << where.line() << ": " << message << '\n';
        }
    }

    template <typename T>
    T require(std::expected<T, std::string> result)
    {
        if (!result)
            throw std::runtime_error(result.error());
        return std::move(*result);
    }

    double dot(const Matrix &left, const Matrix &right)
    {
        return (left.array() * right.array()).sum();
    }

    struct FixedDensityLedger
    {
        std::string policy;
        Eigen::Index points = 0;
        double energy = 0.0;
        double potential_norm = 0.0;
        double first_response = 0.0;
        double kernel_bilinear = 0.0;
    };

    HartreeFock::Calculator make_fixture()
    {
        HartreeFock::Calculator calculator;
        auto &molecule = calculator._molecule;
        molecule.natoms = 3;
        molecule.charge = 1;
        molecule.multiplicity = 2;
        molecule.atomic_numbers.resize(3);
        molecule.atomic_numbers << 8, 1, 1;
        molecule.atomic_masses.resize(3);
        molecule.atomic_masses << 16.0, 1.0, 1.0;
        // Deliberately asymmetric HO2 geometry: the same geometry used by the
        // live G3 rotation audit, but this test never solves an SCF problem.
        molecule.coordinates.resize(3, 3);
        molecule.coordinates << 0.12, -0.07, 0.11,
                                0.91,  0.23, -0.43,
                               -0.34,  0.86, 0.27;
        calculator._basis._basis = HartreeFock::BasisType::Cartesian;
        calculator.prepare_coordinates();
        molecule.set_standard_from_bohr(molecule._coordinates);
        calculator._shells = require(HartreeFock::BasisFunctions::read_gbs_basis(
            (std::filesystem::path(get_basis_path()) / "sto-3g").string(),
            molecule,
            calculator._basis._basis));
        return calculator;
    }

    DFT::XC::Functional functional(const char *name)
    {
        return require(DFT::XC::Functional::create(
            require(DFT::XC::functional_id(name)), DFT::XC::Spin::Unpolarized));
    }

    FixedDensityLedger measure(const HartreeFock::Calculator &calculator,
                               const DFT::ResolvedGridSpec &spec,
                               const Matrix &density,
                               const Matrix &direction,
                               const DFT::XC::Functional &exchange,
                               const DFT::XC::Functional &correlation)
    {
        const auto grid = require(DFT::MakeMolecularGrid(calculator._molecule, spec));
        const auto ao = require(DFT::evaluate_ao_basis_on_grid(calculator._shells, grid));
        const auto xc = require(DFT::evaluate_xc_on_grid(grid, ao, density, exchange, correlation));
        const auto potential = require(DFT::assemble_xc_matrix(grid, ao, xc));

        constexpr double h = 1.0e-5;
        const auto plus = require(DFT::assemble_xc_matrix(
            grid, ao, require(DFT::evaluate_xc_on_grid(grid, ao, density + h * direction, exchange, correlation))));
        const auto minus = require(DFT::assemble_xc_matrix(
            grid, ao, require(DFT::evaluate_xc_on_grid(grid, ao, density - h * direction, exchange, correlation))));

        FixedDensityLedger result;
        result.policy = spec.policy_name;
        result.points = grid.points.rows();
        result.energy = xc.total_energy;
        result.potential_norm = potential.alpha.norm();
        result.first_response = dot(direction, potential.alpha);
        result.kernel_bilinear = dot(direction, (plus.alpha - minus.alpha) / (2.0 * h));
        return result;
    }

    void print(const FixedDensityLedger &value, const FixedDensityLedger &reference)
    {
        std::cout << std::setprecision(12)
                  << "G3_FIXED_XC policy=" << value.policy
                  << " points=" << value.points
                  << " energy=" << value.energy
                  << " potential_norm=" << value.potential_norm
                  << " D_V=" << value.first_response
                  << " D_fxc_D=" << value.kernel_bilinear
                  << " delta_energy=" << value.energy - reference.energy
                  << " delta_potential_norm=" << value.potential_norm - reference.potential_norm
                  << " delta_D_V=" << value.first_response - reference.first_response
                  << " delta_D_fxc_D=" << value.kernel_bilinear - reference.kernel_bilinear
                  << '\n';
    }
}

int main()
{
    try
    {
        const auto calculator = make_fixture();
        const Eigen::Index nbasis = static_cast<Eigen::Index>(calculator._shells.nbasis());
        check(nbasis >= 3, "fixture basis is unexpectedly small");

        // Positive, non-SCF density and a traceless symmetric perturbation.
        // They are intentionally held byte-for-byte fixed for every policy:
        // this ledger measures XC quadrature only, with no orbital, amplitude,
        // Fock, or Z-vector response path available to contaminate it.
        Matrix density = 0.15 * Matrix::Identity(nbasis, nbasis);
        Matrix direction = Matrix::Zero(nbasis, nbasis);
        direction(0, 1) = direction(1, 0) = 0.030;
        direction(0, 2) = direction(2, 0) = -0.017;
        direction(1, 1) = 0.011;
        direction(2, 2) = -0.011;

        const auto exchange = functional("gga_x_pbe");
        const auto correlation = functional("gga_c_pbe");
        struct Candidate
        {
            const char *label;
            DFT::GridRequest request;
        };
        const std::vector<Candidate> candidates{
            {"legacy", {DFT::GridLevel::Normal, 0, 135,
                         DFT::GridPruningPolicy::LegacyOrcaIndexV1}},
            {"legacy-no-light-reduction", {DFT::GridLevel::Normal, 0, 135,
                                             DFT::GridPruningPolicy::LegacyOrcaNoLightReductionV1}},
            {"legacy-unpruned-max", {DFT::GridLevel::Normal, 0, 135,
                                       DFT::GridPruningPolicy::LegacyOrcaUnprunedMaxV1}},
            {"pyscf-nwchem-2.13", {DFT::GridLevel::Normal, 0, 135,
                                     DFT::GridPruningPolicy::PySCFNWChem213}},
            // This G2-validated uniform rule is the independent angular
            // reference. Unlike a named pruning candidate it deliberately
            // has the same 1454 Lebedev order on every radial shell.
            {"uniform-1454-reference", {DFT::GridLevel::Normal, 1454, 135,
                                          DFT::GridPruningPolicy::LegacyOrcaIndexV1}}};

        std::vector<FixedDensityLedger> ledger;
        ledger.reserve(candidates.size());
        for (const auto &candidate : candidates)
        {
            const auto spec = require(DFT::resolve_grid_spec(candidate.request));
            ledger.push_back(measure(calculator, spec, density, direction, exchange, correlation));
            ledger.back().policy = candidate.label;
            check(std::isfinite(ledger.back().energy) && std::isfinite(ledger.back().potential_norm) &&
                      std::isfinite(ledger.back().first_response) && std::isfinite(ledger.back().kernel_bilinear),
                  "non-finite XC ledger value for " + ledger.back().policy);
            check(ledger.back().points > 0, "empty molecular grid for " + ledger.back().policy);
        }

        // Resolution and quadrature are deterministic. Re-measure the
        // physical-radius PySCF candidate to catch an accidental mutable cache
        // or an unrecorded policy fallback before using its numbers as G3 data.
        const auto py_spec = require(DFT::resolve_grid_spec(candidates[3].request));
        const auto repeat = measure(calculator, py_spec, density, direction, exchange, correlation);
        check(repeat.points == ledger[3].points, "repeat changed PySCF point count");
        check(std::abs(repeat.energy - ledger[3].energy) < 1.0e-13,
              "repeat changed PySCF fixed-density energy");
        check(std::abs(repeat.first_response - ledger[3].first_response) < 1.0e-12,
              "repeat changed PySCF D:V contraction");
        check(std::abs(repeat.kernel_bilinear - ledger[3].kernel_bilinear) < 1.0e-9,
              "repeat changed PySCF D:fxc:D contraction");

        for (const auto &value : ledger)
            print(value, ledger.back());
    }
    catch (const std::exception &error)
    {
        std::cerr << "fixed-grid XC ledger exception: " << error.what() << '\n';
        return 1;
    }
    return passed ? 0 : 1;
}
