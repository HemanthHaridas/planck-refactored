#include "populations/population.h"

#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <string>

namespace
{
    bool expect(bool condition, const std::string &message)
    {
        if (condition)
            return true;
        std::cerr << message << '\n';
        return false;
    }

    bool near(double value, double expected, double tol = 1e-12)
    {
        return std::abs(value - expected) <= tol;
    }

    HartreeFock::Molecule make_h2_molecule()
    {
        HartreeFock::Molecule molecule;
        molecule.natoms = 2;
        molecule.atomic_numbers.resize(2);
        molecule.atomic_numbers << 1, 1;
        return molecule;
    }

    void initialize_two_function_basis(HartreeFock::Basis &basis)
    {
        basis._shells.resize(2);
        basis._shells[0]._atom_index = 0;
        basis._shells[1]._atom_index = 1;

        HartreeFock::ContractedView bf0;
        bf0._shell = &basis._shells[0];
        bf0._index = 0;
        basis._basis_functions.push_back(bf0);

        HartreeFock::ContractedView bf1;
        bf1._shell = &basis._shells[1];
        bf1._index = 1;
        basis._basis_functions.push_back(bf1);
    }
} // namespace

int main()
{
    bool ok = true;

    const HartreeFock::Molecule molecule = make_h2_molecule();
    HartreeFock::Basis basis;
    initialize_two_function_basis(basis);

    Eigen::MatrixXd S(2, 2);
    S << 1.0, 0.2,
        0.2, 1.0;

    Eigen::MatrixXd P(2, 2);
    P << 1.0, 0.5,
        0.5, 1.0;

    Eigen::MatrixXd spin(2, 2);
    spin << 0.6, 0.0,
        0.0, 0.2;

    const auto analysis =
        HartreeFock::SCF::mulliken_population_analysis(molecule, basis, S, P, &spin);

    ok &= expect(static_cast<bool>(analysis),
                 "Mulliken analysis should accept compatible molecule, basis, overlap, and density data");
    if (analysis)
    {
        ok &= expect(analysis->atoms.size() == 2,
                     "Mulliken analysis should return one row per atom");
        ok &= expect(near(analysis->atoms[0].electron_population, 1.1),
                     "atom 1 Mulliken population should include half-shared overlap density through the AO row sum");
        ok &= expect(near(analysis->atoms[1].electron_population, 1.1),
                     "atom 2 Mulliken population should include half-shared overlap density through the AO row sum");
        ok &= expect(near(analysis->total_electrons, 2.2),
                     "total Mulliken population should equal Tr(P S)");
        ok &= expect(near(analysis->total_charge, -0.2),
                     "total Mulliken charge should be nuclear charge minus Mulliken electron population");
        ok &= expect(near(analysis->atoms[0].spin_population, 0.6),
                     "atom 1 spin population should be accumulated from the spin-density matrix");
        ok &= expect(near(analysis->atoms[1].spin_population, 0.2),
                     "atom 2 spin population should be accumulated from the spin-density matrix");
        ok &= expect(near(analysis->total_spin_population, 0.8),
                     "total spin population should equal Tr((P_alpha-P_beta) S)");
    }

    const auto lowdin =
        HartreeFock::SCF::lowdin_population_analysis(molecule, basis, S, P, &spin);

    ok &= expect(static_cast<bool>(lowdin),
                 "Löwdin analysis should accept compatible molecule, basis, overlap, and density data");
    if (lowdin)
    {
        ok &= expect(near(lowdin->atoms[0].electron_population, 1.1),
                     "atom 1 Löwdin population should equal the diagonal of S^(1/2) P S^(1/2) for this symmetric toy case");
        ok &= expect(near(lowdin->atoms[1].electron_population, 1.1),
                     "atom 2 Löwdin population should equal the diagonal of S^(1/2) P S^(1/2) for this symmetric toy case");
        ok &= expect(near(lowdin->atoms[0].spin_population, 0.5959591794226539, 1e-10),
                     "atom 1 Löwdin spin population should come from the diagonal of S^(1/2) (P_alpha-P_beta) S^(1/2)");
        ok &= expect(near(lowdin->atoms[1].spin_population, 0.2040408205773456, 1e-10),
                     "atom 2 Löwdin spin population should come from the diagonal of S^(1/2) (P_alpha-P_beta) S^(1/2)");
        ok &= expect(near(lowdin->total_electrons, 2.2),
                     "total Löwdin population should equal Tr(P S)");
        ok &= expect(near(lowdin->total_spin_population, 0.8),
                     "total Löwdin spin population should equal Tr((P_alpha-P_beta) S)");
    }

    Eigen::MatrixXd alpha(2, 2);
    alpha << 0.8, 0.25,
        0.25, 0.6;
    Eigen::MatrixXd beta(2, 2);
    beta << 0.2, 0.25,
        0.25, 0.4;

    const auto mayer =
        HartreeFock::SCF::mayer_bond_order_analysis(molecule, basis, S, P, &alpha, &beta);

    ok &= expect(static_cast<bool>(mayer),
                 "Mayer bond-order analysis should accept compatible molecule, basis, overlap, and spin densities");
    if (mayer)
    {
        ok &= expect(near(mayer->bond_orders(0, 1), 0.2474, 1e-12),
                     "Mayer bond order should be the sum of alpha and beta (P S) cross products for the atom pair");
        ok &= expect(near(mayer->bond_orders(1, 0), 0.2474, 1e-12),
                     "Mayer bond-order matrix should be symmetric");
        ok &= expect(near(mayer->bond_orders(0, 0), 0.0),
                     "Mayer bond-order diagonal should remain zero");
    }

    const Eigen::MatrixXd bad_density = Eigen::MatrixXd::Identity(3, 3);
    const auto bad =
        HartreeFock::SCF::mulliken_population_analysis(molecule, basis, S, bad_density);
    ok &= expect(!bad,
                 "Mulliken analysis should reject density matrices with the wrong AO dimension");

    basis._shells[1]._atom_index = 5;
    const auto bad_atom =
        HartreeFock::SCF::mulliken_population_analysis(molecule, basis, S, P);
    ok &= expect(!bad_atom,
                 "Mulliken analysis should reject basis functions whose atom index is outside the molecule");

    return ok ? 0 : 1;
}
