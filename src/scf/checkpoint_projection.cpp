// SCF::project_checkpoint_density, declared in working_state.h. Kept out of
// working_state.cpp because it needs the SCF orthogonalizer (scf.cpp) and the
// checkpoint reader, which would otherwise pull the whole SCF stack into every
// target that only rebuilds basis-dependent state.
#include "working_state.h"

#include <Eigen/Core>

#include "basis/basis.h"
#include "basis/spherical.h"
#include "integrals/os.h"
#include "scf/scf.h"

std::expected<void, std::string>
HartreeFock::SCF::project_checkpoint_density(HartreeFock::Calculator &calc,
                                             const HartreeFock::Checkpoint::MOData &mos)
{
    // The stored orbitals hold a different electron count when the ECP cores
    // differ (an ECP basis against an all-electron one, or two different cores).
    if (mos.ecp_core_electrons != calc._molecule.ecp_core_electrons)
        return std::unexpected("ECP core electrons differ between the checkpoint and current basis");

    auto small = HartreeFock::BasisFunctions::read_gbs_basis(
        calc._basis._basis_path + "/" + mos.basis_name, calc._molecule, calc._basis._basis);
    if (!small)
        return std::unexpected(small.error());

    Eigen::MatrixXd S_cross = HartreeFock::ObaraSaika::_compute_cross_overlap(calc._shells, *small);
    if (calc._shells._spherical)
    {
        if (!mos.has_basis_type)
            return std::unexpected("checkpoint lacks the basis_type metadata a spherical cross-basis restart needs");
        if (mos.basis_type != HartreeFock::BasisType::Spherical)
            return std::unexpected("a spherical cross-basis restart requires a spherical checkpoint");
        if (!small->_spherical || small->nbasis_sph() != mos.nbasis)
            return std::unexpected("checkpoint MO dimension does not match the re-read spherical basis");
        // Both transforms arrive row-normalized from read_gbs_basis.
        S_cross = calc._shells._cart_to_sph * S_cross * small->_cart_to_sph.transpose();
    }
    if (S_cross.rows() != calc._overlap.rows() || S_cross.cols() != static_cast<Eigen::Index>(mos.nbasis))
        return std::unexpected("cross overlap does not match the working overlap and checkpoint dimensions");

    auto X = HartreeFock::SCF::build_orthogonalizer(calc._overlap);
    if (!X)
        return std::unexpected("orthogonalizer failed: " + X.error());

    const int n_elec = calc._molecule.total_nuclear_charge() - calc._molecule.charge;
    const int n_unpaired = static_cast<int>(calc._molecule.multiplicity) - 1;
    const int n_alpha = (n_elec + n_unpaired) / 2;
    const int n_beta = (n_elec - n_unpaired) / 2;
    const bool spin_resolved = calc._scf._scf != HartreeFock::SCFType::RHF;
    const auto project = [&](const Eigen::MatrixXd &C, int n, double factor)
    { return HartreeFock::Checkpoint::project_density(*X, S_cross, C.leftCols(n), factor); };

    if (mos.is_uhf)
    {
        const Eigen::MatrixXd Pa = project(mos.C_alpha, n_alpha, 1.0);
        const Eigen::MatrixXd Pb = project(mos.C_beta, n_beta, 1.0);
        if (spin_resolved)
        {
            calc._info._scf.alpha.density = Pa;
            calc._info._scf.beta.density = Pb;
        }
        else
            calc._info._scf.alpha.density = Pa + Pb;
    }
    else
    {
        // Restricted checkpoint; its orbitals also seed the beta spin.
        calc._info._scf.alpha.density = project(mos.C_alpha, n_alpha, spin_resolved ? 1.0 : 2.0);
        if (spin_resolved)
            calc._info._scf.beta.density = project(mos.C_alpha, n_beta, 1.0);
    }
    return {};
}
