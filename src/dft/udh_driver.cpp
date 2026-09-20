#include "udh_driver.h"
#include "driver.h"
#include "base/mpi_env.h"
#include "post_hf/integrals.h"
#include <algorithm>
#include <cmath>
#include <format>

namespace DFT::Driver
{
    std::expected<UDHGradientResult,std::string> build_udh_gradient_correction(
        HartreeFock::Calculator &calc,PreparedSystem &prepared,
        const XC::Functional &exchange,const XC::Functional &correlation,
        const HartreeFock::Correlation::UMP2Result &r,double c)
    {
        using namespace Gradient;
        using M=Eigen::MatrixXd;
        const auto error=[](const std::string &s)->std::expected<UDHGradientResult,std::string>
        { return std::unexpected("UDH driver: "+s); };
        const bool combined=exchange.is_combined_exchange_correlation();
        if(calc._scf._scf!=HartreeFock::SCFType::UHF || !calc._info._scf.is_uhf ||
            !calc._info._is_converged || HartreeFock::Mpi::distributed())
            return error("requires a converged serial/OpenMP UKS reference");
        UDHPT2Scope scope;
        scope.reference=calc._scf._scf; scope.basis=calc._basis._basis;
        scope.solvent=calc._solvation._model!=HartreeFock::SolvationModel::None;
        scope.range_separated=exchange.is_range_separated() || (!combined && correlation.is_range_separated());
        // Libxc 7 classifies hybrid meta-GGAs as MGGA too; hybrid metadata
        // is separate from the semilocal family. Use the shared wrapper.
        scope.meta_gga=exchange.is_meta_gga_like() || (!combined && correlation.is_meta_gga_like());
        if(scope.solvent || scope.range_separated || scope.meta_gga ||
            scope.basis!=HartreeFock::BasisType::Cartesian || !std::isfinite(c) ||
            !calc._mp2.frozen.empty() || calc._mp2.use_ri || calc._mp2.level_shift!=0 || !calc._mp2.with_t2)
            return error("requires global Cartesian LDA/GGA, all-active conventional unshifted PT2 with T2, no solvent");
        const auto n=static_cast<Eigen::Index>(calc._shells.nbasis());
        const UDHOrbitalMatrices coeff{r.mo_coeff_alpha,r.mo_coeff_beta};
        const UDHSpinMatrices ground{calc._info._scf.alpha.density,calc._info._scf.beta.density};
        if(n==0 || r.nocca<0 || r.noccb<0 || r.nvira<=0 || r.nvirb<=0 ||
            r.nocca+r.nvira!=n || r.noccb+r.nvirb!=n ||
            coeff.alpha.rows()!=n || coeff.alpha.cols()!=n || coeff.beta.rows()!=n || coeff.beta.cols()!=n ||
            r.mo_energy_alpha.size()!=n || r.mo_energy_beta.size()!=n ||
            ground.alpha.rows()!=n || ground.alpha.cols()!=n || ground.beta.rows()!=n || ground.beta.cols()!=n ||
            !coeff.alpha.allFinite() || !coeff.beta.allFinite() || !ground.alpha.allFinite() || !ground.beta.allFinite())
            return error("inconsistent live UKS/PT2 dimensions or nonfinite data");
        if((ground.alpha-coeff.alpha.leftCols(r.nocca)*coeff.alpha.leftCols(r.nocca).transpose()).cwiseAbs().maxCoeff()>1e-9 ||
            (ground.beta-coeff.beta.leftCols(r.noccb)*coeff.beta.leftCols(r.noccb).transpose()).cwiseAbs().maxCoeff()>1e-9)
            return error("PT2 occupied spaces do not reproduce the live spin densities");
        // Do not trust a cached, possibly iteration-shifted Fock. Rebuild
        // from the same converged P/grid/functionals used by the response.
        auto xc_grid=evaluate_current_density_and_xc(calc,prepared,exchange,correlation);
        if(!xc_grid) return error(xc_grid.error());
        auto potential=assemble_current_ks_potential(calc,prepared,*xc_grid);
        if(!potential) return error(potential.error());
        const UDHOrbitalMatrices f{r.mo_energy_alpha.asDiagonal(),r.mo_energy_beta.asDiagonal()};
        UDHGradientResult out;
        out.physical_fock_error=std::max(
            (coeff.alpha.transpose()*(calc._hcore+potential->alpha)*coeff.alpha-f.alpha).cwiseAbs().maxCoeff(),
            (coeff.beta.transpose()*(calc._hcore+potential->beta)*coeff.beta-f.beta).cwiseAbs().maxCoeff());
        if(!std::isfinite(out.physical_fock_error) || out.physical_fock_error>1e-8)
            return error(std::format("physical unshifted canonical Fock does not match PT2 snapshot (max error {:.6e}, limit 1e-8)",
                out.physical_fock_error));

        std::vector<double> storage;
        const auto &eri=HartreeFock::Correlation::ensure_eri(calc,prepared.shell_pairs,storage,"UDH Eq. 22:");
        UDHFullMOIntegrals full;
        full.n_alpha=n; full.n_beta=n;
        full.aa=HartreeFock::Correlation::transform_eri(eri,n,coeff.alpha,coeff.alpha,coeff.alpha,coeff.alpha);
        full.ab=HartreeFock::Correlation::transform_eri(eri,n,coeff.alpha,coeff.alpha,coeff.beta,coeff.beta);
        full.bb=HartreeFock::Correlation::transform_eri(eri,n,coeff.beta,coeff.beta,coeff.beta,coeff.beta);
        UDHPT2DirectIntegrals direct;
        for(int block=0;block<3;++block)
        {
            int os=block==2?r.noccb:r.nocca,ot=block==0?r.nocca:r.noccb;
            int vs=block==2?r.nvirb:r.nvira,vt=block==0?r.nvira:r.nvirb;
            const auto &g=block==0?full.aa:block==1?full.ab:full.bb;
            auto &d=block==0?direct.aa:block==1?direct.ab:direct.bb;
            for(int i=0;i<os;++i) for(int j=0;j<ot;++j) for(int a=0;a<vs;++a) for(int b=0;b<vt;++b)
                d.push_back(g[((static_cast<std::size_t>(i)*n+os+a)*n+j)*n+ot+b]);
        }
        auto stationary=build_udh_pt2_stationary_contract(r,direct,calc._overlap,calc._mp2,c,scope);
        if(!stationary) return error(stationary.error());
        out.energy=stationary->energy;
        const UDHPT2Amplitudes t{r.nocca,r.noccb,r.nvira,r.nvirb,r.t2_aa,r.t2_ab,r.t2_bb};
        const double ax=exchange.fock_exchange_coefficient();
        auto response=make_udh_direct_ks_response_operator({&prepared.shell_pairs,&prepared.molecular_grid,
            &prepared.ao_grid,ground,&exchange,&correlation,ax,calc._integral._engine,calc._integral._tol_eri});
        if(!response) return error(response.error());
        auto rhs=build_udh_orbital_rhs(t,full,c,coeff,calc._overlap,f,*response);
        if(!rhs) return error(rhs.error());
        auto z=solve_udh_zvector({coeff,f,calc._overlap,r.nocca,r.noccb,*response},rhs->total_ai);
        if(!z) return error(z.error());
        out.rhs=rhs->total_ai; out.z=z->z_ai; out.z_residual=z->residual_max_abs;
        double rhs_max=0;
        for(const M *m:{&out.rhs.alpha,&out.rhs.beta}) if(m->size()) rhs_max=std::max(rhs_max,m->cwiseAbs().maxCoeff());
        out.scaled_z_residual=out.z_residual/(1+rhs_max);
        if(!std::isfinite(out.scaled_z_residual) || out.scaled_z_residual>1e-10)
            return error("scaled Z residual exceeds 1e-10");
        auto relaxed=build_udh_relaxed_difference_density(rhs->dprime,z->z_ai);
        if(!relaxed) return error(relaxed.error());
        auto w=build_udh_overlap_density(rhs->pair,*relaxed,coeff,f,calc._overlap,*response);
        if(!w) return error(w.error());
        auto gamma=build_udh_two_particle_density(t,c,*relaxed,coeff,calc._overlap,ax);
        if(!gamma) return error(gamma.error());
        auto dx=build_dh_eq33_derivative_integrals(calc);
        if(!dx) return error(dx.error());
        auto nonxc=build_udh_non_xc_gradient(*gamma,*w,*dx);
        if(!nonxc) return error(nonxc.error());
        out.non_xc=std::move(*nonxc);
        const UDHXCIIInputs xci{&calc._molecule,&calc._shells,&prepared.molecular_grid,&prepared.ao_grid,
            ground,gamma->relaxed_ao.spin,&exchange,&correlation};
        if(exchange.is_lda_like() && (combined || correlation.is_lda_like()))
        {
            auto xc=build_udh_lda_complete_xc_ii_gradient(xci);
            if(!xc) return error(xc.error());
            out.xc=std::move(*xc);
        }
        else
        {
            auto hessian=evaluate_ao_hessian_on_grid(calc._shells,prepared.molecular_grid);
            if(!hessian) return error(hessian.error());
            auto xc=build_udh_gga_complete_xc_ii_gradient({xci,&*hessian});
            if(!xc) return error(xc.error());
            out.xc=std::move(*xc);
        }
        const auto atoms=static_cast<Eigen::Index>(calc._molecule.natoms);
        if(out.non_xc.total.size()!=3*atoms || out.xc.total.rows()!=atoms || out.xc.total.cols()!=3)
            return error("invalid correction coordinate layout");
        out.correction=out.xc.total;
        for(Eigen::Index a=0;a<atoms;++a) for(int q=0;q<3;++q) out.correction(a,q)+=out.non_xc.total(3*a+q);
        if(!out.correction.allFinite()) return error("nonfinite Eq. 33 correction");
        out.assembled=true;
        return out;
    }
}
