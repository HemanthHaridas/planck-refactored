// U10/U11 test-only front end. The normal driver owns SCF/PT2, assembly,
// requested-frame rotation and ordinary JSON. Additional evidence goes in
// a separate sidecar, never in the production JSON schema or environment.
#include "dft/driver.h"
#include "g4_3_direct_grid.h"
#include "io/io.h"
#include "io/results_json.h"
#include "integrals/os.h"
#include "lookup/loader/license_path.h"
#include "lookup/loader/orchestrator.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{
using M=Eigen::MatrixXd;
void matrix(std::ostream &out,const M &m)
{
    out<<'[';
    for(Eigen::Index i=0;i<m.rows();++i)
    { out<<(i?",[":"["); for(Eigen::Index j=0;j<m.cols();++j) out<<(j?",":"")<<m(i,j); out<<']'; }
    out<<']';
}
void parse(const std::string &path,HartreeFock::Calculator &c)
{
    std::ifstream in(path); if(!in) throw std::runtime_error("Cannot open "+path);
    auto result=HartreeFock::IO::parse_input(in,c);
    if(!result) throw std::runtime_error(result.error());
    c._checkpoint_path=path+".dftchk";
}

template<class T> T require(std::expected<T,std::string> value)
{
    if(!value) throw std::runtime_error(value.error());
    return std::move(*value);
}
double dot(const M &a,const M &b) { return (a.array()*b.array()).sum(); }

DFT::XC::Functional functional(HartreeFock::XCExchangeFunctional f,int explicit_id,DFT::XC::Spin spin)
{
    if(explicit_id>0) return require(DFT::XC::Functional::create(explicit_id,spin));
    const char *name=nullptr;
    switch(f)
    {
    case HartreeFock::XCExchangeFunctional::Slater: name="lda_x"; break;
    case HartreeFock::XCExchangeFunctional::B88: name="gga_x_b88"; break;
    case HartreeFock::XCExchangeFunctional::PW91: name="gga_x_pw91"; break;
    case HartreeFock::XCExchangeFunctional::PBE: name="gga_x_pbe"; break;
    case HartreeFock::XCExchangeFunctional::B3LYP: name="hyb_gga_xc_b3lyp"; break;
    case HartreeFock::XCExchangeFunctional::PBE0: name="hyb_gga_xc_pbeh"; break;
    case HartreeFock::XCExchangeFunctional::Custom:
        throw std::runtime_error("fixed XC ledger requires an explicit exchange libxc id");
    }
    return require(DFT::XC::Functional::create(require(DFT::XC::functional_id(name)),spin));
}
DFT::XC::Functional functional(HartreeFock::XCCorrelationFunctional f,int explicit_id,DFT::XC::Spin spin)
{
    if(explicit_id>0) return require(DFT::XC::Functional::create(explicit_id,spin));
    const char *name=nullptr;
    switch(f)
    {
    case HartreeFock::XCCorrelationFunctional::VWN5: name="lda_c_vwn"; break;
    case HartreeFock::XCCorrelationFunctional::LYP: name="gga_c_lyp"; break;
    case HartreeFock::XCCorrelationFunctional::P86: name="gga_c_p86"; break;
    case HartreeFock::XCCorrelationFunctional::PW91: name="gga_c_pw91"; break;
    case HartreeFock::XCCorrelationFunctional::PBE: name="gga_c_pbe"; break;
    case HartreeFock::XCCorrelationFunctional::Custom:
        throw std::runtime_error("fixed XC ledger requires an explicit correlation libxc id");
    }
    return require(DFT::XC::Functional::create(require(DFT::XC::functional_id(name)),spin));
}
M transition(const M &coeff,int nocc,int ordinal)
{
    if(nocc<=0 || nocc>=coeff.cols()) throw std::runtime_error("fixed XC ledger requires occupied and virtual orbitals");
    const int i=ordinal==0?nocc-1:0;
    const int a=std::min(nocc+ordinal,static_cast<int>(coeff.cols())-1);
    return coeff.col(i)*coeff.col(a).transpose()+coeff.col(a)*coeff.col(i).transpose();
}
void fixed_xc_ledger(const std::string &path,const HartreeFock::Calculator &c,int nalpha,int nbeta,
                     bool reference_ladder_only=false)
{
    using Policy=DFT::GridPruningPolicy;
    const auto spin=c._scf._scf==HartreeFock::SCFType::UHF?DFT::XC::Spin::Polarized:DFT::XC::Spin::Unpolarized;
    const auto x=functional(c._dft._exchange,c._dft._exchange_id,spin);
    const auto corr=functional(c._dft._correlation,c._dft._correlation_id,spin);
    const M pa=c._info._scf.alpha.density;
    const M pb=c._scf._scf==HartreeFock::SCFType::UHF?c._info._scf.beta.density:M::Zero(pa.rows(),pa.cols());
    const M ca=c._info._scf.alpha.mo_coefficients;
    const M cb=c._scf._scf==HartreeFock::SCFType::UHF?c._info._scf.beta.mo_coefficients:ca;
    const std::array<std::pair<M,M>,2> directions{{
        {transition(ca,nalpha,0), c._scf._scf==HartreeFock::SCFType::UHF?transition(cb,nbeta,0):M::Zero(pa.rows(),pa.cols())},
        {transition(ca,nalpha,1), c._scf._scf==HartreeFock::SCFType::UHF?transition(cb,nbeta,1):M::Zero(pa.rows(),pa.cols())}}};
    struct Candidate { const char *name; DFT::GridRequest request; };
    const std::vector<Candidate> candidates=reference_ladder_only
        ? std::vector<Candidate>{{"uniform-2030-a",{DFT::GridLevel::Normal,2030,135,Policy::LegacyOrcaIndexV1}},
                                 {"uniform-2030-b",{DFT::GridLevel::Normal,2030,165,Policy::LegacyOrcaIndexV1}},
                                 {"uniform-2702-c",{DFT::GridLevel::Normal,2702,165,Policy::LegacyOrcaIndexV1}}}
        : std::vector<Candidate>{{"legacy",{DFT::GridLevel::Normal,0,135,Policy::LegacyOrcaIndexV1}},
                                 {"legacy-no-light-reduction",{DFT::GridLevel::Normal,0,135,Policy::LegacyOrcaNoLightReductionV1}},
                                 {"legacy-unpruned-max",{DFT::GridLevel::Normal,0,135,Policy::LegacyOrcaUnprunedMaxV1}},
                                 {"pyscf-nwchem-2.13",{DFT::GridLevel::Normal,0,135,Policy::PySCFNWChem213}},
                                 {"uniform-1454-r1",{DFT::GridLevel::Normal,1454,135,Policy::LegacyOrcaIndexV1}},
                                 {"uniform-2030-r2",{DFT::GridLevel::Normal,2030,165,Policy::LegacyOrcaIndexV1}}};
    std::ofstream out(path); if(!out) throw std::runtime_error("Cannot open fixed XC ledger "+path);
    out<<std::setprecision(17)<<"{\"contract\":\""
       <<(reference_ladder_only?"g4-fixed-xc-reference-ladder-v1":"g4-fixed-stationary-density-v1")
       <<"\",\"polarized\":"
       <<(spin==DFT::XC::Spin::Polarized?"true":"false")<<",\"directions\":2,\"rows\":[";
    constexpr double h=1e-5;
    for(std::size_t k=0;k<candidates.size();++k)
    {
        const auto spec=require(DFT::resolve_grid_spec(candidates[k].request));
        const auto grid=require(DFT::MakeMolecularGrid(c._molecule,spec));
        const auto ao=require(DFT::evaluate_ao_basis_on_grid(c._shells,grid));
        const auto evaluate=[&](const M &alpha,const M &beta)->std::expected<DFT::XCGridEvaluation,std::string>
        {
            if(spin==DFT::XC::Spin::Polarized)
                return DFT::evaluate_xc_on_grid(grid,ao,alpha,beta,x,corr);
            return DFT::evaluate_xc_on_grid(grid,ao,alpha,x,corr);
        };
        const auto xc=require(evaluate(pa,pb));
        const auto v=require(DFT::assemble_xc_matrix(grid,ao,xc));
        std::vector<std::pair<Eigen::Index,Eigen::Index>> owner_ranges;
        for(Eigen::Index lo=0;lo<grid.owner.size();)
        {
            Eigen::Index hi=lo+1;
            while(hi<grid.owner.size() && grid.owner(hi)==grid.owner(lo)) ++hi;
            owner_ranges.emplace_back(lo,hi);
            lo=hi;
        }
        if(owner_ranges.size()!=static_cast<std::size_t>(c._molecule.natoms))
            throw std::runtime_error("fixed XC ledger owner ranges do not match molecule");
        struct Band { int atom,first_shell,last_shell,angular_points,diagnostic_angular_points; Eigen::Index lo,hi; };
        std::vector<Band> bands;
        const bool band_diagnostic=!reference_ladder_only && (std::string_view(candidates[k].name)=="pyscf-nwchem-2.13" ||
                                   std::string_view(candidates[k].name)=="uniform-2030-r2");
        const auto diagnostic_spec=band_diagnostic?require(DFT::resolve_grid_spec(
            DFT::GridRequest{DFT::GridLevel::Normal,0,135,Policy::PySCFNWChem213})):spec;
        if(band_diagnostic) for(int atom=0;atom<c._molecule.natoms;++atom)
        {
            const int z=c._molecule.atomic_numbers(atom);
            const int nr=require(DFT::radial_point_count(z,spec));
            const M radial=DFT::MakeTreutlerAhlrichsGrid(nr,DFT::treutler_xi(z));
            Eigen::Index point=owner_ranges[static_cast<std::size_t>(atom)].first;
            for(int shell=0;shell<nr;)
            {
                const int diagnostic_angular=require(DFT::angular_points_for_shell(
                    z,diagnostic_spec,shell,nr,radial(shell,0)));
                const int first=shell; Eigen::Index lo=point;
                int angular=0;
                do
                {
                    angular=require(DFT::angular_points_for_shell(z,spec,shell,nr,radial(shell,0)));
                    point+=angular; ++shell;
                }
                while(shell<nr && require(DFT::angular_points_for_shell(
                    z,diagnostic_spec,shell,nr,radial(shell,0)))==diagnostic_angular);
                bands.push_back({atom,first,shell-1,angular,diagnostic_angular,lo,point});
            }
            if(point!=owner_ranges[static_cast<std::size_t>(atom)].second)
                throw std::runtime_error("fixed XC ledger radial bands do not cover owner range");
        }
        if(k) out<<',';
        out<<"{\"name\":\""<<candidates[k].name<<"\",\"points\":"<<grid.points.rows()
           <<",\"xc_energy\":"<<xc.total_energy<<",\"alpha_norm\":"<<v.alpha.norm()
           <<",\"beta_norm\":"<<v.beta.norm()<<",\"directions\":[";
        for(std::size_t q=0;q<directions.size();++q)
        {
            const auto &[da,db]=directions[q];
            const auto plus=require(DFT::assemble_xc_matrix(grid,ao,require(evaluate(pa+h*da,pb+h*db))));
            const auto minus=require(DFT::assemble_xc_matrix(grid,ao,require(evaluate(pa-h*da,pb-h*db))));
            if(q) out<<',';
            out<<"{\"D_V\":"<<dot(da,v.alpha)+dot(db,v.beta)
               <<",\"D_fxc_D\":"<<dot(da,(plus.alpha-minus.alpha)/(2*h))+dot(db,(plus.beta-minus.beta)/(2*h))
               <<",\"owners\":[";
            for(std::size_t atom=0;atom<owner_ranges.size();++atom)
            {
                const auto [lo,hi]=owner_ranges[atom];
                const auto owner_v=require(DFT::assemble_xc_matrix(grid,ao,xc,lo,hi));
                const auto owner_plus=require(DFT::assemble_xc_matrix(grid,ao,require(evaluate(pa+h*da,pb+h*db)),lo,hi));
                const auto owner_minus=require(DFT::assemble_xc_matrix(grid,ao,require(evaluate(pa-h*da,pb-h*db)),lo,hi));
                if(atom) out<<',';
                out<<"{\"atom\":"<<atom<<",\"D_V\":"<<dot(da,owner_v.alpha)+dot(db,owner_v.beta)
                   <<",\"D_fxc_D\":"<<dot(da,(owner_plus.alpha-owner_minus.alpha)/(2*h))
                   +dot(db,(owner_plus.beta-owner_minus.beta)/(2*h))<<'}';
            }
            out<<"],\"bands\":[";
            for(std::size_t b=0;b<bands.size();++b)
            {
                const auto &band=bands[b];
                const auto band_v=require(DFT::assemble_xc_matrix(grid,ao,xc,band.lo,band.hi));
                const auto band_plus=require(DFT::assemble_xc_matrix(grid,ao,require(evaluate(pa+h*da,pb+h*db)),band.lo,band.hi));
                const auto band_minus=require(DFT::assemble_xc_matrix(grid,ao,require(evaluate(pa-h*da,pb-h*db)),band.lo,band.hi));
                if(b) out<<',';
                out<<"{\"atom\":"<<band.atom<<",\"first_shell\":"<<band.first_shell
                   <<",\"last_shell\":"<<band.last_shell<<",\"angular_points\":"<<band.angular_points
                   <<",\"diagnostic_angular_points\":"<<band.diagnostic_angular_points
                   <<",\"D_V\":"<<dot(da,band_v.alpha)+dot(db,band_v.beta)
                   <<",\"D_fxc_D\":"<<dot(da,(band_plus.alpha-band_minus.alpha)/(2*h))
                   +dot(db,(band_plus.beta-band_minus.beta)/(2*h))<<'}';
            }
            out<<"]}";
        }
        out<<"]}";
    }
    out<<"]}\n";
}

void write_fixed_xc_shell_layer(std::ostream &out,const HartreeFock::Calculator &c,int nalpha,int nbeta,
                                int angular_points)
{
    using Policy=DFT::GridPruningPolicy;
    const auto spin=c._scf._scf==HartreeFock::SCFType::UHF?DFT::XC::Spin::Polarized:DFT::XC::Spin::Unpolarized;
    const auto x=functional(c._dft._exchange,c._dft._exchange_id,spin);
    const auto corr=functional(c._dft._correlation,c._dft._correlation_id,spin);
    const M pa=c._info._scf.alpha.density;
    const M pb=c._scf._scf==HartreeFock::SCFType::UHF?c._info._scf.beta.density:M::Zero(pa.rows(),pa.cols());
    const M ca=c._info._scf.alpha.mo_coefficients;
    const M cb=c._scf._scf==HartreeFock::SCFType::UHF?c._info._scf.beta.mo_coefficients:ca;
    const std::array<std::pair<M,M>,2> directions{{
        {transition(ca,nalpha,0), c._scf._scf==HartreeFock::SCFType::UHF?transition(cb,nbeta,0):M::Zero(pa.rows(),pa.cols())},
        {transition(ca,nalpha,1), c._scf._scf==HartreeFock::SCFType::UHF?transition(cb,nbeta,1):M::Zero(pa.rows(),pa.cols())}}};
    const auto spec=require(DFT::resolve_grid_spec({DFT::GridLevel::Normal,angular_points,135,Policy::LegacyOrcaIndexV1}));
    const auto grid=require(DFT::MakeMolecularGrid(c._molecule,spec));
    const auto ao=require(DFT::evaluate_ao_basis_on_grid(c._shells,grid));
    const auto evaluate=[&](const M &alpha,const M &beta)->std::expected<DFT::XCGridEvaluation,std::string>
    {
        return spin==DFT::XC::Spin::Polarized?DFT::evaluate_xc_on_grid(grid,ao,alpha,beta,x,corr):
                                                DFT::evaluate_xc_on_grid(grid,ao,alpha,x,corr);
    };
    const auto xc=require(evaluate(pa,pb));
    const auto v=require(DFT::assemble_xc_matrix(grid,ao,xc));
    constexpr double h=1e-5;
    std::array<DFT::XCGridEvaluation,2> plus,minus;
    for(std::size_t q=0;q<directions.size();++q)
    {
        const auto &[da,db]=directions[q];
        plus[q]=require(evaluate(pa+h*da,pb+h*db));
        minus[q]=require(evaluate(pa-h*da,pb-h*db));
    }
    out<<std::setprecision(17)<<"{\"lebedev_points\":"<<angular_points<<",\"polarized\":"
       <<(spin==DFT::XC::Spin::Polarized?"true":"false")<<",\"points\":"<<grid.points.rows()
       <<",\"xc_energy\":"<<xc.total_energy<<",\"directions\":[";
    for(std::size_t q=0;q<directions.size();++q)
    {
        const auto &[da,db]=directions[q];
        const auto vp=require(DFT::assemble_xc_matrix(grid,ao,plus[q]));
        const auto vm=require(DFT::assemble_xc_matrix(grid,ao,minus[q]));
        if(q) out<<',';
        out<<"{\"D_V\":"<<dot(da,v.alpha)+dot(db,v.beta)
           <<",\"D_fxc_D\":"<<dot(da,(vp.alpha-vm.alpha)/(2*h))+dot(db,(vp.beta-vm.beta)/(2*h))<<'}';
    }
    out<<"],\"records\":[";
    Eigen::Index record=0;
    for(int atom=0;atom<c._molecule.natoms;++atom)
    {
        const int z=c._molecule.atomic_numbers(atom);
        const int nr=require(DFT::radial_point_count(z,spec));
        const M radial=DFT::MakeTreutlerAhlrichsGrid(nr,DFT::treutler_xi(z));
        Eigen::Index lo=0;
        for(int previous=0;previous<atom;++previous)
            lo+=static_cast<Eigen::Index>(angular_points)*require(DFT::radial_point_count(c._molecule.atomic_numbers(previous),spec));
        for(int shell=0;shell<nr;++shell,lo+=angular_points)
        {
            const Eigen::Index hi=lo+angular_points;
            const auto slice_v=require(DFT::assemble_xc_matrix(grid,ao,xc,lo,hi));
            if(record++) out<<',';
            out<<"{\"atom\":"<<atom<<",\"atomic_number\":"<<z<<",\"radial_index\":"<<shell
               <<",\"radius_bohr\":"<<radial(shell,0)<<",\"u\":"<<radial(shell,0)/DFT::bragg_radius(z)
               <<",\"lebedev_points\":"<<angular_points<<",\"xc_energy\":"
               <<grid.points.col(3).segment(lo,angular_points).dot(xc.energy_density.segment(lo,angular_points))
               <<",\"direction_records\":[";
            for(std::size_t q=0;q<directions.size();++q)
            {
                const auto &[da,db]=directions[q];
                const auto vp=require(DFT::assemble_xc_matrix(grid,ao,plus[q],lo,hi));
                const auto vm=require(DFT::assemble_xc_matrix(grid,ao,minus[q],lo,hi));
                if(q) out<<',';
                out<<"{\"index\":"<<q<<",\"spin_channels\":[{\"spin\":\"alpha\",\"D_V\":"
                   <<dot(da,slice_v.alpha)<<",\"D_fxc_D\":"<<dot(da,(vp.alpha-vm.alpha)/(2*h))
                   <<"},{\"spin\":\"beta\",\"D_V\":"<<dot(db,slice_v.beta)<<",\"D_fxc_D\":"
                   <<dot(db,(vp.beta-vm.beta)/(2*h))<<"}]}";
            }
            out<<"]}";
        }
    }
    out<<"]}";
}

void fixed_xc_shell_atlas(const std::string &path,const HartreeFock::Calculator &c,int nalpha,int nbeta)
{
    std::ofstream out(path); if(!out) throw std::runtime_error("Cannot open fixed XC shell atlas "+path);
    out<<"{\"contract\":\"g4-fixed-xc-shell-layers-v1\",\"layers\":[";
    write_fixed_xc_shell_layer(out,c,nalpha,nbeta,2030);
    out<<"]}\n";
}

void fixed_xc_shell_residuals(const std::string &path,const HartreeFock::Calculator &c,int nalpha,int nbeta)
{
    constexpr std::array<int,19> angular_sizes{{26,50,86,110,146,170,194,230,266,302,350,434,590,
                                                  770,974,1202,1454,1730,2030}};
    std::ofstream out(path); if(!out) throw std::runtime_error("Cannot open fixed XC shell residuals "+path);
    out<<"{\"contract\":\"g4-fixed-xc-shell-layers-v1\",\"layers\":[";
    for(std::size_t k=0;k<angular_sizes.size();++k)
    {
        if(k) out<<',';
        write_fixed_xc_shell_layer(out,c,nalpha,nbeta,angular_sizes[k]);
    }
    const auto legacy=require(DFT::resolve_grid_spec(
        {DFT::GridLevel::Normal,0,135,DFT::GridPruningPolicy::LegacyOrcaIndexV1}));
    out<<"],\"legacy_selection\":[";
    bool first=true;
    for(int atom=0;atom<c._molecule.natoms;++atom)
    {
        const int z=c._molecule.atomic_numbers(atom);
        const int nr=require(DFT::radial_point_count(z,legacy));
        const M radial=DFT::MakeTreutlerAhlrichsGrid(nr,DFT::treutler_xi(z));
        for(int shell=0;shell<nr;++shell)
        {
            if(!first) out<<',';
            first=false;
            out<<"{\"atom\":"<<atom<<",\"atomic_number\":"<<z<<",\"radial_index\":"<<shell
               <<",\"lebedev_points\":"
               <<require(DFT::angular_points_for_shell(z,legacy,shell,nr,radial(shell,0)))<<'}';
        }
    }
    out<<"]}\n";
}

void fixed_xc_g43_direct_rescore(const std::string &path,const HartreeFock::Calculator &c,int nalpha,int nbeta,
                                  G43Direct::Candidate candidate=G43Direct::Candidate::G437Finalist)
{
    // G4.3.8: one literal candidate realization, evaluated at the fixed parent
    // density.  The normalized Python audit compares this raw direct value to
    // the independent shell-atlas sum; it never substitutes this rule into SCF.
    const auto spin=c._scf._scf==HartreeFock::SCFType::UHF?DFT::XC::Spin::Polarized:DFT::XC::Spin::Unpolarized;
    const auto x=functional(c._dft._exchange,c._dft._exchange_id,spin);
    const auto corr=functional(c._dft._correlation,c._dft._correlation_id,spin);
    const M pa=c._info._scf.alpha.density;
    const M pb=c._scf._scf==HartreeFock::SCFType::UHF?c._info._scf.beta.density:M::Zero(pa.rows(),pa.cols());
    const M ca=c._info._scf.alpha.mo_coefficients;
    const M cb=c._scf._scf==HartreeFock::SCFType::UHF?c._info._scf.beta.mo_coefficients:ca;
    const std::array<std::pair<M,M>,2> directions{{
        {transition(ca,nalpha,0), c._scf._scf==HartreeFock::SCFType::UHF?transition(cb,nbeta,0):M::Zero(pa.rows(),pa.cols())},
        {transition(ca,nalpha,1), c._scf._scf==HartreeFock::SCFType::UHF?transition(cb,nbeta,1):M::Zero(pa.rows(),pa.cols())}}};
    const auto direct=G43Direct::make(c._molecule,candidate);
    const auto ao=require(DFT::evaluate_ao_basis_on_grid(c._shells,direct.values));
    const auto evaluate=[&](const M &alpha,const M &beta)->std::expected<DFT::XCGridEvaluation,std::string>
    {
        return spin==DFT::XC::Spin::Polarized?DFT::evaluate_xc_on_grid(direct.values,ao,alpha,beta,x,corr):
                                                DFT::evaluate_xc_on_grid(direct.values,ao,alpha,x,corr);
    };
    const auto xc=require(evaluate(pa,pb));
    const auto v=require(DFT::assemble_xc_matrix(direct.values,ao,xc));
    constexpr double h=1e-5;
    std::ofstream out(path); if(!out) throw std::runtime_error("Cannot open G4.3 direct rescore "+path);
    out<<std::setprecision(17)<<"{\"contract\":\"g4-fixed-xc-g4-3-direct-v1\",\"fingerprint\":\""
       <<G43Direct::fingerprint(candidate)<<"\",\"radial_points\":"<<G43Direct::radial_points
       <<",\"polarized\":"<<(spin==DFT::XC::Spin::Polarized?"true":"false")
       <<",\"directions\":2,\"candidate\":{\"points\":"<<direct.values.points.rows()
       <<",\"xc_energy\":"<<xc.total_energy<<",\"alpha_norm\":"<<v.alpha.norm()
       <<",\"beta_norm\":"<<v.beta.norm()<<",\"directions\":[";
    for(std::size_t q=0;q<directions.size();++q)
    {
        const auto &[da,db]=directions[q];
        const auto plus=require(DFT::assemble_xc_matrix(direct.values,ao,require(evaluate(pa+h*da,pb+h*db))));
        const auto minus=require(DFT::assemble_xc_matrix(direct.values,ao,require(evaluate(pa-h*da,pb-h*db))));
        if(q) out<<',';
        out<<"{\"D_V\":"<<dot(da,v.alpha)+dot(db,v.beta)
           <<",\"D_fxc_D\":"<<dot(da,(plus.alpha-minus.alpha)/(2*h))+dot(db,(plus.beta-minus.beta)/(2*h))<<'}';
    }
    out<<"],\"shells\":[";
    for(std::size_t k=0;k<direct.shells.size();++k)
    {
        const auto &shell=direct.shells[k];
        if(k) out<<',';
        out<<"{\"atom\":"<<shell.atom<<",\"atomic_number\":"<<shell.atomic_number
           <<",\"radial_index\":"<<shell.radial_index<<",\"radius_bohr\":"<<shell.radius_bohr
           <<",\"u\":"<<shell.u<<",\"lebedev_points\":"<<shell.lebedev_points
           <<",\"begin\":"<<shell.begin<<",\"end\":"<<shell.end<<'}';
    }
    out<<"]}}\n";
}
}
int main(int argc,const char **argv)
{
    try
    {
        const std::string blob_url_template =
            std::string("https://boys-blob-worker.boys-core.workers.dev/{version}/planck_boys-") +
            HartreeFock::Loader::get_platform_string() + ".enc";
        if(auto initialized=HartreeFock::Loader::initialize_boys_library(
               HartreeFock::Loader::get_license_path(), blob_url_template); !initialized)
            throw std::runtime_error("Boys loader initialization failed: "+initialized.error());

        std::string input,json,reference,fixed_xc_ledger_path,fixed_xc_reference_ladder_path,fixed_xc_shell_atlas_path,fixed_xc_shell_residuals_path,fixed_xc_g43_direct_rescore_path,fixed_xc_g43_revised_direct_rescore_path,fixed_xc_g43_dual_floor_direct_rescore_path,fixed_xc_g43_v3_direct_rescore_path,fixed_xc_g43_v4_direct_rescore_path;
        DFT::Driver::Options options;
        DFT::Driver::UDHGradientResult ledger;
        options.udh_gradient_validation=&ledger;
        for(int k=1;k<argc;++k)
        {
            std::string a=argv[k];
            if(a=="--json" && k+1<argc) json=argv[++k];
            else if(a=="--reference" && k+1<argc) reference=argv[++k];
            else if(a=="--scale" && k+1<argc) options.udh_pt2_scale_validation=std::stod(argv[++k]);
            else if(a=="--fixed-xc-ledger" && k+1<argc) fixed_xc_ledger_path=argv[++k];
            else if(a=="--fixed-xc-reference-ladder" && k+1<argc) fixed_xc_reference_ladder_path=argv[++k];
            else if(a=="--fixed-xc-shell-atlas" && k+1<argc) fixed_xc_shell_atlas_path=argv[++k];
            else if(a=="--fixed-xc-shell-residuals" && k+1<argc) fixed_xc_shell_residuals_path=argv[++k];
            else if(a=="--fixed-xc-g4-3-direct-rescore" && k+1<argc) fixed_xc_g43_direct_rescore_path=argv[++k];
            else if(a=="--fixed-xc-g4-3-revised-direct-rescore" && k+1<argc) fixed_xc_g43_revised_direct_rescore_path=argv[++k];
            else if(a=="--fixed-xc-g4-3-dual-floor-direct-rescore" && k+1<argc) fixed_xc_g43_dual_floor_direct_rescore_path=argv[++k];
            else if(a=="--fixed-xc-g4-3-v3-direct-rescore" && k+1<argc) fixed_xc_g43_v3_direct_rescore_path=argv[++k];
            else if(a=="--fixed-xc-g4-3-v4-direct-rescore" && k+1<argc) fixed_xc_g43_v4_direct_rescore_path=argv[++k];
            else if(input.empty() && !a.starts_with("--")) input=a;
            else throw std::runtime_error("Unexpected argument: "+a);
        }
        if(input.empty() || json.empty()) throw std::runtime_error(
            "usage: planck-udh-validation INPUT --json PATH [--reference INPUT] [--scale C] [--fixed-xc-ledger PATH] [--fixed-xc-reference-ladder PATH] [--fixed-xc-shell-atlas PATH] [--fixed-xc-shell-residuals PATH] [--fixed-xc-g4-3-direct-rescore PATH] [--fixed-xc-g4-3-revised-direct-rescore PATH] [--fixed-xc-g4-3-dual-floor-direct-rescore PATH] [--fixed-xc-g4-3-v3-direct-rescore PATH] [--fixed-xc-g4-3-v4-direct-rescore PATH]");
        const int fixed_xc_outputs=(!fixed_xc_ledger_path.empty())+(!fixed_xc_reference_ladder_path.empty())+
                                   (!fixed_xc_shell_atlas_path.empty())+(!fixed_xc_shell_residuals_path.empty())+
                                   (!fixed_xc_g43_direct_rescore_path.empty())+(!fixed_xc_g43_revised_direct_rescore_path.empty())+
                                   (!fixed_xc_g43_dual_floor_direct_rescore_path.empty())+(!fixed_xc_g43_v3_direct_rescore_path.empty())+
                                   (!fixed_xc_g43_v4_direct_rescore_path.empty());
        if(fixed_xc_outputs>1)
            throw std::runtime_error("choose one fixed-XC test sidecar");
        HartreeFock::Calculator c{}; parse(input,c);
        auto result=DFT::Driver::run(c,options);
        if(!result) throw std::runtime_error(result.error());
        if(!result->converged) throw std::runtime_error("UKS/RKS did not converge");
        auto saved=HartreeFock::IO::dump_results_json(c,json);
        if(!saved) throw std::runtime_error(saved.error());
        const bool uks=c._scf._scf==HartreeFock::SCFType::UHF;
        int electrons=-c._molecule.charge;
        for(auto z:c._molecule.atomic_numbers) electrons+=z;
        int na=(electrons+static_cast<int>(c._molecule.multiplicity)-1)/2,nb=electrons-na;
        const M ca=c._info._scf.alpha.mo_coefficients;
        const M cb=uks?c._info._scf.beta.mo_coefficients:ca;
        const M occ_overlap=ca.leftCols(na).transpose()*c._overlap*cb.leftCols(nb);
        const double sz=.5*(na-nb),s2=sz*(sz+1)+nb-occ_overlap.squaredNorm();
        if(!fixed_xc_ledger_path.empty()) fixed_xc_ledger(fixed_xc_ledger_path,c,na,nb);
        if(!fixed_xc_reference_ladder_path.empty())
            fixed_xc_ledger(fixed_xc_reference_ladder_path,c,na,nb,true);
        if(!fixed_xc_shell_atlas_path.empty()) fixed_xc_shell_atlas(fixed_xc_shell_atlas_path,c,na,nb);
        if(!fixed_xc_shell_residuals_path.empty()) fixed_xc_shell_residuals(fixed_xc_shell_residuals_path,c,na,nb);
        if(!fixed_xc_g43_direct_rescore_path.empty()) fixed_xc_g43_direct_rescore(fixed_xc_g43_direct_rescore_path,c,na,nb);
        if(!fixed_xc_g43_revised_direct_rescore_path.empty())
            fixed_xc_g43_direct_rescore(fixed_xc_g43_revised_direct_rescore_path,c,na,nb,
                                         G43Direct::Candidate::G439SecondRowFloorRevision);
        if(!fixed_xc_g43_dual_floor_direct_rescore_path.empty())
            fixed_xc_g43_direct_rescore(fixed_xc_g43_dual_floor_direct_rescore_path,c,na,nb,
                                         G43Direct::Candidate::G439DualEnergyFloorRevision);
        if(!fixed_xc_g43_v3_direct_rescore_path.empty())
            fixed_xc_g43_direct_rescore(fixed_xc_g43_v3_direct_rescore_path,c,na,nb,
                                         G43Direct::Candidate::G43V3ConstrainedCandidate);
        if(!fixed_xc_g43_v4_direct_rescore_path.empty())
            fixed_xc_g43_direct_rescore(fixed_xc_g43_v4_direct_rescore_path,c,na,nb,
                                         G43Direct::Candidate::G43V4RobustCandidate);
        std::ofstream out(json+".state.json");
        if(!out) throw std::runtime_error("Cannot open state sidecar");
        out<<std::setprecision(17)<<"{\"converged\":true,\"uks\":"<<(uks?"true":"false")
           <<",\"nalpha\":"<<na<<",\"nbeta\":"<<nb<<",\"s2\":"<<s2
           <<",\"grid_angular_points\":"<<c._dft._grid_angular_points
           <<",\"grid_radial_points\":"<<c._dft._grid_radial_points
           <<",\"grid_policy\":\""<<c._dft._grid_policy<<"\""
           <<",\"assembled\":"<<(ledger.assembled?"true":"false")<<",\"coordinates_standard\":";
        matrix(out,c._molecule._standard); out<<",\"coordinates_requested\":"; matrix(out,c._molecule._coordinates);
        out<<",\"occupied_alpha\":"; matrix(out,ca.leftCols(na));
        out<<",\"occupied_beta\":"; matrix(out,cb.leftCols(nb));
        out<<",\"overlap\":"; matrix(out,c._overlap);
        if(!reference.empty())
        {
            HartreeFock::Calculator ref{}; parse(reference,ref);
            auto prepared=DFT::Driver::prepare(ref);
            if(!prepared) throw std::runtime_error(prepared.error());
            out<<",\"cross_overlap_reference_current\":";
            matrix(out,HartreeFock::ObaraSaika::_compute_cross_overlap(ref._shells,c._shells));
        }
        if(ledger.assembled)
        {
            out<<",\"scale\":"<<ledger.energy.c_pt2<<",\"pt2_energy\":"<<ledger.energy.correction
               <<",\"scaled_z_residual\":"<<ledger.scaled_z_residual
               <<",\"physical_fock_error\":"<<ledger.physical_fock_error;
            for(auto [name,m]:{std::pair{"ks_standard",&ledger.ks},std::pair{"correction_standard",&ledger.correction},
                std::pair{"total_standard",&ledger.total},std::pair{"xc_p",&ledger.xc.p_side_fixed},
                std::pair{"xc_d",&ledger.xc.d_side_ao},std::pair{"xc_partition",&ledger.xc.becke_partition},
                std::pair{"xc_point",&ledger.xc.point_translation}})
            { out<<",\""<<name<<"\":"; matrix(out,*m); }
            for(auto [name,v]:{std::pair{"h",&ledger.non_xc.one_electron},std::pair{"overlap_term",&ledger.non_xc.overlap},
                std::pair{"J",&ledger.non_xc.coulomb},std::pair{"Ka",&ledger.non_xc.exchange_alpha},
                std::pair{"Kb",&ledger.non_xc.exchange_beta},std::pair{"pair_aa",&ledger.non_xc.pair_aa},
                std::pair{"pair_ab",&ledger.non_xc.pair_ab},std::pair{"pair_bb",&ledger.non_xc.pair_bb}})
            { out<<",\""<<name<<"\":"; matrix(out,*v); }
        }
        out<<"}\n";
        if(!out) throw std::runtime_error("Failed writing state sidecar");
        std::cout<<"Converged : true\nUDH validation-only front end completed\n";
        return 0;
    }
    catch(const std::exception &e) { std::cerr<<"UDH validation failed: "<<e.what()<<'\n'; return 1; }
}
