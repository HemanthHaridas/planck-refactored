// U10/U11 test-only front end. The normal driver owns SCF/PT2, assembly,
// requested-frame rotation and ordinary JSON. Additional evidence goes in
// a separate sidecar, never in the production JSON schema or environment.
#include "dft/driver.h"
#include "io/io.h"
#include "io/results_json.h"
#include "integrals/os.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

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
}
int main(int argc,const char **argv)
{
    try
    {
        std::string input,json,reference;
        DFT::Driver::Options options;
        DFT::Driver::UDHGradientResult ledger;
        options.udh_gradient_validation=&ledger;
        for(int k=1;k<argc;++k)
        {
            std::string a=argv[k];
            if(a=="--json" && k+1<argc) json=argv[++k];
            else if(a=="--reference" && k+1<argc) reference=argv[++k];
            else if(a=="--scale" && k+1<argc) options.udh_pt2_scale_validation=std::stod(argv[++k]);
            else if(input.empty() && !a.starts_with("--")) input=a;
            else throw std::runtime_error("Unexpected argument: "+a);
        }
        if(input.empty() || json.empty()) throw std::runtime_error(
            "usage: planck-udh-validation INPUT --json PATH [--reference INPUT] [--scale C]");
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
        std::ofstream out(json+".state.json");
        if(!out) throw std::runtime_error("Cannot open state sidecar");
        out<<std::setprecision(17)<<"{\"converged\":true,\"uks\":"<<(uks?"true":"false")
           <<",\"nalpha\":"<<na<<",\"nbeta\":"<<nb<<",\"s2\":"<<s2
           <<",\"grid_angular_points\":"<<c._dft._grid_angular_points
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
