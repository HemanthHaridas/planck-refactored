// Unit test for native RI 3-center Coulomb integrals (mu nu | P).
//
// This is a correctness gate against the local PySCF reference available in
// tests/pyscf/.venv. The production compute_3c_eri implementation is native
// C++; PySCF is used here only as the oracle.

#include <Eigen/Core>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unistd.h>

#include "base/types.h"
#include "basis/basis.h"
#include "basis/rifit.h"
#include "integrals/shellpair.h"
#include "lookup/elements.h"
#include "post_hf/ri/ri_eri.h"

namespace
{
    struct PyscfThreeCenterReference
    {
        Eigen::MatrixXd j3c;
        Eigen::VectorXd ao_overlap_diag;
        Eigen::VectorXd aux_overlap_diag;
    };

    bool g_ok = true;

    void fail(const std::string &m)
    {
        std::cerr << "FAIL: " << m << '\n';
        g_ok = false;
    }

    std::filesystem::path repo_root()
    {
        std::filesystem::path here(__FILE__);
        return here.parent_path().parent_path();
    }

    std::filesystem::path unique_temp_path(const std::string &stem, const char *ext)
    {
        const auto tmp = std::filesystem::temp_directory_path();
        const auto pid = static_cast<unsigned long long>(::getpid());
        for (int attempt = 0; attempt < 64; ++attempt)
        {
            const auto candidate = tmp / (stem + "-" + std::to_string(pid) + "-" +
                                          std::to_string(attempt) + ext);
            if (!std::filesystem::exists(candidate))
                return candidate;
        }
        return tmp / (stem + "-" + std::to_string(pid) + ext);
    }

    std::expected<PyscfThreeCenterReference, std::string> pyscf_reference_j3c(
        const HartreeFock::Molecule &mol,
        const std::filesystem::path &basis_file,
        const std::filesystem::path &aux_file,
        const std::vector<std::string> &planck_ao_keys,
        const std::vector<std::string> &planck_aux_keys)
    {
        const auto root = repo_root();
        const auto python = root / "tests" / "pyscf" / ".venv" / "bin" / "python";
        const auto script = root / "scripts" / "export_ri_3c_pyscf.py";
        if (!std::filesystem::exists(python))
            return std::unexpected("PySCF interpreter not found: " + python.string());
        if (!std::filesystem::exists(script))
            return std::unexpected("PySCF export helper not found: " + script.string());

        const auto input = unique_temp_path("planck-ri3c-test", ".json");
        const auto output = unique_temp_path("planck-ri3c-test", ".bin");

        {
            std::ofstream js(input);
            if (!js)
                return std::unexpected("Cannot create PySCF JSON input.");

            js << "{\n";
            js << "  \"basis_file\": \"" << basis_file.string() << "\",\n";
            js << "  \"aux_basis_file\": \"" << aux_file.string() << "\",\n";
            js << "  \"planck_ao_keys\": [\n";
            for (std::size_t i = 0; i < planck_ao_keys.size(); ++i)
            {
                js << "    \"" << planck_ao_keys[i] << "\"";
                if (i + 1 != planck_ao_keys.size())
                    js << ",";
                js << "\n";
            }
            js << "  ],\n";
            js << "  \"planck_aux_keys\": [\n";
            for (std::size_t i = 0; i < planck_aux_keys.size(); ++i)
            {
                js << "    \"" << planck_aux_keys[i] << "\"";
                if (i + 1 != planck_aux_keys.size())
                    js << ",";
                js << "\n";
            }
            js << "  ],\n";
            js << "  \"atoms\": [\n";
            for (std::size_t i = 0; i < mol.natoms; ++i)
            {
                auto coord = mol._standard.row(i);
                const auto element = element_from_z(mol.atomic_numbers[i]);
                if (!element)
                {
                    std::filesystem::remove(input);
                    return std::unexpected("element lookup failed");
                }
                js << "    {\"symbol\": \"" << element->symbol << "\", \"coords_bohr\": ["
                   << coord(0) << ", " << coord(1) << ", " << coord(2) << "]}";
                if (i + 1 != mol.natoms)
                    js << ",";
                js << "\n";
            }
            js << "  ]\n";
            js << "}\n";
        }

        const std::string cmd = "\"" + python.string() + "\" \"" + script.string() +
                                "\" \"" + input.string() + "\" \"" + output.string() + "\"";
        const int status = std::system(cmd.c_str());
        std::filesystem::remove(input);
        if (status != 0)
        {
            std::filesystem::remove(output);
            return std::unexpected("PySCF helper failed.");
        }

        std::ifstream in(output, std::ios::binary);
        std::filesystem::remove(output);
        if (!in)
            return std::unexpected("Cannot open PySCF export output.");

        std::uint64_t rows = 0, cols = 0, nao = 0, naux = 0;
        in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
        in.read(reinterpret_cast<char *>(&cols), sizeof(cols));
        in.read(reinterpret_cast<char *>(&nao), sizeof(nao));
        in.read(reinterpret_cast<char *>(&naux), sizeof(naux));
        if (!in)
            return std::unexpected("Failed reading PySCF export header.");

        PyscfThreeCenterReference out;
        out.j3c.resize(rows, cols);
        out.ao_overlap_diag.resize(nao);
        out.aux_overlap_diag.resize(naux);

        in.read(reinterpret_cast<char *>(out.j3c.data()),
                static_cast<std::streamsize>(rows * cols * sizeof(double)));
        in.read(reinterpret_cast<char *>(out.ao_overlap_diag.data()),
                static_cast<std::streamsize>(nao * sizeof(double)));
        in.read(reinterpret_cast<char *>(out.aux_overlap_diag.data()),
                static_cast<std::streamsize>(naux * sizeof(double)));
        if (!in)
            return std::unexpected("Failed reading PySCF export payload.");
        return out;
    }

    std::string component_label(const Eigen::Vector3i &am)
    {
        std::string out;
        out.append(static_cast<std::size_t>(am[0]), 'x');
        out.append(static_cast<std::size_t>(am[1]), 'y');
        out.append(static_cast<std::size_t>(am[2]), 'z');
        return out;
    }

    template <typename ShellContainer>
    std::vector<std::string> build_basis_keys(const ShellContainer &shells)
    {
        std::vector<std::string> keys;
        std::map<unsigned, unsigned> shell_counts;
        for (const auto &shell : shells)
        {
            const unsigned atom = shell._atom_index;
            const unsigned L = static_cast<unsigned>(shell._shell);
            const unsigned occ = shell_counts[atom]++;
            const auto components = HartreeFock::BasisFunctions::_cartesian_shell_order(L);
            keys.reserve(keys.size() + components.size());
            for (const auto &am : components)
                keys.push_back(std::to_string(atom) + ":" +
                               std::to_string(occ) + ":" + component_label(am));
        }
        return keys;
    }
} // namespace

int main()
{
    using HartreeFock::BasisFunctions::read_gbs_basis;
    using HartreeFock::BasisFunctions::read_ri_basis;
    using HartreeFock::Correlation::RI::compute_3c_eri;
    using HartreeFock::Correlation::RI::ensure_ri_3c_ready;
    using HartreeFock::Correlation::RI::ensure_ri_metric_ready;

    const auto root = repo_root();
    const auto basis_file = root / "basis-sets" / "cc-pVDZ";
    const auto aux_file = root / "basis-sets" / "cc-pVDZ-RIFIT";

    HartreeFock::Molecule mol;
    mol.natoms = 3;
    mol.atomic_numbers.resize(3); mol.atomic_numbers << 8, 1, 1;
    mol._standard.resize(3, 3);
    mol._standard << 0.0,    0.0,    0.0,
                     0.0,    1.43,   1.11,
                     0.0,   -1.43,   1.11;
    mol._standard_is_bohr = true;

    HartreeFock::Calculator calc;
    calc._molecule = mol;
    calc._basis._basis_name = "cc-pVDZ";
    calc._basis._basis_path = (root / "basis-sets").string();
    calc._integral._engine = HartreeFock::IntegralMethod::HeadGordonPople;
    calc._mp2.use_ri = true;
    calc._mp2.ri_basis_name = "cc-pVDZ-RIFIT";
    calc._mp2.ri_basis_path = (root / "basis-sets").string();
    calc._mp2.ri_lindep = 1e-7;

    auto basis_res = read_gbs_basis(basis_file.string(), mol, HartreeFock::BasisType::Cartesian);
    if (!basis_res)
        fail("read_gbs_basis failed: " + basis_res.error());
    else
        calc._shells = std::move(*basis_res);

    auto aux_res = read_ri_basis(aux_file.string(), mol);
    if (!aux_res)
        fail("read_ri_basis failed: " + aux_res.error());
    else
        calc._ri_aux_basis = std::make_shared<HartreeFock::AuxBasis>(std::move(*aux_res));

    if (g_ok)
    {
        auto prep_res = ensure_ri_metric_ready(calc);
        if (!prep_res)
            fail("ensure_ri_metric_ready failed: " + prep_res.error());
    }

    std::expected<Eigen::MatrixXd, std::string> native_res =
        g_ok ? compute_3c_eri(calc) : std::unexpected("skipped");
    if (g_ok && !native_res)
        fail("compute_3c_eri failed: " + native_res.error());

    const auto planck_ao_keys = build_basis_keys(calc._shells._shells);
    const auto planck_aux_keys = build_basis_keys(calc._ri_aux_basis->shells);

    auto ref_res = g_ok ? pyscf_reference_j3c(
                             mol, basis_file, aux_file, planck_ao_keys, planck_aux_keys)
                        : std::unexpected("skipped");
    if (g_ok && !ref_res)
        fail("PySCF reference build failed: " + ref_res.error());

    if (g_ok)
    {
        const auto &native = *native_res;
        const auto &ref_data = *ref_res;
        const auto &ref_raw = ref_data.j3c;
        Eigen::MatrixXd ref = ref_raw;
        if (native.rows() != ref.rows() || native.cols() != ref.cols())
        {
            fail("native/PySCF shape mismatch");
        }
        else
        {
            std::size_t row = 0;
            for (std::size_t mu = 0; mu < calc._shells.nbasis(); ++mu)
            {
                const double mu_norm = 1.0 / std::sqrt(ref_data.ao_overlap_diag(mu));
                for (std::size_t nu = 0; nu <= mu; ++nu, ++row)
                {
                    const double pair_norm =
                        mu_norm / std::sqrt(ref_data.ao_overlap_diag(nu));
                    for (std::size_t col = 0; col < static_cast<std::size_t>(ref_data.aux_overlap_diag.size()); ++col)
                    {
                        ref(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) *=
                            pair_norm / std::sqrt(ref_data.aux_overlap_diag(static_cast<Eigen::Index>(col)));
                    }
                }
            }

            Eigen::Index max_row = 0;
            Eigen::Index max_col = 0;
            const double max_abs =
                (native - ref).cwiseAbs().maxCoeff(&max_row, &max_col);
            if (max_abs > 1e-10)
            {
                std::size_t mu = 0;
                std::size_t nu = 0;
                std::size_t packed = 0;
                for (std::size_t i = 0; i < calc._shells.nbasis(); ++i)
                {
                    for (std::size_t j = 0; j <= i; ++j, ++packed)
                    {
                        if (packed == static_cast<std::size_t>(max_row))
                        {
                            mu = i;
                            nu = j;
                            i = calc._shells.nbasis();
                            break;
                        }
                    }
                }
                const auto &bf_mu = calc._shells._basis_functions[mu];
                const auto &bf_nu = calc._shells._basis_functions[nu];
                const auto mu_center = bf_mu._shell->_center;
                const auto nu_center = bf_nu._shell->_center;
                std::size_t aux_shell_idx = 0;
                for (; aux_shell_idx + 1 < calc._ri_aux_basis->offsets.size(); ++aux_shell_idx)
                {
                    if (calc._ri_aux_basis->offsets[aux_shell_idx + 1] >
                        static_cast<std::size_t>(max_col))
                        break;
                }
                const auto &aux_shell = calc._ri_aux_basis->shells[aux_shell_idx];
                const auto aux_center = aux_shell._center;
                const auto aux_local =
                    static_cast<std::size_t>(max_col) - calc._ri_aux_basis->offsets[aux_shell_idx];
                std::string shell_native = "[";
                std::string shell_ref = "[";
                const auto shell_off = calc._ri_aux_basis->offsets[aux_shell_idx];
                const auto shell_len = (static_cast<std::size_t>(static_cast<int>(aux_shell._shell)) + 1) *
                                       (static_cast<std::size_t>(static_cast<int>(aux_shell._shell)) + 2) / 2;
                for (std::size_t k = 0; k < shell_len; ++k)
                {
                    if (k != 0)
                    {
                        shell_native += ", ";
                        shell_ref += ", ";
                    }
                    shell_native += std::to_string(
                        native(max_row, static_cast<Eigen::Index>(shell_off + k)));
                    shell_ref += std::to_string(
                        ref(max_row, static_cast<Eigen::Index>(shell_off + k)));
                }
                shell_native += "]";
                shell_ref += "]";
                fail("native/PySCF 3c mismatch: max abs = " + std::to_string(max_abs) +
                     " at (" + std::to_string(max_row) + ", " + std::to_string(max_col) +
                     "), native = " + std::to_string(native(max_row, max_col)) +
                     ", ref = " + std::to_string(ref(max_row, max_col)) +
                     ", mu cart = [" + std::to_string(bf_mu._cartesian[0]) + "," +
                     std::to_string(bf_mu._cartesian[1]) + "," +
                     std::to_string(bf_mu._cartesian[2]) + "]" +
                     ", nu cart = [" + std::to_string(bf_nu._cartesian[0]) + "," +
                     std::to_string(bf_nu._cartesian[1]) + "," +
                     std::to_string(bf_nu._cartesian[2]) + "]" +
                     ", mu center = [" + std::to_string(mu_center[0]) + "," +
                     std::to_string(mu_center[1]) + "," +
                     std::to_string(mu_center[2]) + "]" +
                     ", nu center = [" + std::to_string(nu_center[0]) + "," +
                     std::to_string(nu_center[1]) + "," +
                     std::to_string(nu_center[2]) + "]" +
                     ", aux shell L = " + std::to_string(static_cast<int>(aux_shell._shell)) +
                     ", aux local = " + std::to_string(aux_local) +
                     ", aux center = [" + std::to_string(aux_center[0]) + "," +
                     std::to_string(aux_center[1]) + "," +
                     std::to_string(aux_center[2]) + "]" +
                     ", native shell block = " + shell_native +
                     ", ref shell block = " + shell_ref);
            }
        }
    }

    if (g_ok)
    {
        auto cache_res = ensure_ri_3c_ready(calc);
        if (!cache_res)
            fail("ensure_ri_3c_ready failed: " + cache_res.error());
        else if (calc._ri_j3c.rows() == 0 || calc._ri_j3c.cols() == 0)
            fail("ensure_ri_3c_ready did not populate calculator._ri_j3c");
    }

    if (g_ok)
        std::cout << "PASS: ri_3c_eri\n";
    return g_ok ? 0 : 1;
}
