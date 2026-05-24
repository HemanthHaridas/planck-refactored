#include "io/fcidump.h"

#include "io/logging.h"
#include "post_hf/integrals.h"

#include <Eigen/Core>

#include <cmath>
#include <cstdio>
#include <format>
#include <unordered_map>

namespace HartreeFock::IO
{
    using HartreeFock::LogLevel;
    using HartreeFock::Logger::logging;

    namespace
    {
        // Mapping from Mulliken irrep label to MOLPRO ORBSYM number, per point
        // group. This mirrors PySCF's tools/fcidump.py ORBSYM_MAP so an FCIDUMP
        // written here is byte-compatible with the symmetry numbering every
        // FCIDUMP-consuming solver expects. Only one-dimensional (Abelian) point
        // groups are listed; higher groups carry degenerate irreps that the flat
        // FCIDUMP ORBSYM field cannot represent, so they fall back to all-ones.
        const std::unordered_map<std::string, std::unordered_map<std::string, int>> &orbsym_table()
        {
            static const std::unordered_map<std::string, std::unordered_map<std::string, int>> table = {
                {"D2h", {{"Ag", 1}, {"B1g", 4}, {"B2g", 6}, {"B3g", 7}, {"Au", 8}, {"B1u", 5}, {"B2u", 3}, {"B3u", 2}}},
                {"C2v", {{"A1", 1}, {"A2", 4}, {"B1", 2}, {"B2", 3}}},
                {"C2h", {{"Ag", 1}, {"Bg", 4}, {"Au", 2}, {"Bu", 3}}},
                {"D2", {{"A", 1}, {"B1", 4}, {"B2", 3}, {"B3", 2}}},
                {"Cs", {{"A'", 1}, {"A\"", 2}}},
                {"C2", {{"A", 1}, {"B", 2}}},
                {"Ci", {{"Ag", 1}, {"Au", 2}}},
                {"C1", {{"A", 1}}},
            };
            return table;
        }
    } // namespace

    std::vector<int> molpro_orbsym(
        const std::string &point_group,
        const std::vector<std::string> &mo_labels)
    {
        const auto &table = orbsym_table();
        auto pg_it = table.find(point_group);
        if (pg_it == table.end())
            return {}; // unsupported group → caller falls back to all-ones

        const auto &label_map = pg_it->second;
        std::vector<int> orbsym(mo_labels.size(), 0);
        for (std::size_t i = 0; i < mo_labels.size(); ++i)
        {
            auto it = label_map.find(mo_labels[i]);
            if (it == label_map.end())
                return {}; // an unrecognized label means we cannot trust any of them
            orbsym[i] = it->second;
        }
        return orbsym;
    }

    std::expected<void, std::string> write_fcidump(
        HartreeFock::Calculator &calc,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const std::string &path)
    {
        const std::string tag = "FCIDUMP :";

        // ── Reference guards ───────────────────────────────────────────────────
        // The FCIDUMP layout assumes a single common spatial-orbital set for both
        // spins. RHF has one set; ROHF stores one common set in both channels.
        // (A UHF dump would need the unrestricted FCIDUMP extension, which most
        // consumers do not read — out of scope.)
        if (!calc._info._is_converged)
            return std::unexpected(tag + " requires a converged RHF or ROHF reference.");
        if (calc._scf._scf != HartreeFock::SCFType::RHF &&
            calc._scf._scf != HartreeFock::SCFType::ROHF)
            return std::unexpected(tag + " only RHF or ROHF references are supported.");

        // The MO-basis Hamiltonian spans the SCF working basis: in spherical mode
        // that is the (2L+1)-per-shell spherical basis (working_nbasis()), and the
        // MO coefficients, _hcore, and the cached ERI are all spherical to match.
        // In Cartesian mode working_nbasis() == nbasis(), so this is a no-op there.
        const int nbasis = static_cast<int>(calc.working_nbasis());
        if (nbasis <= 0)
            return std::unexpected(tag + " empty basis.");

        const Eigen::MatrixXd &C = calc._info._scf.alpha.mo_coefficients;
        if (C.rows() != nbasis || C.cols() != nbasis)
            return std::unexpected(tag + " MO coefficient matrix has wrong size.");

        // ── Electron count and spin ────────────────────────────────────────────
        const int n_total_elec =
            static_cast<int>(calc._molecule.atomic_numbers.cast<int>().sum()) - calc._molecule.charge;
        if (n_total_elec <= 0)
            return std::unexpected(tag + " non-positive electron count.");
        const int multiplicity = static_cast<int>(calc._molecule.multiplicity);
        const int ms2 = multiplicity - 1; // 2*Sz for the high-spin reference

        // ── MO-basis integrals (Chemists' (ij|kl), MO basis) ───────────────────
        // h(i,j) = Cᵀ H_core C ;  (ij|kl) from the full four-index transform.
        // These are exactly the two ingredients run_fci builds, so the FCIDUMP and
        // Planck's own FCI see an identical Hamiltonian.
        std::vector<double> eri_local;
        const std::vector<double> &eri =
            HartreeFock::Correlation::ensure_eri(calc, shell_pairs, eri_local, tag);

        const Eigen::MatrixXd h_mo = C.transpose() * calc._hcore * C;
        const std::vector<double> g =
            HartreeFock::Correlation::transform_eri_internal(eri, nbasis, C);
        // g is row-major (ij|kl): g[((i*n + j)*n + k)*n + l].

        // ── ORBSYM ─────────────────────────────────────────────────────────────
        std::vector<int> orbsym;
        if (calc._geometry._use_symm)
        {
            const std::vector<std::string> &labels = calc._info._scf.alpha.mo_symmetry;
            if (static_cast<int>(labels.size()) == nbasis)
                orbsym = molpro_orbsym(calc._molecule._point_group, labels);
        }
        const bool have_symmetry = (static_cast<int>(orbsym.size()) == nbasis);
        if (!have_symmetry)
            orbsym.assign(nbasis, 1); // all totally-symmetric → solver ignores symmetry

        // ── Write the file ─────────────────────────────────────────────────────
        std::FILE *fout = std::fopen(path.c_str(), "w");
        if (fout == nullptr)
            return std::unexpected(std::format("{} cannot open '{}' for writing.", tag, path));

        // Header. Indices are 1-based throughout the FCIDUMP body, ISYM=1.
        std::fprintf(fout, " &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n", nbasis, n_total_elec, ms2);
        std::fprintf(fout, "  ORBSYM=");
        for (int i = 0; i < nbasis; ++i)
            std::fprintf(fout, "%d,", orbsym[i]);
        std::fprintf(fout, "\n");
        std::fprintf(fout, "  ISYM=1,\n");
        std::fprintf(fout, " &END\n");

        constexpr double tol = 1e-15;
        // All integral values are written with 15 decimal digits in a fixed-width,
        // right-aligned field ("%23.15f") so the value column lines up regardless
        // of sign or magnitude (width 23 fits "-<two int digits>.<15 frac digits>").
        const auto idx4 = [nbasis](int i, int j, int k, int l) -> std::size_t
        {
            return ((static_cast<std::size_t>(i) * nbasis + j) * nbasis + k) * nbasis + l;
        };

        // Two-electron integrals, 8-fold permutational symmetry. The canonical
        // unique set is i>=j, k>=l, ij>=kl with ij = i(i+1)/2 + j (compound index).
        for (int i = 0; i < nbasis; ++i)
            for (int j = 0; j <= i; ++j)
            {
                const int ij = i * (i + 1) / 2 + j;
                for (int k = 0; k <= i; ++k)
                {
                    const int l_max = (k == i) ? j : k;
                    for (int l = 0; l <= l_max; ++l)
                    {
                        const int kl = k * (k + 1) / 2 + l;
                        if (ij < kl)
                            continue;
                        const double v = g[idx4(i, j, k, l)];
                        if (std::abs(v) > tol)
                            std::fprintf(fout, "%23.15f %4d %4d %4d %4d\n",
                                         v, i + 1, j + 1, k + 1, l + 1);
                    }
                }
            }

        // One-electron integrals h(i,j), lower triangle (k=l=0 marks 1e block).
        for (int i = 0; i < nbasis; ++i)
            for (int j = 0; j <= i; ++j)
            {
                const double v = h_mo(i, j);
                if (std::abs(v) > tol)
                    std::fprintf(fout, "%23.15f %4d %4d  0  0\n", v, i + 1, j + 1);
            }

        // Scalar nuclear repulsion (and any frozen-core constant — zero here).
        std::fprintf(fout, "%23.15f  0  0  0  0\n", calc._nuclear_repulsion);

        std::fclose(fout);

        logging(LogLevel::Info, tag,
                std::format("Wrote {} orbitals, {} electrons (MS2={}) to '{}' [{}]",
                            nbasis, n_total_elec, ms2, path,
                            have_symmetry ? ("ORBSYM: " + calc._molecule._point_group)
                                          : "ORBSYM: none (C1)"));
        return {};
    }

} // namespace HartreeFock::IO
