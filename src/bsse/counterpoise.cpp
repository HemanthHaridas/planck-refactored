#include <format>
#include <string>
#include <vector>

#include "bsse/counterpoise.h"

#include "base/tables.h"
#include "basis/basis.h"
#include "integrals/base.h"
#include "integrals/shellpair.h"
#include "io/logging.h"
#include "scf/scf.h"

namespace HartreeFock
{
    namespace BSSE
    {
        namespace
        {
            // Specification of one SCF sub-calculation: which parent atoms are
            // kept as real atoms, which are kept as ghosts (basis only), and the
            // charge / multiplicity of the resulting (sub)system. Atoms not listed
            // in either set are dropped from the sub-calculation entirely.
            struct SubCalcSpec
            {
                std::string label;
                std::vector<int> real_atoms;  // parent indices kept as real atoms
                std::vector<int> ghost_atoms; // parent indices kept as ghosts
                int charge = 0;
                unsigned int multiplicity = 1;
            };

            // Build a fresh Calculator for one sub-calculation by copying the
            // parent's options and selecting / ghosting the requested atoms.
            // Symmetry and checkpointing are disabled; the geometry frame is taken
            // directly from the parent (already in Bohr) so no re-detection runs.
            HartreeFock::Calculator build_subcalculator(
                const HartreeFock::Calculator &parent, const SubCalcSpec &spec)
            {
                HartreeFock::Calculator sub;

                // Copy the option blocks that drive an SCF energy. Deliberately do
                // NOT copy _bsse (avoid recursion), checkpoint path, or cached
                // matrices — the sub-calc rebuilds everything from scratch.
                sub._scf = parent._scf;
                sub._basis = parent._basis;
                sub._geometry = parent._geometry;
                sub._integral = parent._integral;
                sub._output = parent._output;
                sub._calculation = HartreeFock::CalculationType::SinglePoint;
                sub._correlation = HartreeFock::PostHF::None;

                // Symmetry off, and a core-Hamiltonian guess for every sub-calc.
                // HCore is used (rather than carrying the parent's guess) for two
                // reasons: (1) counterpoise sub-calcs must never read a checkpoint,
                // which would collide across the five runs; (2) the SAD guess
                // false-converges to a wrong SCF minimum for small closed-shell
                // systems such as an isolated He atom (see docs/BSSE_PLAN.md "SAD
                // caveat"), which would corrupt the monomer reference energies.
                // HCore reproduces the PySCF reference to 1e-10 for these small
                // fragments, and the CP sub-systems are small enough that the extra
                // SCF iterations are negligible.
                sub._geometry._use_symm = false;
                sub._scf._guess = HartreeFock::SCFGuess::HCore;
                sub._scf._save_checkpoint = false;

                // Spin treatment: a fragment with unpaired electrons (mult > 1)
                // forces UHF; a closed-shell fragment keeps the parent's RHF/UHF
                // type. ROHF requests are honored as-is.
                sub._scf._scf = parent._scf._scf;
                if (spec.multiplicity > 1 && sub._scf._scf == HartreeFock::SCFType::RHF)
                    sub._scf._scf = HartreeFock::SCFType::UHF;

                // ── Assemble the sub-molecule ────────────────────────────────────
                const std::size_t n_real = spec.real_atoms.size();
                const std::size_t n_ghost = spec.ghost_atoms.size();
                const std::size_t n = n_real + n_ghost;

                HartreeFock::Molecule &m = sub._molecule;
                m.natoms = n;
                m.charge = spec.charge;
                m.multiplicity = spec.multiplicity;
                m.atomic_numbers.resize(static_cast<Eigen::Index>(n));
                m.atomic_masses.resize(static_cast<Eigen::Index>(n));
                m.is_ghost.assign(n, false);
                m.coordinates.resize(static_cast<Eigen::Index>(n), 3);
                m._coordinates.resize(static_cast<Eigen::Index>(n), 3);

                const HartreeFock::Molecule &pm = parent._molecule;
                auto place = [&](std::size_t dst, int src, bool ghost) {
                    m.atomic_numbers[static_cast<Eigen::Index>(dst)] =
                        pm.atomic_numbers[static_cast<Eigen::Index>(src)];
                    m.atomic_masses[static_cast<Eigen::Index>(dst)] =
                        pm.atomic_masses[static_cast<Eigen::Index>(src)];
                    m.is_ghost[dst] = ghost;
                    // Parent _coordinates are already in Bohr (prepare_coordinates
                    // ran before the CP driver). Copy them straight across.
                    m.coordinates.row(static_cast<Eigen::Index>(dst)) =
                        pm._coordinates.row(static_cast<Eigen::Index>(src));
                };

                std::size_t dst = 0;
                for (int src : spec.real_atoms)
                    place(dst++, src, false);
                for (int src : spec.ghost_atoms)
                    place(dst++, src, true);

                // The parent's _coordinates are in Bohr, so the sub-molecule's
                // coordinates are too: mark them and set the standard frame
                // directly (no symmetry detection, no Angstrom conversion).
                m._is_bohr = true;
                m._coordinates = m.coordinates;
                m.set_standard_from_bohr(m.coordinates);
                m._symmetry = false;
                m._point_group = "C1";

                return sub;
            }

            // Run a single sub-calculation's SCF and return its total energy.
            // Mirrors the driver's single-point energy path, minus checkpoint,
            // symmetry, and spherical handling (CP sub-calcs are Cartesian and
            // symmetry-free). Output is silenced; only the returned energy matters.
            std::expected<double, std::string> run_subcalculation(
                HartreeFock::Calculator &sub)
            {
                HartreeFock::Logger::ScopedSilence silence;

                // Basis
                const std::string gbs_file =
                    sub._basis._basis_path + "/" + sub._basis._basis_name;
                auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(
                    gbs_file, sub._molecule, sub._basis._basis);
                if (!basis_res)
                    return std::unexpected(basis_res.error());
                sub._shells = std::move(*basis_res);

                // Nuclear repulsion + SCF data structures
                if (auto res = sub.initialize(); !res)
                    return std::unexpected(res.error());

                // Shell pairs and one-electron integrals
                std::vector<HartreeFock::ShellPair> shellpairs =
                    build_shellpairs(sub._shells);

                auto [S, T] = _compute_1e(shellpairs, sub._shells.nbasis(),
                                          sub._integral._engine, nullptr);
                Eigen::MatrixXd V = _compute_nuclear_attraction(
                    shellpairs, sub._shells.nbasis(), sub._molecule,
                    sub._integral._engine, nullptr);
                sub._overlap = S;
                sub._hcore = T + V;

                // SCF
                switch (sub._scf._scf)
                {
                case HartreeFock::SCFType::RHF:
                    if (auto res = HartreeFock::SCF::run_rhf(sub, shellpairs, nullptr); !res)
                        return std::unexpected(res.error());
                    break;
                case HartreeFock::SCFType::ROHF:
                    if (auto res = HartreeFock::SCF::run_rohf(sub, shellpairs, nullptr); !res)
                        return std::unexpected(res.error());
                    break;
                case HartreeFock::SCFType::UHF:
                    if (auto res = HartreeFock::SCF::run_uhf(sub, shellpairs, nullptr); !res)
                        return std::unexpected(res.error());
                    break;
                }

                return sub._total_energy;
            }

            // Run one SubCalcSpec end to end (build + SCF) with a progress line.
            std::expected<double, std::string> run_spec(
                const HartreeFock::Calculator &parent, const SubCalcSpec &spec)
            {
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Counterpoise :",
                                             std::format("Running sub-calculation: {}", spec.label));
                HartreeFock::Calculator sub = build_subcalculator(parent, spec);
                auto e = run_subcalculation(sub);
                if (!e)
                    return std::unexpected(std::format("{} failed: {}", spec.label, e.error()));
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Counterpoise :",
                                             std::format("  {} : {:.10f} Eh", spec.label, *e));
                return *e;
            }

            void log_report(const CounterpoiseResult &r)
            {
                using HartreeFock::Logger::logging;
                const double kcal = HARTREE_TO_KCALMOL;

                HartreeFock::Logger::blank();
                logging(HartreeFock::LogLevel::Info, "Counterpoise / BSSE Report :", "");
                logging(HartreeFock::LogLevel::Info, "",
                        std::format("E(AB)  dimer basis        : {:18.10f} Eh", r.e_dimer));
                logging(HartreeFock::LogLevel::Info, "",
                        std::format("E(A)   monomer basis      : {:18.10f} Eh", r.e_mono_a));
                logging(HartreeFock::LogLevel::Info, "",
                        std::format("E(B)   monomer basis      : {:18.10f} Eh", r.e_mono_b));
                logging(HartreeFock::LogLevel::Info, "",
                        std::format("E(A)*  dimer basis (CP)   : {:18.10f} Eh", r.e_mono_a_cp));
                logging(HartreeFock::LogLevel::Info, "",
                        std::format("E(B)*  dimer basis (CP)   : {:18.10f} Eh", r.e_mono_b_cp));
                HartreeFock::Logger::blank();
                logging(HartreeFock::LogLevel::Info, "",
                        std::format("BSSE                      : {:18.10f} Eh = {:10.4f} kcal/mol",
                                    r.bsse, r.bsse * kcal));
                logging(HartreeFock::LogLevel::Info, "",
                        std::format("Interaction (uncorrected) : {:18.10f} Eh = {:10.4f} kcal/mol",
                                    r.interaction_raw, r.interaction_raw * kcal));
                logging(HartreeFock::LogLevel::Info, "",
                        std::format("Interaction (CP-corrected): {:18.10f} Eh = {:10.4f} kcal/mol",
                                    r.interaction_cp, r.interaction_cp * kcal));
                HartreeFock::Logger::blank();
            }
        } // namespace

        std::expected<CounterpoiseResult, std::string>
        run_counterpoise(const HartreeFock::Calculator &parent)
        {
            const auto &bsse = parent._bsse;
            if (bsse._fragments.size() != 2)
                return std::unexpected("counterpoise: expected exactly two fragments");

            const std::vector<int> &A = bsse._fragments[0];
            const std::vector<int> &B = bsse._fragments[1];

            auto frag_charge = [&](std::size_t f) -> int {
                return bsse._charges.empty() ? 0 : bsse._charges[f];
            };
            auto frag_mult = [&](std::size_t f) -> unsigned int {
                return bsse._multiplicities.empty()
                           ? 1u
                           : static_cast<unsigned int>(bsse._multiplicities[f]);
            };

            // Dimer uses the parent's own charge / multiplicity.
            std::vector<int> all_atoms = A;
            all_atoms.insert(all_atoms.end(), B.begin(), B.end());

            SubCalcSpec dimer{"E(AB) dimer", all_atoms, {}, parent._molecule.charge,
                              parent._molecule.multiplicity};
            SubCalcSpec mono_a{"E(A) monomer", A, {}, frag_charge(0), frag_mult(0)};
            SubCalcSpec mono_b{"E(B) monomer", B, {}, frag_charge(1), frag_mult(1)};
            SubCalcSpec mono_a_cp{"E(A)* CP", A, B, frag_charge(0), frag_mult(0)};
            SubCalcSpec mono_b_cp{"E(B)* CP", B, A, frag_charge(1), frag_mult(1)};

            CounterpoiseResult r;

            auto e_dimer = run_spec(parent, dimer);
            if (!e_dimer)
                return std::unexpected(e_dimer.error());
            r.e_dimer = *e_dimer;

            auto e_a = run_spec(parent, mono_a);
            if (!e_a)
                return std::unexpected(e_a.error());
            r.e_mono_a = *e_a;

            auto e_b = run_spec(parent, mono_b);
            if (!e_b)
                return std::unexpected(e_b.error());
            r.e_mono_b = *e_b;

            auto e_a_cp = run_spec(parent, mono_a_cp);
            if (!e_a_cp)
                return std::unexpected(e_a_cp.error());
            r.e_mono_a_cp = *e_a_cp;

            auto e_b_cp = run_spec(parent, mono_b_cp);
            if (!e_b_cp)
                return std::unexpected(e_b_cp.error());
            r.e_mono_b_cp = *e_b_cp;

            r.bsse = (r.e_mono_a_cp - r.e_mono_a) + (r.e_mono_b_cp - r.e_mono_b);
            r.interaction_raw = r.e_dimer - r.e_mono_a - r.e_mono_b;
            r.interaction_cp = r.e_dimer - r.e_mono_a_cp - r.e_mono_b_cp;

            log_report(r);
            return r;
        }
    } // namespace BSSE
} // namespace HartreeFock
