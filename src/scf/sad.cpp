#include "sad.h"

#include <Eigen/Eigenvalues>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "basis/basis.h"
#include "integrals/os.h"
#include "integrals/shellpair.h"
#include "io/logging.h"
#include "lookup/elements.h"
#include "scf.h"

// ---------------------------------------------------------------------------
// GBS parsing utilities — mirror of the static helpers in gaussian.cpp.
// Kept file-local so the two TUs remain independent.
// ---------------------------------------------------------------------------
namespace
{
    // ── GBS parsing types ────────────────────────────────────────────────────

    struct GbsPrimitive
    {
        double exponent;
        double coefficient;
    };

    struct GbsShell
    {
        std::string label;
        std::vector<GbsPrimitive> primitives;
    };

    using BasisSet = std::unordered_map<std::string, std::vector<GbsShell>>;

    static bool starts_with_alpha(const std::string &line)
    {
        for (char c : line)
        {
            if (!std::isspace(static_cast<unsigned char>(c)))
                return std::isalpha(static_cast<unsigned char>(c));
        }
        return false;
    }

    static bool is_shell_label(const std::string &s)
    {
        return s == "S" || s == "P" || s == "D" ||
               s == "F" || s == "G" || s == "H" ||
               s == "SP";
    }

    static void normalize_fortran_exponents(std::string &line)
    {
        for (char &c : line)
            if (c == 'D' || c == 'd')
                c = 'E';
    }

    static int double_factorial(int n)
    {
        if (n <= 0)
            return 1;
        int result = 1;
        while (n > 0)
        {
            result *= n;
            n -= 2;
        }
        return result;
    }

    static std::expected<BasisSet, std::string> read_gbs(std::ifstream &input)
    {
        BasisSet basis;
        std::string line;
        std::string current_element;

        while (std::getline(input, line))
        {
            if (line.empty() || line.starts_with("!"))
                continue;

            if (line == "****")
            {
                current_element.clear();
                continue;
            }

            std::istringstream header(line);
            std::string symbol;
            int charge;
            if ((header >> symbol >> charge) && header.eof())
            {
                auto _ = element_from_symbol(symbol);
                current_element = symbol;
                basis.try_emplace(symbol);
                continue;
            }

            if (!starts_with_alpha(line))
                return std::unexpected("Expected shell header, got: " + line);

            if (current_element.empty())
                return std::unexpected("Shell before element header");

            std::istringstream iss(line);
            std::string label;
            std::size_t nprim;
            double scale = 1.0;
            iss >> label >> nprim >> scale;

            if (!iss || !is_shell_label(label))
                return std::unexpected("Malformed shell line: " + line);

            if (label == "SP")
            {
                GbsShell s{"S"}, p{"P"};
                for (std::size_t i = 0; i < nprim; ++i)
                {
                    std::getline(input, line);
                    normalize_fortran_exponents(line);
                    std::istringstream prim(line);
                    double expn, cs, cp;
                    prim >> expn >> cs >> cp;
                    s.primitives.push_back({expn, cs * scale});
                    p.primitives.push_back({expn, cp * scale});
                }
                basis[current_element].push_back(std::move(s));
                basis[current_element].push_back(std::move(p));
            }
            else
            {
                GbsShell shell{label};
                for (std::size_t i = 0; i < nprim; ++i)
                {
                    std::getline(input, line);
                    normalize_fortran_exponents(line);
                    std::istringstream prim(line);
                    double expn, cs;
                    prim >> expn >> cs;
                    shell.primitives.push_back({expn, cs * scale});
                }
                basis[current_element].push_back(std::move(shell));
            }
        }
        return basis;
    }

    // Build a single-atom Basis for element `sym` at the origin.
    // Convention matches read_gbs_basis exactly: primitive norms folded into
    // coefficients, Cartesian AOs, _atom_index = 0.
    static HartreeFock::Basis build_atomic_basis(const std::vector<GbsShell> &gbs_shells)
    {
        HartreeFock::Basis basis;

        for (const GbsShell &gbs_shell : gbs_shells)
        {
            HartreeFock::Shell shell;
            shell._center = Eigen::Vector3d::Zero();
            shell._shell = HartreeFock::BasisFunctions::_map_shell_to_L(gbs_shell.label);
            shell._atom_index = 0;

            const std::size_t nprim = gbs_shell.primitives.size();
            shell._primitives.resize(nprim);
            shell._coefficients.resize(nprim);

            for (std::size_t i = 0; i < nprim; ++i)
            {
                shell._primitives[i] = gbs_shell.primitives[i].exponent;
                shell._coefficients[i] = gbs_shell.primitives[i].coefficient;
            }

            const unsigned int L = static_cast<unsigned int>(shell._shell);
            shell._normalizations = HartreeFock::BasisFunctions::primitive_normalization(
                L, shell._primitives);

            const double Nc = HartreeFock::BasisFunctions::contracted_normalization(
                L, shell._primitives, shell._coefficients, shell._normalizations);
            shell._coefficients *= Nc;

            basis._shells.push_back(std::move(shell));
            const HartreeFock::Shell *shell_ptr = &basis._shells.back();

            for (auto am : HartreeFock::BasisFunctions::_cartesian_shell_order(L))
            {
                const std::size_t idx = basis._basis_functions.size();
                const int df = double_factorial(2 * am[0] - 1) *
                               double_factorial(2 * am[1] - 1) *
                               double_factorial(2 * am[2] - 1);
                basis._basis_functions.emplace_back();
                auto &bf = basis._basis_functions.back();
                bf._shell = shell_ptr;
                bf._index = idx;
                bf._component_norm = HartreeFock::BasisFunctions::component_norm(df);
                bf._cartesian = am;
            }
        }
        return basis;
    }

    // ── Minimal atomic multiplicity ───────────────────────────────────────────
    // Returns the smallest valid neutral-atom multiplicity supported by the
    // current atomic UHF path: doublet if Z is odd, singlet if Z is even.
    //
    // This is intentionally a simple fallback rather than a full neutral-atom
    // term-symbol table. The resulting density is immediately spin-summed and
    // shell-averaged before use in RHF SAD, so the important quantity is the
    // shell population rather than the detailed spin state of the isolated atom.
    static unsigned int atomic_minimal_multiplicity(int Z)
    {
        return (Z % 2 == 0) ? 1u : 2u;
    }

    // ── Numerical helpers ─────────────────────────────────────────────────────

    // Symmetric pseudo-inverse via eigenvalue decomposition.
    // Eigenvalues below `thresh` are treated as zero.
    static Eigen::MatrixXd pseudo_inverse_symmetric(const Eigen::MatrixXd &A, double thresh)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
        const Eigen::VectorXd &evals = es.eigenvalues();
        const Eigen::MatrixXd &evecs = es.eigenvectors();

        Eigen::VectorXd inv_evals(evals.size());
        for (Eigen::Index i = 0; i < evals.size(); ++i)
            inv_evals[i] = (evals[i] > thresh) ? 1.0 / evals[i] : 0.0;

        return evecs * inv_evals.asDiagonal() * evecs.transpose();
    }

    // Thresholded S^{-1/2}: zero eigenvalues of S below `thresh`.
    static Eigen::MatrixXd make_thresholded_inverse_sqrt(const Eigen::MatrixXd &S, double thresh)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
        const Eigen::VectorXd &evals = es.eigenvalues();
        const Eigen::MatrixXd &evecs = es.eigenvectors();

        Eigen::VectorXd inv_sqrt(evals.size());
        for (Eigen::Index i = 0; i < evals.size(); ++i)
            inv_sqrt[i] = (evals[i] > thresh) ? 1.0 / std::sqrt(evals[i]) : 0.0;

        return evecs * inv_sqrt.asDiagonal() * evecs.transpose();
    }

    // ── AO index map ─────────────────────────────────────────────────────────

    // Returns, for each atom A, the list of molecular AO indices (row/col in
    // the full molecular density or overlap matrix) that belong to atom A.
    // Does not assume atom-contiguous ordering of AOs.
    static std::vector<std::vector<int>>
    build_atom_to_ao_indices(const HartreeFock::Basis &mol_basis)
    {
        int n_atoms = 0;
        for (const auto &bf : mol_basis._basis_functions)
            n_atoms = std::max(n_atoms, static_cast<int>(bf._shell->_atom_index) + 1);

        std::vector<std::vector<int>> result(static_cast<std::size_t>(n_atoms));
        const HartreeFock::index_t basis_function_count =
            static_cast<HartreeFock::index_t>(mol_basis._basis_functions.size());
        for (HartreeFock::index_t mu = 0; mu < basis_function_count; ++mu)
        {
            const int A = static_cast<int>(mol_basis._basis_functions[static_cast<std::size_t>(mu)]._shell->_atom_index);
            result[static_cast<std::size_t>(A)].push_back(static_cast<int>(mu));
        }
        return result;
    }

    // ── Atomic UHF ───────────────────────────────────────────────────────────

    // Runs atomic UHF for element Z in the given atomic basis (single atom at
    // the origin).  Returns (P_alpha + P_beta, S_atom) — both in the atomic AO
    // basis ordering (size n_at × n_at).
    // Inherits integral engine and ERI tolerance from the molecular calculator.
    // All Logger output is suppressed via ScopedSilence.
    static std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
    run_atomic_uhf(int Z,
                   const HartreeFock::Basis &atomic_basis,
                   const HartreeFock::Calculator &mol_calc)
    {
        const unsigned int mult = atomic_minimal_multiplicity(Z);

        // ── Build atomic Calculator ───────────────────────────────────────────
        HartreeFock::Calculator atom;

        // Molecule: single neutral atom at the origin (Bohr).
        atom._molecule.natoms = 1;
        atom._molecule.atomic_numbers.resize(1);
        atom._molecule.atomic_numbers[0] = Z;
        atom._molecule.atomic_masses.resize(1);
        atom._molecule.atomic_masses[0] =
            static_cast<double>(element_from_z(static_cast<uint64_t>(Z)).mass);
        atom._molecule.coordinates = Eigen::MatrixXd::Zero(1, 3);
        atom._molecule._coordinates = Eigen::MatrixXd::Zero(1, 3);
        atom._molecule.set_standard_from_bohr(Eigen::MatrixXd::Zero(1, 3));
        atom._molecule._is_bohr = true;
        atom._molecule.charge = 0;
        atom._molecule.multiplicity = mult;
        // Basis
        atom._shells = atomic_basis;

        // SCF options: UHF, HCore guess, conventional mode, quiet DIIS
        atom._scf._scf = HartreeFock::SCFType::UHF;
        atom._scf._guess = HartreeFock::SCFGuess::HCore;
        atom._scf._mode = HartreeFock::SCFMode::Conventional;
        atom._scf._use_DIIS = true;
        atom._scf._DIIS_dim = 6;
        atom._scf._tol_energy = 1e-10;
        atom._scf._tol_density = 1e-10;

        // Integral engine: inherit from molecular calculator
        atom._integral._engine = mol_calc._integral._engine;
        atom._integral._tol_eri = mol_calc._integral._tol_eri;

        // Initialize: sets up DataSCF matrices and nuclear repulsion (= 0 for 1 atom)
        auto init_result = atom.initialize();
        if (!init_result)
            return std::unexpected("SAD: atomic calculator init failed: " + init_result.error());

        // ── Atomic 1e integrals ───────────────────────────────────────────────
        const std::size_t n_at = atom._shells.nbasis();
        auto atom_pairs = build_shellpairs(atom._shells);

        auto [S_at, T_at] = HartreeFock::ObaraSaika::_compute_1e(atom_pairs, n_at, nullptr);
        Eigen::MatrixXd V_at = HartreeFock::ObaraSaika::_compute_nuclear_attraction(
            atom_pairs, n_at, atom._molecule, nullptr);

        atom._overlap = S_at;
        atom._hcore = T_at + V_at;

        // ── Run atomic UHF (suppress all Logger output) ───────────────────────
        {
            HartreeFock::Logger::ScopedSilence silence;
            auto result = HartreeFock::SCF::run_uhf(atom, atom_pairs);
            if (!result)
                return std::unexpected(
                    "SAD: atomic UHF failed for Z = " + std::to_string(Z) + ": " + result.error());
        }

        Eigen::MatrixXd P_atom =
            atom._info._scf.alpha.density + atom._info._scf.beta.density;

        return std::pair{std::move(P_atom), std::move(S_at)};
    }

    // ── Spherical averaging ───────────────────────────────────────────────────

    // Replaces the in-place atomic density P_atom (atomic AO ordering) with its
    // shell-wise spherically averaged form. All cross-shell coupling blocks are
    // zeroed. Each diagonal shell block is replaced by (N_l / g) * S_block^{-1},
    // where N_l is the shell population evaluated in the full atomic AO metric:
    //
    //   N_l = sum_{mu in shell l} (P * S)_{mu,mu}
    //
    // This Mulliken-style partitioning ensures that removing inter-shell density
    // terms does not change the total electron count or the per-shell population.
    static void spherical_average_atomic_density(
        const HartreeFock::Basis &atomic_basis,
        const Eigen::MatrixXd &S_atom,
        Eigen::MatrixXd &P_atom,
        double thresh)
    {
        // Build shell → list of atomic AO indices, preserving discovery order.
        std::vector<const HartreeFock::Shell *> shell_order;
        std::unordered_map<const HartreeFock::Shell *, std::vector<int>> shell_aos;

        const HartreeFock::index_t basis_function_count =
            static_cast<HartreeFock::index_t>(atomic_basis._basis_functions.size());
        for (HartreeFock::index_t mu = 0; mu < basis_function_count; ++mu)
        {
            const HartreeFock::Shell *sp = atomic_basis._basis_functions[static_cast<std::size_t>(mu)]._shell;
            if (shell_aos.find(sp) == shell_aos.end())
                shell_order.push_back(sp);
            shell_aos[sp].push_back(static_cast<int>(mu));
        }

        // Compute shell populations before modifying P_atom.  In a nonorthogonal
        // AO basis, a shell's population is not determined by its diagonal block
        // alone; inter-shell density/overlap terms also contribute. Partition
        // the full Mulliken population by AO row so the shell totals add up to
        // Tr(P * S) even after we later zero the off-diagonal shell blocks.
        const Eigen::MatrixXd PS = P_atom * S_atom;
        std::unordered_map<const HartreeFock::Shell *, double> N_shell;
        for (const HartreeFock::Shell *sp : shell_order)
        {
            const auto &idx = shell_aos[sp];
            double population = 0.0;
            for (int i = 0; i < static_cast<int>(idx.size()); ++i)
            {
                const int mu = idx[static_cast<std::size_t>(i)];
                population += PS(mu, mu);
            }
            N_shell[sp] = population;
        }

        // Zero the entire density (removes all cross-shell couplings).
        P_atom.setZero();

        // Fill each shell's diagonal block with the spherically averaged value.
        for (const HartreeFock::Shell *sp : shell_order)
        {
            const auto &idx = shell_aos[sp];
            const int g = static_cast<int>(idx.size());
            const double Nl = N_shell[sp];

            Eigen::MatrixXd S_blk(g, g);
            for (int i = 0; i < g; ++i)
                for (int j = 0; j < g; ++j)
                    S_blk(i, j) = S_atom(idx[static_cast<std::size_t>(i)],
                                         idx[static_cast<std::size_t>(j)]);

            Eigen::MatrixXd P_sph =
                (Nl / static_cast<double>(g)) *
                pseudo_inverse_symmetric(S_blk, thresh);

            for (int i = 0; i < g; ++i)
                for (int j = 0; j < g; ++j)
                    P_atom(idx[static_cast<std::size_t>(i)],
                           idx[static_cast<std::size_t>(j)]) = P_sph(i, j);
        }
    }

    // ── Molecular assembly ────────────────────────────────────────────────────

    // Assembles the block-diagonal raw SAD density in the full molecular AO
    // basis by inserting each atom's cached averaged atomic block at the
    // appropriate molecular AO positions.
    static Eigen::MatrixXd assemble_raw_sad_density(
        const HartreeFock::Calculator &calc,
        const std::unordered_map<std::string, Eigen::MatrixXd> &atomic_P_cache)
    {
        const int nbasis = static_cast<int>(calc._shells.nbasis());
        Eigen::MatrixXd P_mol = Eigen::MatrixXd::Zero(nbasis, nbasis);

        const auto atom_ao_map = build_atom_to_ao_indices(calc._shells);

        for (std::size_t A = 0; A < calc._molecule.natoms; ++A)
        {
            const std::string sym = std::string(
                element_from_z(static_cast<uint64_t>(
                                   calc._molecule.atomic_numbers[static_cast<Eigen::Index>(A)]))
                    .symbol);

            const Eigen::MatrixXd &P_atom = atomic_P_cache.at(sym);
            const auto &ao_indices = atom_ao_map[A];
            const HartreeFock::index_t n = static_cast<HartreeFock::index_t>(ao_indices.size());

            for (HartreeFock::index_t i = 0; i < n; ++i)
                for (HartreeFock::index_t j = 0; j < n; ++j)
                    P_mol(ao_indices[static_cast<std::size_t>(i)],
                          ao_indices[static_cast<std::size_t>(j)]) = P_atom(i, j);
        }

        return P_mol;
    }

    // ── RHF reconstruction ────────────────────────────────────────────────────

    // Projects the raw (non-idempotent) SAD density onto the nearest proper
    // RHF density by orthogonalizing, diagonalizing, and rebuilding from the
    // top n_occ = n_electrons/2 natural orbitals.
    //
    // Guarantees Tr(P * S) = n_electrons (up to the linear-dependence threshold).
    static std::expected<Eigen::MatrixXd, std::string> project_raw_sad_to_rhf_density(
        const Eigen::MatrixXd &P_raw,
        const Eigen::MatrixXd &S_mol,
        int n_electrons,
        double thresh)
    {
        const int n_occ = n_electrons / 2;

        // X = S^{-1/2} with eigenvalue thresholding
        const Eigen::MatrixXd X = make_thresholded_inverse_sqrt(S_mol, thresh);

        // P_bar = X^T P_raw X  (orthonormal basis representation)
        Eigen::MatrixXd P_bar = X.transpose() * P_raw * X;
        P_bar = 0.5 * (P_bar + P_bar.transpose());

        // Diagonalize: eigenvalues are natural occupation numbers in [0,1] (ideally)
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(P_bar);
        if (es.info() != Eigen::Success)
            return std::unexpected("SAD: failed to diagonalize orthogonalized density");

        // Eigenvalues sorted ascending; take the top n_occ (rightmost columns).
        const Eigen::MatrixXd U_occ = es.eigenvectors().rightCols(n_occ);
        const Eigen::MatrixXd C_occ = X * U_occ;

        Eigen::MatrixXd P = 2.0 * C_occ * C_occ.transpose();
        return 0.5 * (P + P.transpose());
    }

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

namespace HartreeFock
{
    namespace SCF
    {
        std::expected<std::unordered_map<std::string, HartreeFock::Basis>, std::string>
        read_gbs_basis_atomic(const std::string &file_name,
                              const HartreeFock::Molecule &molecule,
                              const HartreeFock::BasisType &basis_type)
        {
            if (basis_type != HartreeFock::BasisType::Cartesian)
                return std::unexpected(
                    "Spherical Harmonics are not supported. "
                    "Only Cartesian basis functions are currently supported");

            std::ifstream file(file_name);
            if (!file)
                return std::unexpected("Cannot open basis file: " + file_name);

            auto gbs_res = read_gbs(file);
            if (!gbs_res)
                return std::unexpected(gbs_res.error());
            const BasisSet &gbs = *gbs_res;

            std::unordered_set<std::string> seen;
            std::unordered_map<std::string, HartreeFock::Basis> result;

            for (std::size_t i = 0; i < molecule.natoms; ++i)
            {
                const std::string sym = std::string(
                    element_from_z(static_cast<uint64_t>(molecule.atomic_numbers[static_cast<Eigen::Index>(i)])).symbol);

                if (!seen.insert(sym).second)
                    continue;

                auto it = gbs.find(sym);
                if (it == gbs.end())
                    return std::unexpected("Element not found in basis file: " + sym);

                result.emplace(sym, build_atomic_basis(it->second));
            }

            return result;
        }

        std::expected<Eigen::MatrixXd, std::string> compute_sad_guess_rhf(
            const HartreeFock::Calculator &calc)
        {
            const int n_electrons = static_cast<int>(
                calc._molecule.atomic_numbers.cast<int>().sum() - calc._molecule.charge);

            if (n_electrons % 2 != 0)
                return std::unexpected("SAD: RHF requires an even number of electrons");

            // ── Step 1: read atomic bases (one per unique element) ────────────
            const std::string gbs_file =
                calc._basis._basis_path + "/" + calc._basis._basis_name;
            auto atomic_bases_res = read_gbs_basis_atomic(
                gbs_file,
                calc._molecule,
                calc._basis._basis);
            if (!atomic_bases_res)
                return std::unexpected(atomic_bases_res.error());
            auto atomic_bases = std::move(*atomic_bases_res);

            // ── Step 2: for each unique element, run atomic UHF and average ───
            std::unordered_map<std::string, Eigen::MatrixXd> atomic_P_cache;

            for (auto &[sym, atomic_basis] : atomic_bases)
            {
                const int Z = static_cast<int>(element_from_symbol(sym).Z);

                auto atomic_res = run_atomic_uhf(Z, atomic_basis, calc);
                if (!atomic_res)
                    return std::unexpected(atomic_res.error());
                auto [P_atom, S_atom] = std::move(*atomic_res);

                spherical_average_atomic_density(atomic_basis, S_atom, P_atom, 1e-10);
                P_atom = 0.5 * (P_atom + P_atom.transpose());

                atomic_P_cache[sym] = std::move(P_atom);
            }

            // ── Step 3: assemble block-diagonal molecular density ─────────────
            Eigen::MatrixXd P_raw = assemble_raw_sad_density(calc, atomic_P_cache);
            P_raw = 0.5 * (P_raw + P_raw.transpose());

            // ── Step 4: reconstruct proper RHF density ────────────────────────
            return project_raw_sad_to_rhf_density(
                P_raw, calc._overlap, n_electrons, 1e-8);
        }

    } // namespace SCF
} // namespace HartreeFock
