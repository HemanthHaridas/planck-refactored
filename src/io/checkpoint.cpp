#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <fstream>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <Eigen/SVD>

#include "checkpoint.h"

// ─── Low-level binary helpers ─────────────────────────────────────────────────

// Primitive scalar write/read helpers
// The checkpoint payload is currently host-endian and assumes IEEE-754 doubles.
// The magic/version banner below is used to reject incompatible or stale files
// explicitly rather than silently mis-decoding them.

template <typename T>
static void write_pod(std::ostream &out, T val)
{
    out.write(reinterpret_cast<const char *>(&val), sizeof(T));
}

template <typename T>
static std::expected<T, std::string> read_pod(std::istream &in)
{
    T val{};
    in.read(reinterpret_cast<char *>(&val), sizeof(T));
    if (!in)
    {
        return std::unexpected(
            std::format("Checkpoint truncated while reading {} bytes", sizeof(T)));
    }
    return val;
}

static std::expected<void, std::string> read_exact(
    std::istream &in,
    char *data,
    std::size_t bytes,
    std::string_view label)
{
    in.read(data, static_cast<std::streamsize>(bytes));
    if (!in)
        return std::unexpected(std::format("Checkpoint truncated while reading {}", label));
    return {};
}

static constexpr std::uint32_t MAX_CHECKPOINT_STRING_BYTES = 4096;
static constexpr std::uint64_t MAX_CHECKPOINT_NATOMS = 1'000'000;

static std::expected<int, std::string> checked_atom_count(std::uint64_t natoms, std::string_view label)
{
    if (natoms > MAX_CHECKPOINT_NATOMS)
    {
        return std::unexpected(std::format(
            "Checkpoint {} atom count {} exceeds supported limit {}",
            label,
            natoms,
            MAX_CHECKPOINT_NATOMS));
    }
    if (natoms > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    {
        return std::unexpected(std::format(
            "Checkpoint {} atom count {} exceeds Eigen int indexing limit {}",
            label,
            natoms,
            std::numeric_limits<int>::max()));
    }
    return static_cast<int>(natoms);
}

static std::expected<std::uint64_t, std::string> checked_element_count(
    std::int64_t rows,
    std::int64_t cols,
    std::uint64_t max_dimension,
    std::string_view label)
{
    if (rows < 0 || cols < 0)
        return std::unexpected(std::format("Checkpoint {} has negative dimensions", label));

    const auto urows = static_cast<std::uint64_t>(rows);
    const auto ucols = static_cast<std::uint64_t>(cols);
    if (urows > max_dimension || ucols > max_dimension)
    {
        return std::unexpected(std::format(
            "Checkpoint {} dimensions {}x{} exceed supported limit {}",
            label,
            rows,
            cols,
            max_dimension));
    }

    if (urows != 0 && ucols > std::numeric_limits<std::uint64_t>::max() / urows)
        return std::unexpected(std::format("Checkpoint {} dimensions overflow", label));

    const std::uint64_t count = urows * ucols;
    if (count > max_dimension * max_dimension)
    {
        return std::unexpected(std::format(
            "Checkpoint {} element count {} exceeds supported limit {}",
            label,
            count,
            max_dimension * max_dimension));
    }

    return count;
}

// Eigen::MatrixXd (column-major storage → write as-is)
static void write_matrix(std::ostream &out, const Eigen::MatrixXd &m)
{
    const int64_t rows = static_cast<int64_t>(m.rows());
    const int64_t cols = static_cast<int64_t>(m.cols());
    out.write(reinterpret_cast<const char *>(&rows), 8);
    out.write(reinterpret_cast<const char *>(&cols), 8);
    out.write(reinterpret_cast<const char *>(m.data()), rows * cols * sizeof(double));
}

static std::expected<Eigen::MatrixXd, std::string> read_matrix(
    std::istream &in,
    std::uint64_t max_dimension,
    std::string_view label)
{
    int64_t rows = 0, cols = 0;
    auto row_read = read_exact(in, reinterpret_cast<char *>(&rows), 8, std::format("{} row count", label));
    if (!row_read)
        return std::unexpected(row_read.error());
    auto col_read = read_exact(in, reinterpret_cast<char *>(&cols), 8, std::format("{} column count", label));
    if (!col_read)
        return std::unexpected(col_read.error());
    auto count = checked_element_count(rows, cols, max_dimension, label);
    if (!count)
        return std::unexpected(count.error());
    Eigen::MatrixXd m(rows, cols);
    if (*count > 0)
    {
        auto data_read = read_exact(in,
                                    reinterpret_cast<char *>(m.data()),
                                    static_cast<std::size_t>(*count) * sizeof(double),
                                    std::format("{} data", label));
        if (!data_read)
            return std::unexpected(data_read.error());
    }
    return m;
}

// Eigen::VectorXd stored as n×1 matrix
static void write_vector(std::ostream &out, const Eigen::VectorXd &v)
{
    const int64_t rows = static_cast<int64_t>(v.size());
    const int64_t cols = 1;
    out.write(reinterpret_cast<const char *>(&rows), 8);
    out.write(reinterpret_cast<const char *>(&cols), 8);
    out.write(reinterpret_cast<const char *>(v.data()), rows * sizeof(double));
}

static std::expected<Eigen::VectorXd, std::string> read_vector(
    std::istream &in,
    std::uint64_t max_dimension,
    std::string_view label)
{
    int64_t rows = 0, cols = 0;
    auto row_read = read_exact(in, reinterpret_cast<char *>(&rows), 8, std::format("{} row count", label));
    if (!row_read)
        return std::unexpected(row_read.error());
    auto col_read = read_exact(in, reinterpret_cast<char *>(&cols), 8, std::format("{} column count", label));
    if (!col_read)
        return std::unexpected(col_read.error());
    if (cols != 1)
        return std::unexpected(std::format("Checkpoint {} is not stored as an n x 1 vector", label));
    auto count = checked_element_count(rows, cols, max_dimension, label);
    if (!count)
        return std::unexpected(count.error());
    Eigen::VectorXd v(rows);
    if (*count > 0)
    {
        auto data_read = read_exact(in,
                                    reinterpret_cast<char *>(v.data()),
                                    static_cast<std::size_t>(*count) * sizeof(double),
                                    std::format("{} data", label));
        if (!data_read)
            return std::unexpected(data_read.error());
    }
    return v;
}

// Fixed-length string helper for basis name
static void write_string(std::ostream &out, const std::string &s)
{
    const uint32_t len = static_cast<uint32_t>(s.size());
    out.write(reinterpret_cast<const char *>(&len), 4);
    out.write(s.data(), len);
}

static std::expected<std::string, std::string> read_string(std::istream &in)
{
    uint32_t len = 0;
    auto len_read = read_exact(in, reinterpret_cast<char *>(&len), 4, "string length");
    if (!len_read)
        return std::unexpected(len_read.error());
    if (len > MAX_CHECKPOINT_STRING_BYTES)
    {
        return std::unexpected(std::format(
            "Checkpoint string length {} exceeds supported limit {}",
            len,
            MAX_CHECKPOINT_STRING_BYTES));
    }
    std::string s(len, '\0');
    if (len > 0)
    {
        auto data_read = read_exact(in, s.data(), len, "string data");
        if (!data_read)
            return std::unexpected(data_read.error());
    }
    return s;
}

// Write one SpinChannel's matrices
static void write_spin_channel(std::ostream &out, const HartreeFock::SpinChannel &ch)
{
    write_matrix(out, ch.density);
    write_matrix(out, ch.fock);
    write_vector(out, ch.mo_energies);
    write_matrix(out, ch.mo_coefficients);
}

// Read into a SpinChannel
static std::expected<void, std::string> read_spin_channel(
    std::istream &in,
    HartreeFock::SpinChannel &ch,
    std::uint64_t max_dimension,
    std::string_view label)
{
    auto density = read_matrix(in, max_dimension, std::format("{} density", label));
    if (!density)
        return std::unexpected(density.error());
    auto fock = read_matrix(in, max_dimension, std::format("{} fock", label));
    if (!fock)
        return std::unexpected(fock.error());
    auto orbital_energies = read_vector(in, max_dimension, std::format("{} orbital energies", label));
    if (!orbital_energies)
        return std::unexpected(orbital_energies.error());
    auto mo_coefficients = read_matrix(in, max_dimension, std::format("{} MO coefficients", label));
    if (!mo_coefficients)
        return std::unexpected(mo_coefficients.error());

    ch.density = std::move(*density);
    ch.fock = std::move(*fock);
    ch.mo_energies = std::move(*orbital_energies);
    ch.mo_coefficients = std::move(*mo_coefficients);
    return {};
}

static std::pair<int, int> current_spin_occupations(const HartreeFock::Calculator &calc)
{
    const int n_electrons = static_cast<int>(
        calc._molecule.atomic_numbers.cast<int>().sum() - calc._molecule.charge);

    if (calc._scf._scf == HartreeFock::SCFType::RHF)
        return {n_electrons / 2, n_electrons / 2};

    const int n_unpaired = static_cast<int>(calc._molecule.multiplicity) - 1;
    return {
        (n_electrons + n_unpaired) / 2,
        (n_electrons - n_unpaired) / 2};
}

static std::expected<Eigen::MatrixXd, std::string> density_from_mos(
    const Eigen::MatrixXd &coefficients,
    int n_occ,
    double factor)
{
    if (n_occ <= 0 || coefficients.rows() == 0 || coefficients.cols() == 0)
        return Eigen::MatrixXd::Zero(coefficients.rows(), coefficients.rows());

    const Eigen::Index n_cols = coefficients.cols();
    const Eigen::Index n_occ_eigen = static_cast<Eigen::Index>(n_occ);
    if (n_occ_eigen > n_cols)
    {
        return std::unexpected(std::format(
            "Checkpoint orbital block has {} columns but {} occupied orbitals are required",
            n_cols,
            n_occ));
    }

    const Eigen::MatrixXd occ = coefficients.leftCols(n_occ_eigen);
    return factor * occ * occ.transpose();
}

static std::expected<void, std::string> adapt_restart_spin_state(
    HartreeFock::Calculator &calc,
    bool checkpoint_spin_resolved,
    const HartreeFock::SpinChannel &stored_alpha,
    const HartreeFock::SpinChannel &stored_beta)
{
    const bool target_spin_resolved = (calc._scf._scf != HartreeFock::SCFType::RHF);
    const auto [n_alpha, n_beta] = current_spin_occupations(calc);

    if (checkpoint_spin_resolved)
    {
        if (target_spin_resolved)
        {
            calc._info._scf.alpha = stored_alpha;
            calc._info._scf.beta = stored_beta;
            return {};
        }

        calc._info._scf.alpha = stored_alpha;
        calc._info._scf.alpha.density = stored_alpha.density + stored_beta.density;
        calc._info._scf.alpha.fock = 0.5 * (stored_alpha.fock + stored_beta.fock);
        return {};
    }

    if (!target_spin_resolved)
    {
        calc._info._scf.alpha = stored_alpha;
        return {};
    }

    auto alpha_density = density_from_mos(stored_alpha.mo_coefficients, n_alpha, 1.0);
    if (!alpha_density)
        return std::unexpected(alpha_density.error());
    auto beta_density = density_from_mos(stored_alpha.mo_coefficients, n_beta, 1.0);
    if (!beta_density)
        return std::unexpected(beta_density.error());

    calc._info._scf.alpha = stored_alpha;
    calc._info._scf.beta = stored_alpha;
    calc._info._scf.alpha.density = std::move(*alpha_density);
    calc._info._scf.beta.density = std::move(*beta_density);
    return {};
}

// ─── Public API ───────────────────────────────────────────────────────────────

// Checkpoint payloads are written in native host byte order and with native
// IEEE-754 `double` layout. MAGIC/VERSION let us reject incompatible files
// explicitly rather than silently interpreting foreign-endian data.
static constexpr char MAGIC[8] = {'P', 'L', 'N', 'K', 'C', 'H', 'K', '\0'};
static constexpr uint32_t VERSION = 6;

std::expected<void, std::string> HartreeFock::Checkpoint::save(
    const HartreeFock::Calculator &calc,
    const std::string &path)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        return std::unexpected(std::format("Cannot open checkpoint file for writing: {}", path));

    const std::size_t nb = static_cast<std::size_t>(calc._overlap.rows());
    const bool is_uhf = calc._info._scf.is_uhf;
    const uint32_t iters = 0; // last iteration not tracked; store 0

    // ── Header ────────────────────────────────────────────────────────────────
    out.write(MAGIC, 8);
    write_pod<uint32_t>(out, VERSION);
    write_pod<uint64_t>(out, static_cast<uint64_t>(nb));
    write_pod<uint8_t>(out, static_cast<uint8_t>(is_uhf ? 1 : 0));
    write_pod<uint8_t>(out, static_cast<uint8_t>(calc._info._is_converged ? 1 : 0));
    write_pod<uint32_t>(out, iters);
    write_pod<double>(out, calc._total_energy);
    write_pod<double>(out, calc._nuclear_repulsion);

    // ── Molecule ──────────────────────────────────────────────────────────────
    const std::size_t natoms = calc._molecule.natoms;
    write_pod<uint64_t>(out, static_cast<uint64_t>(natoms));
    write_pod<int32_t>(out, static_cast<int32_t>(calc._molecule.charge));
    write_pod<uint32_t>(out, static_cast<uint32_t>(calc._molecule.multiplicity));

    for (std::size_t i = 0; i < natoms; ++i)
        write_pod<int32_t>(out, static_cast<int32_t>(calc._molecule.atomic_numbers[i]));

    // Coordinates in Bohr (standard frame, natoms × 3, row-major)
    for (std::size_t i = 0; i < natoms; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            write_pod<double>(out, calc._molecule._standard(i, j));

    // Basis name for informational validation
    write_string(out, calc._basis._basis_name);

    // has_opt_coords: 1 if these coordinates came from a converged geometry optimization
    const bool is_opt = ((calc._calculation == HartreeFock::CalculationType::GeomOpt ||
                          calc._calculation == HartreeFock::CalculationType::GeomOptFrequency) &&
                         calc._info._is_converged);
    write_pod<uint8_t>(out, static_cast<uint8_t>(is_opt ? 1 : 0));

    // ── One-electron matrices ─────────────────────────────────────────────────
    write_matrix(out, calc._overlap);
    write_matrix(out, calc._hcore);

    // ── SCF results ───────────────────────────────────────────────────────────
    write_spin_channel(out, calc._info._scf.alpha);
    if (is_uhf)
        write_spin_channel(out, calc._info._scf.beta);

    // ── Optional post-HF restart data ────────────────────────────────────────
    const bool has_casscf_mos =
        calc._cas_mo_coefficients.rows() == static_cast<int>(nb) &&
        calc._cas_mo_coefficients.cols() == static_cast<int>(nb);
    write_pod<uint8_t>(out, static_cast<uint8_t>(has_casscf_mos ? 1 : 0));
    if (has_casscf_mos)
        write_matrix(out, calc._cas_mo_coefficients);

    // ── v4: Basis shell data (enables cube file generation in chkdump) ────────
    write_pod<uint8_t>(out, 1u); // has_basis = 1

    const auto &shells = calc._shells._shells;
    const auto &bfs = calc._shells._basis_functions;
    write_pod<uint64_t>(out, static_cast<uint64_t>(shells.size()));

    for (const auto &sh : shells)
    {
        write_pod<int32_t>(out, static_cast<int32_t>(sh._shell));
        write_pod<uint32_t>(out, static_cast<uint32_t>(sh._primitives.size()));
        write_pod<double>(out, sh._center.x());
        write_pod<double>(out, sh._center.y());
        write_pod<double>(out, sh._center.z());
        for (Eigen::Index k = 0; k < sh._primitives.size(); ++k)
            write_pod<double>(out, sh._primitives[k]);
        for (Eigen::Index k = 0; k < sh._coefficients.size(); ++k)
            write_pod<double>(out, sh._coefficients[k]);
        for (Eigen::Index k = 0; k < sh._normalizations.size(); ++k)
            write_pod<double>(out, sh._normalizations[k]);
    }

    write_pod<uint64_t>(out, static_cast<uint64_t>(bfs.size()));

    std::unordered_map<const HartreeFock::Shell *, uint64_t> shell_indices;
    shell_indices.reserve(shells.size());
    uint64_t shell_index = 0;
    for (const auto &sh : shells)
        shell_indices.emplace(&sh, shell_index++);

    for (const auto &bf : bfs)
    {
        const auto it = shell_indices.find(bf._shell);
        if (it == shell_indices.end())
            return std::unexpected("Checkpoint save failed: basis function references an unknown shell");
        const uint64_t shell_idx = it->second;
        write_pod<uint64_t>(out, shell_idx);
        write_pod<int32_t>(out, bf._cartesian.x());
        write_pod<int32_t>(out, bf._cartesian.y());
        write_pod<int32_t>(out, bf._cartesian.z());
        write_pod<double>(out, bf._component_norm);
    }

    // ── v5: CASSCF active-space orbital densities / occupations ─────────────
    const bool has_casscf_active_densities = calc._cas_nat_occ.size() > 0;
    write_pod<uint8_t>(out, static_cast<uint8_t>(has_casscf_active_densities ? 1 : 0));
    if (has_casscf_active_densities)
        write_vector(out, calc._cas_nat_occ);

    // ── v6: CASSCF active-orbital range metadata for cube export ────────────
    const int n_total_elec =
        static_cast<int>(calc._molecule.atomic_numbers.cast<int>().sum()) - calc._molecule.charge;
    const bool has_valid_active_window =
        has_casscf_mos &&
        calc._active_space.nactorb > 0 &&
        calc._active_space.nactele > 0 &&
        (n_total_elec - calc._active_space.nactele) >= 0 &&
        ((n_total_elec - calc._active_space.nactele) % 2 == 0);
    const int active_start =
        has_valid_active_window ? (n_total_elec - calc._active_space.nactele) / 2 : 0;
    const int active_count =
        has_valid_active_window ? calc._active_space.nactorb : 0;
    const bool has_casscf_active_orbitals =
        has_valid_active_window &&
        active_start >= 0 &&
        active_count > 0 &&
        active_start + active_count <= static_cast<int>(nb);
    write_pod<uint8_t>(out, static_cast<uint8_t>(has_casscf_active_orbitals ? 1 : 0));
    if (has_casscf_active_orbitals)
    {
        write_pod<int32_t>(out, static_cast<int32_t>(active_start));
        write_pod<int32_t>(out, static_cast<int32_t>(active_count));
    }

    if (!out)
        return std::unexpected(std::format("I/O error while writing checkpoint: {}", path));

    return {};
}

std::expected<void, std::string> HartreeFock::Checkpoint::load(
    HartreeFock::Calculator &calc,
    const std::string &path,
    bool load_1e_matrices)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return std::unexpected(std::format("Cannot open checkpoint file: {}", path));

    // ── Validate header ───────────────────────────────────────────────────
    char magic[8] = {};
    auto magic_read = read_exact(in, magic, 8, "magic");
    if (!magic_read)
        return std::unexpected(magic_read.error());
    if (std::memcmp(magic, MAGIC, 8) != 0)
        return std::unexpected("Not a valid Planck checkpoint file (bad magic)");

    auto version = read_pod<uint32_t>(in);
    if (!version)
        return std::unexpected(version.error());
    if (*version < 2 || *version > VERSION)
        return std::unexpected(
            std::format("Checkpoint version mismatch: file={}, expected={}", *version, VERSION));

    auto chk_nb = read_pod<uint64_t>(in);
    if (!chk_nb)
        return std::unexpected(chk_nb.error());
    auto chk_uhf = read_pod<uint8_t>(in);
    if (!chk_uhf)
        return std::unexpected(chk_uhf.error());
    auto chk_conv = read_pod<uint8_t>(in);
    if (!chk_conv)
        return std::unexpected(chk_conv.error());
    auto chk_iters = read_pod<uint32_t>(in);
    if (!chk_iters)
        return std::unexpected(chk_iters.error());
    [[maybe_unused]] const uint32_t chk_iters_value = *chk_iters; // informational only
    auto tot_e = read_pod<double>(in);
    if (!tot_e)
        return std::unexpected(tot_e.error());
    auto nuc_e = read_pod<double>(in);
    if (!nuc_e)
        return std::unexpected(nuc_e.error());

    // ── Molecule ──────────────────────────────────────────────────────────
    auto natoms = read_pod<uint64_t>(in);
    if (!natoms)
        return std::unexpected(natoms.error());
    auto natoms_checked = checked_atom_count(*natoms, "geometry block");
    if (!natoms_checked)
        return std::unexpected(natoms_checked.error());
    auto chk_charge = read_pod<int32_t>(in);
    if (!chk_charge)
        return std::unexpected(chk_charge.error());
    [[maybe_unused]] const int32_t chk_charge_value = *chk_charge;
    auto chk_mult = read_pod<uint32_t>(in);
    if (!chk_mult)
        return std::unexpected(chk_mult.error());
    [[maybe_unused]] const uint32_t chk_mult_value = *chk_mult;

    for (uint64_t i = 0; i < *natoms; ++i)
    {
        auto atomic_number = read_pod<int32_t>(in);
        if (!atomic_number)
            return std::unexpected(atomic_number.error());
    }

    for (uint64_t i = 0; i < *natoms; ++i)
    {
        for (int k = 0; k < 3; ++k)
        {
            auto coordinate = read_pod<double>(in);
            if (!coordinate)
                return std::unexpected(coordinate.error());
        }
    }

    auto chk_basis = read_string(in);
    if (!chk_basis)
        return std::unexpected(chk_basis.error());
    if (*chk_basis != calc._basis._basis_name)
    {
        // Warn but do not abort — the user may have intentionally changed the basis
        // (e.g., converging in a small basis and reusing the density in a larger one).
        // The nbasis check below will catch incompatible sizes.
    }

    auto has_opt_coords = read_pod<uint8_t>(in);
    if (!has_opt_coords)
        return std::unexpected(has_opt_coords.error());
    [[maybe_unused]] const uint8_t has_opt_coords_value = *has_opt_coords;

    // ── Validate basis size ───────────────────────────────────────────────
    const std::size_t cur_nb = calc._shells.nbasis();
    if (*chk_nb != static_cast<uint64_t>(cur_nb))
        return std::unexpected(
            std::format("Checkpoint nbasis ({}) does not match current nbasis ({}); "
                        "use the same basis set or remove the checkpoint file.",
                        *chk_nb, cur_nb));

    // ── One-electron matrices ─────────────────────────────────────────────
    // Only apply when the stored geometry matches current geometry (guess full).
    // For guess density the caller recomputes fresh integrals; skip the matrices.
    if (load_1e_matrices)
    {
        auto overlap = read_matrix(in, *chk_nb, "overlap matrix");
        if (!overlap)
            return std::unexpected(overlap.error());
        auto hcore = read_matrix(in, *chk_nb, "core Hamiltonian");
        if (!hcore)
            return std::unexpected(hcore.error());
        calc._overlap = std::move(*overlap);
        calc._hcore = std::move(*hcore);
    }
    else
    {
        auto overlap = read_matrix(in, *chk_nb, "overlap matrix");
        if (!overlap)
            return std::unexpected(overlap.error());
        auto hcore = read_matrix(in, *chk_nb, "core Hamiltonian");
        if (!hcore)
            return std::unexpected(hcore.error());
    }

    // ── SCF results ───────────────────────────────────────────────────────
    HartreeFock::SpinChannel stored_alpha;
    HartreeFock::SpinChannel stored_beta;

    auto alpha_read = read_spin_channel(in, stored_alpha, *chk_nb, "alpha");
    if (!alpha_read)
        return std::unexpected(alpha_read.error());
    if (*chk_uhf)
    {
        auto beta_read = read_spin_channel(in, stored_beta, *chk_nb, "beta");
        if (!beta_read)
            return std::unexpected(beta_read.error());
    }

    auto adapted = adapt_restart_spin_state(
        calc,
        static_cast<bool>(*chk_uhf),
        stored_alpha,
        stored_beta);
    if (!adapted)
        return std::unexpected(adapted.error());

    calc._cas_mo_coefficients.resize(0, 0);
    if (*version >= 3)
    {
        auto has_casscf_mos = read_pod<uint8_t>(in);
        if (!has_casscf_mos)
            return std::unexpected(has_casscf_mos.error());
        if (*has_casscf_mos != 0)
        {
            auto casscf_mos = read_matrix(in, *chk_nb, "CASSCF MO coefficients");
            if (!casscf_mos)
                return std::unexpected(casscf_mos.error());
            calc._cas_mo_coefficients = std::move(*casscf_mos);
        }
    }

    calc._total_energy = *tot_e;
    calc._correlated_total_energy = 0.0;
    calc._have_correlated_total_energy = false;
    calc._nuclear_repulsion = *nuc_e;
    calc._info._is_converged = static_cast<bool>(*chk_conv);

    if (!in)
        return std::unexpected(std::format("I/O error while reading checkpoint: {}", path));

    return {};
}

std::expected<HartreeFock::Checkpoint::MOData, std::string>
HartreeFock::Checkpoint::load_mos(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return std::unexpected(std::format("Cannot open checkpoint file: {}", path));

    // ── Validate magic and version ─────────────────────────────────────────
    char magic[8] = {};
    auto magic_read = read_exact(in, magic, 8, "magic");
    if (!magic_read)
        return std::unexpected(magic_read.error());
    if (std::memcmp(magic, MAGIC, 8) != 0)
        return std::unexpected("Not a valid Planck checkpoint file (bad magic)");

    auto version = read_pod<uint32_t>(in);
    if (!version)
        return std::unexpected(version.error());
    if (*version < 2 || *version > VERSION)
        return std::unexpected(
            std::format("Checkpoint version mismatch: file={}, expected={}", *version, VERSION));

    MOData result;
    auto nbasis = read_pod<uint64_t>(in);
    if (!nbasis)
        return std::unexpected(nbasis.error());
    result.nbasis = static_cast<std::size_t>(*nbasis);
    auto is_uhf = read_pod<uint8_t>(in);
    if (!is_uhf)
        return std::unexpected(is_uhf.error());
    result.is_uhf = static_cast<bool>(*is_uhf);

    auto converged = read_pod<uint8_t>(in);
    if (!converged)
        return std::unexpected(converged.error());
    auto last_iter = read_pod<uint32_t>(in);
    if (!last_iter)
        return std::unexpected(last_iter.error());
    auto total_energy = read_pod<double>(in);
    if (!total_energy)
        return std::unexpected(total_energy.error());
    auto nuclear_repulsion = read_pod<double>(in);
    if (!nuclear_repulsion)
        return std::unexpected(nuclear_repulsion.error());

    auto natoms = read_pod<uint64_t>(in);
    if (!natoms)
        return std::unexpected(natoms.error());
    auto natoms_checked = checked_atom_count(*natoms, "geometry block");
    if (!natoms_checked)
        return std::unexpected(natoms_checked.error());
    auto charge = read_pod<int32_t>(in);
    if (!charge)
        return std::unexpected(charge.error());
    auto multiplicity = read_pod<uint32_t>(in);
    if (!multiplicity)
        return std::unexpected(multiplicity.error());

    for (uint64_t i = 0; i < *natoms; ++i)
    {
        auto atomic_number = read_pod<int32_t>(in);
        if (!atomic_number)
            return std::unexpected(atomic_number.error());
    }

    for (uint64_t i = 0; i < *natoms; ++i)
    {
        for (int k = 0; k < 3; ++k)
        {
            auto coordinate = read_pod<double>(in);
            if (!coordinate)
                return std::unexpected(coordinate.error());
        }
    }

    auto basis_name = read_string(in);
    if (!basis_name)
        return std::unexpected(basis_name.error());
    result.basis_name = std::move(*basis_name);
    auto opt_coords = read_pod<uint8_t>(in);
    if (!opt_coords)
        return std::unexpected(opt_coords.error());

    auto overlap = read_matrix(in, result.nbasis, "overlap matrix");
    if (!overlap)
        return std::unexpected(overlap.error());
    auto core_hamiltonian = read_matrix(in, result.nbasis, "core Hamiltonian");
    if (!core_hamiltonian)
        return std::unexpected(core_hamiltonian.error());

    auto alpha_density = read_matrix(in, result.nbasis, "alpha density");
    if (!alpha_density)
        return std::unexpected(alpha_density.error());
    auto alpha_fock = read_matrix(in, result.nbasis, "alpha fock");
    if (!alpha_fock)
        return std::unexpected(alpha_fock.error());
    auto alpha_orbital_energies = read_vector(in, result.nbasis, "alpha orbital energies");
    if (!alpha_orbital_energies)
        return std::unexpected(alpha_orbital_energies.error());
    auto alpha_coefficients = read_matrix(in, result.nbasis, "alpha MO coefficients");
    if (!alpha_coefficients)
        return std::unexpected(alpha_coefficients.error());
    result.C_alpha = std::move(*alpha_coefficients);

    if (result.is_uhf)
    {
        auto beta_density = read_matrix(in, result.nbasis, "beta density");
        if (!beta_density)
            return std::unexpected(beta_density.error());
        auto beta_fock = read_matrix(in, result.nbasis, "beta fock");
        if (!beta_fock)
            return std::unexpected(beta_fock.error());
        auto beta_orbital_energies = read_vector(in, result.nbasis, "beta orbital energies");
        if (!beta_orbital_energies)
            return std::unexpected(beta_orbital_energies.error());
        auto beta_coefficients = read_matrix(in, result.nbasis, "beta MO coefficients");
        if (!beta_coefficients)
            return std::unexpected(beta_coefficients.error());
        result.C_beta = std::move(*beta_coefficients);
    }

    result.C_casscf.resize(0, 0);
    if (*version >= 3)
    {
        auto has_casscf_mos = read_pod<uint8_t>(in);
        if (!has_casscf_mos)
            return std::unexpected(has_casscf_mos.error());
        if (*has_casscf_mos != 0)
        {
            auto casscf_mos = read_matrix(in, result.nbasis, "CASSCF MO coefficients");
            if (!casscf_mos)
                return std::unexpected(casscf_mos.error());
            result.C_casscf = std::move(*casscf_mos);
        }
    }

    if (!in)
        return std::unexpected(
            std::format("I/O error while reading MOs from checkpoint: {}", path));

    return result;
}

std::expected<HartreeFock::Checkpoint::GeometryData, std::string>
HartreeFock::Checkpoint::load_geometry(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return std::unexpected(std::format("Cannot open checkpoint file: {}", path));

    char magic[8] = {};
    auto magic_read = read_exact(in, magic, 8, "magic");
    if (!magic_read)
        return std::unexpected(magic_read.error());
    if (std::memcmp(magic, MAGIC, 8) != 0)
        return std::unexpected("Not a valid Planck checkpoint file (bad magic)");

    auto version = read_pod<uint32_t>(in);
    if (!version)
        return std::unexpected(version.error());
    if (*version < 2 || *version > VERSION)
        return std::unexpected(
            std::format("Checkpoint version mismatch: file={}, expected={}", *version, VERSION));

    auto nbasis = read_pod<uint64_t>(in);
    if (!nbasis)
        return std::unexpected(nbasis.error());
    auto is_uhf = read_pod<uint8_t>(in);
    if (!is_uhf)
        return std::unexpected(is_uhf.error());
    auto is_converged = read_pod<uint8_t>(in);
    if (!is_converged)
        return std::unexpected(is_converged.error());
    auto last_iter = read_pod<uint32_t>(in);
    if (!last_iter)
        return std::unexpected(last_iter.error());
    auto total_energy = read_pod<double>(in);
    if (!total_energy)
        return std::unexpected(total_energy.error());
    auto nuclear_repulsion = read_pod<double>(in);
    if (!nuclear_repulsion)
        return std::unexpected(nuclear_repulsion.error());

    GeometryData geo;
    auto natoms = read_pod<uint64_t>(in);
    if (!natoms)
        return std::unexpected(natoms.error());
    auto natoms_int = checked_atom_count(*natoms, "geometry block");
    if (!natoms_int)
        return std::unexpected(natoms_int.error());
    geo.natoms = static_cast<std::size_t>(*natoms_int);
    auto charge = read_pod<int32_t>(in);
    if (!charge)
        return std::unexpected(charge.error());
    geo.charge = static_cast<int>(*charge);
    auto multiplicity = read_pod<uint32_t>(in);
    if (!multiplicity)
        return std::unexpected(multiplicity.error());
    geo.multiplicity = static_cast<unsigned int>(*multiplicity);

    geo.atomic_numbers.resize(*natoms_int);
    for (std::size_t i = 0; i < geo.natoms; ++i)
    {
        auto atomic_number = read_pod<int32_t>(in);
        if (!atomic_number)
            return std::unexpected(atomic_number.error());
        geo.atomic_numbers[static_cast<int>(i)] = *atomic_number;
    }

    geo.coords_bohr.resize(*natoms_int, 3);
    for (std::size_t i = 0; i < geo.natoms; ++i)
    {
        for (int k = 0; k < 3; ++k)
        {
            auto coordinate = read_pod<double>(in);
            if (!coordinate)
                return std::unexpected(coordinate.error());
            geo.coords_bohr(static_cast<int>(i), k) = *coordinate;
        }
    }

    auto basis_name = read_string(in);
    if (!basis_name)
        return std::unexpected(basis_name.error());

    auto has_opt_coords = read_pod<uint8_t>(in);
    if (!has_opt_coords)
        return std::unexpected(has_opt_coords.error());
    geo.has_opt_coords = (*has_opt_coords != 0);

    if (!in)
        return std::unexpected(
            std::format("I/O error while reading geometry from checkpoint: {}", path));

    return geo;
}

Eigen::MatrixXd HartreeFock::Checkpoint::project_density(
    const Eigen::MatrixXd &X_large,
    const Eigen::MatrixXd &S_cross,
    const Eigen::MatrixXd &C_occ,
    double factor)
{
    // O = X^T * S_cross * C_occ  (large orthogonal frame, nb_large × n_occ)
    const Eigen::MatrixXd O = X_large.transpose() * S_cross * C_occ;

    // Thin SVD: O = U Σ V^T
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(O,
                                                Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Orthonormal projected MOs in the large basis (AO frame)
    const Eigen::MatrixXd C_proj = X_large * svd.matrixU() * svd.matrixV().transpose();

    return factor * C_proj * C_proj.transpose();
}
