#include <cstdint>
#include <cstring>
#include <fstream>
#include <format>

#include <Eigen/SVD>

#include "checkpoint.h"

// ─── Low-level binary helpers ─────────────────────────────────────────────────

// Primitive scalar write/read helpers

template<typename T>
static void write_pod(std::ostream& out, T val)
{
    out.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

template<typename T>
static T read_pod(std::istream& in)
{
    T val{};
    in.read(reinterpret_cast<char*>(&val), sizeof(T));
    return val;
}

// Eigen::MatrixXd (column-major storage → write as-is)
static void write_matrix(std::ostream& out, const Eigen::MatrixXd& m)
{
    const int64_t rows = static_cast<int64_t>(m.rows());
    const int64_t cols = static_cast<int64_t>(m.cols());
    out.write(reinterpret_cast<const char*>(&rows), 8);
    out.write(reinterpret_cast<const char*>(&cols), 8);
    out.write(reinterpret_cast<const char*>(m.data()), rows * cols * sizeof(double));
}

static Eigen::MatrixXd read_matrix(std::istream& in)
{
    int64_t rows = 0, cols = 0;
    in.read(reinterpret_cast<char*>(&rows), 8);
    in.read(reinterpret_cast<char*>(&cols), 8);
    Eigen::MatrixXd m(rows, cols);
    in.read(reinterpret_cast<char*>(m.data()), rows * cols * sizeof(double));
    return m;
}

// Eigen::VectorXd stored as n×1 matrix
static void write_vector(std::ostream& out, const Eigen::VectorXd& v)
{
    const int64_t rows = static_cast<int64_t>(v.size());
    const int64_t cols = 1;
    out.write(reinterpret_cast<const char*>(&rows), 8);
    out.write(reinterpret_cast<const char*>(&cols), 8);
    out.write(reinterpret_cast<const char*>(v.data()), rows * sizeof(double));
}

static Eigen::VectorXd read_vector(std::istream& in)
{
    int64_t rows = 0, cols = 0;
    in.read(reinterpret_cast<char*>(&rows), 8);
    in.read(reinterpret_cast<char*>(&cols), 8);
    Eigen::VectorXd v(rows);
    in.read(reinterpret_cast<char*>(v.data()), rows * sizeof(double));
    return v;
}

// Fixed-length string helper for basis name
static void write_string(std::ostream& out, const std::string& s)
{
    const uint32_t len = static_cast<uint32_t>(s.size());
    out.write(reinterpret_cast<const char*>(&len), 4);
    out.write(s.data(), len);
}

static std::string read_string(std::istream& in)
{
    uint32_t len = 0;
    in.read(reinterpret_cast<char*>(&len), 4);
    std::string s(len, '\0');
    in.read(s.data(), len);
    return s;
}

// Write one SpinChannel's matrices
static void write_spin_channel(std::ostream& out, const HartreeFock::SpinChannel& ch)
{
    write_matrix(out, ch.density);
    write_matrix(out, ch.fock);
    write_vector(out, ch.mo_energies);
    write_matrix(out, ch.mo_coefficients);
}

// Read into a SpinChannel
static void read_spin_channel(std::istream& in, HartreeFock::SpinChannel& ch)
{
    ch.density         = read_matrix(in);
    ch.fock            = read_matrix(in);
    ch.mo_energies     = read_vector(in);
    ch.mo_coefficients = read_matrix(in);
}

// ─── Public API ───────────────────────────────────────────────────────────────

static constexpr char MAGIC[8] = {'P','L','N','K','C','H','K','\0'};
static constexpr uint32_t VERSION = 2;

std::expected<void, std::string> HartreeFock::Checkpoint::save(
    const HartreeFock::Calculator& calc,
    const std::string& path)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        return std::unexpected(std::format("Cannot open checkpoint file for writing: {}", path));

    const std::size_t nb     = static_cast<std::size_t>(calc._overlap.rows());
    const bool        is_uhf = calc._info._scf.is_uhf;
    const uint32_t    iters  = 0;  // last iteration not tracked; store 0

    // ── Header ────────────────────────────────────────────────────────────────
    out.write(MAGIC, 8);
    write_pod<uint32_t>(out, VERSION);
    write_pod<uint64_t>(out, static_cast<uint64_t>(nb));
    write_pod<uint8_t> (out, static_cast<uint8_t>(is_uhf ? 1 : 0));
    write_pod<uint8_t> (out, static_cast<uint8_t>(calc._info._is_converged ? 1 : 0));
    write_pod<uint32_t>(out, iters);
    write_pod<double>  (out, calc._total_energy);
    write_pod<double>  (out, calc._nuclear_repulsion);

    // ── Molecule ──────────────────────────────────────────────────────────────
    const std::size_t natoms = calc._molecule.natoms;
    write_pod<uint64_t>(out, static_cast<uint64_t>(natoms));
    write_pod<int32_t> (out, static_cast<int32_t>(calc._molecule.charge));
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
                          calc._calculation == HartreeFock::CalculationType::GeomOptFrequency)
                         && calc._info._is_converged);
    write_pod<uint8_t>(out, static_cast<uint8_t>(is_opt ? 1 : 0));

    // ── One-electron matrices ─────────────────────────────────────────────────
    write_matrix(out, calc._overlap);
    write_matrix(out, calc._hcore);

    // ── SCF results ───────────────────────────────────────────────────────────
    write_spin_channel(out, calc._info._scf.alpha);
    if (is_uhf)
        write_spin_channel(out, calc._info._scf.beta);

    if (!out)
        return std::unexpected(std::format("I/O error while writing checkpoint: {}", path));

    return {};
}

std::expected<void, std::string> HartreeFock::Checkpoint::load(
    HartreeFock::Calculator& calc,
    const std::string& path,
    bool load_1e_matrices)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return std::unexpected(std::format("Cannot open checkpoint file: {}", path));

    // ── Validate header ───────────────────────────────────────────────────────
    char magic[8] = {};
    in.read(magic, 8);
    if (std::memcmp(magic, MAGIC, 8) != 0)
        return std::unexpected("Not a valid Planck checkpoint file (bad magic)");

    const uint32_t version = read_pod<uint32_t>(in);
    if (version != VERSION)
        return std::unexpected(
            std::format("Checkpoint version mismatch: file={}, expected={}", version, VERSION));

    const uint64_t chk_nb    = read_pod<uint64_t>(in);
    const uint8_t  chk_uhf   = read_pod<uint8_t>(in);
    const uint8_t  chk_conv  = read_pod<uint8_t>(in);
    const uint32_t chk_iters = read_pod<uint32_t>(in);
    const double   tot_e     = read_pod<double>(in);
    const double   nuc_e     = read_pod<double>(in);

    (void)chk_iters; // informational only

    // ── Molecule ──────────────────────────────────────────────────────────────
    const uint64_t natoms      = read_pod<uint64_t>(in);
    const int32_t  chk_charge  = read_pod<int32_t>(in);
    const uint32_t chk_mult    = read_pod<uint32_t>(in);

    (void)chk_charge; (void)chk_mult;  // user input takes precedence; just skip

    for (uint64_t i = 0; i < natoms; ++i)
        read_pod<int32_t>(in);  // skip stored atomic numbers

    for (uint64_t i = 0; i < natoms * 3; ++i)
        read_pod<double>(in);   // skip stored coordinates

    const std::string chk_basis = read_string(in);
    if (chk_basis != calc._basis._basis_name)
    {
        // Warn but do not abort — the user may have intentionally changed the basis
        // (e.g., converging in a small basis and reusing the density in a larger one).
        // The nbasis check below will catch incompatible sizes.
    }

    read_pod<uint8_t>(in);   // has_opt_coords (informational; geometry handled by load_geometry)

    // ── Validate basis size ───────────────────────────────────────────────────
    const std::size_t cur_nb = calc._shells.nbasis();
    if (chk_nb != static_cast<uint64_t>(cur_nb))
        return std::unexpected(
            std::format("Checkpoint nbasis ({}) does not match current nbasis ({}); "
                        "use the same basis set or remove the checkpoint file.",
                        chk_nb, cur_nb));

    // ── One-electron matrices ─────────────────────────────────────────────────
    // Only apply when the stored geometry matches current geometry (guess full).
    // For guess density the caller recomputes fresh integrals; skip the matrices.
    if (load_1e_matrices)
    {
        calc._overlap = read_matrix(in);
        calc._hcore   = read_matrix(in);
    }
    else
    {
        read_matrix(in);  // overlap — discard
        read_matrix(in);  // hcore   — discard
    }

    // ── SCF results ───────────────────────────────────────────────────────────
    read_spin_channel(in, calc._info._scf.alpha);
    if (chk_uhf)
        read_spin_channel(in, calc._info._scf.beta);

    calc._total_energy      = tot_e;
    calc._nuclear_repulsion = nuc_e;
    calc._info._is_converged = static_cast<bool>(chk_conv);

    if (!in)
        return std::unexpected(std::format("I/O error while reading checkpoint: {}", path));

    return {};
}

std::expected<HartreeFock::Checkpoint::MOData, std::string>
HartreeFock::Checkpoint::load_mos(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return std::unexpected(std::format("Cannot open checkpoint file: {}", path));

    // ── Validate magic and version ─────────────────────────────────────────────
    char magic[8] = {};
    in.read(magic, 8);
    if (std::memcmp(magic, MAGIC, 8) != 0)
        return std::unexpected("Not a valid Planck checkpoint file (bad magic)");

    const uint32_t version = read_pod<uint32_t>(in);
    if (version != VERSION)
        return std::unexpected(
            std::format("Checkpoint version mismatch: file={}, expected={}", version, VERSION));

    MOData result;
    result.nbasis = static_cast<std::size_t>(read_pod<uint64_t>(in));
    result.is_uhf = static_cast<bool>(read_pod<uint8_t>(in));

    // Skip: is_converged, last_iter, total_energy, nuclear_repulsion
    read_pod<uint8_t>(in);   // is_converged
    read_pod<uint32_t>(in);  // last_iter
    read_pod<double>(in);    // total_energy
    read_pod<double>(in);    // nuclear_repulsion

    // ── Skip molecule ──────────────────────────────────────────────────────────
    const uint64_t natoms = read_pod<uint64_t>(in);
    read_pod<int32_t>(in);   // charge
    read_pod<uint32_t>(in);  // multiplicity

    for (uint64_t i = 0; i < natoms; ++i)
        read_pod<int32_t>(in);   // atomic_numbers

    for (uint64_t i = 0; i < natoms * 3; ++i)
        read_pod<double>(in);    // coordinates

    result.basis_name = read_string(in);
    read_pod<uint8_t>(in);   // has_opt_coords

    // ── Skip 1e matrices ───────────────────────────────────────────────────────
    read_matrix(in);  // overlap
    read_matrix(in);  // hcore

    // ── Read alpha MO coefficients ─────────────────────────────────────────────
    read_matrix(in);  // density  (discard)
    read_matrix(in);  // fock     (discard)
    read_vector(in);  // mo_energies (discard)
    result.C_alpha = read_matrix(in);

    // ── Read beta MO coefficients if UHF ──────────────────────────────────────
    if (result.is_uhf)
    {
        read_matrix(in);  // density  (discard)
        read_matrix(in);  // fock     (discard)
        read_vector(in);  // mo_energies (discard)
        result.C_beta = read_matrix(in);
    }

    if (!in)
        return std::unexpected(
            std::format("I/O error while reading MOs from checkpoint: {}", path));

    return result;
}

std::expected<HartreeFock::Checkpoint::GeometryData, std::string>
HartreeFock::Checkpoint::load_geometry(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return std::unexpected(std::format("Cannot open checkpoint file: {}", path));

    char magic[8] = {};
    in.read(magic, 8);
    if (std::memcmp(magic, MAGIC, 8) != 0)
        return std::unexpected("Not a valid Planck checkpoint file (bad magic)");

    const uint32_t version = read_pod<uint32_t>(in);
    if (version != VERSION)
        return std::unexpected(
            std::format("Checkpoint version mismatch: file={}, expected={}", version, VERSION));

    // Skip: nbasis, is_uhf, is_converged, last_iter, total_energy, nuclear_repulsion
    read_pod<uint64_t>(in);  // nbasis
    read_pod<uint8_t>(in);   // is_uhf
    read_pod<uint8_t>(in);   // is_converged
    read_pod<uint32_t>(in);  // last_iter
    read_pod<double>(in);    // total_energy
    read_pod<double>(in);    // nuclear_repulsion

    GeometryData geo;
    geo.natoms       = static_cast<std::size_t>(read_pod<uint64_t>(in));
    geo.charge       = static_cast<int>(read_pod<int32_t>(in));
    geo.multiplicity = static_cast<unsigned int>(read_pod<uint32_t>(in));

    geo.atomic_numbers.resize(static_cast<int>(geo.natoms));
    for (std::size_t i = 0; i < geo.natoms; ++i)
        geo.atomic_numbers[static_cast<int>(i)] = read_pod<int32_t>(in);

    geo.coords_bohr.resize(static_cast<int>(geo.natoms), 3);
    for (std::size_t i = 0; i < geo.natoms; ++i)
        for (int k = 0; k < 3; ++k)
            geo.coords_bohr(static_cast<int>(i), k) = read_pod<double>(in);

    read_string(in);  // basis_name (not needed here)

    geo.has_opt_coords = (read_pod<uint8_t>(in) != 0);

    if (!in)
        return std::unexpected(
            std::format("I/O error while reading geometry from checkpoint: {}", path));

    return geo;
}

Eigen::MatrixXd HartreeFock::Checkpoint::project_density(
    const Eigen::MatrixXd& X_large,
    const Eigen::MatrixXd& S_cross,
    const Eigen::MatrixXd& C_occ,
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
