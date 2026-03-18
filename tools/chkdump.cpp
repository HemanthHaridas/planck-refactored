// chkdump -- export a Planck binary checkpoint (.hfchk) as a Gaussian-style
// formatted checkpoint text file (.fchk-like).
//
// Usage:
//   chkdump <file.hfchk> [output.fchk]
//
// Notes:
// - The Planck checkpoint does not store the full Gaussian basis metadata that a
//   canonical Gaussian .fchk would contain, so this exporter writes the common
//   scalar, vector, and matrix sections that can be reconstructed exactly from
//   the checkpoint contents.
// - Symmetric matrices are emitted in packed lower-triangular form, matching
//   the usual formatted-checkpoint convention.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

template<typename T>
static T read_pod(std::istream& in)
{
    T v{};
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    return v;
}

static std::string read_string(std::istream& in)
{
    const uint32_t len = read_pod<uint32_t>(in);
    std::string s(len, '\0');
    in.read(s.data(), len);
    return s;
}

struct Matrix
{
    int64_t rows = 0;
    int64_t cols = 0;
    std::vector<double> data; // column-major

    double operator()(int64_t r, int64_t c) const
    {
        return data[static_cast<std::size_t>(c * rows + r)];
    }
};

static Matrix read_matrix(std::istream& in)
{
    Matrix m;
    in.read(reinterpret_cast<char*>(&m.rows), 8);
    in.read(reinterpret_cast<char*>(&m.cols), 8);
    m.data.resize(static_cast<std::size_t>(m.rows * m.cols));
    in.read(reinterpret_cast<char*>(m.data.data()),
            static_cast<std::streamsize>(m.rows * m.cols * static_cast<int64_t>(sizeof(double))));
    return m;
}

static Matrix read_vector_as_matrix(std::istream& in)
{
    return read_matrix(in);
}

static std::string uppercase(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    return s;
}

static std::vector<double> packed_lower_triangle(const Matrix& m)
{
    std::vector<double> packed;
    const int64_t n = std::min(m.rows, m.cols);
    packed.reserve(static_cast<std::size_t>(n * (n + 1) / 2));
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j <= i; ++j)
            packed.push_back(m(i, j));
    return packed;
}

static std::vector<double> flatten_matrix(const Matrix& m)
{
    return m.data;
}

static std::vector<int32_t> to_int32_vector(const std::vector<int32_t>& values)
{
    return values;
}

static std::vector<double> to_double_vector(const std::vector<int32_t>& values)
{
    std::vector<double> out;
    out.reserve(values.size());
    for (int32_t v : values)
        out.push_back(static_cast<double>(v));
    return out;
}

static void write_scalar_int(std::ostream& out, const std::string& label, long long value)
{
    out << std::left << std::setw(43) << label
        << "I"
        << std::right << std::setw(12) << value << "\n";
}

static void write_scalar_real(std::ostream& out, const std::string& label, double value)
{
    out << std::left << std::setw(43) << label
        << "R"
        << std::right << std::setw(27) << std::uppercase << std::scientific
        << std::setprecision(15) << value << "\n";
}

static void write_array_header(std::ostream& out, const std::string& label, char kind, std::size_t n)
{
    out << std::left << std::setw(43) << label
        << kind
        << "   N="
        << std::right << std::setw(12) << n << "\n";
}

static void write_integer_array(std::ostream& out, const std::string& label,
                                const std::vector<int32_t>& values)
{
    write_array_header(out, label, 'I', values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        out << std::setw(12) << values[i];
        if ((i + 1) % 6 == 0 || i + 1 == values.size())
            out << "\n";
    }
}

static void write_real_array(std::ostream& out, const std::string& label,
                             const std::vector<double>& values)
{
    write_array_header(out, label, 'R', values.size());
    out << std::uppercase << std::scientific << std::setprecision(8);
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        out << std::setw(16) << values[i];
        if ((i + 1) % 5 == 0 || i + 1 == values.size())
            out << "\n";
    }
}

int main(int argc, char* argv[])
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: chkdump <file.hfchk> [output.fchk]\n";
        return EXIT_FAILURE;
    }

    const std::string in_path = argv[1];
    const bool to_file = (argc == 3);
    const std::string out_path = to_file ? argv[2] : "";

    std::ifstream in(in_path, std::ios::binary);
    if (!in)
    {
        std::cerr << "chkdump: cannot open '" << in_path << "'\n";
        return EXIT_FAILURE;
    }

    std::ostream* outp = &std::cout;
    std::ofstream fout;
    if (to_file)
    {
        fout.open(out_path);
        if (!fout)
        {
            std::cerr << "chkdump: cannot open output file '" << out_path << "'\n";
            return EXIT_FAILURE;
        }
        outp = &fout;
    }
    std::ostream& out = *outp;

    char magic[8] = {};
    in.read(magic, 8);
    if (std::memcmp(magic, "PLNKCHK\0", 8) != 0)
    {
        std::cerr << "chkdump: not a Planck checkpoint file (bad magic)\n";
        return EXIT_FAILURE;
    }

    const uint32_t version = read_pod<uint32_t>(in);
    if (version != 2)
    {
        std::cerr << "chkdump: unsupported checkpoint version " << version
                  << " (expected 2)\n";
        return EXIT_FAILURE;
    }

    const uint64_t nbasis    = read_pod<uint64_t>(in);
    const uint8_t  is_uhf    = read_pod<uint8_t>(in);
    const uint8_t  converged = read_pod<uint8_t>(in);
    const uint32_t last_iter = read_pod<uint32_t>(in);
    const double   tot_energy = read_pod<double>(in);
    const double   nuc_rep    = read_pod<double>(in);

    const uint64_t natoms = read_pod<uint64_t>(in);
    const int32_t  charge = read_pod<int32_t>(in);
    const uint32_t mult   = read_pod<uint32_t>(in);

    std::vector<int32_t> atomic_numbers(natoms);
    for (uint64_t i = 0; i < natoms; ++i)
        atomic_numbers[static_cast<std::size_t>(i)] = read_pod<int32_t>(in);

    std::vector<double> coords(natoms * 3);
    for (uint64_t i = 0; i < natoms * 3; ++i)
        coords[static_cast<std::size_t>(i)] = read_pod<double>(in);

    const std::string basis_name = read_string(in);
    const uint8_t has_opt_coords = read_pod<uint8_t>(in);

    const Matrix overlap = read_matrix(in);
    const Matrix hcore   = read_matrix(in);

    const Matrix alpha_density = read_matrix(in);
    const Matrix alpha_fock    = read_matrix(in);
    const Matrix alpha_mo_e    = read_vector_as_matrix(in);
    const Matrix alpha_mo_c    = read_matrix(in);

    Matrix beta_density, beta_fock, beta_mo_e, beta_mo_c;
    if (is_uhf)
    {
        beta_density = read_matrix(in);
        beta_fock    = read_matrix(in);
        beta_mo_e    = read_vector_as_matrix(in);
        beta_mo_c    = read_matrix(in);
    }

    if (!in)
    {
        std::cerr << "chkdump: I/O error while reading checkpoint\n";
        return EXIT_FAILURE;
    }

    int z_total = 0;
    for (int32_t z : atomic_numbers)
        z_total += z;
    const int n_elec = z_total - charge;
    const int n_unpaired = static_cast<int>(mult) - 1;
    const int n_alpha = (n_elec + n_unpaired) / 2;
    const int n_beta  = (n_elec - n_unpaired) / 2;

    std::vector<double> nuclear_charges = to_double_vector(atomic_numbers);
    std::vector<double> overlap_packed  = packed_lower_triangle(overlap);
    std::vector<double> hcore_packed    = packed_lower_triangle(hcore);
    std::vector<double> alpha_density_packed = packed_lower_triangle(alpha_density);
    std::vector<double> total_density_packed = alpha_density_packed;

    if (is_uhf)
    {
        total_density_packed.clear();
        const int64_t n = std::min(alpha_density.rows, alpha_density.cols);
        total_density_packed.reserve(static_cast<std::size_t>(n * (n + 1) / 2));
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j <= i; ++j)
                total_density_packed.push_back(alpha_density(i, j) + beta_density(i, j));
    }

    out << "Planck checkpoint export\n";
    out << std::left << std::setw(10) << "SP"
        << std::setw(10) << (is_uhf ? "UHF" : "RHF")
        << uppercase(basis_name) << "\n";

    write_scalar_int(out, "Number of atoms", static_cast<long long>(natoms));
    write_scalar_int(out, "Charge", static_cast<long long>(charge));
    write_scalar_int(out, "Multiplicity", static_cast<long long>(mult));
    write_scalar_int(out, "Number of electrons", static_cast<long long>(n_elec));
    write_scalar_int(out, "Number of alpha electrons", static_cast<long long>(n_alpha));
    write_scalar_int(out, "Number of beta electrons", static_cast<long long>(n_beta));
    write_scalar_int(out, "Number of basis functions", static_cast<long long>(nbasis));
    write_scalar_int(out, "Number of independent functions", static_cast<long long>(nbasis));
    write_scalar_int(out, "SCF converged", static_cast<long long>(converged));
    write_scalar_int(out, "Last SCF iteration", static_cast<long long>(last_iter));
    write_scalar_int(out, "Has optimized geometry", static_cast<long long>(has_opt_coords));
    write_scalar_real(out, "SCF Energy", tot_energy);
    write_scalar_real(out, "Nuclear repulsion energy", nuc_rep);

    write_integer_array(out, "Atomic numbers", to_int32_vector(atomic_numbers));
    write_real_array(out, "Nuclear charges", nuclear_charges);
    write_real_array(out, "Current cartesian coordinates", coords);
    write_real_array(out, "Alpha Orbital Energies", flatten_matrix(alpha_mo_e));
    write_real_array(out, "Alpha MO coefficients", flatten_matrix(alpha_mo_c));

    if (is_uhf)
    {
        write_real_array(out, "Beta Orbital Energies", flatten_matrix(beta_mo_e));
        write_real_array(out, "Beta MO coefficients", flatten_matrix(beta_mo_c));
    }

    write_real_array(out, "Total SCF Density", total_density_packed);
    write_real_array(out, "Alpha Density Matrix", alpha_density_packed);
    if (is_uhf)
        write_real_array(out, "Beta Density Matrix", packed_lower_triangle(beta_density));

    write_real_array(out, "Alpha Fock Matrix", packed_lower_triangle(alpha_fock));
    if (is_uhf)
        write_real_array(out, "Beta Fock Matrix", packed_lower_triangle(beta_fock));

    write_real_array(out, "Core Hamiltonian Matrix", hcore_packed);
    write_real_array(out, "Overlap Matrix", overlap_packed);

    return EXIT_SUCCESS;
}
