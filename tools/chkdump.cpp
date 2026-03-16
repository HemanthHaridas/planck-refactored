// chkdump — convert a Planck binary checkpoint (.hfchk) to formatted text
//
// Usage:  chkdump <file.hfchk> [output.txt]
//
// When no output file is given, the formatted text is written to stdout.
// The tool is self-contained and requires no link dependencies beyond the
// C++ standard library.

#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

// ─── Binary helpers ───────────────────────────────────────────────────────────

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

// Read an int64 × int64 matrix stored column-major
struct Matrix {
    int64_t rows = 0, cols = 0;
    std::vector<double> data;   // column-major
    double operator()(int r, int c) const { return data[c * rows + r]; }
};

static Matrix read_matrix(std::istream& in)
{
    Matrix m;
    in.read(reinterpret_cast<char*>(&m.rows), 8);
    in.read(reinterpret_cast<char*>(&m.cols), 8);
    m.data.resize(static_cast<std::size_t>(m.rows * m.cols));
    in.read(reinterpret_cast<char*>(m.data.data()),
            m.rows * m.cols * static_cast<int64_t>(sizeof(double)));
    return m;
}

static Matrix read_vector_as_matrix(std::istream& in)
{
    return read_matrix(in);   // stored as n × 1
}

// ─── Element symbol lookup (Z = 1 … 118) ─────────────────────────────────────

static const char* element_symbol(int Z)
{
    static constexpr const char* sym[] = {
        "?",
        "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
        "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
        "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    };
    if (Z < 1 || Z > 118) return sym[0];
    return sym[Z];
}

// ─── Formatting helpers ───────────────────────────────────────────────────────

static const int LINE_WIDTH = 78;

static void rule(std::ostream& out)
{
    out << std::string(LINE_WIDTH, '=') << "\n";
}

static void section(std::ostream& out, const std::string& title)
{
    out << "\n";
    rule(out);
    out << "  " << title << "\n";
    rule(out);
}

static void kv(std::ostream& out, const std::string& key, const std::string& val)
{
    out << "  " << std::setw(22) << std::left << key << ": " << val << "\n";
}

static void print_matrix(std::ostream& out, const Matrix& m, int max_cols = 8)
{
    // Print at most max_cols columns per pass to keep lines manageable.
    // Column header
    const int w = 14;
    const int label_w = 8;

    for (int64_t col_start = 0; col_start < m.cols; col_start += max_cols)
    {
        const int64_t col_end = std::min(col_start + max_cols, m.cols);

        // Column indices
        out << std::string(label_w, ' ');
        for (int64_t c = col_start; c < col_end; ++c)
            out << std::setw(w) << std::right << (c + 1);
        out << "\n";

        // Separator
        out << std::string(label_w + (col_end - col_start) * w, '-') << "\n";

        // Rows
        for (int64_t r = 0; r < m.rows; ++r)
        {
            out << std::setw(label_w) << std::right << (r + 1);
            for (int64_t c = col_start; c < col_end; ++c)
            {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(7) << m(static_cast<int>(r), static_cast<int>(c));
                out << std::setw(w) << std::right << oss.str();
            }
            out << "\n";
        }
        out << "\n";
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: chkdump <file.hfchk> [output.txt]\n";
        return EXIT_FAILURE;
    }

    const std::string in_path  = argv[1];
    const bool        to_file  = (argc == 3);
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

    // ── Header ────────────────────────────────────────────────────────────────
    char magic[8] = {};
    in.read(magic, 8);
    if (std::memcmp(magic, "PLNKCHK\0", 8) != 0)
    {
        std::cerr << "chkdump: not a Planck checkpoint file (bad magic)\n";
        return EXIT_FAILURE;
    }

    const uint32_t version   = read_pod<uint32_t>(in);
    if (version != 2)
    {
        std::cerr << "chkdump: unsupported checkpoint version " << version
                  << " (expected 2)\n";
        return EXIT_FAILURE;
    }

    const uint64_t nbasis      = read_pod<uint64_t>(in);
    const uint8_t  is_uhf      = read_pod<uint8_t>(in);
    const uint8_t  is_conv     = read_pod<uint8_t>(in);
    const uint32_t last_iter   = read_pod<uint32_t>(in);
    const double   tot_energy  = read_pod<double>(in);
    const double   nuc_rep     = read_pod<double>(in);

    // ── Molecule ──────────────────────────────────────────────────────────────
    const uint64_t natoms   = read_pod<uint64_t>(in);
    const int32_t  charge   = read_pod<int32_t>(in);
    const uint32_t mult     = read_pod<uint32_t>(in);

    std::vector<int32_t> atomic_numbers(natoms);
    for (uint64_t i = 0; i < natoms; ++i)
        atomic_numbers[i] = read_pod<int32_t>(in);

    std::vector<double> coords(natoms * 3);
    for (uint64_t i = 0; i < natoms * 3; ++i)
        coords[i] = read_pod<double>(in);

    const std::string basis_name     = read_string(in);
    const uint8_t     has_opt_coords = read_pod<uint8_t>(in);

    // ── 1e matrices ───────────────────────────────────────────────────────────
    const Matrix overlap = read_matrix(in);
    const Matrix hcore   = read_matrix(in);

    // ── Alpha spin channel ────────────────────────────────────────────────────
    const Matrix alpha_density = read_matrix(in);
    const Matrix alpha_fock    = read_matrix(in);
    const Matrix alpha_mo_e    = read_vector_as_matrix(in);
    const Matrix alpha_mo_c    = read_matrix(in);

    // ── Beta spin channel (UHF only) ──────────────────────────────────────────
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

    // ── Derive electron counts ────────────────────────────────────────────────
    int Z_total = 0;
    for (auto z : atomic_numbers) Z_total += z;
    const int n_elec   = Z_total - charge;
    const int n_unpair = static_cast<int>(mult) - 1;
    const int n_alpha  = (n_elec + n_unpair) / 2;
    const int n_beta   = (n_elec - n_unpair) / 2;
    const int homo_a   = n_alpha - 1;
    const int lumo_a   = n_alpha;
    const int homo_b   = n_beta  - 1;
    const int lumo_b   = n_beta;

    // ── Output ────────────────────────────────────────────────────────────────
    rule(out);
    out << "  Planck Checkpoint File  —  " << in_path << "\n";
    rule(out);
    out << "\n";

    // General info
    kv(out, "Format version",  std::to_string(version));
    kv(out, "SCF type",        is_uhf ? "UHF" : "RHF");
    kv(out, "Converged",       is_conv ? "yes" : "no");
    if (last_iter > 0) kv(out, "Last SCF iter", std::to_string(last_iter));
    {
        std::ostringstream s;
        s << std::fixed << std::setprecision(10) << tot_energy << " Eh";
        kv(out, "Total energy", s.str());
    }
    {
        std::ostringstream s;
        s << std::fixed << std::setprecision(10) << nuc_rep << " Eh";
        kv(out, "Nuclear repulsion", s.str());
    }
    kv(out, "Basis set",       basis_name);
    kv(out, "Nbasis",          std::to_string(nbasis));
    kv(out, "Geometry source", has_opt_coords
                               ? "optimized (converged geomopt)"
                               : "input (single-point or gradient)");

    // Molecule
    section(out, "Molecule");
    kv(out, "Natoms",       std::to_string(natoms));
    kv(out, "Charge",       std::to_string(charge));
    kv(out, "Multiplicity", std::to_string(mult));
    kv(out, "Electrons",    std::to_string(n_elec)
                            + "  (" + std::to_string(n_alpha) + " alpha, "
                            + std::to_string(n_beta)  + " beta)");

    out << "\n";
    out << "  " << std::setw(4) << "Atom"
        << std::setw(4) << "Z"
        << std::setw(5) << "Sym"
        << std::setw(18) << "X (Bohr)"
        << std::setw(16) << "Y (Bohr)"
        << std::setw(16) << "Z (Bohr)" << "\n";
    out << "  " << std::string(59, '-') << "\n";
    for (uint64_t i = 0; i < natoms; ++i)
    {
        const int    Z  = atomic_numbers[i];
        const double x  = coords[i * 3 + 0];
        const double y  = coords[i * 3 + 1];
        const double z  = coords[i * 3 + 2];
        out << "  "
            << std::setw(4) << (i + 1)
            << std::setw(4) << Z
            << std::setw(5) << element_symbol(Z)
            << std::setw(18) << std::fixed << std::setprecision(8) << x
            << std::setw(16) << std::fixed << std::setprecision(8) << y
            << std::setw(16) << std::fixed << std::setprecision(8) << z
            << "\n";
    }

    // MO energies
    auto print_mo_energies = [&](const Matrix& mo_e, int homo, int lumo,
                                 const std::string& spin_label)
    {
        section(out, spin_label + " MO Energies (Eh)");
        out << "  " << std::setw(6)  << "MO"
            << std::setw(18) << "Energy (Eh)"
            << "  \n";
        out << "  " << std::string(26, '-') << "\n";
        for (int64_t i = 0; i < mo_e.rows; ++i)
        {
            out << "  " << std::setw(6) << (i + 1)
                << std::setw(18) << std::fixed << std::setprecision(6) << mo_e.data[i];
            if (i == homo) out << "  <-- HOMO";
            if (i == lumo) out << "  <-- LUMO";
            out << "\n";
        }
    };

    if (is_uhf)
    {
        print_mo_energies(alpha_mo_e, homo_a, lumo_a, "Alpha");
        print_mo_energies(beta_mo_e,  homo_b, lumo_b, "Beta");
    }
    else
    {
        print_mo_energies(alpha_mo_e, homo_a, lumo_a, "");
    }

    // 1e matrices
    section(out, "Overlap Matrix  (" + std::to_string(overlap.rows)
                 + " x " + std::to_string(overlap.cols) + ")");
    print_matrix(out, overlap);

    section(out, "Core Hamiltonian  (" + std::to_string(hcore.rows)
                 + " x " + std::to_string(hcore.cols) + ")");
    print_matrix(out, hcore);

    // SCF results
    auto print_spin = [&](const Matrix& dens, const Matrix& fock,
                          const Matrix& mo_c, const std::string& label)
    {
        const std::string prefix = label.empty() ? "" : label + " ";

        section(out, prefix + "Density Matrix  ("
                + std::to_string(dens.rows) + " x " + std::to_string(dens.cols) + ")");
        print_matrix(out, dens);

        section(out, prefix + "Fock Matrix  ("
                + std::to_string(fock.rows) + " x " + std::to_string(fock.cols) + ")");
        print_matrix(out, fock);

        section(out, prefix + "MO Coefficients  ("
                + std::to_string(mo_c.rows) + " x " + std::to_string(mo_c.cols) + ")");
        print_matrix(out, mo_c);
    };

    if (is_uhf)
    {
        print_spin(alpha_density, alpha_fock, alpha_mo_c, "Alpha");
        print_spin(beta_density,  beta_fock,  beta_mo_c,  "Beta");
    }
    else
    {
        print_spin(alpha_density, alpha_fock, alpha_mo_c, "");
    }

    out << "\n";
    rule(out);
    out << "  End of checkpoint dump\n";
    rule(out);
    out << "\n";

    if (to_file)
        std::cerr << "chkdump: wrote " << out_path << "\n";

    return EXIT_SUCCESS;
}
