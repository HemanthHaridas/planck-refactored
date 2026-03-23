// chkdump -- export a Planck binary checkpoint (.hfchk) as a Gaussian-style
// formatted checkpoint text file (.fchk-like), or generate a Gaussian cube file.
//
// Usage:
//   chkdump <file.hfchk> [output.fchk]
//   chkdump <file.hfchk> --density [output.cube]
//   chkdump <file.hfchk> --mo N   [output.cube]
//       [--spin alpha|beta]   UHF spin channel (default: alpha)
//       [--spacing F]         grid spacing in Bohr (default: 0.2)
//       [--pad F]             padding around molecule in Bohr (default: 5.0)
//       [--casscf]            use CASSCF MOs instead of SCF MOs (requires --mo)
//
// Notes:
// - The Planck checkpoint does not store the full Gaussian basis metadata that a
//   canonical Gaussian .fchk would contain, so this exporter writes the common
//   scalar, vector, and matrix sections that can be reconstructed exactly from
//   the checkpoint contents.
// - Symmetric matrices are emitted in packed lower-triangular form, matching
//   the usual formatted-checkpoint convention.
// - Cube generation requires a v4 checkpoint (re-run the calculation to produce one).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
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

// ─── Cube file support ────────────────────────────────────────────────────────

struct ChkShell
{
    int shell_type;                       // L = shell_type (0=S,1=P,2=D,...)
    double cx, cy, cz;                   // center in Bohr
    std::vector<double> primitives;      // exponents alpha_k
    std::vector<double> coefficients;   // c_k × Nc (contracted norm already folded)
    std::vector<double> normalizations; // per-primitive N_k
};

struct ChkBF
{
    std::size_t shell_index;
    int lx, ly, lz;
    double component_norm; // 1/sqrt((2lx-1)!!(2ly-1)!!(2lz-1)!!)
};

struct Grid
{
    double xmin, ymin, zmin; // origin in Bohr
    int    nx, ny, nz;
    double dx, dy, dz;       // spacing in Bohr
};

struct CubeOptions
{
    bool        cube_mode  = false;
    bool        do_density = false;
    int         mo_index   = 0;     // 1-indexed; 0 = not set
    bool        spin_beta  = false;
    bool        use_casscf = false;
    double      spacing    = 0.2;   // Bohr
    double      pad        = 5.0;   // Bohr
    std::string out_path;
};

static Grid make_grid(const std::vector<double>& coords, std::size_t natoms,
                      double spacing, double pad)
{
    double xlo = coords[0], xhi = coords[0];
    double ylo = coords[1], yhi = coords[1];
    double zlo = coords[2], zhi = coords[2];
    for (std::size_t i = 0; i < natoms; ++i)
    {
        xlo = std::min(xlo, coords[3*i]);   xhi = std::max(xhi, coords[3*i]);
        ylo = std::min(ylo, coords[3*i+1]); yhi = std::max(yhi, coords[3*i+1]);
        zlo = std::min(zlo, coords[3*i+2]); zhi = std::max(zhi, coords[3*i+2]);
    }
    Grid g;
    g.xmin = xlo - pad;  g.ymin = ylo - pad;  g.zmin = zlo - pad;
    g.dx = g.dy = g.dz = spacing;
    g.nx = static_cast<int>(std::ceil((xhi - g.xmin + pad) / spacing)) + 1;
    g.ny = static_cast<int>(std::ceil((yhi - g.ymin + pad) / spacing)) + 1;
    g.nz = static_cast<int>(std::ceil((zhi - g.zmin + pad) / spacing)) + 1;
    return g;
}

// Evaluate all nbf contracted GTOs at point (px, py, pz).
// phi_mu(r) = component_norm * (dx^lx * dy^ly * dz^lz)
//           * sum_k [ normalizations[k] * coefficients[k] * exp(-primitives[k] * r2) ]
static std::vector<double> eval_basis(double px, double py, double pz,
                                      const std::vector<ChkShell>& shells,
                                      const std::vector<ChkBF>& bfs)
{
    std::vector<double> phi(bfs.size(), 0.0);
    for (std::size_t mu = 0; mu < bfs.size(); ++mu)
    {
        const ChkBF&    bf = bfs[mu];
        const ChkShell& sh = shells[bf.shell_index];
        const double dx = px - sh.cx;
        const double dy = py - sh.cy;
        const double dz = pz - sh.cz;
        const double r2 = dx*dx + dy*dy + dz*dz;

        // Angular part: integer multiplication loops avoid std::pow(0.0, 0) edge cases
        double ang = 1.0;
        for (int i = 0; i < bf.lx; ++i) ang *= dx;
        for (int i = 0; i < bf.ly; ++i) ang *= dy;
        for (int i = 0; i < bf.lz; ++i) ang *= dz;

        // Contracted radial part
        double rad = 0.0;
        for (std::size_t k = 0; k < sh.primitives.size(); ++k)
            rad += sh.normalizations[k] * sh.coefficients[k]
                   * std::exp(-sh.primitives[k] * r2);

        phi[mu] = bf.component_norm * ang * rad;
    }
    return phi;
}

// rho(r) = sum_{mu,nu} P_{mu,nu} * phi_mu(r) * phi_nu(r)
static double eval_density(const std::vector<double>& phi, const Matrix& P)
{
    double rho = 0.0;
    const int64_t n = P.rows;
    for (int64_t mu = 0; mu < n; ++mu)
    {
        double acc = 0.0;
        for (int64_t nu = 0; nu < n; ++nu)
            acc += P(mu, nu) * phi[static_cast<std::size_t>(nu)];
        rho += acc * phi[static_cast<std::size_t>(mu)];
    }
    return rho;
}

// psi_i(r) = sum_mu C_{mu,i} * phi_mu(r)  (i is 0-indexed)
static double eval_mo(const std::vector<double>& phi, const Matrix& C, int mo_idx)
{
    double psi = 0.0;
    for (int64_t mu = 0; mu < C.rows; ++mu)
        psi += C(mu, mo_idx) * phi[static_cast<std::size_t>(mu)];
    return psi;
}

// Write a standard Gaussian cube file.
// Loop order: X slowest, Y middle, Z fastest.
// Newline after every 6 values and always at the end of each Z-row.
static void write_cube(std::ostream& out,
                       const std::string& comment1,
                       const std::string& comment2,
                       const std::vector<int32_t>& atomic_numbers,
                       const std::vector<double>& coords, // natoms×3, Bohr
                       std::size_t natoms,
                       const Grid& g,
                       const std::vector<double>& values) // nx×ny×nz, x-major
{
    out << comment1 << "\n";
    out << comment2 << "\n";

    out << std::fixed << std::setprecision(6);

    // natoms and grid origin
    out << std::setw(5) << static_cast<int>(natoms)
        << std::setw(12) << g.xmin
        << std::setw(12) << g.ymin
        << std::setw(12) << g.zmin << "\n";

    // Grid vectors (axis-aligned)
    out << std::setw(5) << g.nx
        << std::setw(12) << g.dx << std::setw(12) << 0.0 << std::setw(12) << 0.0 << "\n";
    out << std::setw(5) << g.ny
        << std::setw(12) << 0.0 << std::setw(12) << g.dy << std::setw(12) << 0.0 << "\n";
    out << std::setw(5) << g.nz
        << std::setw(12) << 0.0 << std::setw(12) << 0.0 << std::setw(12) << g.dz << "\n";

    // Atom records: Z  0.0  x  y  z  (charge field is 0.0 by Gaussian convention)
    for (std::size_t i = 0; i < natoms; ++i)
        out << std::setw(5) << atomic_numbers[i]
            << std::setw(12) << 0.0
            << std::setw(12) << coords[3*i]
            << std::setw(12) << coords[3*i + 1]
            << std::setw(12) << coords[3*i + 2] << "\n";

    // Volumetric data
    out << std::scientific << std::setprecision(5);
    const std::size_t nz = static_cast<std::size_t>(g.nz);
    for (std::size_t idx = 0; idx < values.size(); ++idx)
    {
        out << std::setw(13) << values[idx];
        // Break after every 6 values, or at the end of each Z-row
        if ((idx + 1) % 6 == 0 || (idx + 1) % nz == 0)
            out << "\n";
    }
}

// ─── CLI parsing ─────────────────────────────────────────────────────────────

static std::string stem(const std::string& path)
{
    return std::filesystem::path(path).stem().string();
}

static bool parse_double(const char* s, double& out)
{
    char* end = nullptr;
    out = std::strtod(s, &end);
    return end != s && *end == '\0';
}

static bool parse_int(const char* s, int& out)
{
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') return false;
    out = static_cast<int>(v);
    return true;
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage:\n"
                     "  chkdump <file.hfchk> [output.fchk]\n"
                     "  chkdump <file.hfchk> --density [output.cube]\n"
                     "  chkdump <file.hfchk> --mo N   [output.cube]\n"
                     "      [--spin alpha|beta] [--spacing F] [--pad F] [--casscf]\n";
        return EXIT_FAILURE;
    }

    const std::string in_path = argv[1];

    // Parse remaining arguments
    CubeOptions opts;
    std::string explicit_out;
    bool has_explicit_out = false;

    for (int i = 2; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--density")
        {
            opts.cube_mode  = true;
            opts.do_density = true;
        }
        else if (arg == "--mo")
        {
            if (i + 1 >= argc) { std::cerr << "chkdump: --mo requires an integer argument\n"; return EXIT_FAILURE; }
            if (!parse_int(argv[++i], opts.mo_index) || opts.mo_index < 1)
            {
                std::cerr << "chkdump: --mo argument must be a positive integer\n";
                return EXIT_FAILURE;
            }
            opts.cube_mode = true;
        }
        else if (arg == "--spin")
        {
            if (i + 1 >= argc) { std::cerr << "chkdump: --spin requires alpha or beta\n"; return EXIT_FAILURE; }
            std::string spin = argv[++i];
            if (spin == "beta")       opts.spin_beta = true;
            else if (spin != "alpha") { std::cerr << "chkdump: --spin must be alpha or beta\n"; return EXIT_FAILURE; }
        }
        else if (arg == "--spacing")
        {
            if (i + 1 >= argc || !parse_double(argv[++i], opts.spacing) || opts.spacing <= 0.0)
            { std::cerr << "chkdump: --spacing requires a positive float (Bohr)\n"; return EXIT_FAILURE; }
        }
        else if (arg == "--pad")
        {
            if (i + 1 >= argc || !parse_double(argv[++i], opts.pad) || opts.pad < 0.0)
            { std::cerr << "chkdump: --pad requires a non-negative float (Bohr)\n"; return EXIT_FAILURE; }
        }
        else if (arg == "--casscf")
        {
            opts.use_casscf = true;
        }
        else if (arg[0] == '-' && arg[1] == '-')
        {
            std::cerr << "chkdump: unknown option: " << arg << "\n";
            return EXIT_FAILURE;
        }
        else
        {
            // Positional: output path
            if (has_explicit_out) { std::cerr << "chkdump: unexpected argument: " << arg << "\n"; return EXIT_FAILURE; }
            explicit_out     = arg;
            has_explicit_out = true;
        }
    }

    // Validate cube option combinations
    if (opts.cube_mode && opts.do_density && opts.mo_index > 0)
    {
        std::cerr << "chkdump: specify exactly one of --density or --mo N\n";
        return EXIT_FAILURE;
    }
    if (opts.cube_mode && !opts.do_density && opts.mo_index == 0)
    {
        std::cerr << "chkdump: cube mode requires --density or --mo N\n";
        return EXIT_FAILURE;
    }
    if (opts.use_casscf && opts.mo_index == 0)
    {
        std::cerr << "chkdump: --casscf requires --mo N\n";
        return EXIT_FAILURE;
    }

    // ── Open input ────────────────────────────────────────────────────────────

    std::ifstream in(in_path, std::ios::binary);
    if (!in)
    {
        std::cerr << "chkdump: cannot open '" << in_path << "'\n";
        return EXIT_FAILURE;
    }

    // ── Read header ───────────────────────────────────────────────────────────

    char magic[8] = {};
    in.read(magic, 8);
    if (std::memcmp(magic, "PLNKCHK\0", 8) != 0)
    {
        std::cerr << "chkdump: not a Planck checkpoint file (bad magic)\n";
        return EXIT_FAILURE;
    }

    const uint32_t version = read_pod<uint32_t>(in);
    if (version < 2 || version > 4)
    {
        std::cerr << "chkdump: unsupported checkpoint version " << version
                  << " (expected 2–4)\n";
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

    uint8_t has_casscf_mos = 0;
    Matrix casscf_mo_c;
    if (version >= 3)
    {
        has_casscf_mos = read_pod<uint8_t>(in);
        if (has_casscf_mos)
            casscf_mo_c = read_matrix(in);
    }

    // ── v4: Basis shell data ──────────────────────────────────────────────────

    std::vector<ChkShell> shells;
    std::vector<ChkBF>    bfs;
    bool has_basis = false;

    if (version >= 4)
    {
        has_basis = (read_pod<uint8_t>(in) != 0);
        if (has_basis)
        {
            const uint64_t nshells = read_pod<uint64_t>(in);
            shells.resize(nshells);
            for (auto& sh : shells)
            {
                sh.shell_type = read_pod<int32_t>(in);
                const uint32_t np = read_pod<uint32_t>(in);
                sh.cx = read_pod<double>(in);
                sh.cy = read_pod<double>(in);
                sh.cz = read_pod<double>(in);
                sh.primitives.resize(np);
                sh.coefficients.resize(np);
                sh.normalizations.resize(np);
                for (auto& v : sh.primitives)    v = read_pod<double>(in);
                for (auto& v : sh.coefficients)  v = read_pod<double>(in);
                for (auto& v : sh.normalizations) v = read_pod<double>(in);
            }
            const uint64_t nbf2 = read_pod<uint64_t>(in);
            bfs.resize(nbf2);
            for (auto& bf : bfs)
            {
                bf.shell_index    = static_cast<std::size_t>(read_pod<uint64_t>(in));
                bf.lx             = read_pod<int32_t>(in);
                bf.ly             = read_pod<int32_t>(in);
                bf.lz             = read_pod<int32_t>(in);
                bf.component_norm = read_pod<double>(in);
            }
        }
    }

    if (!in)
    {
        std::cerr << "chkdump: I/O error while reading checkpoint\n";
        return EXIT_FAILURE;
    }

    // ── Cube mode ─────────────────────────────────────────────────────────────

    if (opts.cube_mode)
    {
        if (!has_basis)
        {
            std::cerr << "chkdump: cube generation requires a v4 checkpoint; "
                         "re-run the calculation to regenerate.\n";
            return EXIT_FAILURE;
        }

        // Validate options that depend on checkpoint contents
        if (opts.spin_beta && !is_uhf)
        {
            std::cerr << "chkdump: --spin beta: beta spin channel not available for RHF checkpoint\n";
            return EXIT_FAILURE;
        }
        if (opts.mo_index > 0 && static_cast<uint64_t>(opts.mo_index) > nbasis)
        {
            std::cerr << "chkdump: --mo " << opts.mo_index
                      << " out of range [1, " << nbasis << "]\n";
            return EXIT_FAILURE;
        }
        if (opts.use_casscf && !has_casscf_mos)
        {
            std::cerr << "chkdump: --casscf: no CASSCF orbitals found in checkpoint\n";
            return EXIT_FAILURE;
        }

        // Determine output path
        if (!has_explicit_out)
        {
            const std::string s = stem(in_path);
            if (opts.do_density)
                opts.out_path = s + "_density.cube";
            else
            {
                const std::string spin_tag = opts.spin_beta ? "_beta" : "_alpha";
                const std::string prefix   = opts.use_casscf ? "_casscf_mo" : "_mo";
                opts.out_path = s + prefix + std::to_string(opts.mo_index) + spin_tag + ".cube";
            }
        }
        else
        {
            opts.out_path = explicit_out;
        }

        // Build grid
        const Grid g = make_grid(coords, static_cast<std::size_t>(natoms),
                                 opts.spacing, opts.pad);

        const std::size_t nx    = static_cast<std::size_t>(g.nx);
        const std::size_t ny    = static_cast<std::size_t>(g.ny);
        const std::size_t nz    = static_cast<std::size_t>(g.nz);
        const std::size_t total = nx * ny * nz;

        // Build UHF total density matrix if needed
        Matrix total_density;
        if (opts.do_density && is_uhf)
        {
            total_density.rows = alpha_density.rows;
            total_density.cols = alpha_density.cols;
            total_density.data.resize(alpha_density.data.size());
            for (std::size_t k = 0; k < alpha_density.data.size(); ++k)
                total_density.data[k] = alpha_density.data[k] + beta_density.data[k];
        }

        const Matrix& density_ref = (opts.do_density && is_uhf) ? total_density : alpha_density;
        const Matrix& mo_c_ref    = opts.use_casscf ? casscf_mo_c :
                                    (opts.spin_beta  ? beta_mo_c   : alpha_mo_c);

        // Evaluate grid
        std::vector<double> values(total);
        for (std::size_t ix = 0; ix < nx; ++ix)
        {
            const double px = g.xmin + static_cast<double>(ix) * g.dx;
            for (std::size_t iy = 0; iy < ny; ++iy)
            {
                const double py = g.ymin + static_cast<double>(iy) * g.dy;
                for (std::size_t iz = 0; iz < nz; ++iz)
                {
                    const double pz = g.zmin + static_cast<double>(iz) * g.dz;
                    const auto phi  = eval_basis(px, py, pz, shells, bfs);
                    values[ix*ny*nz + iy*nz + iz] = opts.do_density
                        ? eval_density(phi, density_ref)
                        : eval_mo(phi, mo_c_ref, opts.mo_index - 1);
                }
            }
        }

        // Build comment lines
        std::string comment1, comment2;
        if (opts.do_density)
        {
            comment1 = "Planck cube file - total electron density";
            comment2 = "Generated by chkdump from " + in_path;
        }
        else
        {
            const std::string spin_str = opts.spin_beta ? "beta" : "alpha";
            const std::string mo_src   = opts.use_casscf ? "CASSCF" : "SCF";
            comment1 = "Planck cube file - " + mo_src + " " + spin_str
                       + " MO " + std::to_string(opts.mo_index);
            comment2 = "Generated by chkdump from " + in_path;
        }

        // Open output
        std::ofstream fout(opts.out_path);
        if (!fout)
        {
            std::cerr << "chkdump: cannot open output file '" << opts.out_path << "'\n";
            return EXIT_FAILURE;
        }

        write_cube(fout, comment1, comment2,
                   atomic_numbers, coords, static_cast<std::size_t>(natoms),
                   g, values);

        std::cerr << "chkdump: wrote " << opts.out_path
                  << "  [" << g.nx << " × " << g.ny << " × " << g.nz << " pts]\n";
        return EXIT_SUCCESS;
    }

    // ── fchk export mode ──────────────────────────────────────────────────────

    std::ostream* outp = &std::cout;
    std::ofstream fout;
    if (has_explicit_out)
    {
        fout.open(explicit_out);
        if (!fout)
        {
            std::cerr << "chkdump: cannot open output file '" << explicit_out << "'\n";
            return EXIT_FAILURE;
        }
        outp = &fout;
    }
    std::ostream& out = *outp;

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
    write_scalar_int(out, "Checkpoint version", static_cast<long long>(version));
    write_scalar_int(out, "Has CASSCF orbitals", static_cast<long long>(has_casscf_mos));
    write_scalar_real(out, "SCF Energy", tot_energy);
    write_scalar_real(out, "Nuclear repulsion energy", nuc_rep);

    write_integer_array(out, "Atomic numbers", to_int32_vector(atomic_numbers));
    write_real_array(out, "Nuclear charges", nuclear_charges);
    write_real_array(out, "Current cartesian coordinates", coords);

    // Gaussian fchk ordering: Overlap and Core Hamiltonian precede orbital data
    write_real_array(out, "Overlap Matrix", overlap_packed);
    write_real_array(out, "Core Hamiltonian Matrix", hcore_packed);

    write_real_array(out, "Alpha Orbital Energies", flatten_matrix(alpha_mo_e));
    write_real_array(out, "Alpha MO coefficients", flatten_matrix(alpha_mo_c));

    if (is_uhf)
    {
        write_real_array(out, "Beta Orbital Energies", flatten_matrix(beta_mo_e));
        write_real_array(out, "Beta MO coefficients", flatten_matrix(beta_mo_c));
    }

    if (has_casscf_mos)
        write_real_array(out, "CASSCF MO coefficients", flatten_matrix(casscf_mo_c));

    write_real_array(out, "Total SCF Density", total_density_packed);
    write_real_array(out, "Alpha Density Matrix", alpha_density_packed);
    if (is_uhf)
        write_real_array(out, "Beta Density Matrix", packed_lower_triangle(beta_density));

    write_real_array(out, "Alpha Fock Matrix", packed_lower_triangle(alpha_fock));
    if (is_uhf)
        write_real_array(out, "Beta Fock Matrix", packed_lower_triangle(beta_fock));

    return EXIT_SUCCESS;
}
