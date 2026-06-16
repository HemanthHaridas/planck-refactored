// Per-(L_AB, L_CD) timing sweep for OS / HGP / Rys.
//
// Drives the calibration step of docs/AUTO_DISPATCH_PLAN.md: we need
// ms/quartet broken down by angular-momentum bucket so the auto-dispatch
// predicate can be fit against measured cost curves instead of a textbook
// estimate. Output is a CSV under docs/ with one row per
// (basis, engine, L_AB, L_CD).
//
// Molecules are kept small at high L on purpose: He at cc-pVQZ / cc-pV5Z
// gives the diagonal g/h buckets without paying for the N^4 cost of a
// multi-atom run. Water tops out at cc-pVTZ (f shells); cc-pVQZ/5Z on
// water are intentionally not in the sweep — they exist but are too
// expensive for the calibration to need.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "base/basis.h"
#include "base/types.h"
#include "basis/basis.h"
#include "integrals/hgp.h"
#include "integrals/os.h"
#include "integrals/rys.h"
#include "integrals/shellpair.h"

namespace
{
    using Clock = std::chrono::steady_clock;

    // (L_AB, L_CD) bucket key.
    using Bucket = std::pair<int, int>;

    int shell_L(const HartreeFock::ContractedView &cv) noexcept
    {
        return cv._cartesian[0] + cv._cartesian[1] + cv._cartesian[2];
    }

    int pair_L(const HartreeFock::ShellPair &sp) noexcept
    {
        return shell_L(sp.A) + shell_L(sp.B);
    }

    double median_ms(int reps, const std::function<void()> &fn)
    {
        fn(); // warmup
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(reps));
        for (int r = 0; r < reps; ++r)
        {
            const auto t0 = Clock::now();
            fn();
            const auto t1 = Clock::now();
            samples.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    }

    // Precomputed quartet sample (pair-index pairs) for one (L_AB, L_CD) bucket.
    using QuartetSample = std::vector<std::pair<std::size_t, std::size_t>>;

    // Build the quartet sample for a bucket. If the bucket has more than
    // `max_quartets` total entries, stride-sample uniformly. The ms/quartet
    // figure the fit needs is stable once the sample is in the ~10^3 range,
    // and the higher-L cc-pVTZ buckets hit 10^5 quartets at full population.
    QuartetSample bucket_quartets(const std::vector<HartreeFock::ShellPair> &pairs,
                                  int L_AB, int L_CD,
                                  std::size_t max_quartets)
    {
        std::vector<std::size_t> idx_AB;
        std::vector<std::size_t> idx_CD;
        idx_AB.reserve(pairs.size());
        idx_CD.reserve(pairs.size());
        for (std::size_t i = 0; i < pairs.size(); ++i)
        {
            const int L = pair_L(pairs[i]);
            if (L == L_AB) idx_AB.push_back(i);
            if (L == L_CD) idx_CD.push_back(i);
        }
        const std::size_t total = idx_AB.size() * idx_CD.size();

        QuartetSample sample;
        if (total <= max_quartets)
        {
            sample.reserve(total);
            for (std::size_t a : idx_AB)
                for (std::size_t c : idx_CD)
                    sample.emplace_back(a, c);
            return sample;
        }
        // Stride-sample. Use coprime-ish strides so coverage is even.
        const double stride = static_cast<double>(total) / static_cast<double>(max_quartets);
        sample.reserve(max_quartets);
        for (std::size_t k = 0; k < max_quartets; ++k)
        {
            const std::size_t flat = static_cast<std::size_t>(k * stride);
            const std::size_t a = idx_AB[flat / idx_CD.size()];
            const std::size_t c = idx_CD[flat % idx_CD.size()];
            sample.emplace_back(a, c);
        }
        return sample;
    }

    template <typename Kernel>
    void bucket_pass(const std::vector<HartreeFock::ShellPair> &pairs,
                     const QuartetSample &sample,
                     Kernel &&kernel)
    {
        volatile double sink = 0.0;
        for (auto [ij, kl] : sample)
            sink += kernel(pairs[ij], pairs[kl]);
        (void)sink;
    }

    double eri_os(const HartreeFock::ShellPair &spAB,
                  const HartreeFock::ShellPair &spCD) noexcept
    {
        return HartreeFock::ObaraSaika::_contracted_eri_elem(
            spAB, spCD,
            spAB.A._cartesian[0], spAB.A._cartesian[1], spAB.A._cartesian[2],
            spAB.B._cartesian[0], spAB.B._cartesian[1], spAB.B._cartesian[2],
            spCD.A._cartesian[0], spCD.A._cartesian[1], spCD.A._cartesian[2],
            spCD.B._cartesian[0], spCD.B._cartesian[1], spCD.B._cartesian[2]);
    }

    double eri_hgp(const HartreeFock::ShellPair &spAB,
                   const HartreeFock::ShellPair &spCD) noexcept
    {
        return HartreeFock::HeadGordonPople::_contracted_eri_elem(
            spAB, spCD,
            spAB.A._cartesian[0], spAB.A._cartesian[1], spAB.A._cartesian[2],
            spAB.B._cartesian[0], spAB.B._cartesian[1], spAB.B._cartesian[2],
            spCD.A._cartesian[0], spCD.A._cartesian[1], spCD.A._cartesian[2],
            spCD.B._cartesian[0], spCD.B._cartesian[1], spCD.B._cartesian[2]);
    }

    double eri_rys(const HartreeFock::ShellPair &spAB,
                   const HartreeFock::ShellPair &spCD) noexcept
    {
        return HartreeFock::RysQuad::_rys_contracted_eri(
            spAB, spCD,
            spAB.A._cartesian[0], spAB.A._cartesian[1], spAB.A._cartesian[2],
            spAB.B._cartesian[0], spAB.B._cartesian[1], spAB.B._cartesian[2],
            spCD.A._cartesian[0], spCD.A._cartesian[1], spCD.A._cartesian[2],
            spCD.B._cartesian[0], spCD.B._cartesian[1], spCD.B._cartesian[2]);
    }

    std::expected<HartreeFock::Calculator, std::string>
    make_water(const std::string &basis_name)
    {
        HartreeFock::Calculator calc;
        HartreeFock::Molecule &mol = calc._molecule;
        mol.natoms = 3;
        mol.charge = 0;
        mol.multiplicity = 1;
        mol.atomic_numbers.resize(3);
        mol.atomic_numbers << 8, 1, 1;
        mol.atomic_masses.resize(3);
        mol.atomic_masses << 16.0, 1.0, 1.0;
        mol.coordinates.resize(3, 3);
        mol.coordinates <<
            0.000000, 0.000000,  0.117176,
            0.000000, 0.757200, -0.468704,
            0.000000,-0.757200, -0.468704;
        calc._basis._basis = HartreeFock::BasisType::Cartesian;
        calc.prepare_coordinates();
        mol.set_standard_from_bohr(mol._coordinates);

        const std::filesystem::path gbs =
            std::filesystem::path(get_basis_path()) / basis_name;
        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(
            gbs.string(), mol, calc._basis._basis);
        if (!basis_res)
            return std::unexpected("read_gbs_basis failed for " + basis_name + ": " + basis_res.error());
        calc._shells = std::move(*basis_res);
        return calc;
    }

    std::expected<HartreeFock::Calculator, std::string>
    make_helium(const std::string &basis_name)
    {
        HartreeFock::Calculator calc;
        HartreeFock::Molecule &mol = calc._molecule;
        mol.natoms = 1;
        mol.charge = 0;
        mol.multiplicity = 1;
        mol.atomic_numbers.resize(1);
        mol.atomic_numbers << 2;
        mol.atomic_masses.resize(1);
        mol.atomic_masses << 4.0;
        mol.coordinates.resize(1, 3);
        mol.coordinates << 0.0, 0.0, 0.0;
        calc._basis._basis = HartreeFock::BasisType::Cartesian;
        calc.prepare_coordinates();
        mol.set_standard_from_bohr(mol._coordinates);

        const std::filesystem::path gbs =
            std::filesystem::path(get_basis_path()) / basis_name;
        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(
            gbs.string(), mol, calc._basis._basis);
        if (!basis_res)
            return std::unexpected("read_gbs_basis failed for " + basis_name + ": " + basis_res.error());
        calc._shells = std::move(*basis_res);
        return calc;
    }

    struct Case
    {
        std::string label;     // molecule
        std::string basis;     // gbs filename
        std::function<std::expected<HartreeFock::Calculator, std::string>(const std::string &)> build;
    };

    void sweep_case(const Case &c, int reps, std::size_t max_quartets, std::ofstream &csv)
    {
        auto calc_res = c.build(c.basis);
        if (!calc_res)
        {
            std::fprintf(stderr, "[SKIP] %s / %s: %s\n",
                         c.label.c_str(), c.basis.c_str(),
                         calc_res.error().c_str());
            return;
        }

        const auto pairs = build_shellpairs(calc_res->_shells);

        // Histogram (L_AB, L_CD) → full quartet count.
        std::map<Bucket, std::size_t> counts;
        for (const auto &spAB : pairs)
        {
            const int lab = pair_L(spAB);
            for (const auto &spCD : pairs)
            {
                const int lcd = pair_L(spCD);
                ++counts[{lab, lcd}];
            }
        }

        std::fprintf(stdout, "%s / %s: %zu shell-pairs, %zu populated buckets (cap=%zu quartets/bucket)\n",
                     c.label.c_str(), c.basis.c_str(),
                     pairs.size(), counts.size(), max_quartets);

        for (const auto &[bucket, full_count] : counts)
        {
            const int L_AB = bucket.first;
            const int L_CD = bucket.second;

            // Stride-sample if the bucket is huge. ms/quartet is what the fit
            // needs — it stays stable once the sample is large enough.
            const auto sample = bucket_quartets(pairs, L_AB, L_CD, max_quartets);
            const std::size_t n = sample.size();
            if (n == 0)
                continue;

            const double t_os  = median_ms(reps, [&] { bucket_pass(pairs, sample, eri_os); });
            const double t_hgp = median_ms(reps, [&] { bucket_pass(pairs, sample, eri_hgp); });
            const double t_rys = median_ms(reps, [&] { bucket_pass(pairs, sample, eri_rys); });

            const double per_os  = t_os  / static_cast<double>(n);
            const double per_hgp = t_hgp / static_cast<double>(n);
            const double per_rys = t_rys / static_cast<double>(n);

            // CSV `count` column reports the bench *sample size* — the number
            // of quartets the per-quartet timing was averaged over. The
            // full-population bucket count is on stdout for context.
            csv << c.label  << ',' << c.basis << ",os,"  << L_AB << ',' << L_CD
                << ',' << n << ',' << t_os  << ',' << per_os  << '\n';
            csv << c.label  << ',' << c.basis << ",hgp," << L_AB << ',' << L_CD
                << ',' << n << ',' << t_hgp << ',' << per_hgp << '\n';
            csv << c.label  << ',' << c.basis << ",rys," << L_AB << ',' << L_CD
                << ',' << n << ',' << t_rys << ',' << per_rys << '\n';

            std::fprintf(stdout,
                "  (%d,%d) full=%zu sampled=%zu  OS=%9.3f ms (%.3e/q)  HGP=%9.3f ms (%.3e/q)  Rys=%9.3f ms (%.3e/q)\n",
                L_AB, L_CD, full_count, n, t_os, per_os, t_hgp, per_hgp, t_rys, per_rys);
        }
    }

    // Header block. Keeps host/compiler/OMP/git context with the data so the
    // dataset is self-describing for future re-tuning, per plan step 2.
    void write_header(std::ofstream &csv)
    {
        const char *host = std::getenv("HOSTNAME");
        const char *omp  = std::getenv("OMP_NUM_THREADS");
        csv << "# planck auto-dispatch per-bucket timing sweep\n";
        csv << "# host=" << (host ? host : "(unset)") << '\n';
        csv << "# omp_num_threads=" << (omp ? omp : "(unset)") << '\n';
#ifdef __VERSION__
        csv << "# compiler=" << __VERSION__ << '\n';
#endif
        csv << "# columns: molecule,basis,engine,L_AB,L_CD,count,total_ms,ms_per_quartet\n";
        csv << "molecule,basis,engine,L_AB,L_CD,count,total_ms,ms_per_quartet\n";
    }
} // namespace

int main(int argc, char **argv)
{
    int reps = 3;
    std::size_t max_quartets = 2000;
    std::string csv_path = "docs/auto_dispatch_timings.csv";
    if (argc > 1) reps = std::max(1, std::atoi(argv[1]));
    if (argc > 2) csv_path = argv[2];
    if (argc > 3) max_quartets = static_cast<std::size_t>(std::max(1, std::atoi(argv[3])));

    std::ofstream csv(csv_path);
    if (!csv)
    {
        std::fprintf(stderr, "Failed to open %s for writing\n", csv_path.c_str());
        return 1;
    }
    write_header(csv);

    const std::vector<Case> cases = {
        {"water",  "sto-3g",   make_water},
        {"water",  "6-31g*",   make_water},
        {"water",  "cc-pVDZ",  make_water},
        {"water",  "cc-pVTZ",  make_water},
        {"helium", "cc-pVQZ",  make_helium},
        {"helium", "cc-pV5Z",  make_helium},
    };

    std::fprintf(stdout,
                 "Auto-dispatch per-bucket benchmark, reps=%d (median of), max %zu quartets/bucket\n",
                 reps, max_quartets);
    for (const auto &c : cases)
        sweep_case(c, reps, max_quartets, csv);

    std::fprintf(stdout, "wrote %s\n", csv_path.c_str());
    return 0;
}
