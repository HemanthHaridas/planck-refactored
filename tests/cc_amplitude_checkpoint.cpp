// Round-trip gate for the generated-CC amplitude sidecar (.ccamp).
// Builds a small 2-rank ArbitraryOrderRCCAmplitudes, saves it, loads it back,
// and asserts bytewise-equal dims/data plus metadata. Also checks that a bad
// magic and a truncated file are rejected (errored, not crashed).
//
// C0: extended to cover the version-2 sector block -- the higher independent
// Sz sectors (amplitudes.sectors) that version 1 silently dropped on write.
// Covers: a sector round-trips bytewise identically to by_rank; a version-1
// file (no sector block, no reference_type byte) still loads with zero
// sectors and RHF defaulted; a version-2 file with a TRUNCATED sector block
// (as opposed to no sector block at all) still errors rather than silently
// reading as "zero sectors" -- that distinction is the actual defect this
// gate exists to catch, since "ends right after by_rank" and "ends partway
// through n_sectors/a sector" must NOT be treated the same way.
//
// U0/U1: extended further to cover the version-3 fields (n_by_rank + UCC's
// four occupation counts). Covers: a sectors-only amplitude set (empty
// by_rank, the exact UCC shape) round-trips correctly, which the version-2
// format could not represent at all -- a writer-only fix for this was tried
// first and found broken on round-trip (see docs/CC_AMPLITUDE_CHECKPOINT.md,
// "History: two defects and one capability gap"), so this test exists
// specifically to keep that regression from recurring; a hand-built
// version-2 file (no n_by_rank, no UHF counts in the byte stream) loads
// with n_by_rank defaulted to THAT FILE'S OWN max_rank, not to 0 -- 0 would
// silently discard every existing version-2 sidecar's by_rank data.

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "post_hf/cc/cc_amplitude_checkpoint.h"

using namespace HartreeFock::Correlation::CC;

namespace
{
    ArbitraryOrderRCCAmplitudes make_sample()
    {
        // rank 1: [n_occ=2, n_virt=3] -> 6 elems; rank 2: [2,2,3,3] -> 36 elems.
        ArbitraryOrderRCCAmplitudes amps;
        TensorND t1({2, 3}, 0.0);
        for (std::size_t i = 0; i < t1.data.size(); ++i)
            t1.data[i] = 0.1 * static_cast<double>(i) - 0.3;
        TensorND t2({2, 2, 3, 3}, 0.0);
        for (std::size_t i = 0; i < t2.data.size(); ++i)
            t2.data[i] = -0.05 * static_cast<double>(i) + 1.7;
        amps.by_rank.push_back(std::move(t1));
        amps.by_rank.push_back(std::move(t2));
        return amps;
    }

    // A rank-4 amplitude set carrying one higher Sz sector, mirroring the
    // real CCSDTQ shape this defect was found on: by_rank[3] is the balanced
    // t4 sector, sectors holds the independent (4, "aaabaaab") block.
    ArbitraryOrderRCCAmplitudes make_sample_with_sector()
    {
        ArbitraryOrderRCCAmplitudes amps;
        // Ranks 1-4, tiny dims so the test stays fast: [n_occ=2, n_virt=2]
        // per excitation, i.e. rank r has dims [2]*r + [2]*r.
        for (int rank = 1; rank <= 4; ++rank)
        {
            std::vector<int> dims(static_cast<std::size_t>(2 * rank), 2);
            TensorND t(dims, 0.0);
            for (std::size_t i = 0; i < t.data.size(); ++i)
                t.data[i] = 0.01 * static_cast<double>(rank) * static_cast<double>(i) - 0.5;
            amps.by_rank.push_back(std::move(t));
        }
        std::vector<int> sector_dims(8, 2); // rank-4 shape, same as by_rank[3]
        TensorND sector(sector_dims, 0.0);
        for (std::size_t i = 0; i < sector.data.size(); ++i)
            sector.data[i] = 7.0 - 0.003 * static_cast<double>(i); // distinct from by_rank[3]
        amps.sectors.push_back({{4, "aaabaaab"}, std::move(sector)});
        return amps;
    }

    // U0/U1: the exact UCC shape -- by_rank EMPTY, all real data in sectors.
    // Mirrors prepare_generated_ucc_state's own documented state ("No
    // amplitudes at all: by_rank stays empty... the sectors are filled by
    // ensure_amplitude_sectors"). Two rank-2 spin blocks, matching
    // ucc_amplitude_blocks(2)'s tag shape ("aaaa"/"abab"/"bbbb"-style).
    ArbitraryOrderRCCAmplitudes make_ucc_sample()
    {
        ArbitraryOrderRCCAmplitudes amps;
        std::vector<int> dims{2, 2, 2, 2};
        TensorND aaaa(dims, 0.0);
        for (std::size_t i = 0; i < aaaa.data.size(); ++i)
            aaaa.data[i] = 0.02 * static_cast<double>(i) - 0.1;
        TensorND abab(dims, 0.0);
        for (std::size_t i = 0; i < abab.data.size(); ++i)
            abab.data[i] = 0.03 * static_cast<double>(i) + 0.4; // distinct from aaaa
        amps.sectors.push_back({{2, "aaaa"}, std::move(aaaa)});
        amps.sectors.push_back({{2, "abab"}, std::move(abab)});
        // by_rank deliberately left empty.
        return amps;
    }

    std::string temp_path(const char *name)
    {
        return (std::filesystem::temp_directory_path() / name).string();
    }
} // namespace

int main()
{
    const std::string path = temp_path("planck_ccamp_roundtrip.ccamp");

    const ArbitraryOrderRCCAmplitudes original = make_sample();
    CCAmplitudeCheckpointMeta meta{
        .max_rank = 2, .method = "cc2", .basis_name = "sto-3g", .n_occ = 2, .n_virt = 3};

    auto saved = save_cc_amplitudes(path, original, meta);
    assert(saved && "save should succeed");

    auto loaded = load_cc_amplitudes(path);
    assert(loaded && "load should succeed");

    // Metadata survives.
    assert(loaded->meta.max_rank == 2);
    assert(loaded->meta.method == "cc2");
    assert(loaded->meta.basis_name == "sto-3g");
    assert(loaded->meta.n_occ == 2 && loaded->meta.n_virt == 3);

    // Amplitudes are bytewise-equal.
    assert(loaded->amplitudes.by_rank.size() == original.by_rank.size());
    for (std::size_t r = 0; r < original.by_rank.size(); ++r)
    {
        assert(loaded->amplitudes.by_rank[r].dims == original.by_rank[r].dims);
        assert(loaded->amplitudes.by_rank[r].data == original.by_rank[r].data);
    }

    // Bad magic errors, does not crash.
    {
        const std::string bad = temp_path("planck_ccamp_badmagic.ccamp");
        std::ofstream out(bad, std::ios::binary | std::ios::trunc);
        const char junk[16] = {'N', 'O', 'T', 'C', 'C', 'A', 'M', 'P', 0, 0, 0, 0, 0, 0, 0, 0};
        out.write(junk, 16);
        out.close();
        auto r = load_cc_amplitudes(bad);
        assert(!r && "bad magic must error");
        std::filesystem::remove(bad);
    }

    // Truncated file errors, does not crash.
    {
        const std::string trunc = temp_path("planck_ccamp_trunc.ccamp");
        {
            std::ifstream in(path, std::ios::binary);
            std::string bytes((std::istreambuf_iterator<char>(in)), {});
            std::ofstream out(trunc, std::ios::binary | std::ios::trunc);
            out.write(bytes.data(), static_cast<std::streamsize>(bytes.size() / 2));
        }
        auto r = load_cc_amplitudes(trunc);
        assert(!r && "truncated file must error");
        std::filesystem::remove(trunc);
    }

    // Missing file errors.
    {
        auto r = load_cc_amplitudes(temp_path("planck_ccamp_does_not_exist.ccamp"));
        assert(!r && "missing file must error");
    }

    std::filesystem::remove(path);

    // C0: a sector round-trips bytewise identically to by_rank, and RHF is
    // the correct default reference_type when none is set explicitly.
    {
        const std::string sector_path = temp_path("planck_ccamp_sector_roundtrip.ccamp");
        const ArbitraryOrderRCCAmplitudes original = make_sample_with_sector();
        CCAmplitudeCheckpointMeta meta{
            .max_rank = 4, .method = "cc4", .basis_name = "sto-3g", .n_occ = 2, .n_virt = 2};

        auto saved = save_cc_amplitudes(sector_path, original, meta);
        assert(saved && "save with a sector should succeed");

        auto loaded = load_cc_amplitudes(sector_path);
        assert(loaded && "load with a sector should succeed");
        assert(loaded->meta.reference_type == CCReferenceType::RHF);

        assert(loaded->amplitudes.by_rank.size() == original.by_rank.size());
        for (std::size_t r = 0; r < original.by_rank.size(); ++r)
        {
            assert(loaded->amplitudes.by_rank[r].dims == original.by_rank[r].dims);
            assert(loaded->amplitudes.by_rank[r].data == original.by_rank[r].data);
        }

        assert(loaded->amplitudes.sectors.size() == 1);
        assert(loaded->amplitudes.sectors[0].first.first == 4);
        assert(loaded->amplitudes.sectors[0].first.second == "aaabaaab");
        assert(loaded->amplitudes.sectors[0].second.dims == original.sectors[0].second.dims);
        assert(loaded->amplitudes.sectors[0].second.data == original.sectors[0].second.data);
        // The sector must be genuinely distinct from by_rank[3] in the
        // round-tripped data too -- otherwise this gate could pass even if
        // the loader accidentally aliased the two.
        assert(loaded->amplitudes.sectors[0].second.data != loaded->amplitudes.by_rank[3].data);

        std::filesystem::remove(sector_path);
    }

    // C0: a hand-built version-1 file (no reference_type byte, no sector
    // block -- the exact shape every sidecar written before this fix has on
    // disk) still loads: zero sectors, RHF reference_type, by_rank intact.
    // This is the compatibility contract the whole point of versioning is
    // for; if this regresses, every pre-existing .ccamp becomes unreadable.
    {
        const std::string v1_path = temp_path("planck_ccamp_v1_compat.ccamp");
        {
            std::ofstream out(v1_path, std::ios::binary | std::ios::trunc);
            const char magic[8] = {'P', 'L', 'N', 'K', 'C', 'C', 'A', '\0'};
            out.write(magic, 8);
            const std::uint32_t version = 1;
            out.write(reinterpret_cast<const char *>(&version), 4);
            const std::int32_t max_rank = 1;
            out.write(reinterpret_cast<const char *>(&max_rank), 4);
            const std::uint32_t method_len = 3;
            out.write(reinterpret_cast<const char *>(&method_len), 4);
            out.write("cc1", 3);
            const std::uint32_t basis_len = 6;
            out.write(reinterpret_cast<const char *>(&basis_len), 4);
            out.write("sto-3g", 6);
            const std::uint64_t n_occ = 2, n_virt = 2;
            out.write(reinterpret_cast<const char *>(&n_occ), 8);
            out.write(reinterpret_cast<const char *>(&n_virt), 8);
            // rank 1: order=2, dims=[2,2], count=4, data
            const std::int32_t order = 2;
            out.write(reinterpret_cast<const char *>(&order), 4);
            const std::int32_t d0 = 2, d1 = 2;
            out.write(reinterpret_cast<const char *>(&d0), 4);
            out.write(reinterpret_cast<const char *>(&d1), 4);
            const std::uint64_t count = 4;
            out.write(reinterpret_cast<const char *>(&count), 8);
            const double data[4] = {1.0, 2.0, 3.0, 4.0};
            out.write(reinterpret_cast<const char *>(data), sizeof(data));
            // File ends here -- no reference_type byte, no sector block.
        }

        auto loaded = load_cc_amplitudes(v1_path);
        assert(loaded && "a version-1 file must still load");
        assert(loaded->meta.method == "cc1");
        assert(loaded->meta.reference_type == CCReferenceType::RHF);
        assert(loaded->amplitudes.by_rank.size() == 1);
        assert((loaded->amplitudes.by_rank[0].dims == std::vector<int>{2, 2}));
        assert((loaded->amplitudes.by_rank[0].data == std::vector<double>{1.0, 2.0, 3.0, 4.0}));
        assert(loaded->amplitudes.sectors.empty());

        std::filesystem::remove(v1_path);
    }

    // C0: a version-2 file truncated PARTWAY THROUGH the sector block (as
    // opposed to a version-1 file, which correctly ends with no sector block
    // at all) must still error. This is the mutation the "no sector block"
    // vs "truncated sector block" distinction exists to catch: a naive
    // "stream ended, so zero sectors" rule would silently accept this too.
    {
        const std::string sector_path = temp_path("planck_ccamp_sector_trunc_src.ccamp");
        const ArbitraryOrderRCCAmplitudes original = make_sample_with_sector();
        CCAmplitudeCheckpointMeta meta{
            .max_rank = 4, .method = "cc4", .basis_name = "sto-3g", .n_occ = 2, .n_virt = 2};
        auto saved = save_cc_amplitudes(sector_path, original, meta);
        assert(saved && "save should succeed");

        const std::string trunc_path = temp_path("planck_ccamp_sector_trunc.ccamp");
        {
            std::ifstream in(sector_path, std::ios::binary);
            std::string bytes((std::istreambuf_iterator<char>(in)), {});
            // Drop only the final few bytes -- deep enough to land inside the
            // sector's tag string or tensor body, never at the exact
            // by_rank/sector boundary (which would just look like a
            // version-1 file and is correctly accepted elsewhere).
            std::ofstream out(trunc_path, std::ios::binary | std::ios::trunc);
            out.write(bytes.data(), static_cast<std::streamsize>(bytes.size() - 5));
        }

        auto r = load_cc_amplitudes(trunc_path);
        assert(!r && "a sector block truncated mid-way must error, not read as zero sectors");

        std::filesystem::remove(sector_path);
        std::filesystem::remove(trunc_path);
    }

    // C0: the sharpest form of the same distinction -- truncated 2 of 4 bytes
    // into `n_sectors` ITSELF (not deeper inside a sector). A version-2 file
    // with NO sectors ends with exactly this 4-byte field; a version-1 file
    // ends without it at all. Those two "the stream ended" cases must be
    // told apart: the first is truncation (error), the second is the normal
    // version-1 shape (fine). A mutation that swallows a failed n_sectors
    // read as "zero sectors" passes every other test in this file (the
    // deep-truncation case above never reaches that code path) and is caught
    // only here.
    {
        const std::string zero_sector_path = temp_path("planck_ccamp_zero_sector_v2.ccamp");
        const ArbitraryOrderRCCAmplitudes original = make_sample(); // no sectors
        CCAmplitudeCheckpointMeta meta{
            .max_rank = 2, .method = "cc2", .basis_name = "sto-3g", .n_occ = 2, .n_virt = 3};
        auto saved = save_cc_amplitudes(zero_sector_path, original, meta);
        assert(saved && "save should succeed");

        const std::string trunc_n_sectors_path = temp_path("planck_ccamp_trunc_n_sectors.ccamp");
        {
            std::ifstream in(zero_sector_path, std::ios::binary);
            std::string bytes((std::istreambuf_iterator<char>(in)), {});
            // The file ends with the 4-byte n_sectors=0 field (no sectors
            // present). Drop the last 2 of those 4 bytes -- the file now
            // ends partway through n_sectors, not before it.
            std::ofstream out(trunc_n_sectors_path, std::ios::binary | std::ios::trunc);
            out.write(bytes.data(), static_cast<std::streamsize>(bytes.size() - 2));
        }

        auto r = load_cc_amplitudes(trunc_n_sectors_path);
        assert(!r && "truncation partway through n_sectors itself must error");

        std::filesystem::remove(zero_sector_path);
        std::filesystem::remove(trunc_n_sectors_path);
    }

    // U0/U1: the sectors-only (UCC) shape round-trips correctly -- empty
    // by_rank, real data entirely in sectors, non-zero UHF occupation
    // counts. This is the exact case a writer-only fix was tried and found
    // broken on: save_cc_amplitudes succeeding is not evidence
    // load_cc_amplitudes can read what it wrote back correctly, since the
    // reader's by_rank loop trip count used to be silently coupled to
    // max_rank rather than to an independent field.
    {
        const std::string ucc_path = temp_path("planck_ccamp_ucc_roundtrip.ccamp");
        const ArbitraryOrderRCCAmplitudes original = make_ucc_sample();
        CCAmplitudeCheckpointMeta meta{
            .max_rank = 2,
            .method = "ucc2",
            .basis_name = "sto-3g",
            .n_occ = 0,
            .n_virt = 0,
            .reference_type = CCReferenceType::UHF,
            .n_occ_alpha = 3,
            .n_occ_beta = 2,
            .n_virt_alpha = 4,
            .n_virt_beta = 5,
        };

        auto saved = save_cc_amplitudes(ucc_path, original, meta);
        assert(saved && "save with an empty by_rank and populated sectors should succeed");

        auto loaded = load_cc_amplitudes(ucc_path);
        assert(loaded && "load of a sectors-only (UCC-shaped) file should succeed");

        assert(loaded->meta.max_rank == 2);
        assert(loaded->meta.method == "ucc2");
        assert(loaded->meta.reference_type == CCReferenceType::UHF);
        assert(loaded->meta.n_occ_alpha == 3);
        assert(loaded->meta.n_occ_beta == 2);
        assert(loaded->meta.n_virt_alpha == 4);
        assert(loaded->meta.n_virt_beta == 5);

        // The actual defect: by_rank must be genuinely empty, not
        // reconstructed or padded from max_rank.
        assert(loaded->amplitudes.by_rank.empty());

        assert(loaded->amplitudes.sectors.size() == 2);
        for (std::size_t s = 0; s < original.sectors.size(); ++s)
        {
            assert(loaded->amplitudes.sectors[s].first == original.sectors[s].first);
            assert(loaded->amplitudes.sectors[s].second.dims == original.sectors[s].second.dims);
            assert(loaded->amplitudes.sectors[s].second.data == original.sectors[s].second.data);
        }
        // The two sectors must be genuinely distinct in the round-tripped
        // data -- catches an aliasing bug where both keys end up pointing
        // at the same underlying tensor.
        assert(loaded->amplitudes.sectors[0].second.data != loaded->amplitudes.sectors[1].second.data);

        std::filesystem::remove(ucc_path);
    }

    // U0/U1: a hand-built version-2 file (no n_by_rank, no UHF counts in the
    // byte stream -- the exact shape every sidecar written before this
    // scope's fix has on disk) must still load, with n_by_rank defaulted to
    // THAT FILE'S OWN max_rank (not 0 -- 0 would silently discard the
    // by_rank data every existing version-2 sidecar actually carries) and
    // the four UHF counts defaulted to 0.
    {
        const std::string v2_path = temp_path("planck_ccamp_v2_compat.ccamp");
        {
            std::ofstream out(v2_path, std::ios::binary | std::ios::trunc);
            const char magic[8] = {'P', 'L', 'N', 'K', 'C', 'C', 'A', '\0'};
            out.write(magic, 8);
            const std::uint32_t version = 2;
            out.write(reinterpret_cast<const char *>(&version), 4);
            const std::int32_t max_rank = 1;
            out.write(reinterpret_cast<const char *>(&max_rank), 4);
            const std::uint32_t method_len = 3;
            out.write(reinterpret_cast<const char *>(&method_len), 4);
            out.write("cc1", 3);
            const std::uint32_t basis_len = 6;
            out.write(reinterpret_cast<const char *>(&basis_len), 4);
            out.write("sto-3g", 6);
            const std::uint64_t n_occ = 2, n_virt = 2;
            out.write(reinterpret_cast<const char *>(&n_occ), 8);
            out.write(reinterpret_cast<const char *>(&n_virt), 8);
            const std::uint8_t reference_type_u8 = 0; // RHF
            out.write(reinterpret_cast<const char *>(&reference_type_u8), 1);
            // rank 1: order=2, dims=[2,2], count=4, data
            const std::int32_t order = 2;
            out.write(reinterpret_cast<const char *>(&order), 4);
            const std::int32_t d0 = 2, d1 = 2;
            out.write(reinterpret_cast<const char *>(&d0), 4);
            out.write(reinterpret_cast<const char *>(&d1), 4);
            const std::uint64_t count = 4;
            out.write(reinterpret_cast<const char *>(&count), 8);
            const double data[4] = {5.0, 6.0, 7.0, 8.0};
            out.write(reinterpret_cast<const char *>(data), sizeof(data));
            const std::int32_t n_sectors = 0;
            out.write(reinterpret_cast<const char *>(&n_sectors), 4);
            // File ends here -- no n_by_rank, no UHF counts.
        }

        auto loaded = load_cc_amplitudes(v2_path);
        assert(loaded && "a version-2 file must still load");
        assert(loaded->meta.method == "cc1");
        assert(loaded->meta.reference_type == CCReferenceType::RHF);
        // The default under test: n_by_rank must come from max_rank (1),
        // not from a hardcoded 0.
        assert(loaded->amplitudes.by_rank.size() == 1);
        assert((loaded->amplitudes.by_rank[0].dims == std::vector<int>{2, 2}));
        assert((loaded->amplitudes.by_rank[0].data == std::vector<double>{5.0, 6.0, 7.0, 8.0}));
        assert(loaded->amplitudes.sectors.empty());
        assert(loaded->meta.n_occ_alpha == 0);
        assert(loaded->meta.n_occ_beta == 0);
        assert(loaded->meta.n_virt_alpha == 0);
        assert(loaded->meta.n_virt_beta == 0);

        std::filesystem::remove(v2_path);
    }

    std::cout << "cc_amplitude_checkpoint: all round-trip and error cases passed\n";
    return 0;
}
