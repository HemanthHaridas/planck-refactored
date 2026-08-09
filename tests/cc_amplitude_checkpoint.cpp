// Round-trip gate for the generated-CC amplitude sidecar (.ccamp).
// Builds a small 2-rank ArbitraryOrderRCCAmplitudes, saves it, loads it back,
// and asserts bytewise-equal dims/data plus metadata. Also checks that a bad
// magic and a truncated file are rejected (errored, not crashed).

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
    std::cout << "cc_amplitude_checkpoint: all round-trip and error cases passed\n";
    return 0;
}
