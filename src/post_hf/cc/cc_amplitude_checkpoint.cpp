#include "post_hf/cc/cc_amplitude_checkpoint.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <format>

namespace HartreeFock::Correlation::CC
{
    namespace
    {
        constexpr char CCAMP_MAGIC[8] = {'P', 'L', 'N', 'K', 'C', 'C', 'A', '\0'};
        constexpr std::uint32_t CCAMP_VERSION = 1;
        constexpr std::uint32_t MAX_TAG_BYTES = 4096;
        // A single rank tensor is o^r v^r doubles; cap the per-rank element count
        // so a corrupt `count` can't drive an unbounded allocation. 2^40 doubles
        // (8 TiB) is far past any real CC tensor and still rejects garbage.
        constexpr std::uint64_t MAX_TENSOR_ELEMENTS = (std::uint64_t{1} << 40);

        std::expected<void, std::string> read_exact(
            std::istream &in, char *data, std::size_t bytes, std::string_view label)
        {
            in.read(data, static_cast<std::streamsize>(bytes));
            if (!in)
                return std::unexpected(std::format(".ccamp truncated while reading {}", label));
            return {};
        }

        void write_string(std::ostream &out, const std::string &s)
        {
            const std::uint32_t len = static_cast<std::uint32_t>(s.size());
            out.write(reinterpret_cast<const char *>(&len), 4);
            out.write(s.data(), len);
        }

        std::expected<std::string, std::string> read_string(std::istream &in, std::string_view label)
        {
            std::uint32_t len = 0;
            if (auto r = read_exact(in, reinterpret_cast<char *>(&len), 4, label); !r)
                return std::unexpected(r.error());
            if (len > MAX_TAG_BYTES)
                return std::unexpected(std::format(".ccamp {} length {} exceeds limit {}", label, len, MAX_TAG_BYTES));
            std::string s(len, '\0');
            if (len > 0)
                if (auto r = read_exact(in, s.data(), len, label); !r)
                    return std::unexpected(r.error());
            return s;
        }
    } // namespace

    std::expected<void, std::string> save_cc_amplitudes(
        const std::string &path,
        const ArbitraryOrderRCCAmplitudes &amplitudes,
        const CCAmplitudeCheckpointMeta &meta)
    {
        const int max_rank = static_cast<int>(amplitudes.by_rank.size());
        if (max_rank < 1)
            return std::unexpected("save_cc_amplitudes: amplitudes are empty.");

        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out)
            return std::unexpected(std::format("save_cc_amplitudes: cannot open '{}' for writing.", path));

        out.write(CCAMP_MAGIC, 8);
        out.write(reinterpret_cast<const char *>(&CCAMP_VERSION), 4);
        const std::int32_t max_rank_i32 = max_rank;
        out.write(reinterpret_cast<const char *>(&max_rank_i32), 4);
        write_string(out, meta.method);
        write_string(out, meta.basis_name);
        out.write(reinterpret_cast<const char *>(&meta.n_occ), 8);
        out.write(reinterpret_cast<const char *>(&meta.n_virt), 8);

        for (const TensorND &t : amplitudes.by_rank)
        {
            const std::int32_t order = static_cast<std::int32_t>(t.dims.size());
            out.write(reinterpret_cast<const char *>(&order), 4);
            for (int d : t.dims)
            {
                const std::int32_t di = d;
                out.write(reinterpret_cast<const char *>(&di), 4);
            }
            const std::uint64_t count = t.data.size();
            out.write(reinterpret_cast<const char *>(&count), 8);
            out.write(reinterpret_cast<const char *>(t.data.data()),
                      static_cast<std::streamsize>(count * sizeof(double)));
        }

        if (!out)
            return std::unexpected(std::format("save_cc_amplitudes: write to '{}' failed.", path));
        return {};
    }

    std::expected<CCAmplitudeCheckpoint, std::string> load_cc_amplitudes(
        const std::string &path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            return std::unexpected(std::format("load_cc_amplitudes: cannot open '{}'.", path));

        char magic[8];
        if (auto r = read_exact(in, magic, 8, "magic"); !r)
            return std::unexpected(r.error());
        if (std::memcmp(magic, CCAMP_MAGIC, 8) != 0)
            return std::unexpected("load_cc_amplitudes: bad magic (not a .ccamp file).");

        std::uint32_t version = 0;
        if (auto r = read_exact(in, reinterpret_cast<char *>(&version), 4, "version"); !r)
            return std::unexpected(r.error());
        if (version != CCAMP_VERSION)
            return std::unexpected(std::format(
                "load_cc_amplitudes: version {} unsupported (expected {}).", version, CCAMP_VERSION));

        CCAmplitudeCheckpoint chk;
        std::int32_t max_rank = 0;
        if (auto r = read_exact(in, reinterpret_cast<char *>(&max_rank), 4, "max_rank"); !r)
            return std::unexpected(r.error());
        if (max_rank < 1)
            return std::unexpected(std::format("load_cc_amplitudes: max_rank {} is invalid.", max_rank));
        chk.meta.max_rank = max_rank;

        auto method = read_string(in, "method tag");
        if (!method)
            return std::unexpected(method.error());
        chk.meta.method = std::move(*method);

        auto basis = read_string(in, "basis name");
        if (!basis)
            return std::unexpected(basis.error());
        chk.meta.basis_name = std::move(*basis);

        if (auto r = read_exact(in, reinterpret_cast<char *>(&chk.meta.n_occ), 8, "n_occ"); !r)
            return std::unexpected(r.error());
        if (auto r = read_exact(in, reinterpret_cast<char *>(&chk.meta.n_virt), 8, "n_virt"); !r)
            return std::unexpected(r.error());

        chk.amplitudes.by_rank.reserve(static_cast<std::size_t>(max_rank));
        for (int rank = 1; rank <= max_rank; ++rank)
        {
            std::int32_t order = 0;
            if (auto r = read_exact(in, reinterpret_cast<char *>(&order), 4,
                                    std::format("rank-{} order", rank)); !r)
                return std::unexpected(r.error());
            if (order < 0 || order > 2 * max_rank)
                return std::unexpected(std::format(
                    "load_cc_amplitudes: rank-{} order {} is invalid.", rank, order));

            std::vector<int> dims(static_cast<std::size_t>(order));
            std::uint64_t expected = order == 0 ? 0 : 1;
            for (std::int32_t d = 0; d < order; ++d)
            {
                std::int32_t di = 0;
                if (auto r = read_exact(in, reinterpret_cast<char *>(&di), 4,
                                        std::format("rank-{} dim", rank)); !r)
                    return std::unexpected(r.error());
                if (di < 0)
                    return std::unexpected(std::format(
                        "load_cc_amplitudes: rank-{} has negative dim {}.", rank, di));
                dims[static_cast<std::size_t>(d)] = di;
                if (di != 0 && expected > MAX_TENSOR_ELEMENTS / static_cast<std::uint64_t>(di))
                    return std::unexpected(std::format(
                        "load_cc_amplitudes: rank-{} element count overflows.", rank));
                expected *= static_cast<std::uint64_t>(di);
            }

            std::uint64_t count = 0;
            if (auto r = read_exact(in, reinterpret_cast<char *>(&count), 8,
                                    std::format("rank-{} count", rank)); !r)
                return std::unexpected(r.error());
            if (count != expected)
                return std::unexpected(std::format(
                    "load_cc_amplitudes: rank-{} count {} disagrees with dims product {}.",
                    rank, count, expected));
            if (count > MAX_TENSOR_ELEMENTS)
                return std::unexpected(std::format(
                    "load_cc_amplitudes: rank-{} count {} exceeds limit.", rank, count));

            TensorND tensor(dims, 0.0);
            if (count > 0)
                if (auto r = read_exact(in, reinterpret_cast<char *>(tensor.data.data()),
                                        static_cast<std::size_t>(count) * sizeof(double),
                                        std::format("rank-{} data", rank)); !r)
                    return std::unexpected(r.error());
            chk.amplitudes.by_rank.push_back(std::move(tensor));
        }

        return chk;
    }
} // namespace HartreeFock::Correlation::CC
