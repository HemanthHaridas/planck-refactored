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
        // C0: version 2 appends a sector block after `by_rank` (see
        // write_sectors/read_sectors below) plus a one-byte reference-type field
        // in the header, folded in now per C4 so a future UCC sidecar does not
        // need a version 3 -- "one spare byte in the header beats a second
        // version bump." Version 1 sidecars (no sectors, no reference-type byte)
        // still load: read_sectors on a truncated-at-EOF stream after `by_rank`
        // is treated as "no sectors", not an error.
        constexpr std::uint32_t CCAMP_VERSION = 2;
        constexpr std::uint32_t MAX_TAG_BYTES = 4096;
        constexpr std::uint32_t MAX_SECTORS = 1u << 20; // generous; real counts are floor(n/2)
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

        // C0: the tensor body (order + dims + count + data) is byte-identical
        // between a `by_rank` entry and a `sectors` entry -- one codec, two
        // callers, so the two paths cannot silently drift apart.
        void write_tensor(std::ostream &out, const TensorND &t)
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

        std::expected<TensorND, std::string> read_tensor(
            std::istream &in, std::string_view label, int max_rank)
        {
            std::int32_t order = 0;
            if (auto r = read_exact(in, reinterpret_cast<char *>(&order), 4,
                                    std::format("{} order", label)); !r)
                return std::unexpected(r.error());
            if (order < 0 || order > 2 * max_rank)
                return std::unexpected(std::format(
                    "load_cc_amplitudes: {} order {} is invalid.", label, order));

            std::vector<int> dims(static_cast<std::size_t>(order));
            std::uint64_t expected = order == 0 ? 0 : 1;
            for (std::int32_t d = 0; d < order; ++d)
            {
                std::int32_t di = 0;
                if (auto r = read_exact(in, reinterpret_cast<char *>(&di), 4,
                                        std::format("{} dim", label)); !r)
                    return std::unexpected(r.error());
                if (di < 0)
                    return std::unexpected(std::format(
                        "load_cc_amplitudes: {} has negative dim {}.", label, di));
                dims[static_cast<std::size_t>(d)] = di;
                if (di != 0 && expected > MAX_TENSOR_ELEMENTS / static_cast<std::uint64_t>(di))
                    return std::unexpected(std::format(
                        "load_cc_amplitudes: {} element count overflows.", label));
                expected *= static_cast<std::uint64_t>(di);
            }

            std::uint64_t count = 0;
            if (auto r = read_exact(in, reinterpret_cast<char *>(&count), 8,
                                    std::format("{} count", label)); !r)
                return std::unexpected(r.error());
            if (count != expected)
                return std::unexpected(std::format(
                    "load_cc_amplitudes: {} count {} disagrees with dims product {}.",
                    label, count, expected));
            if (count > MAX_TENSOR_ELEMENTS)
                return std::unexpected(std::format(
                    "load_cc_amplitudes: {} count {} exceeds limit.", label, count));

            TensorND tensor(dims, 0.0);
            if (count > 0)
                if (auto r = read_exact(in, reinterpret_cast<char *>(tensor.data.data()),
                                        static_cast<std::size_t>(count) * sizeof(double),
                                        std::format("{} data", label)); !r)
                    return std::unexpected(r.error());
            return tensor;
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

        if (amplitudes.sectors.size() > MAX_SECTORS)
            return std::unexpected(std::format(
                "save_cc_amplitudes: {} sectors exceeds limit {}.",
                amplitudes.sectors.size(), MAX_SECTORS));

        out.write(CCAMP_MAGIC, 8);
        out.write(reinterpret_cast<const char *>(&CCAMP_VERSION), 4);
        const std::int32_t max_rank_i32 = max_rank;
        out.write(reinterpret_cast<const char *>(&max_rank_i32), 4);
        write_string(out, meta.method);
        write_string(out, meta.basis_name);
        out.write(reinterpret_cast<const char *>(&meta.n_occ), 8);
        out.write(reinterpret_cast<const char *>(&meta.n_virt), 8);
        const std::uint8_t reference_type_u8 = static_cast<std::uint8_t>(meta.reference_type);
        out.write(reinterpret_cast<const char *>(&reference_type_u8), 1);

        for (const TensorND &t : amplitudes.by_rank)
            write_tensor(out, t);

        // C0: the sectors amplitudes.by_rank silently dropped. Same tensor
        // codec as by_rank, keyed by (excitation_rank, tag) so the loader can
        // route each block back to sector_tensor(rank, tag).
        const std::int32_t n_sectors = static_cast<std::int32_t>(amplitudes.sectors.size());
        out.write(reinterpret_cast<const char *>(&n_sectors), 4);
        for (const auto &[key, tensor] : amplitudes.sectors)
        {
            const auto &[excitation_rank, tag] = key;
            const std::int32_t rank_i32 = excitation_rank;
            out.write(reinterpret_cast<const char *>(&rank_i32), 4);
            write_string(out, tag);
            write_tensor(out, tensor);
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
        if (version != 1 && version != CCAMP_VERSION)
            return std::unexpected(std::format(
                "load_cc_amplitudes: version {} unsupported (expected 1 or {}).", version, CCAMP_VERSION));

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

        // C4: version 1 never wrote this byte -- it only ever wrote RHF
        // amplitudes, so that is the correct default rather than an error.
        if (version >= 2)
        {
            std::uint8_t reference_type_u8 = 0;
            if (auto r = read_exact(in, reinterpret_cast<char *>(&reference_type_u8), 1,
                                    "reference_type"); !r)
                return std::unexpected(r.error());
            if (reference_type_u8 > static_cast<std::uint8_t>(CCReferenceType::UHF))
                return std::unexpected(std::format(
                    "load_cc_amplitudes: reference_type {} is invalid.", reference_type_u8));
            chk.meta.reference_type = static_cast<CCReferenceType>(reference_type_u8);
        }
        else
        {
            chk.meta.reference_type = CCReferenceType::RHF;
        }

        chk.amplitudes.by_rank.reserve(static_cast<std::size_t>(max_rank));
        for (int rank = 1; rank <= max_rank; ++rank)
        {
            auto tensor = read_tensor(in, std::format("rank-{}", rank), max_rank);
            if (!tensor)
                return std::unexpected(tensor.error());
            chk.amplitudes.by_rank.push_back(std::move(*tensor));
        }

        // C0: the higher Sz sectors, present from version 2 onward. A
        // version-1 file (or a version-2 file with no sectors) ends here --
        // reading `n_sectors` then hits EOF cleanly, which is "zero sectors",
        // not an error. `peek()` distinguishes a real end-of-stream from a
        // genuine truncation partway through the field: if the stream is NOT
        // at EOF, whatever bytes are there must parse as a valid n_sectors,
        // so a truncated version-2 sector block still errors as truncation
        // rather than silently reading as zero sectors.
        if (in.peek() != std::char_traits<char>::eof())
        {
            std::int32_t n_sectors = 0;
            if (auto r = read_exact(in, reinterpret_cast<char *>(&n_sectors), 4, "n_sectors"); !r)
                return std::unexpected(r.error());
            if (n_sectors < 0 || static_cast<std::uint32_t>(n_sectors) > MAX_SECTORS)
                return std::unexpected(std::format(
                    "load_cc_amplitudes: n_sectors {} is invalid.", n_sectors));

            chk.amplitudes.sectors.reserve(static_cast<std::size_t>(n_sectors));
            for (int s = 0; s < n_sectors; ++s)
            {
                std::int32_t excitation_rank = 0;
                if (auto r = read_exact(in, reinterpret_cast<char *>(&excitation_rank), 4,
                                        std::format("sector {} rank", s)); !r)
                    return std::unexpected(r.error());

                auto tag = read_string(in, std::format("sector {} tag", s));
                if (!tag)
                    return std::unexpected(tag.error());

                auto tensor = read_tensor(
                    in, std::format("sector ({},{})", excitation_rank, *tag), max_rank);
                if (!tensor)
                    return std::unexpected(tensor.error());

                chk.amplitudes.sectors.push_back(
                    {{excitation_rank, std::move(*tag)}, std::move(*tensor)});
            }
        }

        return chk;
    }
} // namespace HartreeFock::Correlation::CC
