#ifndef HF_LOOKUP_LOADER_CACHE_H
#define HF_LOOKUP_LOADER_CACHE_H

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace HartreeFock
{
    namespace Loader
    {
        // Reads a cache entry for (library_version, platform_asset) under
        // cache_dir, if one exists and is no older than max_age. Returns
        // std::nullopt on a cache miss, a stale entry, or any read error —
        // every one of those is just "go fetch instead", not a hard error,
        // so this never returns std::expected. This does NOT check license
        // expiry; that's the caller's job (load_and_verify_license already
        // does it), since this function has no License to check.
        //
        // platform_asset is the basename of the blob URL (e.g.
        // "planck_boys-macos-arm64.enc") — it already encodes platform, so
        // no separate platform parameter is needed.
        std::optional<std::vector<std::uint8_t>> read_cached_blob(
            const std::string &cache_dir,
            const std::string &library_version,
            const std::string &platform_asset,
            std::chrono::seconds max_age);

        // Writes bytes to the cache entry for (library_version,
        // platform_asset), atomically (write to a pid-suffixed temp file,
        // then rename over the final path — survives concurrent writers,
        // and the rename refreshes the entry's mtime for read_cached_blob's
        // age check). Failure (read-only directory, disk full, ...) is
        // silently ignored: caching is a convenience, never a reason to
        // fail a run that already has the bytes in hand.
        void write_cached_blob(
            const std::string &cache_dir,
            const std::string &library_version,
            const std::string &platform_asset,
            const std::vector<std::uint8_t> &bytes);

        // Deletes a cache entry, e.g. after it fails to decrypt (corrupted
        // or truncated on disk). Best-effort; a failure to delete is not
        // reported, since the caller's next step is to overwrite the same
        // path via write_cached_blob() regardless.
        void evict_cached_blob(
            const std::string &cache_dir,
            const std::string &library_version,
            const std::string &platform_asset);

    } // namespace Loader
} // namespace HartreeFock

#endif // !HF_LOOKUP_LOADER_CACHE_H
