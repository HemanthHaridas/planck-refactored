#include "lookup/loader/orchestrator.h"

#include <cassert>
#include <chrono>

#include "lookup/boys.h"
#include "lookup/loader/cache.h"
#include "lookup/loader/cache_path.h"
#include "lookup/loader/decrypt.h"
#include "lookup/loader/fetch.h"
#include "lookup/loader/license.h"
#include "lookup/loader/memory_load.h"

namespace HartreeFock
{
    namespace Loader
    {
        namespace
        {
            using BoysFn = double (*)(int, double);
            using BoysVecFn = void (*)(double, double *, std::size_t);
            using BoysBatchFn = void (*)(const double *, std::size_t, int, double *);

            // Populated by initialize_boys_library(); read (never written)
            // by every boys()/boys_vec() call thereafter.
            BoysFn g_boys_fn = nullptr;
            BoysVecFn g_boys_vec_fn = nullptr;
            BoysBatchFn g_boys_batch_fn = nullptr;
            void *g_library_handle = nullptr;

            std::string substitute_version(const std::string &url_template, const std::string &version)
            {
                std::string out = url_template;
                const std::string placeholder = "{version}";
                auto pos = out.find(placeholder);
                if (pos != std::string::npos)
                    out.replace(pos, placeholder.size(), version);
                return out;
            }

            // The part of the (already version-substituted) URL after the
            // last '/' — e.g. "planck_boys-macos-arm64.enc". Used as the
            // cache entry's filename; it already encodes platform, so the
            // cache layout needs no separate platform segment.
            std::string url_basename(const std::string &url)
            {
                const auto slash = url.find_last_of('/');
                return slash == std::string::npos ? url : url.substr(slash + 1);
            }

            // How long a cached blob is trusted without re-fetching, even
            // under a still-valid license. Not tied to license expiry alone
            // (which could be years away) — a periodic re-fetch keeps the
            // door open for a future revocation check to piggyback on it,
            // per the design note in the license-gate handoff.
            constexpr std::chrono::seconds kBoysCacheMaxAge = std::chrono::hours(24 * 7);

            // Shared by initialize_boys_library_from_blob(), the
            // decrypt-retry path in initialize_boys_library(), and the
            // public boys_library_blob_decrypts() validity check — all
            // three need to distinguish "decrypt failed" (worth retrying
            // past a stale cache entry) from every other failure (a
            // load/symbol problem the same plaintext would hit again
            // regardless of where the ciphertext came from) without
            // resorting to string matching on the error message.
            std::expected<std::vector<std::uint8_t>, std::string> decrypt_step(
                const License &license,
                const std::vector<std::uint8_t> &blob)
            {
                return decrypt_library_blob(blob, license.decryption_key);
            }

            std::expected<void, std::string> load_and_resolve(const std::vector<std::uint8_t> &plaintext)
            {
                auto handle = load_library_from_memory(plaintext);
                if (!handle)
                    return std::unexpected("Failed to load library: " + handle.error());

                auto boys_sym = resolve_symbol(*handle, "planck_boys");
                if (!boys_sym)
                {
                    unload_library(*handle);
                    return std::unexpected("Failed to resolve planck_boys: " + boys_sym.error());
                }

                auto boys_vec_sym = resolve_symbol(*handle, "planck_boys_vec");
                if (!boys_vec_sym)
                {
                    unload_library(*handle);
                    return std::unexpected("Failed to resolve planck_boys_vec: " + boys_vec_sym.error());
                }

                // Core v1.2.0+. A v1.1.0 library (license) fails here, loudly:
                // the ERI kernels call boys_batch on every block.
                auto boys_batch_sym = resolve_symbol(*handle, "planck_boys_batch");
                if (!boys_batch_sym)
                {
                    unload_library(*handle);
                    return std::unexpected("Failed to resolve planck_boys_batch (needs Boys library v1.2.0 or "
                                           "later; install a v1.2.0 license): " +
                                           boys_batch_sym.error());
                }

                g_library_handle = *handle;
                g_boys_fn = reinterpret_cast<BoysFn>(*boys_sym);
                g_boys_vec_fn = reinterpret_cast<BoysVecFn>(*boys_vec_sym);
                g_boys_batch_fn = reinterpret_cast<BoysBatchFn>(*boys_batch_sym);
                return {};
            }
        }

        std::expected<std::vector<std::uint8_t>, std::string> fetch_boys_library_blob(
            const License &license,
            const std::string &blob_url_template,
            bool force_network)
        {
            const std::string url = substitute_version(blob_url_template, license.library_version);
            const std::string asset = url_basename(url);
            const std::string cache_dir = get_boys_cache_dir();

            if (!force_network)
            {
                if (auto cached = read_cached_blob(cache_dir, license.library_version, asset, kBoysCacheMaxAge))
                    return std::move(*cached);
            }

            auto blob = fetch_url(url);
            if (!blob)
                return std::unexpected("Failed to fetch library blob: " + blob.error());

            write_cached_blob(cache_dir, license.library_version, asset, *blob);
            return *blob;
        }

        void evict_boys_library_cache_entry(
            const License &license,
            const std::string &blob_url_template)
        {
            const std::string url = substitute_version(blob_url_template, license.library_version);
            evict_cached_blob(get_boys_cache_dir(), license.library_version, url_basename(url));
        }

        std::expected<void, std::string> boys_library_blob_decrypts(
            const License &license,
            const std::vector<std::uint8_t> &blob)
        {
            auto plaintext = decrypt_step(license, blob);
            if (!plaintext)
                return std::unexpected(plaintext.error());
            return {};
        }

        std::expected<void, std::string> initialize_boys_library_from_blob(
            const License &license,
            const std::vector<std::uint8_t> &blob)
        {
            if (g_boys_fn != nullptr)
                return std::unexpected("initialize_boys_library_from_blob() called more than once");

            auto plaintext = decrypt_step(license, blob);
            if (!plaintext)
                return std::unexpected("Failed to decrypt library blob: " + plaintext.error());

            return load_and_resolve(*plaintext);
        }

        std::expected<void, std::string> initialize_boys_library(
            const std::string &license_path,
            const std::string &blob_url_template)
        {
            if (g_boys_fn != nullptr)
                return std::unexpected("initialize_boys_library() called more than once");

            auto license = load_and_verify_license(license_path);
            if (!license)
                return std::unexpected("License check failed: " + license.error());

            auto blob = fetch_boys_library_blob(*license, blob_url_template);
            if (!blob)
                return std::unexpected(blob.error());

            auto plaintext = decrypt_step(*license, *blob);
            if (!plaintext)
            {
                // A cached blob that fails to decrypt is corrupt (bit rot,
                // truncated write, ...) rather than wrong — the GCM tag
                // already rejects tampering, and a network fetch that
                // decrypted fine last time doesn't spontaneously become the
                // wrong key. Evict the entry and retry exactly once,
                // forcing past the cache. A load/symbol failure below is
                // NOT retried here: it means the plaintext itself is wrong,
                // which the same key against the same host would reproduce
                // regardless of where the ciphertext came from.
                evict_boys_library_cache_entry(*license, blob_url_template);

                auto retried_blob = fetch_boys_library_blob(*license, blob_url_template, /*force_network=*/true);
                if (!retried_blob)
                    return std::unexpected(retried_blob.error());

                plaintext = decrypt_step(*license, *retried_blob);
                if (!plaintext)
                    return std::unexpected("Failed to decrypt library blob: " + plaintext.error());
            }

            return load_and_resolve(*plaintext);
        }

    } // namespace Loader

    namespace Lookup
    {
        // Definitions matching the declarations in lookup/boys.h. Dispatch
        // to the function pointers resolved by initialize_boys_library().
        // Calling either before initialization is a programming error in
        // the caller (driver startup must call initialize_boys_library()
        // before any calculation runs) — asserts rather than silently
        // returning garbage, since a silent wrong-Fock-matrix failure mode
        // is far worse than a hard crash naming the actual bug.

        double boys(int n, double x) noexcept
        {
            assert(Loader::g_boys_fn && "boys() called before initialize_boys_library()");
            return Loader::g_boys_fn(n, x);
        }

        void boys_vec(double x, std::span<double> out) noexcept
        {
            assert(Loader::g_boys_vec_fn && "boys_vec() called before initialize_boys_library()");
            Loader::g_boys_vec_fn(x, out.data(), out.size());
        }

        void boys_batch(std::span<const double> x, int nmax, std::span<double> out) noexcept
        {
            assert(Loader::g_boys_batch_fn && "boys_batch() called before initialize_boys_library()");
            assert(nmax >= 0 && out.size() >= x.size() * (static_cast<std::size_t>(nmax) + 1));
            Loader::g_boys_batch_fn(x.data(), x.size(), nmax, out.data());
        }

    } // namespace Lookup
} // namespace HartreeFock
