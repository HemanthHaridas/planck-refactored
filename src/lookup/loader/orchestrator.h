#ifndef HF_LOOKUP_LOADER_ORCHESTRATOR_H
#define HF_LOOKUP_LOADER_ORCHESTRATOR_H

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "lookup/loader/license.h"

namespace HartreeFock
{
    namespace Loader
    {
        // Runs the full startup sequence: read the local license file,
        // verify its signature, fetch the matching encrypted library blob,
        // decrypt it, and dlopen it from memory. On success, boys()/
        // boys_vec() (declared in lookup/boys.h) become callable — they
        // dispatch to the function pointers this resolves.
        //
        // Must be called exactly once, before any code calls boys()/
        // boys_vec(). Not thread-safe by construction (startup-only).
        //
        // license_path: local path to the signed license.json.
        // blob_url_template: a URL containing the literal substring
        //   "{version}", replaced with the license's library_version
        //   before fetching (e.g. "https://.../boys-v{version}-macos-arm64.enc").
        //
        // No fallback: failure here is unconditional and terminal for the
        // calling binary — by design, not an oversight.
        //
        // Implemented as fetch_boys_library_blob() +
        // initialize_boys_library_from_blob() below — use those two directly
        // instead of this convenience wrapper when the network fetch needs to
        // happen somewhere other than the calling process (e.g. planck-mpi:
        // one MPI rank fetches once and MPI_Bcasts the encrypted bytes to the
        // rest, so an N-rank job costs one blob-host request, not N — every
        // rank still calls load_and_verify_license() and
        // initialize_boys_library_from_blob() itself, since decrypt/dlopen
        // happen in that process's own address space and cannot be shared
        // across processes the way the fetched bytes can).
        std::expected<void, std::string> initialize_boys_library(
            const std::string &license_path,
            const std::string &blob_url_template);

        // Substitutes license.library_version into blob_url_template
        // (replacing the literal "{version}") and returns the blob's bytes.
        // The network half of initialize_boys_library() — the one step that
        // should run once per job, not once per MPI rank.
        //
        // Checks a local cache first (see cache.h): a hit — an entry no
        // older than the compile-time max age — is returned without
        // touching the network at all. license.expires_at is NOT checked
        // here, because every caller already obtained `license` via
        // load_and_verify_license(), which rejects an expired license
        // before this function is ever reached — there is no path by which
        // an expired license's cache entry gets served.
        //
        // force_network skips the cache lookup unconditionally (but still
        // writes a fresh cache entry on success) — used by the retry path
        // in initialize_boys_library() after a cached blob fails to
        // decrypt, so the retry can't just re-read the same corrupt entry.
        std::expected<std::vector<std::uint8_t>, std::string> fetch_boys_library_blob(
            const License &license,
            const std::string &blob_url_template,
            bool force_network = false);

        // Deletes the cache entry fetch_boys_library_blob() would have read
        // for this (license, blob_url_template) pair. For callers that
        // split fetch from decrypt across process boundaries (planck-mpi:
        // rank 0 fetches, every rank decrypts independently) and so can't
        // use initialize_boys_library()'s own built-in retry — call this
        // then fetch_boys_library_blob(..., force_network=true) to retry
        // past a cache entry that failed to decrypt.
        void evict_boys_library_cache_entry(
            const License &license,
            const std::string &blob_url_template);

        // Decrypts (but does not load/dlopen) a fetched blob — a pure
        // validity check for callers that need to know "would this blob
        // decrypt" without the call-once, process-global side effects of
        // initialize_boys_library_from_blob(). planck-mpi's rank 0 uses
        // this to validate a fetched/cached blob before broadcasting it,
        // so a stale cache entry is caught and retried once on rank 0
        // alone instead of failing identically on every rank after the
        // broadcast.
        std::expected<void, std::string> boys_library_blob_decrypts(
            const License &license,
            const std::vector<std::uint8_t> &blob);

        // Decrypts an already-fetched blob (license.decryption_key, AES-256-GCM)
        // and dlopens the result, resolving boys()/boys_vec() the same way
        // initialize_boys_library() does. The per-process half — must be
        // called in every process that will call boys()/boys_vec(), since
        // dlopen's result is only valid in the calling process's own address
        // space. Same call-once, not-thread-safe, no-fallback contract as
        // initialize_boys_library().
        std::expected<void, std::string> initialize_boys_library_from_blob(
            const License &license,
            const std::vector<std::uint8_t> &blob);

        // Compile-time OS+architecture tag matching planck-boys-core's release
        // asset naming (planck_boys-<platform>.enc, e.g. "macos-arm64",
        // "linux-x86_64") — this identifies which binary is running, a fact
        // fixed at compile time, not something to detect at runtime. Callers
        // build the full blob_url_template for initialize_boys_library() by
        // substituting this in, e.g.:
        //   "https://.../{version}/planck_boys-" + get_platform_string() + ".enc"
        //
        // #error's at compile time on an OS/arch combination with no
        // corresponding release asset, rather than silently resolving to the
        // wrong platform's blob — same policy as memory_load.cpp's own
        // #else #error for unsupported platforms.
        constexpr const char *get_platform_string()
        {
#if defined(__APPLE__) && defined(__aarch64__)
            return "macos-arm64";
#elif defined(__linux__) && defined(__x86_64__)
            return "linux-x86_64";
#else
#error "get_platform_string: no planck-boys-core release asset for this OS/architecture combination — see .github/workflows/release.yml in planck-boys-core for the supported matrix"
#endif
        }

    } // namespace Loader
} // namespace HartreeFock

#endif // !HF_LOOKUP_LOADER_ORCHESTRATOR_H
