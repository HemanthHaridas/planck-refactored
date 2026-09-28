#include "lookup/loader/orchestrator.h"

#include <cassert>

#include "lookup/boys.h"
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

            // Populated by initialize_boys_library(); read (never written)
            // by every boys()/boys_vec() call thereafter.
            BoysFn g_boys_fn = nullptr;
            BoysVecFn g_boys_vec_fn = nullptr;
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
        }

        std::expected<std::vector<std::uint8_t>, std::string> fetch_boys_library_blob(
            const License &license,
            const std::string &blob_url_template)
        {
            const std::string url = substitute_version(blob_url_template, license.library_version);
            auto blob = fetch_url(url);
            if (!blob)
                return std::unexpected("Failed to fetch library blob: " + blob.error());
            return *blob;
        }

        std::expected<void, std::string> initialize_boys_library_from_blob(
            const License &license,
            const std::vector<std::uint8_t> &blob)
        {
            if (g_boys_fn != nullptr)
                return std::unexpected("initialize_boys_library_from_blob() called more than once");

            auto plaintext = decrypt_library_blob(blob, license.decryption_key);
            if (!plaintext)
                return std::unexpected("Failed to decrypt library blob: " + plaintext.error());

            auto handle = load_library_from_memory(*plaintext);
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

            g_library_handle = *handle;
            g_boys_fn = reinterpret_cast<BoysFn>(*boys_sym);
            g_boys_vec_fn = reinterpret_cast<BoysVecFn>(*boys_vec_sym);

            return {};
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

            return initialize_boys_library_from_blob(*license, *blob);
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

    } // namespace Lookup
} // namespace HartreeFock
