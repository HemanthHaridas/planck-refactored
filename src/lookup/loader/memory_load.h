#ifndef HF_LOOKUP_LOADER_MEMORY_LOAD_H
#define HF_LOOKUP_LOADER_MEMORY_LOAD_H

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace HartreeFock
{
    namespace Loader
    {
        // Loads a shared library from an in-memory buffer and returns an
        // opaque handle (the dlopen() return value). On Linux this uses
        // memfd_create() — the library bytes never touch durable storage.
        // On platforms without an anonymous-fd primitive (macOS: no
        // equivalent exists), this falls back to a restrictively-permissioned
        // temp file that is unlinked immediately after dlopen() succeeds —
        // best-effort ephemeral, not a durable-storage guarantee. See
        // docs/ or session notes for why this asymmetry is inherent to the
        // platform, not a shortcut taken here.
        std::expected<void *, std::string> load_library_from_memory(const std::vector<std::uint8_t> &bytes);

        // Resolves a symbol from a handle returned by load_library_from_memory.
        std::expected<void *, std::string> resolve_symbol(void *handle, const std::string &symbol_name);

        // Closes a handle returned by load_library_from_memory.
        void unload_library(void *handle);

    } // namespace Loader
} // namespace HartreeFock

#endif // !HF_LOOKUP_LOADER_MEMORY_LOAD_H
