#ifndef HF_BASE_MALLOC_ENV_H
#define HF_BASE_MALLOC_ENV_H

#ifdef __APPLE__
#include <cstdint>
#include <cstdlib>
#include <unistd.h>

// Declared here rather than through <mach-o/dyld.h>, whose Mach headers g++
// cannot parse.
extern "C" int _NSGetExecutablePath(char *buf, std::uint32_t *bufsize);
#endif

namespace HartreeFock
{
    // Re-exec this process with macOS libmalloc's large-block cache off.
    //
    // With the cache on, a freed large block stays dirty in the process and is
    // handed back whole for a later, smaller request: the freed 2.18 GB
    // Cartesian ERI tensor came back for the 1.45 GB spherical one, pinning a
    // 0.73 GB tail for the rest of the run. Off, freed blocks go back to the OS
    // (water cc-pVTZ CCSD(T) peak footprint 624 -> 261 MiB, same wall time;
    // docs/CCSD_T_SIZE_SCOPE.md, step 3b). libmalloc reads the switch only at
    // process start, hence the re-exec. Call first thing in main(). A value the
    // user already set is respected; if the exec fails the run continues
    // unchanged. Not for planck-mpi: re-exec under an MPI launcher is unsafe.
    // No-op off macOS, where glibc returns blocks this large on free.
    inline void reexec_without_malloc_large_cache(const char *const argv[]) noexcept
    {
#ifdef __APPLE__
        if (std::getenv("MallocLargeCache") != nullptr)
            return;
        char path[4096];
        std::uint32_t size = sizeof path;
        if (_NSGetExecutablePath(path, &size) != 0)
            return;
        if (setenv("MallocLargeCache", "0", 0) != 0)
            return;
        execv(path, const_cast<char *const *>(argv));
        unsetenv("MallocLargeCache"); // exec failed: carry on as before
#else
        (void)argv;
#endif
    }
} // namespace HartreeFock

#endif
