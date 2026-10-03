#include "lookup/loader/memory_load.h"

#include <dlfcn.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/syscall.h>
#elif defined(__APPLE__)
#include <sys/stat.h>
#include <cstdlib>
#else
#error "load_library_from_memory: unsupported platform — needs a memfd_create or restrictive-temp-file backend"
#endif

namespace HartreeFock
{
    namespace Loader
    {
#if defined(__linux__)

        std::expected<void *, std::string> load_library_from_memory(const std::vector<std::uint8_t> &bytes)
        {
            // memfd_create gives a file descriptor backed only by anonymous
            // memory (tmpfs-like, never written to a real filesystem). We
            // dlopen it via /proc/self/fd/<n>, which the kernel resolves
            // straight back to the memfd.
            const int fd = static_cast<int>(syscall(SYS_memfd_create, "planck_boys_lib", 0));
            if (fd < 0)
                return std::unexpected("memfd_create failed");

            std::size_t written = 0;
            while (written < bytes.size())
            {
                const ssize_t n = write(fd, bytes.data() + written, bytes.size() - written);
                if (n < 0)
                {
                    close(fd);
                    return std::unexpected("Failed writing library bytes to memfd");
                }
                written += static_cast<std::size_t>(n);
            }

            char path[64];
            std::snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);

            void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
            close(fd); // dlopen has already mmap'd it; the fd itself can close
            if (!handle)
                return std::unexpected(std::string("dlopen (memfd) failed: ") + dlerror());

            return handle;
        }

#elif defined(__APPLE__)

        std::expected<void *, std::string> load_library_from_memory(const std::vector<std::uint8_t> &bytes)
        {
            // macOS has no anonymous-fd-backed dlopen path. Best effort:
            // write to a private, restrictively-permissioned temp file,
            // dlopen it, then unlink immediately — the bytes are on disk for
            // the shortest practical window, not held there durably, but
            // this is a real (if brief) disk touch, unlike the Linux path.
            char path[] = "/tmp/.planck_boys_XXXXXX";
            const int fd = mkstemp(path);
            if (fd < 0)
                return std::unexpected("mkstemp failed for temporary library file");

            if (fchmod(fd, S_IRUSR | S_IWUSR) != 0)
            {
                close(fd);
                unlink(path);
                return std::unexpected("Failed to set restrictive permissions on temp library file");
            }

            std::size_t written = 0;
            while (written < bytes.size())
            {
                const ssize_t n = write(fd, bytes.data() + written, bytes.size() - written);
                if (n < 0)
                {
                    close(fd);
                    unlink(path);
                    return std::unexpected("Failed writing library bytes to temp file");
                }
                written += static_cast<std::size_t>(n);
            }
            close(fd);

            void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
            unlink(path); // remove the directory entry now; the mapped pages survive via the open mapping
            if (!handle)
                return std::unexpected(std::string("dlopen (temp file) failed: ") + dlerror());

            return handle;
        }

#endif

        std::expected<void *, std::string> resolve_symbol(void *handle, const std::string &symbol_name)
        {
            dlerror(); // clear any prior error
            void *sym = dlsym(handle, symbol_name.c_str());
            const char *err = dlerror();
            if (err)
                return std::unexpected(std::string("dlsym('" + symbol_name + "') failed: ") + err);
            if (!sym)
                return std::unexpected("Symbol '" + symbol_name + "' resolved to null");
            return sym;
        }

        void unload_library(void *handle)
        {
            if (handle)
                dlclose(handle);
        }

    } // namespace Loader
} // namespace HartreeFock
