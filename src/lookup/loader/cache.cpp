#include "lookup/loader/cache.h"

#include <cerrno>
#include <fstream>

#include <sys/stat.h>
#include <unistd.h>

namespace HartreeFock
{
    namespace Loader
    {
        namespace
        {
            // <cache_dir>/<library_version>/<platform_asset> — the
            // version segment matters because different library versions'
            // blobs must never shadow each other; the platform_asset
            // basename already disambiguates platform within a version.
            std::string entry_path(
                const std::string &cache_dir,
                const std::string &library_version,
                const std::string &platform_asset)
            {
                return cache_dir + "/" + library_version + "/" + platform_asset;
            }

            // mkdir -p, best-effort. Returns false only on an error other
            // than "already exists", which the caller treats as "caching
            // isn't going to work this time" rather than a hard failure.
            bool ensure_parent_dir(const std::string &path)
            {
                const auto slash = path.find_last_of('/');
                if (slash == std::string::npos)
                    return true;
                const std::string dir = path.substr(0, slash);

                std::string partial;
                std::size_t pos = 0;
                bool all_ok = true;
                while (pos <= dir.size())
                {
                    const auto next = dir.find('/', pos);
                    const std::string segment = dir.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
                    if (!segment.empty())
                    {
                        partial += segment;
                        if (::mkdir(partial.c_str(), 0700) != 0 && errno != EEXIST)
                            all_ok = false;
                    }
                    partial += "/";
                    if (next == std::string::npos)
                        break;
                    pos = next + 1;
                }
                return all_ok;
            }
        }

        std::optional<std::vector<std::uint8_t>> read_cached_blob(
            const std::string &cache_dir,
            const std::string &library_version,
            const std::string &platform_asset,
            std::chrono::seconds max_age)
        {
            const std::string path = entry_path(cache_dir, library_version, platform_asset);

            struct stat st{};
            if (::stat(path.c_str(), &st) != 0)
                return std::nullopt; // cache miss

            const auto mtime = std::chrono::system_clock::from_time_t(st.st_mtime);
            const auto age = std::chrono::system_clock::now() - mtime;
            if (age > max_age)
                return std::nullopt; // stale — treated exactly like a miss

            std::ifstream in(path, std::ios::binary);
            if (!in)
                return std::nullopt;
            std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            in.close();
            if (bytes.empty())
                return std::nullopt;
            return bytes;
        }

        void write_cached_blob(
            const std::string &cache_dir,
            const std::string &library_version,
            const std::string &platform_asset,
            const std::vector<std::uint8_t> &bytes)
        {
            const std::string path = entry_path(cache_dir, library_version, platform_asset);
            if (!ensure_parent_dir(path))
                return; // caching isn't available — not an error for the caller

            const std::string tmp_path = path + ".tmp." + std::to_string(::getpid());

            {
                std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
                if (!out)
                    return;
                out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
                if (!out)
                {
                    out.close();
                    ::unlink(tmp_path.c_str());
                    return;
                }
            }

            // Atomic on POSIX: rename() replaces any existing entry in one
            // step, so a concurrent reader sees either the old file or the
            // new one in full, never a partial write. Also refreshes the
            // entry's mtime, which is exactly what read_cached_blob's
            // age check wants after a fresh fetch.
            if (::rename(tmp_path.c_str(), path.c_str()) != 0)
                ::unlink(tmp_path.c_str());
        }

        void evict_cached_blob(
            const std::string &cache_dir,
            const std::string &library_version,
            const std::string &platform_asset)
        {
            const std::string path = entry_path(cache_dir, library_version, platform_asset);
            ::unlink(path.c_str());
        }

    } // namespace Loader
} // namespace HartreeFock
