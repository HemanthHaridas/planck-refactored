#ifndef HF_LOOKUP_LOADER_FETCH_H
#define HF_LOOKUP_LOADER_FETCH_H

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace HartreeFock
{
    namespace Loader
    {
        // Downloads a URL to memory via a `curl` subprocess (no libcurl
        // dependency — curl the binary is already present on every
        // dev/CI/cluster host this project targets). Supports plain http(s)
        // and file:// URLs (the latter used for local testing).
        //
        // Fails closed: any nonzero curl exit status, or a response written
        // to stderr indicating an HTTP error (curl's --fail flag), is
        // reported as an error rather than returning partial/empty bytes.
        std::expected<std::vector<std::uint8_t>, std::string> fetch_url(const std::string &url);

    } // namespace Loader
} // namespace HartreeFock

#endif // !HF_LOOKUP_LOADER_FETCH_H
