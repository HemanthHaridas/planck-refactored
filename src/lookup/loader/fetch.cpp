#include "lookup/loader/fetch.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include <unistd.h>
#include <sys/wait.h>

namespace HartreeFock
{
    namespace Loader
    {
        namespace
        {
            // Quote a string for safe inclusion in a single-quoted shell
            // argument. Prevents a malicious/malformed URL from breaking out
            // of the argument (e.g. embedded quotes, `;`, `$()`). The only
            // untrusted input reaching this function is the URL, which is
            // either a compile-time constant (production) or a locally-
            // supplied test URL — this exists as defense in depth either way.
            std::string shell_quote(const std::string &s)
            {
                std::string out = "'";
                for (char c : s)
                {
                    if (c == '\'')
                        out += "'\\''";
                    else
                        out += c;
                }
                out += "'";
                return out;
            }
        }

        std::expected<std::vector<std::uint8_t>, std::string> fetch_url(const std::string &url)
        {
            char out_path_template[] = "/tmp/.planck_fetch_XXXXXX";
            const int out_fd = mkstemp(out_path_template);
            if (out_fd < 0)
                return std::unexpected("Failed to create temp file for curl output");
            close(out_fd);
            const std::string out_path(out_path_template);

            // --fail: nonzero exit on HTTP >= 400, instead of writing the
            // error page's body to the output file as if it were the blob.
            // --silent --show-error: no progress meter, but errors still
            // reach stderr for the diagnostic message below.
            // --location: follow redirects (GitHub Releases assets redirect
            // through a signed S3-style URL).
            const std::string cmd = "curl --fail --silent --show-error --location "
                                    "-o " +
                                    shell_quote(out_path) + " " + shell_quote(url) + " 2>&1";

            FILE *pipe = popen(cmd.c_str(), "r");
            if (!pipe)
            {
                unlink(out_path.c_str());
                return std::unexpected("Failed to spawn curl");
            }

            std::ostringstream stderr_capture;
            char buf[256];
            while (fgets(buf, sizeof(buf), pipe) != nullptr)
                stderr_capture << buf;

            const int status = pclose(pipe);
            const int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;

            if (exit_code != 0)
            {
                unlink(out_path.c_str());
                std::ostringstream err;
                err << "curl fetch of '" << url << "' failed (exit " << exit_code << "): "
                    << stderr_capture.str();
                return std::unexpected(err.str());
            }

            std::ifstream in(out_path, std::ios::binary);
            if (!in)
            {
                unlink(out_path.c_str());
                return std::unexpected("curl reported success but output file is unreadable: " + out_path);
            }
            std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            in.close();
            unlink(out_path.c_str());

            if (bytes.empty())
                return std::unexpected("curl fetch of '" + url + "' returned an empty response");

            return bytes;
        }

    } // namespace Loader
} // namespace HartreeFock
