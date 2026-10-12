#ifndef HF_LOOKUP_LOADER_LICENSE_H
#define HF_LOOKUP_LOADER_LICENSE_H

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace HartreeFock
{
    namespace Loader
    {
        struct License
        {
            int version = 0;
            std::string library_version;
            std::string issued_to;
            std::int64_t issued_at = 0;
            std::int64_t expires_at = 0; // 0 = no expiry
            std::vector<std::uint8_t> decryption_key; // raw 32 bytes, AES-256
        };

        struct ParsedLicenseFile
        {
            License license;
            std::vector<std::uint8_t> signature; // raw bytes, base64-decoded
        };

        // Reads and parses a license file (the JSON format written by
        // issue_license.py). Does NOT verify the signature — call
        // verify_license_signature() separately so parsing and trust are
        // distinct, testable steps.
        std::expected<ParsedLicenseFile, std::string> parse_license_file(const std::string &path);

        // Verifies the license's Ed25519 signature against the embedded
        // public key, using the exact same field-order/delimiter scheme
        // issue_license.py signs with ("version|library_version|issued_to|
        // issued_at|expires_at|decryption_key", pipe-joined field values).
        // The raw signature bytes are passed separately (not re-extracted
        // from the License struct) so this function has no file I/O of its
        // own and is trivially unit-testable against fixed byte strings.
        std::expected<void, std::string> verify_license_signature(
            const License &license,
            const std::vector<std::uint8_t> &signature);

        // Convenience: parse + verify + expiry check in one call. This is
        // what production call sites use; the split functions above exist
        // for testing.
        std::expected<License, std::string> load_and_verify_license(const std::string &path);

    } // namespace Loader
} // namespace HartreeFock

#endif // !HF_LOOKUP_LOADER_LICENSE_H
