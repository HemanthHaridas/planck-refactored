#include "lookup/loader/license.h"

#include <chrono>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include <openssl/evp.h>
#include <openssl/x509.h>

namespace HartreeFock
{
    namespace Loader
    {
        namespace
        {
            // Base64 decode via OpenSSL's EVP_DecodeBlock, trimming the
            // padding-induced zero bytes it doesn't account for itself.
            std::expected<std::vector<std::uint8_t>, std::string> base64_decode(const std::string &in)
            {
                if (in.empty())
                    return std::vector<std::uint8_t>{};

                std::vector<std::uint8_t> out(in.size()); // upper bound
                const int decoded_len = EVP_DecodeBlock(
                    out.data(),
                    reinterpret_cast<const unsigned char *>(in.data()),
                    static_cast<int>(in.size()));
                if (decoded_len < 0)
                    return std::unexpected("Invalid base64 in license file");

                // EVP_DecodeBlock always outputs a multiple of 3 bytes and
                // does not strip padding; trim trailing zero bytes implied
                // by '=' padding characters in the input.
                int pad = 0;
                if (in.size() >= 2 && in[in.size() - 1] == '=')
                    ++pad;
                if (in.size() >= 2 && in[in.size() - 2] == '=')
                    ++pad;
                out.resize(static_cast<std::size_t>(decoded_len - pad));
                return out;
            }

            // Minimal flat-object JSON parser for the fixed license schema:
            // {"key": <string|number>, ...} with no nesting, no arrays. This
            // is not a general JSON parser — it deliberately only handles the
            // shape issue_license.py emits, matching this codebase's existing
            // convention of hand-written JSON I/O (see io/results_json.h)
            // rather than pulling in a JSON library for one small fixed file.
            std::expected<std::unordered_map<std::string, std::string>, std::string>
            parse_flat_json_object(const std::string &text)
            {
                std::unordered_map<std::string, std::string> fields;
                std::size_t i = 0;
                const std::size_t n = text.size();

                auto skip_ws = [&]()
                {
                    while (i < n && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r' || text[i] == ',' || text[i] == '{'))
                        ++i;
                };
                auto read_quoted = [&]() -> std::expected<std::string, std::string>
                {
                    if (i >= n || text[i] != '"')
                        return std::unexpected("Expected '\"' in license JSON");
                    ++i;
                    std::string value;
                    while (i < n && text[i] != '"')
                    {
                        if (text[i] == '\\' && i + 1 < n)
                            ++i;
                        value.push_back(text[i]);
                        ++i;
                    }
                    if (i >= n)
                        return std::unexpected("Unterminated string in license JSON");
                    ++i; // closing quote
                    return value;
                };

                while (true)
                {
                    skip_ws();
                    if (i >= n || text[i] == '}')
                        break;

                    auto key = read_quoted();
                    if (!key)
                        return std::unexpected(key.error());

                    skip_ws();
                    if (i >= n || text[i] != ':')
                        return std::unexpected("Expected ':' after key '" + *key + "' in license JSON");
                    ++i;
                    skip_ws();

                    std::string value;
                    if (i < n && text[i] == '"')
                    {
                        auto v = read_quoted();
                        if (!v)
                            return std::unexpected(v.error());
                        value = *v;
                    }
                    else
                    {
                        // bare number token (int, no decimals used in this schema)
                        std::size_t start = i;
                        while (i < n && (std::isdigit(static_cast<unsigned char>(text[i])) || text[i] == '-'))
                            ++i;
                        if (i == start)
                            return std::unexpected("Expected value for key '" + *key + "' in license JSON");
                        value = text.substr(start, i - start);
                    }

                    fields[*key] = value;
                }

                return fields;
            }

            std::expected<std::int64_t, std::string> to_i64(const std::unordered_map<std::string, std::string> &f, const std::string &key)
            {
                auto it = f.find(key);
                if (it == f.end())
                    return std::unexpected("License JSON missing field: " + key);
                try
                {
                    return std::stoll(it->second);
                }
                catch (...)
                {
                    return std::unexpected("License JSON field '" + key + "' is not an integer");
                }
            }

            std::expected<std::string, std::string> to_str(const std::unordered_map<std::string, std::string> &f, const std::string &key)
            {
                auto it = f.find(key);
                if (it == f.end())
                    return std::unexpected("License JSON missing field: " + key);
                return it->second;
            }

        } // namespace

        std::expected<ParsedLicenseFile, std::string> parse_license_file(const std::string &path)
        {
            std::ifstream in(path, std::ios::binary);
            if (!in)
                return std::unexpected("Cannot open license file: " + path);

            std::ostringstream ss;
            ss << in.rdbuf();
            const std::string text = ss.str();

            auto fields = parse_flat_json_object(text);
            if (!fields)
                return std::unexpected(fields.error());

            License lic;

            auto version = to_i64(*fields, "version");
            if (!version)
                return std::unexpected(version.error());
            lic.version = static_cast<int>(*version);

            auto library_version = to_str(*fields, "library_version");
            if (!library_version)
                return std::unexpected(library_version.error());
            lic.library_version = *library_version;

            auto issued_to = to_str(*fields, "issued_to");
            if (!issued_to)
                return std::unexpected(issued_to.error());
            lic.issued_to = *issued_to;

            auto issued_at = to_i64(*fields, "issued_at");
            if (!issued_at)
                return std::unexpected(issued_at.error());
            lic.issued_at = *issued_at;

            auto expires_at = to_i64(*fields, "expires_at");
            if (!expires_at)
                return std::unexpected(expires_at.error());
            lic.expires_at = *expires_at;

            auto decryption_key_b64 = to_str(*fields, "decryption_key");
            if (!decryption_key_b64)
                return std::unexpected(decryption_key_b64.error());
            auto key_bytes = base64_decode(*decryption_key_b64);
            if (!key_bytes)
                return std::unexpected(key_bytes.error());
            if (key_bytes->size() != 32)
                return std::unexpected("License decryption_key is not 32 bytes (AES-256)");
            lic.decryption_key = *key_bytes;

            auto signature_b64 = to_str(*fields, "signature");
            if (!signature_b64)
                return std::unexpected(signature_b64.error());
            auto signature_bytes = base64_decode(*signature_b64);
            if (!signature_bytes)
                return std::unexpected(signature_bytes.error());

            return ParsedLicenseFile{lic, *signature_bytes};
        }

        std::expected<void, std::string> verify_license_signature(
            const License &license,
            const std::vector<std::uint8_t> &signature)
        {
            // Must match issue_license.py's FIELD_ORDER and delimiter exactly.
            std::ostringstream payload_stream;
            payload_stream << license.version << "|"
                           << license.library_version << "|"
                           << license.issued_to << "|"
                           << license.issued_at << "|"
                           << license.expires_at << "|";
            // decryption_key is signed as its base64 text representation
            // (issue_license.py signs the same string it writes to JSON,
            // not the raw bytes), so re-encode here to match.
            std::vector<std::uint8_t> key_b64_bytes(license.decryption_key.size() * 4 / 3 + 4);
            const int encoded_len = EVP_EncodeBlock(
                key_b64_bytes.data(),
                license.decryption_key.data(),
                static_cast<int>(license.decryption_key.size()));
            const std::string key_b64(reinterpret_cast<const char *>(key_b64_bytes.data()),
                                      static_cast<std::size_t>(encoded_len));
            payload_stream << key_b64;
            const std::string payload = payload_stream.str();

            // Embedded Ed25519 public key (see src/lookup/keys/).
            extern const unsigned char kPlanckSigningPublicKeyDer[];
            extern const std::size_t kPlanckSigningPublicKeyDerLen;

            const unsigned char *pubkey_ptr = kPlanckSigningPublicKeyDer;
            EVP_PKEY *pkey = d2i_PUBKEY(nullptr, &pubkey_ptr, static_cast<long>(kPlanckSigningPublicKeyDerLen));
            if (!pkey)
                return std::unexpected("Failed to load embedded signing public key");

            EVP_MD_CTX *ctx = EVP_MD_CTX_new();
            if (!ctx)
            {
                EVP_PKEY_free(pkey);
                return std::unexpected("Failed to allocate EVP_MD_CTX");
            }

            std::expected<void, std::string> result;
            if (EVP_DigestVerifyInit(ctx, nullptr, nullptr, nullptr, pkey) != 1)
            {
                result = std::unexpected("EVP_DigestVerifyInit failed");
            }
            else
            {
                const int rc = EVP_DigestVerify(
                    ctx,
                    signature.data(), signature.size(),
                    reinterpret_cast<const unsigned char *>(payload.data()), payload.size());
                if (rc != 1)
                    result = std::unexpected("License signature verification failed");
            }

            EVP_MD_CTX_free(ctx);
            EVP_PKEY_free(pkey);
            return result;
        }

        std::expected<License, std::string> load_and_verify_license(const std::string &path)
        {
            auto parsed = parse_license_file(path);
            if (!parsed)
                return std::unexpected(parsed.error());

            auto verify_result = verify_license_signature(parsed->license, parsed->signature);
            if (!verify_result)
                return std::unexpected(verify_result.error());

            if (parsed->license.expires_at != 0)
            {
                const auto now = std::chrono::duration_cast<std::chrono::seconds>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count();
                if (now > parsed->license.expires_at)
                    return std::unexpected("License has expired");
            }

            return parsed->license;
        }

    } // namespace Loader
} // namespace HartreeFock
