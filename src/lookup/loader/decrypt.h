#ifndef HF_LOOKUP_LOADER_DECRYPT_H
#define HF_LOOKUP_LOADER_DECRYPT_H

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace HartreeFock
{
    namespace Loader
    {
        // Decrypts an AES-256-GCM blob laid out as produced by
        // encrypt_library.py: [12-byte nonce][16-byte tag][ciphertext].
        // `key` must be exactly 32 bytes (AES-256), as parsed from a
        // license's decryption_key field. Fails closed on any integrity
        // error — a corrupted or tampered blob does not partially decrypt.
        std::expected<std::vector<std::uint8_t>, std::string> decrypt_library_blob(
            const std::vector<std::uint8_t> &blob,
            const std::vector<std::uint8_t> &key);

    } // namespace Loader
} // namespace HartreeFock

#endif // !HF_LOOKUP_LOADER_DECRYPT_H
