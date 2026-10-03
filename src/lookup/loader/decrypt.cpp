#include "lookup/loader/decrypt.h"

#include <openssl/evp.h>

namespace HartreeFock
{
    namespace Loader
    {
        namespace
        {
            constexpr std::size_t kNonceLen = 12;
            constexpr std::size_t kTagLen = 16;
        }

        std::expected<std::vector<std::uint8_t>, std::string> decrypt_library_blob(
            const std::vector<std::uint8_t> &blob,
            const std::vector<std::uint8_t> &key)
        {
            if (key.size() != 32)
                return std::unexpected("Decryption key must be 32 bytes (AES-256)");
            if (blob.size() < kNonceLen + kTagLen)
                return std::unexpected("Encrypted blob too small to contain nonce+tag");

            const std::uint8_t *nonce = blob.data();
            const std::uint8_t *tag = blob.data() + kNonceLen;
            const std::uint8_t *ciphertext = blob.data() + kNonceLen + kTagLen;
            const std::size_t ciphertext_len = blob.size() - kNonceLen - kTagLen;

            EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
            if (!ctx)
                return std::unexpected("Failed to allocate EVP_CIPHER_CTX");

            std::expected<std::vector<std::uint8_t>, std::string> result;
            std::vector<std::uint8_t> plaintext(ciphertext_len);
            int out_len = 0;
            int total_len = 0;

            if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1)
            {
                result = std::unexpected("EVP_DecryptInit_ex (cipher select) failed");
            }
            else if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, static_cast<int>(kNonceLen), nullptr) != 1)
            {
                result = std::unexpected("Failed to set GCM nonce length");
            }
            else if (EVP_DecryptInit_ex(ctx, nullptr, nullptr, key.data(), nonce) != 1)
            {
                result = std::unexpected("EVP_DecryptInit_ex (key+nonce) failed");
            }
            else if (ciphertext_len > 0 &&
                     EVP_DecryptUpdate(ctx, plaintext.data(), &out_len, ciphertext, static_cast<int>(ciphertext_len)) != 1)
            {
                result = std::unexpected("EVP_DecryptUpdate failed");
            }
            else
            {
                total_len = out_len;
                // Tag must be set before the final call so EVP_DecryptFinal_ex
                // can verify it; a mismatched tag makes that call fail, which
                // is the actual integrity check — a tampered ciphertext or
                // wrong key surfaces here, not as garbage plaintext.
                if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, static_cast<int>(kTagLen),
                                        const_cast<std::uint8_t *>(tag)) != 1)
                {
                    result = std::unexpected("Failed to set GCM tag");
                }
                else if (EVP_DecryptFinal_ex(ctx, plaintext.data() + out_len, &out_len) != 1)
                {
                    result = std::unexpected("GCM authentication failed — blob is corrupted, tampered, or the key is wrong");
                }
                else
                {
                    total_len += out_len;
                    plaintext.resize(static_cast<std::size_t>(total_len));
                    result = plaintext;
                }
            }

            EVP_CIPHER_CTX_free(ctx);
            return result;
        }

    } // namespace Loader
} // namespace HartreeFock
