#ifndef HF_POSTHF_CC_TENSOR_BACKEND_INTERNAL_H
#define HF_POSTHF_CC_TENSOR_BACKEND_INTERNAL_H

#include <cstddef>
#include <expected>
#include <initializer_list>
#include <string>
#include <vector>

#include "post_hf/cc/tensor_backend.h"

namespace HartreeFock::Correlation::CC::detail
{
    [[nodiscard]] std::expected<std::size_t, std::string> checked_product(
        std::initializer_list<int> dims);

    [[nodiscard]] std::expected<std::size_t, std::string> bytes_for_tensor(
        std::initializer_list<int> dims);

    [[nodiscard]] std::string format_bytes(std::size_t bytes);

    std::expected<void, std::string> append_block_memory(
        std::vector<TensorMemoryBlock> &report,
        std::size_t &total_bytes,
        const std::string &label,
        std::initializer_list<int> dims);
} // namespace HartreeFock::Correlation::CC::detail

#endif // HF_POSTHF_CC_TENSOR_BACKEND_INTERNAL_H
