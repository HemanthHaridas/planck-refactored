#ifndef HF_WRAPPER_H
#define HF_WRAPPER_H

#include <cstdlib>
#include <expected>
#include <memory>
#include <string>
#include <vector>

#include "external/libmsym/install/include/libmsym/msym.h"

namespace HartreeFock
{
    namespace Symmetry
    {
        class SymmetryContext
        {
        public:
            static std::expected<SymmetryContext, std::string> create()
            {
                msym_context ctx = msymCreateContext();
                if (!ctx)
                    return std::unexpected("Failed to create msym_context");
                return SymmetryContext(ctx);
            }

            ~SymmetryContext()
            {
                if (_ctx)
                {
                    msymReleaseContext(_ctx);
                }
            }

            // Non-copyable
            SymmetryContext(const SymmetryContext &) = delete;
            SymmetryContext &operator=(const SymmetryContext &) = delete;

            // Movable
            SymmetryContext(SymmetryContext &&other) noexcept : _ctx(other._ctx)
            {
                other._ctx = nullptr;
            }
            SymmetryContext &operator=(SymmetryContext &&other) noexcept
            {
                if (this != &other)
                {
                    if (_ctx)
                        msymReleaseContext(_ctx);
                    _ctx = other._ctx;
                    other._ctx = nullptr;
                }
                return *this;
            }

            // Accessor
            msym_context get() const
            {
                return _ctx;
            }

        private:
            explicit SymmetryContext(msym_context ctx) noexcept : _ctx(ctx)
            {
            }

            msym_context _ctx = nullptr;
        };

        class SymmetryElements
        {
        public:
            explicit SymmetryElements(size_t n_atoms) : elems_(n_atoms)
            {
            }

            // Non-copyable
            SymmetryElements(const SymmetryElements &) = delete;
            SymmetryElements &operator=(const SymmetryElements &) = delete;

            // Movable
            SymmetryElements(SymmetryElements &&other) noexcept = default;
            SymmetryElements &operator=(SymmetryElements &&other) noexcept
            {
                if (this != &other)
                    elems_ = std::move(other.elems_);
                return *this;
            }

            msym_element_t *data()
            {
                return elems_.data();
            }
            size_t size() const
            {
                return elems_.size();
            }

        private:
            std::vector<msym_element_t> elems_;
        };
    } // namespace Symmetry
} // namespace HartreeFock

#endif // !HF_WRAPPER_H
