#ifndef HF_WRAPPER_H
#define HF_WRAPPER_H

#include <memory>
#include <stdexcept>
#include <string>
#include <expected>
#include <cstdlib>

#include "external/libmsym/install/include/libmsym/msym.h"

namespace HartreeFock
{
    namespace Symmetry
    {
        class SymmetryContext
        {
        public:
            // Constructor
            SymmetryContext()
            {
                _ctx = msymCreateContext();
                if (!_ctx)
                {
                    throw std::runtime_error("Failed to create msym_context");
                }
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
            msym_context _ctx;
        };

        class SymmetryElements
        {
        public:
            explicit SymmetryElements(size_t n_atoms)
            {
                elems_ = static_cast <msym_element_t *>(malloc(n_atoms * sizeof(msym_element_t)));
                if (!elems_)
                {
                    throw std::bad_alloc();
                }
                memset(elems_, 0, n_atoms * sizeof(msym_element_t));
                n_atoms_ = n_atoms;
            }

            ~SymmetryElements()
            {
                free(elems_);
            }

            // Non-copyable
            SymmetryElements(const SymmetryElements &) = delete;
            SymmetryElements &operator=(const SymmetryElements &) = delete;

            // Movable
            SymmetryElements(SymmetryElements &&other) noexcept : elems_(other.elems_), n_atoms_(other.n_atoms_)
            {
                other.elems_ = nullptr;
                other.n_atoms_ = 0;
            }
            SymmetryElements &operator=(SymmetryElements &&other) noexcept
            {
                if (this != &other)
                {
                    free(elems_);
                    elems_ = other.elems_;
                    n_atoms_ = other.n_atoms_;
                    other.elems_ = nullptr;
                    other.n_atoms_ = 0;
                }
                return *this;
            }

            msym_element_t *data()
            {
                return elems_;
            }
            size_t size() const {
                return n_atoms_;
            }

        private:
            msym_element_t *elems_;
            size_t n_atoms_;
        };
    }
}

#endif // !HF_WRAPPER_H
