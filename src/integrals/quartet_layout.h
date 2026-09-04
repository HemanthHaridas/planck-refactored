// quartet_layout.h — the 6-axis spatial layout shared by the OS, HGP, and Rys
// per-quartet ERI scratch structs.
//
// All three engines size a per-quartet accumulator over the six Cartesian
// angular-momentum axes (ax, ay, az | cx, cy, cz) with the identical row-major
// stride convention and the identical flat index. That core — six dims, six
// strides, the stride computation, and spatial_index — is factored here.
//
// Everything else stays engine-specific and is NOT in this struct: the Boys `m`
// axis (OS/HGP), the vrr/hrr/a0c0 buffers and their zero-init policies, HGP's
// cd_block_size / v_ptr / h_block_ptr, and Rys's single buf/at. Each scratch
// EMBEDS a SpatialQuartetLayout as a member rather than inheriting it — a plain
// POD member, not a base class, so the engines keep their own buffers and
// accessors and only the duplicated layout math is shared.
#pragma once

#include <cstddef>

namespace HartreeFock::Integrals
{
    struct SpatialQuartetLayout
    {
        int ax_dim = 0;
        int ay_dim = 0;
        int az_dim = 0;
        int cx_dim = 0;
        int cy_dim = 0;
        int cz_dim = 0;
        std::size_t ax_stride = 0;
        std::size_t ay_stride = 0;
        std::size_t az_stride = 0;
        std::size_t cx_stride = 0;
        std::size_t cy_stride = 0;
        std::size_t cz_stride = 0;
        std::size_t spatial_size = 0;

        // Set the six dims from the per-axis total angular momenta (lAB*, lCD*)
        // and compute the row-major strides. Returns spatial_size so callers can
        // size their buffers in one line. Identical in all three engines.
        std::size_t configure(int lABx, int lABy, int lABz,
                              int lCDx, int lCDy, int lCDz) noexcept
        {
            ax_dim = lABx + 1;
            ay_dim = lABy + 1;
            az_dim = lABz + 1;
            cx_dim = lCDx + 1;
            cy_dim = lCDy + 1;
            cz_dim = lCDz + 1;
            cz_stride = 1;
            cy_stride = static_cast<std::size_t>(cz_dim) * cz_stride;
            cx_stride = static_cast<std::size_t>(cy_dim) * cy_stride;
            az_stride = static_cast<std::size_t>(cx_dim) * cx_stride;
            ay_stride = static_cast<std::size_t>(az_dim) * az_stride;
            ax_stride = static_cast<std::size_t>(ay_dim) * ay_stride;
            spatial_size =
                static_cast<std::size_t>(ax_dim) * ay_dim * az_dim *
                cx_dim * cy_dim * cz_dim;
            return spatial_size;
        }

        std::size_t spatial_index(int ax, int ay, int az,
                                  int cx, int cy, int cz) const noexcept
        {
            return static_cast<std::size_t>(ax) * ax_stride +
                   static_cast<std::size_t>(ay) * ay_stride +
                   static_cast<std::size_t>(az) * az_stride +
                   static_cast<std::size_t>(cx) * cx_stride +
                   static_cast<std::size_t>(cy) * cy_stride +
                   static_cast<std::size_t>(cz) * cz_stride;
        }
    };
}
