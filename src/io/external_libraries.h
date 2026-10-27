#ifndef HF_IO_EXTERNAL_LIBRARIES_H
#define HF_IO_EXTERNAL_LIBRARIES_H

// The startup report of every external library a binary uses: name, version,
// licence and role. Included only by the binaries' main.cpp files, so each
// binary's own build flags decide the rows: libxc with USING_Libxc
// (planck-dft, planck-mpi), MPI with USE_MPI (planck-mpi), OpenMP with
// USE_OPENMP. Versions come from the libraries where they can report one
// (Eigen's macros, libxc and MPI at run time, the Boys loader), otherwise from
// the commits CMake pins (PLANCK_LIBMSYM_*, PLANCK_LIBXC_COMMIT). Call after
// the Boys library has loaded, so its version is known.

#include <format>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>

#ifdef USING_Libxc
#include <xc.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "io/logging.h"
#include "lookup/loader/orchestrator.h"

namespace HartreeFock::Logger
{
    namespace detail
    {
        struct ExternalLibrary
        {
            std::string name, version, licence, role;
        };

        inline std::string short_commit(const char *commit) { return std::string(commit).substr(0, 7); }

#ifdef USE_OPENMP
        // _OPENMP is the spec date (yyyymm) the compiler implements.
        inline std::string openmp_spec(int date)
        {
            if (date >= 202111) return "5.2";
            if (date >= 202011) return "5.1";
            if (date >= 201811) return "5.0";
            if (date >= 201511) return "4.5";
            if (date >= 201307) return "4.0";
            if (date >= 201107) return "3.1";
            return std::to_string(date);
        }
#endif

        inline std::vector<ExternalLibrary> external_libraries()
        {
            std::vector<ExternalLibrary> libs;
            libs.push_back({"Eigen",
                            std::format("{}.{}.{}", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION),
                            "MPL-2.0", "dense linear algebra"});
            libs.push_back({"libmsym",
                            std::format("{} (git {})", PLANCK_LIBMSYM_VERSION, short_commit(PLANCK_LIBMSYM_COMMIT)),
                            "MIT", "point-group symmetry"});
#ifdef USING_Libxc
            libs.push_back({"libxc", std::format("{} (git {})", xc_version_string(), short_commit(PLANCK_LIBXC_COMMIT)),
                            "MPL-2.0", "exchange-correlation functionals"});
#endif
            const std::string &boys = HartreeFock::Loader::loaded_boys_library_version();
            libs.push_back({"Planck Boys core", boys.empty() ? "not loaded" : boys,
                            "proprietary, runtime licence", "Boys function"});
#ifdef USE_OPENMP
#if defined(__clang__)
            const char *omp_runtime = "libomp";
#elif defined(__GNUC__)
            const char *omp_runtime = "libgomp";
#else
            const char *omp_runtime = "OpenMP runtime";
#endif
            libs.push_back({"OpenMP",
                            std::format("{} ({}), {} thread{}", openmp_spec(_OPENMP), omp_runtime,
                                        omp_get_max_threads(), omp_get_max_threads() == 1 ? "" : "s"),
                            "compiler runtime", "shared-memory threading"});
#endif
#ifdef USE_MPI
            char mpi_lib[MPI_MAX_LIBRARY_VERSION_STRING];
            int len = 0;
            MPI_Get_library_version(mpi_lib, &len);
            std::string impl(mpi_lib, static_cast<std::size_t>(len));
            impl = impl.substr(0, impl.find_first_of("\n,")); // first line / clause
            int ranks = 1;
            MPI_Comm_size(MPI_COMM_WORLD, &ranks);
            libs.push_back({"MPI", std::format("{}, {} rank{}", impl, ranks, ranks == 1 ? "" : "s"), "implementation's",
                            "distributed builds"});
#endif
            return libs;
        }
    }

    inline void external_libraries()
    {
        if (is_silenced())
            return;
        const auto libs = detail::external_libraries();
        std::lock_guard<std::mutex> lock(log_mutex);
        std::cout << "     External libraries\n"
                  << std::format("       {:<18}{:<34}{:<30}{}\n", "Library", "Version", "Licence", "Used for");
        for (const auto &l : libs)
            std::cout << std::format("       {:<18}{:<34}{:<30}{}\n", l.name, l.version, l.licence, l.role);
#if defined(__clang__)
        std::cout << std::format("     Built with clang {}\n\n", __clang_version__);
#elif defined(__GNUC__)
        std::cout << std::format("     Built with g++ {}\n\n", __VERSION__);
#else
        std::cout << '\n';
#endif
    }
}

#endif // HF_IO_EXTERNAL_LIBRARIES_H
