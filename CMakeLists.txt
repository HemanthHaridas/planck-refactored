cmake_minimum_required(VERSION 3.5)
project(planck LANGUAGES CXX)

include(CTest)

# C++ standard settings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to -O3 when no CMAKE_BUILD_TYPE is set. Debug/RelWithDebInfo/etc.
# keep their standard flags so debugging and profiling builds aren't clobbered.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

message(STATUS "Building on : ${CMAKE_SYSTEM_NAME} using following compiler flags : ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_INSTALL_PREFIX : ${CMAKE_INSTALL_PREFIX}")

# GPU / CUDA acceleration (disabled by default; enable with -DUSE_CUDA=ON)
option(USE_CUDA "Enable CUDA GPU acceleration (requires CUDA toolkit)" OFF)

# Coupled-cluster kernel generation via ccgen.
# PLANCK_CC_MAXORDER sets the highest excitation rank to emit Planck-compatible
# kernels for at build time. All CC methods from CCSD up to the requested
# truncation are generated (e.g. MAXORDER=4 emits ccsd, ccsdt, ccsdtq).
# Valid range: 2..6. Override from the command line:
#   cmake .. -DPLANCK_CC_MAXORDER=4
set(PLANCK_CC_MAXORDER "3" CACHE STRING
    "Max CC excitation rank to generate Planck kernels for (2=CCSD, 3=CCSDT, 4=CCSDTQ, 5=CC5, 6=CC6)")
option(PLANCK_CC_INCLUDE_INTERMEDIATES
    "Emit ccgen intermediate builders in generated CC kernels" ON)
set(PLANCK_CC_INTERMEDIATE_THRESHOLD "5" CACHE STRING
    "Min usage count for extracted ccgen intermediates")

if(NOT PLANCK_CC_MAXORDER MATCHES "^[2-6]$")
    message(FATAL_ERROR
        "PLANCK_CC_MAXORDER must be an integer in [2, 6]; got '${PLANCK_CC_MAXORDER}'")
endif()

# Build the ccgen method list up to PLANCK_CC_MAXORDER.
# Rank -> method name mapping follows ccgen's standard CC level strings.
set(_planck_cc_method_by_rank "ccsd" "ccsdt" "ccsdtq" "cc5" "cc6")
set(PLANCK_CC_METHODS "")
math(EXPR _planck_cc_method_count "${PLANCK_CC_MAXORDER} - 2")
foreach(_rank_index RANGE 0 ${_planck_cc_method_count})
    list(GET _planck_cc_method_by_rank ${_rank_index} _planck_cc_method)
    list(APPEND PLANCK_CC_METHODS ${_planck_cc_method})
endforeach()

# OpenMP support (enabled by default; pass -DUSE_OPENMP=OFF to disable)
option(USE_OPENMP "Enable OpenMP parallelism" ON)
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP : Found (${OpenMP_CXX_VERSION})")
    else()
        message(WARNING "OpenMP : Requested but not found — building without it")
    endif()
else()
    message(STATUS "OpenMP : Disabled")
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET TRUE)

find_package(Python3 COMPONENTS Interpreter)

# Source directory
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

set(BASIS_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/install/share/basis-sets")

# Install basis sets
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/basis-sets
    DESTINATION share
)

# Configure generated header
configure_file(
    ${SRC_DIR}/base/basis.h.in
    ${SRC_DIR}/base/basis.h @ONLY
)

# Fetch Eigen
fetchcontent_declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0 # You can also use master
)
fetchcontent_makeavailable(eigen)

# Run Eigen's GEMMs serially on the calling thread. This code owns its own
# OpenMP parallelism at the ERI/Fock level; letting Eigen spawn a second thread
# pool nests against ours and oversubscribes the cores. Disabling it measured
# ~26% faster on 8-thread conventional RHF (water dimer/cc-pVTZ), energies
# bit-identical. ponytail: global define, revisit only if a hot path is proven
# GEMM-bound and would benefit from parallel Eigen in its own scope.
add_compile_definitions(EIGEN_DONT_PARALLELIZE)

# External project: libmsym
set(MSYM_INSTALL_DIR ${SRC_DIR}/external/libmsym/install)
set(MSYM_BUILD_DIR ${SRC_DIR}/external/libmsym/build)
set(LIBXC_INSTALL_DIR ${SRC_DIR}/external/libxc/install)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)

include(ExternalProject)

externalproject_add(
    libmsym
    PREFIX ${SRC_DIR}/external/libmsym
    GIT_REPOSITORY https://github.com/mcodev31/libmsym.git
    UPDATE_DISCONNECTED TRUE
    CMAKE_ARGS 
        -DCMAKE_INSTALL_PREFIX=${MSYM_INSTALL_DIR} 
        -DBUILD_SHARED_LIBS:BOOL=OFF 
        -DMSYM_BUILD_EXAMPLES:BOOL=OFF
)

externalproject_add(
    libxc
    PREFIX ${SRC_DIR}/external/libxc
    # libxc moved off GitHub; the old ElectronicStructureLibrary/libxc path now
    # 404s (and git then prompts for HTTPS credentials, since over HTTPS a 404 is
    # indistinguishable from a private repo needing auth). GitLab is the live
    # home. Pinned to the EXACT commit the existing validated checkout was on
    # (5.0.0-799-g4bd0e1e36), which still exists on GitLab — a URL-only fix with
    # zero version drift, so the DFT wrapper (src/dft/base/wrapper.h) and XC
    # regressions stay on the libxc they were validated against. The previous
    # unpinned default-branch fetch is what drifted onto the dead URL. (Bumping
    # to 6.x/7.x would need a DFT build + regression check first.)
    GIT_REPOSITORY https://gitlab.com/libxc/libxc.git
    GIT_TAG 4bd0e1e36347c6d0a4e378a2c8d891ae43f8c951
    UPDATE_DISCONNECTED TRUE
    GIT_PROGRESS TRUE
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${LIBXC_INSTALL_DIR}
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DBUILD_TESTING:BOOL=OFF
        -DNAMESPACE_INSTALL_INCLUDEDIR:STRING=/
        -DCMAKE_INSTALL_LIBDIR:STRING=lib
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    BUILD_BYPRODUCTS ${LIBXC_INSTALL_DIR}/lib/libxc.a
)

# Collect sources automatically
file(GLOB BASE_SRC CONFIGURE_DEPENDS
    ${SRC_DIR}/base/*.h
    ${SRC_DIR}/base/*.cpp
)

# Collect sources automatically
file(GLOB IO_SRC CONFIGURE_DEPENDS
    ${SRC_DIR}/io/*.h
    ${SRC_DIR}/io/*.cpp
)

# Collect sources automatically
file(GLOB LOOKUP_SRC CONFIGURE_DEPENDS
    ${SRC_DIR}/lookup/*.h
    ${SRC_DIR}/lookup/*.cpp
)

# Collect sources automatically
file(GLOB SYMM_SRC CONFIGURE_DEPENDS
    ${SRC_DIR}/symmetry/*.h
    ${SRC_DIR}/symmetry/*.cpp
)

# Collect sources automatically
file(GLOB BASIS_SRC CONFIGURE_DEPENDS
    ${SRC_DIR}/basis/*.h
    ${SRC_DIR}/basis/*.cpp
)

# Collect sources automatically
file(GLOB INT_SRC CONFIGURE_DEPENDS
    ${SRC_DIR}/integrals/*.h
    ${SRC_DIR}/integrals/*.cpp
)

# Collect sources automatically
file(GLOB SCF_SRC CONFIGURE_DEPENDS
    ${SRC_DIR}/scf/*.h
    ${SRC_DIR}/scf/*.cpp
)

file(GLOB POPULATIONS_SRC CONFIGURE_DEPENDS
    ${SRC_DIR}/populations/*.h
    ${SRC_DIR}/populations/*.cpp
)

# Collect sources automatically
file(GLOB POSTHF_SRC CONFIGURE_DEPENDS
    ${SRC_DIR}/post_hf/*.h
    ${SRC_DIR}/post_hf/*.cpp
    ${SRC_DIR}/post_hf/cc/*.h
    ${SRC_DIR}/post_hf/cc/*.cpp
    ${SRC_DIR}/post_hf/casscf/*.h
    ${SRC_DIR}/post_hf/casscf/*.cpp
    ${SRC_DIR}/post_hf/ci/*.h
    ${SRC_DIR}/post_hf/ci/*.cpp
    ${SRC_DIR}/post_hf/ri/*.h
    ${SRC_DIR}/post_hf/ri/*.cpp
)
list(REMOVE_ITEM POSTHF_SRC ${SRC_DIR}/post_hf/casscf.cpp)

# planck-dft only needs a subset of post_hf:
#   integrals.cpp        — AO-ERI / AO→MO transforms used for hybrid-XC exact
#                          exchange (dft/driver.cpp) and stability analysis
#                          (scf/stability.cpp)
#   mp2.cpp              — MP2 driver entry points
#   mp2_internal.cpp     — shared ChemistsERIs / frozen-orbital helpers
#   mp2_rmp2.cpp         — RMP2 kernel + RDM builders
#   mp2_ump2.cpp         — UMP2 kernel + RDM builders
#   mp2_gradient.cpp     — RMP2/UMP2 gradient intermediates, referenced from
#                          gradient/gradient.cpp
#   rhf_response.cpp     — solve_rhf_cphf, used by mp2_gradient.cpp
# CASSCF, RASSCF, and coupled-cluster sources are intentionally excluded.
set(POSTHF_DFT_SRC
    ${SRC_DIR}/post_hf/integrals.h
    ${SRC_DIR}/post_hf/integrals.cpp
    ${SRC_DIR}/post_hf/mp2.h
    ${SRC_DIR}/post_hf/mp2.cpp
    ${SRC_DIR}/post_hf/mp2_internal.h
    ${SRC_DIR}/post_hf/mp2_internal.cpp
    ${SRC_DIR}/post_hf/mp2_rmp2.cpp
    ${SRC_DIR}/post_hf/mp2_ump2.cpp
    ${SRC_DIR}/post_hf/mp2_gradient.h
    ${SRC_DIR}/post_hf/mp2_gradient.cpp
    ${SRC_DIR}/post_hf/ri/ri_eri.h
    ${SRC_DIR}/post_hf/ri/ri_eri.cpp
    ${SRC_DIR}/post_hf/cc/diis.h
    ${SRC_DIR}/post_hf/cc/diis.cpp
    ${SRC_DIR}/post_hf/rhf_response.h
    ${SRC_DIR}/post_hf/rhf_response.cpp
    ${SRC_DIR}/post_hf/uhf_response.h
    ${SRC_DIR}/post_hf/uhf_response.cpp
)

file(GLOB GRADIENT_SRC CONFIGURE_DEPENDS
    ${SRC_DIR}/gradient/*.h
    ${SRC_DIR}/gradient/*.cpp
)

file(GLOB OPT_SRC CONFIGURE_DEPENDS
    ${SRC_DIR}/opt/*.h
    ${SRC_DIR}/opt/*.cpp
)

file(GLOB FREQ_SRC CONFIGURE_DEPENDS
    ${SRC_DIR}/freq/*.h
    ${SRC_DIR}/freq/*.cpp
)

file(GLOB SOLVATION_SRC CONFIGURE_DEPENDS
    ${SRC_DIR}/solvation/*.h
    ${SRC_DIR}/solvation/*.cpp
)

file(GLOB BSSE_SRC CONFIGURE_DEPENDS
    ${SRC_DIR}/bsse/*.h
    ${SRC_DIR}/bsse/*.cpp
)

# The HF workflow (HartreeFock::Driver::run) is shared by every front end;
# each front end supplies its own thin main().
set(HF_DRIVER_SRC
    ${SRC_DIR}/hf_driver.cpp
)
set(HF_MAIN_SRC
    ${SRC_DIR}/driver.cpp
    ${HF_DRIVER_SRC}
)

# DFT compute/CLI sources without the entry point, so multiple front ends
# (planck-dft, planck-mpi) can share the DFT driver.
set(DFT_LIB_SRC
    ${SRC_DIR}/dft/ao_grid.h
    ${SRC_DIR}/dft/dft_gradient.cpp
    ${SRC_DIR}/dft/dft_gradient.h
    ${SRC_DIR}/dft/driver.cpp
    ${SRC_DIR}/dft/driver.h
    ${SRC_DIR}/dft/ks_matrix.cpp
    ${SRC_DIR}/dft/ks_matrix.h
    ${SRC_DIR}/dft/xc_grid.cpp
    ${SRC_DIR}/dft/xc_grid.h
    ${SRC_DIR}/dft/base/angular.h
    ${SRC_DIR}/dft/base/radial.h
    ${SRC_DIR}/dft/base/grid.h
    ${SRC_DIR}/dft/base/wrapper.h
)
set(DFT_SRC
    ${SRC_DIR}/dft/main.cpp
    ${DFT_LIB_SRC}
)

# Executable target
add_executable(hartree-fock
    ${BASE_SRC}
    ${HF_MAIN_SRC}
    ${LOOKUP_SRC}
    ${SYMM_SRC}
    ${BASIS_SRC}
    ${INT_SRC}
    ${IO_SRC}
    ${SCF_SRC}
    ${SOLVATION_SRC}
    ${BSSE_SRC}
    ${POPULATIONS_SRC}
    ${POSTHF_SRC}
    ${GRADIENT_SRC}
    ${OPT_SRC}
    ${FREQ_SRC}
)

target_include_directories(hartree-fock PUBLIC
    ${SRC_DIR}
    ${CMAKE_BINARY_DIR}
    ${MSYM_INSTALL_DIR}/include
    ${eigen_SOURCE_DIR}
)

target_link_libraries(hartree-fock
    ${MSYM_INSTALL_DIR}/lib/libmsym.a
)

add_dependencies(hartree-fock libmsym)

# OpenMP linkage
if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(hartree-fock OpenMP::OpenMP_CXX)
    target_compile_definitions(hartree-fock PRIVATE USE_OPENMP)
endif()
target_compile_definitions(hartree-fock PRIVATE PLANCK_CC_MAXORDER=${PLANCK_CC_MAXORDER})

add_executable(planck-dft
    ${BASE_SRC}
    ${DFT_SRC}
    ${LOOKUP_SRC}
    ${SYMM_SRC}
    ${BASIS_SRC}
    ${INT_SRC}
    ${IO_SRC}
    ${SCF_SRC}
    ${SOLVATION_SRC}
    ${POPULATIONS_SRC}
    ${POSTHF_DFT_SRC}
    ${GRADIENT_SRC}
    ${OPT_SRC}
    ${FREQ_SRC}
)

if(Python3_Interpreter_FOUND)
    set(CCGEN_SPINORBITAL_WARM_START_INC
        ${CMAKE_BINARY_DIR}/generated/cc/ccsd_spinorbital_warm_start.inc)

    add_custom_command(
        OUTPUT ${CCGEN_SPINORBITAL_WARM_START_INC}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/generated/cc
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/python/generate_spinorbital_ccsd_warm_start.py
                --output ${CCGEN_SPINORBITAL_WARM_START_INC}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python
        DEPENDS
            ${CMAKE_SOURCE_DIR}/python/generate_spinorbital_ccsd_warm_start.py
            ${CMAKE_SOURCE_DIR}/python/ccgen/generate.py
            ${CMAKE_SOURCE_DIR}/python/ccgen/project.py
            ${CMAKE_SOURCE_DIR}/python/ccgen/wick.py
        COMMENT "Generating spin-orbital CCSD warm-start kernels with ccgen"
    )

    add_custom_target(ccgen-spinorbital-warm-start
        DEPENDS ${CCGEN_SPINORBITAL_WARM_START_INC}
    )

    set(_ccgen_planck_kernel_args
        --output-dir ${CMAKE_BINARY_DIR}/generated/cc
        --methods ${PLANCK_CC_METHODS}
        --intermediate-threshold ${PLANCK_CC_INTERMEDIATE_THRESHOLD}
    )
    if(PLANCK_CC_INCLUDE_INTERMEDIATES)
        list(APPEND _ccgen_planck_kernel_args --include-intermediates)
    endif()

    add_custom_target(ccgen-planck-kernels
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/generated/cc
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/python/generate_planck_cc_kernels.py
                ${_ccgen_planck_kernel_args}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python
        COMMENT "Generating Planck-compatible CC kernels with ccgen (up to rank ${PLANCK_CC_MAXORDER}: ${PLANCK_CC_METHODS})"
    )

    message(STATUS "ccgen     : emitting Planck CC kernels up to rank ${PLANCK_CC_MAXORDER} [${PLANCK_CC_METHODS}]")
endif()

if(Python3_Interpreter_FOUND)
    add_dependencies(hartree-fock ccgen-spinorbital-warm-start)
    add_dependencies(hartree-fock ccgen-planck-kernels)
endif()

target_include_directories(planck-dft PUBLIC
    ${SRC_DIR}
    ${CMAKE_BINARY_DIR}
    ${MSYM_INSTALL_DIR}/include
    ${LIBXC_INSTALL_DIR}/include
    ${eigen_SOURCE_DIR}
)

target_link_libraries(planck-dft
    ${MSYM_INSTALL_DIR}/lib/libmsym.a
    ${LIBXC_INSTALL_DIR}/lib/libxc.a
)

add_dependencies(planck-dft libmsym libxc)
if(Python3_Interpreter_FOUND)
    add_dependencies(planck-dft ccgen-spinorbital-warm-start)
    add_dependencies(planck-dft ccgen-planck-kernels)
endif()
target_compile_definitions(planck-dft PRIVATE USING_Libxc PLANCK_CC_MAXORDER=${PLANCK_CC_MAXORDER})

if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(planck-dft OpenMP::OpenMP_CXX)
    target_compile_definitions(planck-dft PRIVATE USE_OPENMP)
endif()

# Install executable
install(TARGETS hartree-fock DESTINATION bin)
install(TARGETS planck-dft DESTINATION bin)

# ── planck-mpi: unified hybrid MPI/OpenMP front end (opt-in) ───────────────────
# One binary that dispatches HF vs DFT on calculator.is_dft_run(). Built when
# BUILD_MPI=ON. If MPI is found it links MPI and defines USE_MPI; otherwise it
# still builds as a plain serial binary so the dispatch logic is testable without
# an MPI toolchain (single rank 0). See docs/HPC_MPI_EXECUTABLE_SCOPE.md.
option(BUILD_MPI "Build the unified planck-mpi front end" OFF)
if(BUILD_MPI)
    add_executable(planck-mpi
        ${SRC_DIR}/mpi/main.cpp
        ${HF_DRIVER_SRC}
        ${DFT_LIB_SRC}
        ${BASE_SRC}
        ${LOOKUP_SRC}
        ${SYMM_SRC}
        ${BASIS_SRC}
        ${INT_SRC}
        ${IO_SRC}
        ${SCF_SRC}
        ${SOLVATION_SRC}
        ${BSSE_SRC}
        ${POPULATIONS_SRC}
        ${POSTHF_SRC}
        ${GRADIENT_SRC}
        ${OPT_SRC}
        ${FREQ_SRC}
    )

    target_include_directories(planck-mpi PUBLIC
        ${SRC_DIR}
        ${CMAKE_BINARY_DIR}
        ${MSYM_INSTALL_DIR}/include
        ${LIBXC_INSTALL_DIR}/include
        ${eigen_SOURCE_DIR}
    )

    target_link_libraries(planck-mpi
        ${MSYM_INSTALL_DIR}/lib/libmsym.a
        ${LIBXC_INSTALL_DIR}/lib/libxc.a
    )
    add_dependencies(planck-mpi libmsym libxc)
    if(Python3_Interpreter_FOUND)
        add_dependencies(planck-mpi ccgen-spinorbital-warm-start)
        add_dependencies(planck-mpi ccgen-planck-kernels)
    endif()
    # USING_Libxc enables the libxc-backed XC path (same as planck-dft).
    target_compile_definitions(planck-mpi PRIVATE USING_Libxc PLANCK_CC_MAXORDER=${PLANCK_CC_MAXORDER})

    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(planck-mpi OpenMP::OpenMP_CXX)
        target_compile_definitions(planck-mpi PRIVATE USE_OPENMP)
    endif()

    find_package(MPI QUIET)
    if(MPI_CXX_FOUND)
        target_link_libraries(planck-mpi MPI::MPI_CXX)
        target_compile_definitions(planck-mpi PRIVATE USE_MPI)
        message(STATUS "planck-mpi : MPI found — hybrid MPI/OpenMP build")
    else()
        message(STATUS "planck-mpi : MPI not found — serial dispatch build (single rank)")
    endif()

    install(TARGETS planck-mpi DESTINATION bin)
endif()

# ── Optional utility tools ────────────────────────────────────────────────────
option(BUILD_TOOLS "Build utility tools (chkdump checkpoint formatter)" ON)

if(BUILD_TOOLS)
    add_executable(chkdump
        ${CMAKE_SOURCE_DIR}/tools/chkdump.cpp
    )

    target_include_directories(chkdump PRIVATE
        ${eigen_SOURCE_DIR}
    )

    set_target_properties(chkdump PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    install(TARGETS chkdump DESTINATION bin)

    message(STATUS "Tools : chkdump enabled")
else()
    message(STATUS "Tools : disabled (BUILD_TOOLS=OFF)")
endif()

find_package(Python3 COMPONENTS Interpreter)

if(BUILD_TESTING)
    add_executable(planck-casscf-internal
        ${CMAKE_SOURCE_DIR}/tests/casscf_internal.cpp
        ${SRC_DIR}/post_hf/ci/ci.cpp
        ${SRC_DIR}/post_hf/casscf/orbital.cpp
        ${SRC_DIR}/post_hf/ci/rdm.cpp
        ${SRC_DIR}/post_hf/casscf/response.cpp
        ${SRC_DIR}/post_hf/ci/strings.cpp
        ${SRC_DIR}/post_hf/integrals.cpp
        # integrals.cpp now references the RI transform (transform_eri_ri), which
        # lives in ri_eri.cpp — link it plus its rifit/basis deps even though this
        # test only exercises the dense path.
        ${SRC_DIR}/post_hf/ri/ri_eri.cpp
        ${SRC_DIR}/basis/rifit.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${INT_SRC}
        ${LOOKUP_SRC}
        ${SYMM_SRC}
        # SYMM_SRC (mo_symmetry.cpp) now calls BasisFunctions::cart_to_sph_block,
        # defined in src/basis/spherical.cpp — link it (Eigen-only, no msym/xc dep).
        # spherical.cpp in turn delegates L=6 to cart_to_sph_block_recurrence
        # (spherical_recurrence.cpp), so that must be linked too.
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
    )

    add_executable(planck-dft-grid-invariants
        ${CMAKE_SOURCE_DIR}/tests/dft_grid_invariants.cpp
    )

    add_executable(planck-dft-functional-support
        ${CMAKE_SOURCE_DIR}/tests/dft_functional_support.cpp
    )

    add_executable(planck-eri-derivative-kernels
        ${CMAKE_SOURCE_DIR}/tests/eri_derivative_kernels.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${SRC_DIR}/integrals/hgp.cpp
        ${SRC_DIR}/integrals/os.cpp
        ${SRC_DIR}/integrals/rys.cpp
        ${SRC_DIR}/integrals/rys_roots.cpp
        ${SRC_DIR}/integrals/screening.cpp
        ${SRC_DIR}/integrals/shellpair.cpp
        ${SRC_DIR}/lookup/elements.cpp
        ${SRC_DIR}/lookup/boys.cpp
    )

    add_executable(planck-multipole-integrals
        ${CMAKE_SOURCE_DIR}/tests/multipole_integrals.cpp
        ${INT_SRC}
        ${POPULATIONS_SRC}
        ${LOOKUP_SRC}
    )

    add_executable(planck-hgp-engine-smoke
        ${CMAKE_SOURCE_DIR}/tests/hgp_engine_smoke.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${INT_SRC}
        ${LOOKUP_SRC}
    )

    # Characterization test for the one-shot full ERI tensor build
    # (ObaraSaika::_compute_2e), ahead of flattening its triangular parallel
    # loop. Pins permutational symmetry + golden checksum/elements.
    add_executable(planck-compute-2e
        ${CMAKE_SOURCE_DIR}/tests/compute_2e_unit.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${INT_SRC}
        ${LOOKUP_SRC}
    )

    add_executable(planck-hgp-quartet-validation
        ${CMAKE_SOURCE_DIR}/tests/hgp_quartet_validation.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${INT_SRC}
        ${LOOKUP_SRC}
    )

    add_executable(planck-os-block-kernel
        ${CMAKE_SOURCE_DIR}/tests/os_block_kernel.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${INT_SRC}
        ${LOOKUP_SRC}
    )

    add_executable(planck-hgp-triangular-contract
        ${CMAKE_SOURCE_DIR}/tests/hgp_triangular_contract.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${INT_SRC}
        ${LOOKUP_SRC}
    )

    add_executable(planck-rys-box-invariance
        ${CMAKE_SOURCE_DIR}/tests/rys_box_invariance.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${INT_SRC}
        ${LOOKUP_SRC}
    )

    add_executable(planck-hgp-benchmark
        ${CMAKE_SOURCE_DIR}/tests/hgp_benchmark.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${INT_SRC}
        ${LOOKUP_SRC}
    )

    add_executable(planck-auto-dispatch-benchmark
        ${CMAKE_SOURCE_DIR}/tests/auto_dispatch_benchmark.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${INT_SRC}
        ${LOOKUP_SRC}
    )

    add_executable(planck-population-analysis
        ${CMAKE_SOURCE_DIR}/tests/population_analysis.cpp
        ${POPULATIONS_SRC}
    )

    add_executable(planck-symmetry-units
        ${CMAKE_SOURCE_DIR}/tests/symmetry_units.cpp
        ${SRC_DIR}/symmetry/symmetry.cpp
    )

    add_executable(planck-diis-conditioning
        ${CMAKE_SOURCE_DIR}/tests/diis_conditioning.cpp
    )

    add_executable(planck-cc-arbitrary-solver
        ${CMAKE_SOURCE_DIR}/tests/cc_arbitrary_solver.cpp
        ${SRC_DIR}/post_hf/cc/common.cpp
        ${SRC_DIR}/post_hf/cc/amplitudes.cpp
        ${SRC_DIR}/post_hf/cc/diis.cpp
        ${SRC_DIR}/post_hf/cc/generated_arbitrary_runtime.cpp
        ${SRC_DIR}/post_hf/cc/solver_arbitrary.cpp
    )

    add_executable(planck-spherical-transform
        ${CMAKE_SOURCE_DIR}/tests/spherical_transform.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
    )

    add_executable(planck-spherical-density-lift
        ${CMAKE_SOURCE_DIR}/tests/spherical_density_lift.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
    )

    # Characterization test for the whole-ERI Cartesian→spherical transform.
    # Links only the spherical transform sources (lightweight).
    add_executable(planck-transform-eri-sph
        ${CMAKE_SOURCE_DIR}/tests/transform_eri_sph_unit.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
    )

    # Characterization test for the AO→MO four-index transform (transform_eri).
    # Pulls the integral sources because integrals.cpp shares a TU with
    # ensure_eri/_compute_2e; only transform_eri is exercised by the test.
    add_executable(planck-transform-eri
        ${CMAKE_SOURCE_DIR}/tests/transform_eri_unit.cpp
        ${SRC_DIR}/post_hf/integrals.cpp
        # integrals.cpp references transform_eri_ri (ri_eri.cpp) since Step 2.
        ${SRC_DIR}/post_hf/ri/ri_eri.cpp
        ${SRC_DIR}/basis/rifit.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${INT_SRC}
        ${LOOKUP_SRC}
    )

    # Structural test for the RI auxiliary basis loader (no PySCF reference
    # values — this is the file-I/O and shell-construction gate, not a
    # validation against any specific shipped aux basis).
    add_executable(planck-aux-basis-load
        ${CMAKE_SOURCE_DIR}/tests/aux_basis_load.cpp
        ${SRC_DIR}/basis/rifit.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/lookup/elements.cpp
    )

    # Analytical + structural test for the RI 2-center Coulomb metric (P|Q).
    # Validates against closed-form s-Gaussian integrals plus symmetry /
    # positive-definiteness on the shipped cc-pVDZ-RIFIT aux basis.
    add_executable(planck-ri-2c-eri
        ${CMAKE_SOURCE_DIR}/tests/ri_2c_eri.cpp
        ${SRC_DIR}/post_hf/ri/ri_eri.cpp
        ${SRC_DIR}/basis/rifit.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/integrals/os.cpp
        ${SRC_DIR}/integrals/hgp.cpp
        ${SRC_DIR}/integrals/rys.cpp
        ${SRC_DIR}/integrals/rys_roots.cpp
        ${SRC_DIR}/integrals/screening.cpp
        ${SRC_DIR}/integrals/shellpair.cpp
        ${SRC_DIR}/lookup/elements.cpp
        ${SRC_DIR}/lookup/boys.cpp
    )

    # Native RI 3-center (mu nu | P) gate, validated against local PySCF.
    add_executable(planck-ri-3c-eri
        ${CMAKE_SOURCE_DIR}/tests/ri_3c_eri.cpp
        ${SRC_DIR}/post_hf/ri/ri_eri.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/rifit.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${SRC_DIR}/integrals/os.cpp
        ${SRC_DIR}/integrals/hgp.cpp
        ${SRC_DIR}/integrals/rys.cpp
        ${SRC_DIR}/integrals/rys_roots.cpp
        ${SRC_DIR}/integrals/screening.cpp
        ${SRC_DIR}/integrals/shellpair.cpp
        ${SRC_DIR}/lookup/elements.cpp
        ${SRC_DIR}/lookup/boys.cpp
    )

    # Step 2 seam: transform_eri_ri must reproduce the dense transform_eri.
    # Self-contained (no PySCF), so it is registered as a ctest below.
    add_executable(planck-ri-transform-equivalence
        ${CMAKE_SOURCE_DIR}/tests/ri_transform_equivalence.cpp
        ${SRC_DIR}/post_hf/integrals.cpp
        ${SRC_DIR}/post_hf/ri/ri_eri.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/rifit.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${SRC_DIR}/integrals/os.cpp
        ${SRC_DIR}/integrals/hgp.cpp
        ${SRC_DIR}/integrals/rys.cpp
        ${SRC_DIR}/integrals/rys_roots.cpp
        ${SRC_DIR}/integrals/screening.cpp
        ${SRC_DIR}/integrals/shellpair.cpp
        ${SRC_DIR}/lookup/elements.cpp
        ${SRC_DIR}/lookup/boys.cpp
    )

    # Step J0: RI-JK builder harness. Same self-contained fixture; the RI J/K
    # equivalence assertions land in J1/J3.
    add_executable(planck-ri-jk-equivalence
        ${CMAKE_SOURCE_DIR}/tests/ri_jk_equivalence.cpp
        ${SRC_DIR}/post_hf/ri/ri_eri.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/rifit.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${SRC_DIR}/integrals/os.cpp
        ${SRC_DIR}/integrals/hgp.cpp
        ${SRC_DIR}/integrals/rys.cpp
        ${SRC_DIR}/integrals/rys_roots.cpp
        ${SRC_DIR}/integrals/screening.cpp
        ${SRC_DIR}/integrals/shellpair.cpp
        ${SRC_DIR}/lookup/elements.cpp
        ${SRC_DIR}/lookup/boys.cpp
    )

    # G0/G1: RI caches must rebuild when the geometry moves.
    add_executable(planck-ri-cache-invalidation
        ${CMAKE_SOURCE_DIR}/tests/ri_cache_invalidation.cpp
        ${SRC_DIR}/post_hf/ri/ri_eri.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/rifit.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${SRC_DIR}/integrals/os.cpp
        ${SRC_DIR}/integrals/hgp.cpp
        ${SRC_DIR}/integrals/rys.cpp
        ${SRC_DIR}/integrals/rys_roots.cpp
        ${SRC_DIR}/integrals/screening.cpp
        ${SRC_DIR}/integrals/shellpair.cpp
        ${SRC_DIR}/lookup/elements.cpp
        ${SRC_DIR}/lookup/boys.cpp
    )

    # RG0: finite-difference oracle for the 3-center derivative d/dR (μν|Q),
    # the reference the analytic RI-MP2-gradient derivative (RG1) is gated on.
    add_executable(planck-ri-3c-derivative-fd
        ${CMAKE_SOURCE_DIR}/tests/ri_3c_derivative_fd.cpp
        ${SRC_DIR}/post_hf/ri/ri_eri.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/rifit.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${SRC_DIR}/integrals/os.cpp
        ${SRC_DIR}/integrals/hgp.cpp
        ${SRC_DIR}/integrals/rys.cpp
        ${SRC_DIR}/integrals/rys_roots.cpp
        ${SRC_DIR}/integrals/screening.cpp
        ${SRC_DIR}/integrals/shellpair.cpp
        ${SRC_DIR}/lookup/elements.cpp
        ${SRC_DIR}/lookup/boys.cpp
    )

    # Contract test for the working-state rebuild helper that will drive
    # spherical geomopt and freq in Phase 3. Currently the helper has no live
    # caller (the driver keeps its inline copy for ordering reasons around the
    # checkpoint path); this test is the contract that pins its behavior.
    add_executable(planck-working-state-rebuild
        ${CMAKE_SOURCE_DIR}/tests/working_state_rebuild.cpp
        ${SRC_DIR}/scf/working_state.cpp
        ${SRC_DIR}/symmetry/symmetry.cpp
        ${SRC_DIR}/symmetry/integral_symmetry.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${SRC_DIR}/integrals/os.cpp
        ${SRC_DIR}/integrals/hgp.cpp
        ${SRC_DIR}/integrals/rys.cpp
        ${SRC_DIR}/integrals/rys_roots.cpp
        ${SRC_DIR}/integrals/screening.cpp
        ${SRC_DIR}/integrals/shellpair.cpp
        ${SRC_DIR}/lookup/elements.cpp
        ${SRC_DIR}/lookup/boys.cpp
    )

    add_executable(planck-group-operations
        ${CMAKE_SOURCE_DIR}/tests/group_operations.cpp
        ${SRC_DIR}/symmetry/group_operations.cpp
        ${SRC_DIR}/symmetry/symmetry.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${SRC_DIR}/lookup/elements.cpp
        # group_operations.cpp now computes the Cartesian overlap to correct O_R into
        # the spherical AO basis (Step 1', spherical full-symmetry), so it needs the
        # integral engine.
        ${SRC_DIR}/integrals/os.cpp
        ${SRC_DIR}/integrals/hgp.cpp
        ${SRC_DIR}/integrals/rys.cpp
        ${SRC_DIR}/integrals/rys_roots.cpp
        ${SRC_DIR}/integrals/screening.cpp
        ${SRC_DIR}/integrals/shellpair.cpp
        ${SRC_DIR}/lookup/boys.cpp
    )

    add_executable(planck-fock-symmetrization
        ${CMAKE_SOURCE_DIR}/tests/fock_symmetrization.cpp
        ${SRC_DIR}/symmetry/fock_symmetrization.cpp
        ${SRC_DIR}/symmetry/group_operations.cpp
        ${SRC_DIR}/symmetry/symmetry.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
        ${SRC_DIR}/lookup/elements.cpp
        # group_operations.cpp needs the integral engine (Cartesian overlap) for the
        # spherical O_R correction (Step 1').
        ${SRC_DIR}/integrals/os.cpp
        ${SRC_DIR}/integrals/hgp.cpp
        ${SRC_DIR}/integrals/rys.cpp
        ${SRC_DIR}/integrals/rys_roots.cpp
        ${SRC_DIR}/integrals/screening.cpp
        ${SRC_DIR}/integrals/shellpair.cpp
        ${SRC_DIR}/lookup/boys.cpp
    )

    add_executable(planck-symm-fock-benchmark
        ${CMAKE_SOURCE_DIR}/tests/symm_fock_benchmark.cpp
        ${INT_SRC}
        ${LOOKUP_SRC}
        ${SRC_DIR}/symmetry/hgp_symm.cpp
        ${SRC_DIR}/symmetry/os_symm.cpp
        ${SRC_DIR}/symmetry/rys_symm.cpp
        ${SRC_DIR}/symmetry/fock_symmetrization.cpp
        ${SRC_DIR}/symmetry/group_operations.cpp
        ${SRC_DIR}/symmetry/integral_symmetry.cpp
        ${SRC_DIR}/symmetry/symmetry.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
    )

    add_executable(planck-symm-fock-equivalence
        ${CMAKE_SOURCE_DIR}/tests/symm_fock_equivalence.cpp
        ${INT_SRC}
        ${LOOKUP_SRC}
        ${SRC_DIR}/symmetry/os_symm.cpp
        ${SRC_DIR}/symmetry/rys_symm.cpp
        ${SRC_DIR}/symmetry/fock_symmetrization.cpp
        ${SRC_DIR}/symmetry/group_operations.cpp
        ${SRC_DIR}/symmetry/symmetry.cpp
        ${SRC_DIR}/basis/basis.cpp
        ${SRC_DIR}/basis/gaussian.cpp
        ${SRC_DIR}/basis/gbs_parser.cpp
        ${SRC_DIR}/basis/spherical.cpp
        ${SRC_DIR}/basis/spherical_recurrence.cpp
    )

    target_include_directories(planck-casscf-internal PRIVATE
        ${SRC_DIR}
        ${MSYM_INSTALL_DIR}/include
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-dft-grid-invariants PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-dft-functional-support PRIVATE
        ${SRC_DIR}
        ${LIBXC_INSTALL_DIR}/include
    )

    target_include_directories(planck-eri-derivative-kernels PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-multipole-integrals PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-hgp-engine-smoke PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-compute-2e PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-hgp-quartet-validation PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-os-block-kernel PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-hgp-triangular-contract PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-rys-box-invariance PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-hgp-benchmark PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-auto-dispatch-benchmark PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-population-analysis PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-symmetry-units PRIVATE
        ${SRC_DIR}
        ${MSYM_INSTALL_DIR}/include
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-diis-conditioning PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-cc-arbitrary-solver PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-group-operations PRIVATE
        ${SRC_DIR}
        ${MSYM_INSTALL_DIR}/include
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-fock-symmetrization PRIVATE
        ${SRC_DIR}
        ${MSYM_INSTALL_DIR}/include
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-symm-fock-benchmark PRIVATE
        ${SRC_DIR}
        ${MSYM_INSTALL_DIR}/include
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-symm-fock-equivalence PRIVATE
        ${SRC_DIR}
        ${MSYM_INSTALL_DIR}/include
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-spherical-transform PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-spherical-density-lift PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-transform-eri-sph PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-transform-eri PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-aux-basis-load PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-ri-2c-eri PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-ri-3c-eri PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-ri-transform-equivalence PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-ri-jk-equivalence PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-ri-cache-invalidation PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-ri-3c-derivative-fd PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-working-state-rebuild PRIVATE
        ${SRC_DIR}
        ${MSYM_INSTALL_DIR}/include
        ${eigen_SOURCE_DIR}
    )

    set_target_properties(planck-casscf-internal PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-dft-grid-invariants PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-dft-functional-support PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-eri-derivative-kernels PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-multipole-integrals PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-hgp-engine-smoke PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-compute-2e PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-hgp-quartet-validation PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-os-block-kernel PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-hgp-triangular-contract PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-rys-box-invariance PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-hgp-benchmark PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-auto-dispatch-benchmark PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-population-analysis PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-diis-conditioning PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-symmetry-units PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-cc-arbitrary-solver PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-spherical-transform PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-spherical-density-lift PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-transform-eri-sph PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-transform-eri PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-aux-basis-load PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-ri-2c-eri PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-working-state-rebuild PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-group-operations PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-fock-symmetrization PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-symm-fock-equivalence PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(planck-multipole-integrals OpenMP::OpenMP_CXX)
        target_compile_definitions(planck-multipole-integrals PRIVATE USE_OPENMP)
    endif()
    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(planck-hgp-engine-smoke OpenMP::OpenMP_CXX)
        target_compile_definitions(planck-hgp-engine-smoke PRIVATE USE_OPENMP)
    endif()
    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(planck-compute-2e OpenMP::OpenMP_CXX)
        target_compile_definitions(planck-compute-2e PRIVATE USE_OPENMP)
    endif()
    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(planck-hgp-quartet-validation OpenMP::OpenMP_CXX)
        target_compile_definitions(planck-hgp-quartet-validation PRIVATE USE_OPENMP)
    endif()
    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(planck-os-block-kernel OpenMP::OpenMP_CXX)
        target_compile_definitions(planck-os-block-kernel PRIVATE USE_OPENMP)
    endif()
    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(planck-hgp-triangular-contract OpenMP::OpenMP_CXX)
        target_compile_definitions(planck-hgp-triangular-contract PRIVATE USE_OPENMP)
    endif()
    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(planck-rys-box-invariance OpenMP::OpenMP_CXX)
        target_compile_definitions(planck-rys-box-invariance PRIVATE USE_OPENMP)
    endif()
    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(planck-hgp-benchmark OpenMP::OpenMP_CXX)
        target_compile_definitions(planck-hgp-benchmark PRIVATE USE_OPENMP)
    endif()
    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(planck-auto-dispatch-benchmark OpenMP::OpenMP_CXX)
        target_compile_definitions(planck-auto-dispatch-benchmark PRIVATE USE_OPENMP)
    endif()
    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(planck-transform-eri OpenMP::OpenMP_CXX)
        target_compile_definitions(planck-transform-eri PRIVATE USE_OPENMP)
    endif()
    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(planck-transform-eri-sph OpenMP::OpenMP_CXX)
        target_compile_definitions(planck-transform-eri-sph PRIVATE USE_OPENMP)
    endif()

    target_link_libraries(planck-casscf-internal
        ${MSYM_INSTALL_DIR}/lib/libmsym.a
    )
    add_dependencies(planck-casscf-internal libmsym)

    target_link_libraries(planck-symmetry-units
        ${MSYM_INSTALL_DIR}/lib/libmsym.a
    )
    add_dependencies(planck-symmetry-units libmsym)

    target_link_libraries(planck-group-operations
        ${MSYM_INSTALL_DIR}/lib/libmsym.a
    )
    add_dependencies(planck-group-operations libmsym)

    target_link_libraries(planck-working-state-rebuild
        ${MSYM_INSTALL_DIR}/lib/libmsym.a
    )
    add_dependencies(planck-working-state-rebuild libmsym)

    target_link_libraries(planck-fock-symmetrization
        ${MSYM_INSTALL_DIR}/lib/libmsym.a
    )
    add_dependencies(planck-fock-symmetrization libmsym)

    target_link_libraries(planck-symm-fock-benchmark
        ${MSYM_INSTALL_DIR}/lib/libmsym.a
    )
    add_dependencies(planck-symm-fock-benchmark libmsym)
    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(planck-symm-fock-benchmark OpenMP::OpenMP_CXX)
        target_compile_definitions(planck-symm-fock-benchmark PRIVATE USE_OPENMP)
    endif()

    target_link_libraries(planck-symm-fock-equivalence
        ${MSYM_INSTALL_DIR}/lib/libmsym.a
    )
    add_dependencies(planck-symm-fock-equivalence libmsym)
    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(planck-symm-fock-equivalence OpenMP::OpenMP_CXX)
        target_compile_definitions(planck-symm-fock-equivalence PRIVATE USE_OPENMP)
    endif()

    target_link_libraries(planck-dft-functional-support
        ${LIBXC_INSTALL_DIR}/lib/libxc.a
    )
    add_dependencies(planck-dft-functional-support libxc)

    add_test(
        NAME planck-casscf-internal
        COMMAND planck-casscf-internal
    )

    add_test(
        NAME planck-dft-grid-invariants
        COMMAND planck-dft-grid-invariants
    )

    add_test(
        NAME planck-dft-functional-support
        COMMAND planck-dft-functional-support
    )

    add_test(
        NAME planck-eri-derivative-kernels
        COMMAND planck-eri-derivative-kernels
    )

    add_test(
        NAME planck-multipole-integrals
        COMMAND planck-multipole-integrals
    )

    add_test(
        NAME planck-hgp-engine-smoke
        COMMAND planck-hgp-engine-smoke
    )

    add_test(
        NAME planck-compute-2e
        COMMAND planck-compute-2e
    )

    add_test(
        NAME planck-hgp-quartet-validation
        COMMAND planck-hgp-quartet-validation
    )

    add_test(
        NAME planck-os-block-kernel
        COMMAND planck-os-block-kernel
    )

    add_test(
        NAME planck-hgp-triangular-contract
        COMMAND planck-hgp-triangular-contract
    )

    add_test(
        NAME planck-rys-box-invariance
        COMMAND planck-rys-box-invariance
    )

    add_test(
        NAME planck-population-analysis
        COMMAND planck-population-analysis
    )

    add_test(
        NAME planck-diis-conditioning
        COMMAND planck-diis-conditioning
    )

    add_test(
        NAME planck-symmetry-units
        COMMAND planck-symmetry-units
    )

    add_test(
        NAME planck-cc-arbitrary-solver
        COMMAND planck-cc-arbitrary-solver
    )

    add_test(
        NAME planck-spherical-transform
        COMMAND planck-spherical-transform
    )

    add_test(
        NAME planck-spherical-density-lift
        COMMAND planck-spherical-density-lift
    )

    add_test(
        NAME planck-transform-eri-sph
        COMMAND planck-transform-eri-sph
    )

    add_test(
        NAME planck-transform-eri
        COMMAND planck-transform-eri
    )

    # These two read RI auxiliary basis files via the relative path
    # `basis-sets/...`, which only resolves from the repository root, so run
    # them from there (matching the regression tests below).
    add_test(
        NAME planck-aux-basis-load
        COMMAND planck-aux-basis-load
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME planck-ri-2c-eri
        COMMAND planck-ri-2c-eri
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME planck-ri-transform-equivalence
        COMMAND planck-ri-transform-equivalence
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME planck-ri-jk-equivalence
        COMMAND planck-ri-jk-equivalence
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME planck-ri-cache-invalidation
        COMMAND planck-ri-cache-invalidation
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME planck-ri-3c-derivative-fd
        COMMAND planck-ri-3c-derivative-fd
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME planck-working-state-rebuild
        COMMAND planck-working-state-rebuild
    )

    add_test(
        NAME planck-group-operations
        COMMAND planck-group-operations
    )

    add_test(
        NAME planck-fock-symmetrization
        COMMAND planck-fock-symmetrization
    )

    add_test(
        NAME planck-symm-fock-equivalence
        COMMAND planck-symm-fock-equivalence
    )
endif()

if(BUILD_TESTING AND Python3_Interpreter_FOUND)
    set(PLANCK_PYSCF_PYTHON "" CACHE FILEPATH
        "Python interpreter with PySCF installed for tests/pyscf")
    set(_planck_pyscf_default "${CMAKE_SOURCE_DIR}/tests/pyscf/.venv/bin/python")
    if(NOT PLANCK_PYSCF_PYTHON)
        if(EXISTS "${_planck_pyscf_default}")
            set(PLANCK_PYSCF_PYTHON "${_planck_pyscf_default}")
        else()
            set(PLANCK_PYSCF_PYTHON "${Python3_EXECUTABLE}")
        endif()
    endif()

    execute_process(
        COMMAND ${PLANCK_PYSCF_PYTHON} -c "import pyscf"
        RESULT_VARIABLE PLANCK_PYSCF_IMPORT_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )

    add_test(
        NAME planck-regression-smoke
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tests/run_regressions.py
                --build-dir ${CMAKE_BINARY_DIR}
                --suite smoke
    )

    add_test(
        NAME planck-dft-gradient-fd
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tests/dft_gradient_fd.py
                --build-dir ${CMAKE_BINARY_DIR}
    )

    add_test(
        NAME planck-regression-core
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tests/run_regressions.py
                --build-dir ${CMAKE_BINARY_DIR}
                --suite core
    )

    add_test(
        NAME planck-regression-extended
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tests/run_regressions.py
                --build-dir ${CMAKE_BINARY_DIR}
                --suite extended
    )

    add_custom_target(regression-smoke
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tests/run_regressions.py
                --build-dir ${CMAKE_BINARY_DIR}
                --suite smoke
        DEPENDS hartree-fock
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        USES_TERMINAL
    )

    add_custom_target(regression-core
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tests/run_regressions.py
                --build-dir ${CMAKE_BINARY_DIR}
                --suite core
        DEPENDS hartree-fock
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        USES_TERMINAL
    )

    add_custom_target(regression-extended
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tests/run_regressions.py
                --build-dir ${CMAKE_BINARY_DIR}
                --suite extended
        DEPENDS hartree-fock
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        USES_TERMINAL
    )

    add_custom_target(teaching-site
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/docs/build_teaching_site.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        USES_TERMINAL
    )

    if(PLANCK_PYSCF_IMPORT_RESULT EQUAL 0)
        add_test(
            NAME planck-pyscf-equivalence
            COMMAND ${PLANCK_PYSCF_PYTHON}
                    ${CMAKE_SOURCE_DIR}/tests/pyscf/run_all.py
                    --python ${PLANCK_PYSCF_PYTHON}
        )

        add_custom_target(pyscf-equivalence
            COMMAND ${PLANCK_PYSCF_PYTHON}
                    ${CMAKE_SOURCE_DIR}/tests/pyscf/run_all.py
                    --python ${PLANCK_PYSCF_PYTHON}
            DEPENDS hartree-fock
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            USES_TERMINAL
        )
    else()
        message(STATUS
            "PySCF tests : skipped (set PLANCK_PYSCF_PYTHON to a Python with PySCF installed)")
    endif()
endif()

# ── Basis set fetcher ─────────────────────────────────────────────────────────
# Reads basis-sets/basis at configure time; for each name that is not yet on
# disk, emits a custom command that runs:
#   bse get-basis <name> > basis-sets/<name>
# The ALL target runs automatically during the normal build; already-present
# files are never re-fetched.  Manual trigger: cmake --build build --target fetch-basis-sets
# file(STRINGS "${CMAKE_SOURCE_DIR}/basis-sets/basis" _BASIS_NAMES)

# set(_BASIS_OUTPUTS "")
# foreach(_bname IN LISTS _BASIS_NAMES)
#     string(STRIP "${_bname}" _bname)
#     if(_bname STREQUAL "")
#         continue()
#     endif()
#     set(_bfile "${CMAKE_SOURCE_DIR}/basis-sets/${_bname}")
#     add_custom_command(
#         OUTPUT  "${_bfile}"
#         COMMAND ${CMAKE_COMMAND}
#                 -D BSE_NAME=${_bname}
#                 -D BSE_OUT=${_bfile}
#                 -P ${CMAKE_SOURCE_DIR}/cmake/fetch_basis.cmake
#         COMMENT "Fetching basis set: ${_bname}"
#         VERBATIM
#     )
#     list(APPEND _BASIS_OUTPUTS "${_bfile}")
# endforeach()

# add_custom_target(fetch-basis-sets ALL
#     DEPENDS ${_BASIS_OUTPUTS}
# )

# add_dependencies(hartree-fock fetch-basis-sets)
