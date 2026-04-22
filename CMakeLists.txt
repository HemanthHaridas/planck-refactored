cmake_minimum_required(VERSION 3.5)
project(planck LANGUAGES CXX)

include(CTest)

# C++ standard settings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
    GIT_REPOSITORY https://github.com/ElectronicStructureLibrary/libxc.git
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

# Collect sources automatically
file(GLOB POSTHF_SRC CONFIGURE_DEPENDS
    ${SRC_DIR}/post_hf/*.h
    ${SRC_DIR}/post_hf/*.cpp
    ${SRC_DIR}/post_hf/cc/*.h
    ${SRC_DIR}/post_hf/cc/*.cpp
    ${SRC_DIR}/post_hf/casscf/*.h
    ${SRC_DIR}/post_hf/casscf/*.cpp
)
list(REMOVE_ITEM POSTHF_SRC ${SRC_DIR}/post_hf/casscf.cpp)

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

set(HF_MAIN_SRC
    ${SRC_DIR}/driver.cpp
)

set(DFT_SRC
    ${SRC_DIR}/dft/main.cpp
    ${SRC_DIR}/dft/ao_grid.h
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
    ${POSTHF_SRC}
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

# ── GPU acceleration (optional) ───────────────────────────────────────────────
# Must be added after Eigen is fetched (gpu/CMakeLists.txt uses eigen_SOURCE_DIR)
# and after SRC_DIR is defined, but before the install rules.
if(USE_CUDA)
    add_subdirectory(gpu)
    target_include_directories(hartree-fock PUBLIC ${PLANCK_GPU_INCLUDE})
    target_link_libraries(hartree-fock ${PLANCK_GPU_LIB})
    target_compile_definitions(hartree-fock PRIVATE USE_CUDA)
    message(STATUS "GPU        : CUDA acceleration enabled")
else()
    message(STATUS "GPU        : disabled (USE_CUDA=OFF)")
endif()

# Install executable
install(TARGETS hartree-fock DESTINATION bin)
install(TARGETS planck-dft DESTINATION bin)

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
        ${SRC_DIR}/post_hf/casscf/ci.cpp
        ${SRC_DIR}/post_hf/casscf/orbital.cpp
        ${SRC_DIR}/post_hf/casscf/rdm.cpp
        ${SRC_DIR}/post_hf/casscf/response.cpp
        ${SRC_DIR}/post_hf/casscf/strings.cpp
        ${SRC_DIR}/post_hf/integrals.cpp
        ${INT_SRC}
        ${LOOKUP_SRC}
        ${SYMM_SRC}
    )

    add_executable(planck-dft-grid-invariants
        ${CMAKE_SOURCE_DIR}/tests/dft_grid_invariants.cpp
    )

    add_executable(planck-multipole-integrals
        ${CMAKE_SOURCE_DIR}/tests/multipole_integrals.cpp
        ${INT_SRC}
        ${LOOKUP_SRC}
    )

    add_executable(planck-population-analysis
        ${CMAKE_SOURCE_DIR}/tests/population_analysis.cpp
        ${SRC_DIR}/scf/population.cpp
    )

    add_executable(planck-cc-arbitrary-solver
        ${CMAKE_SOURCE_DIR}/tests/cc_arbitrary_solver.cpp
        ${SRC_DIR}/post_hf/cc/common.cpp
        ${SRC_DIR}/post_hf/cc/amplitudes.cpp
        ${SRC_DIR}/post_hf/cc/diis.cpp
        ${SRC_DIR}/post_hf/cc/generated_arbitrary_runtime.cpp
        ${SRC_DIR}/post_hf/cc/solver_arbitrary.cpp
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

    target_include_directories(planck-multipole-integrals PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-population-analysis PRIVATE
        ${SRC_DIR}
        ${eigen_SOURCE_DIR}
    )

    target_include_directories(planck-cc-arbitrary-solver PRIVATE
        ${SRC_DIR}
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

    set_target_properties(planck-multipole-integrals PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-population-analysis PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    set_target_properties(planck-cc-arbitrary-solver PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(planck-multipole-integrals OpenMP::OpenMP_CXX)
        target_compile_definitions(planck-multipole-integrals PRIVATE USE_OPENMP)
    endif()

    target_link_libraries(planck-casscf-internal
        ${MSYM_INSTALL_DIR}/lib/libmsym.a
    )
    add_dependencies(planck-casscf-internal libmsym)

    add_test(
        NAME planck-casscf-internal
        COMMAND planck-casscf-internal
    )

    add_test(
        NAME planck-dft-grid-invariants
        COMMAND planck-dft-grid-invariants
    )

    add_test(
        NAME planck-multipole-integrals
        COMMAND planck-multipole-integrals
    )

    add_test(
        NAME planck-population-analysis
        COMMAND planck-population-analysis
    )

    add_test(
        NAME planck-cc-arbitrary-solver
        COMMAND planck-cc-arbitrary-solver
    )
endif()

if(BUILD_TESTING AND Python3_Interpreter_FOUND)
    add_test(
        NAME planck-regression-smoke
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tests/run_regressions.py
                --build-dir ${CMAKE_BINARY_DIR}
                --suite smoke
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
