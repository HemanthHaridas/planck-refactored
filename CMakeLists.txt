cmake_minimum_required(VERSION 3.15)
project(planck LANGUAGES CXX)

# C++ standard settings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "Building on : ${CMAKE_SYSTEM_NAME} using following compiler flags : ${CMAKE_CXX_FLAGS}")

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
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0 # You can also use master
)
FetchContent_MakeAvailable(eigen)

# External project: libmsym
set(MSYM_INSTALL_DIR ${SRC_DIR}/external/libmsym/install)
set(MSYM_BUILD_DIR ${SRC_DIR}/external/libmsym/build)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)

include(ExternalProject)

ExternalProject_Add(
    libmsym
    PREFIX ${SRC_DIR}/external/libmsym
    GIT_REPOSITORY https://github.com/mcodev31/libmsym.git
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${MSYM_INSTALL_DIR} -DBUILD_SHARED_LIBS:BOOL=OFF -DMSYM_BUILD_EXAMPLES:BOOL=OFF
)

# Collect sources automatically
file(GLOB BASE_SRC
    ${SRC_DIR}/base/*.h
    ${SRC_DIR}/base/*.cpp
)

# Collect sources automatically
file(GLOB IO_SRC
    ${SRC_DIR}/io/*.h
    ${SRC_DIR}/io/*.cpp
)

# Collect sources automatically
file(GLOB LOOKUP_SRC
    ${SRC_DIR}/lookup/*.h
    ${SRC_DIR}/lookup/*.cpp
)

# Collect sources automatically
file(GLOB SYMM_SRC
    ${SRC_DIR}/symmetry/*.h
    ${SRC_DIR}/symmetry/*.cpp
)

# Collect sources automatically
file(GLOB MAIN_SRC
    ${SRC_DIR}/*.h
    ${SRC_DIR}/*.cpp
)

# Collect sources automatically
file(GLOB BASIS_SRC
    ${SRC_DIR}/basis/*.h
    ${SRC_DIR}/basis/*.cpp
)

# Collect sources automatically
file(GLOB INT_SRC
    ${SRC_DIR}/integrals/*.h
    ${SRC_DIR}/integrals/*.cpp
)

# Collect sources automatically
file(GLOB SCF_SRC
    ${SRC_DIR}/scf/*.h
    ${SRC_DIR}/scf/*.cpp
)

# Collect sources automatically
file(GLOB POSTHF_SRC
    ${SRC_DIR}/post_hf/*.h
    ${SRC_DIR}/post_hf/*.cpp
)

# Executable target
add_executable(hartree-fock
    ${BASE_SRC}
    ${MAIN_SRC}
    ${LOOKUP_SRC}
    ${SYMM_SRC}
    ${BASIS_SRC}
    ${INT_SRC}
    ${IO_SRC}
    ${SCF_SRC}
    ${POSTHF_SRC}
)

# Include directories
target_include_directories(hartree-fock PUBLIC
    ${SRC_DIR}
    ${MSYM_INSTALL_DIR}/include
    ${eigen_SOURCE_DIR}
)

# Link external library
target_link_libraries(hartree-fock
    ${MSYM_INSTALL_DIR}/lib/libmsym.a
)

# OpenMP linkage
if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(hartree-fock OpenMP::OpenMP_CXX)
    target_compile_definitions(hartree-fock PRIVATE USE_OPENMP)
endif()

# Install executable
install(TARGETS hartree-fock DESTINATION bin)
