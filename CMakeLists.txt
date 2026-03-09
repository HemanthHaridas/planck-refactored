cmake_minimum_required(VERSION 3.15)
project(planck LANGUAGES CXX)

# C++ standard settings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "Building on : ${CMAKE_SYSTEM_NAME} using following compiler flags : ${CMAKE_CXX_FLAGS}")

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

# Executable target
add_executable(hartree-fock
    ${BASE_SRC}
    ${MAIN_SRC}
    ${LOOKUP_SRC}
    ${SYMM_SRC}
    ${BASIS_SRC}
    ${INT_SRC}
    ${IO_SRC}
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

# Install executable
install(TARGETS hartree-fock DESTINATION bin)
