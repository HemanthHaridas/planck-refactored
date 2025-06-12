cmake_minimum_required(VERSION 3.10)
project(planck_cpp)

# Set the minimum C++ standard and use the most recent standard supported by the compiler
set(CMAKE_CXX_STANDARD 20)  # Minimum C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "/opt/homebrew/Cellar/boost/1.87.0_1/include")
# set(CMAKE_CXX_EXTENSIONS OFF)  # Disable compiler-specific extensions

# Print the C++ standard being used
message(STATUS "Using C++ Standard: ${CMAKE_CXX_STANDARD}")

# Set optimization flags based on OS
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++${CMAKE_CXX_STANDARD} -O3 -Wall -pedantic -ftree-vectorize")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin") # macOS
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++${CMAKE_CXX_STANDARD} -Wall -pedantic")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++${CMAKE_CXX_STANDARD} /O2 /W4 /permissive-")
else()
    message(WARNING "Unknown system; defaulting to generic flags.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++${CMAKE_CXX_STANDARD} -O3 -Wall -pedantic")
endif()

message(STATUS "Building on : ${CMAKE_SYSTEM_NAME} using following compiler flags : ${CMAKE_CXX_FLAGS}")

# FetchContent setup
include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

# Fetch Eigen
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0 # You can also use master
)
FetchContent_MakeAvailable(eigen)

# Fetch Boost
FetchContent_Declare(
  Boost
  GIT_REPOSITORY https://github.com/boostorg/boost.git
  GIT_TAG boost-1.88.0
#   GIT_TAG b6928ae5c92e21a04bbe17a558e6e066dbe632f6
)
FetchContent_MakeAvailable(Boost)

# Gather source files
file(GLOB MAIN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB BASE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/base/*.hpp)

# Add the executable target
add_executable(planck 
    ${MAIN_SOURCES}
    ${BASE_SOURCES}
)

# Get the dependencies
set(MSYM_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/external/libmsym/install)
set(MSYM_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/external/libmsym/build)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)

include(ExternalProject)

ExternalProject_Add(
    libmsym
    PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/src/external/libmsym
    GIT_REPOSITORY https://github.com/mcodev31/libmsym.git
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${MSYM_INSTALL_DIR} -DBUILD_SHARED_LIBS:BOOL=OFF -DMSYM_BUILD_EXAMPLES:BOOL=OFF
)

ExternalProject_Get_Property(libmsym install_dir)

# Include directories
target_include_directories(planck PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/base 
    ${MSYM_INSTALL_DIR}/include  # Include the external library headers
    ${eigen_SOURCE_DIR}
    # ${Boost_INCLUDE_DIR}
)

# Link the external library
target_link_libraries(planck 
                        ${MSYM_INSTALL_DIR}/lib/libmsym.a 
                        Boost::date_time 
                        Boost::property_tree 
                        Boost::foreach 
                        Boost::math
                        Boost::json
                        )

# Add dependencies to ensure the external project is built first
add_dependencies(planck libmsym eigen)