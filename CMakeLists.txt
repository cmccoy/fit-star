cmake_minimum_required (VERSION 2.8.3)
project (fit_gtr)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/lib/cmake/cmake-modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/lib/cmake/cxx11/Modules")

include(GetGitRevisionDescription)
git_describe(GIT_DESCRIPTION --always --long)
configure_file (
  "${PROJECT_SOURCE_DIR}/src/fit_star_config.h.in"
  "${PROJECT_SOURCE_DIR}/src/fit_star_config.h")

include(FindCXXFeatures)

if(NOT _HAS_CXX11_FLAG)
   message( FATAL_ERROR "-std=c++11 flag not supported." )
endif()

include_directories("src")
include_directories("lib/jsoncpp-src-0.6.0-rc2/include")
include_directories(SYSTEM "lib/cpplog")
include_directories(SYSTEM "lib/eigen_3.2.0")
include_directories("lib/samtools")
include_directories("lib/boost_1.55.0")
include_directories("lib/nlopt-2.4.1/api")
set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -O2 ${CXX11_COMPILER_FLAGS}")

# BEAGLE
find_package(PkgConfig REQUIRED)
pkg_check_modules(HMS_BEAGLE hmsbeagle-1)
if(NOT HMS_BEAGLE_FOUND)
  message(STATUS "Trying beagle-2")
  pkg_check_modules(HMS_BEAGLE hmsbeagle-2 REQUIRED)
endif()


enable_testing()
add_subdirectory(lib/samtools)
add_subdirectory(lib/boost_1.55.0/libs/program_options)
add_subdirectory(lib/boost_1.55.0/libs/iostreams)
add_subdirectory(lib/jsoncpp-src-0.6.0-rc2/src/lib_json)
add_subdirectory(lib/nlopt-2.4.1)
add_subdirectory(src)

# Testing
add_subdirectory(lib/gtest-1.7.0)
add_subdirectory(test)
