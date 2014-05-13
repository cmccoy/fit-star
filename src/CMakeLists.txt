# Protocol buffers
find_package(Protobuf REQUIRED)
PROTOBUF_GENERATE_CPP(ProtoSources ProtoHeaders mutationio.proto)

# zlib
find_package(ZLIB REQUIRED)

# BEAGLE
find_package(PkgConfig REQUIRED)
pkg_check_modules(HMS_BEAGLE hmsbeagle-1 REQUIRED)
include_directories(${HMS_BEAGLE_INCLUDE_DIRS})
link_directories(${HMS_BEAGLE_LIBRARY_DIRS})

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
  add_definitions("-DHAVE_OMP")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_library(libstaropt
            mutationio.pb.cc
            star_tree_optimizer.cpp
            kmer_model.cpp)
set_target_properties(libstaropt PROPERTIES OUTPUT_NAME staropt)
target_link_libraries(libstaropt
  protobuf
  ${ZLIB_LIBRARIES}
  c
  pthread
  bam
  nlopt
  bpp-phyl
  bpp-core
  bpp-seq
  ${HMS_BEAGLE_LIBRARIES})

# Build mutation matrices
add_executable(build-mutation-matrices build_mutation_matrices.cpp)
target_link_libraries(build-mutation-matrices libstaropt)
target_link_libraries(build-mutation-matrices boost-programoptions)

# kgt1
add_executable(fit-kgt1 fit_kgt1.cpp kmer_model.cpp)
target_link_libraries(fit-kgt1 libstaropt)
target_link_libraries(fit-kgt1 boost-programoptions)

# model fitting
add_executable(fit-star fit_star.cpp)
target_link_libraries(fit-star boost-programoptions boost-iostream libstaropt)
target_link_libraries(fit-star json)
target_link_libraries(fit-star nlopt)

add_executable(show-mutation-matrices show_mutation_matrices.cpp mutationio.pb.cc)
target_link_libraries(show-mutation-matrices z c protobuf boost-programoptions)
