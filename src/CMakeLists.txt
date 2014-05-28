# Protocol buffers
find_package(Protobuf REQUIRED)
#PROTOBUF_GENERATE_CPP(ProtoSources ProtoHeaders mutationio.proto)
get_filename_component(PROTOBUF_LIB_DIR ${PROTOBUF_LIBRARY} PATH)
set(PROTOBUF_A "${PROTOBUF_LIB_DIR}/libprotobuf.a")

# zlib
find_package(ZLIB REQUIRED)

# BEAGLE
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
            star_tree_optimizer.cpp)
set_target_properties(libstaropt PROPERTIES OUTPUT_NAME staropt)
target_link_libraries(libstaropt
  #protobuf
  ${PROTOBUF_A}
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
add_executable(build-mutation-matrices
  build_mutation_matrices.cpp
  mutationio.pb.cc)
target_link_libraries(build-mutation-matrices libstaropt)
target_link_libraries(build-mutation-matrices boost-programoptions)

# model fitting
add_executable(fit-star fit_star.cpp)
target_link_libraries(fit-star boost-programoptions boost-iostream libstaropt)
target_link_libraries(fit-star json)
target_link_libraries(fit-star nlopt)

add_executable(show-mutation-matrices show_mutation_matrices.cpp mutationio.pb.cc)
target_link_libraries(show-mutation-matrices z c protobuf boost-programoptions)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/build-mutation-matrices
  ${CMAKE_CURRENT_BINARY_DIR}/fit-star
  DESTINATION bin)
