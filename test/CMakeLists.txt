set(PROJECT_TEST_NAME "fit_gtr_test")
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Beagle
include_directories(${HMS_BEAGLE_INCLUDE_DIRS})
link_directories(${HMS_BEAGLE_LIBRARY_DIRS})

add_executable(${PROJECT_TEST_NAME} EXCLUDE_FROM_ALL test-main.cpp test-star_optim.cpp test-log_tricks.cpp
    test-bpp-gamma.cpp)
target_link_libraries(${PROJECT_TEST_NAME} gtest gtest_main libstaropt)
target_link_libraries(${PROJECT_TEST_NAME} bpp-phyl bpp-seq bpp-core)
set_property(TARGET ${PROJECT_TEST_MAIN} PROPERTY COMPILE_FLAGS "-fopenmp")
set_property(TARGET ${PROJECT_TEST_MAIN} PROPERTY LINK_FLAGS "-fopenmp")

add_test(${PROJECT_TEST_NAME} ${PROJECT_TEST_NAME})
