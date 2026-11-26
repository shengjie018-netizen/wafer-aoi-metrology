cmake_minimum_required(VERSION 3.16)
project(wafer_aoi_metrology VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(WAFER_AOI_USE_OPENMP "Enable OpenMP parallelization" ON)
option(WAFER_AOI_BUILD_TESTS "Build unit tests" ON)

add_library(wafer_aoi STATIC
    src/core/image.cpp
    src/core/image_ops.cpp
    src/algo/shape_matcher.cpp
    src/algo/subpixel_metrology.cpp
    src/algo/d2d_inspector.cpp
)

target_include_directories(wafer_aoi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(wafer_aoi PUBLIC /W4 /O2)
else()
    target_compile_options(wafer_aoi PUBLIC -Wall -Wextra -O3)
endif()

if(WAFER_AOI_USE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(wafer_aoi PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(wafer_aoi PUBLIC WAFER_AOI_OPENMP=1)
        message(STATUS "OpenMP enabled for wafer_aoi")
    endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(wafer_aoi PUBLIC Threads::Threads)

# Optional OpenCV detection for future extension
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV ${OpenCV_VERSION} found (optional bridge available)")
endif()

add_executable(wafer_aoi_demo src/main_demo.cpp)
target_link_libraries(wafer_aoi_demo PRIVATE wafer_aoi)

if(WAFER_AOI_BUILD_TESTS)
    enable_testing()
    add_executable(wafer_aoi_tests tests/test_all.cpp)
    target_link_libraries(wafer_aoi_tests PRIVATE wafer_aoi)
    add_test(NAME wafer_aoi_unit_tests COMMAND wafer_aoi_tests)
endif()
