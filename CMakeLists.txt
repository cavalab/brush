# Brush
# copyright 2020 William La Cava 
# license: GNU/GPL v3

get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
link_directories( $ENV{LD_LIBRARY_PATH})
# version requirement for Eigen
cmake_minimum_required (VERSION 3.15...3.19)
# use c++ 20
set (CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set (CUDA_PROPAGATE_HOST_FLAGS ON)

  
# project name
project(_brush)

if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
    message(STATUS "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
endif()
# # add Eigen include paths
if (DEFINED ENV{EIGEN3_INCLUDE_DIR})
    set(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
    include_directories($ENV{EIGEN3_INCLUDE_DIR})
    message(STATUS "EIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR}")
elseif(DEFINED EIGEN3_INCLUDE_DIR)
    include_directories("${EIGEN3_INCLUDE_DIR}")
    message(STATUS "EIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR}")
elseif(DEFINED ENV{CONDA_PREFIX})
    include_directories($ENV{CONDA_PREFIX}/include/eigen3/)
else()
    message(STATUS "No EIGEN3_INCLUDE_DIR found, using find_package...")
    find_package(Eigen3 3.4 REQUIRED NO_MODULE)
endif()


if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
endif()


# autocomplete for YouCompleteMe needs commands to be exported
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set a default build type if none was specified
# set(default_build_type "Release")
set(default_build_type "Debug")
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${default_build_type}' since none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# add OpenMP support
option(OMP "use OpenMP" ON)
if (OMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

if (MACOSX)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp=libgomp -fexperimental-library")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libgomp -fexperimental-library")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes")
########################################
# fmt
find_package(fmt REQUIRED)
message("fmt_FOUND: ${fmt_FOUND}")
########################################
# Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

message("Python_FOUND:${Python_FOUND}")
message("Python_VERSION:${Python_VERSION}")
message("Python_Development_FOUND:${Python_Development_FOUND}")
message("Python_LIBRARIES:${Python_LIBRARIES}")
message("Python_INCLUDE_DIRS:${Python_INCLUDE_DIRS}")

# Find pybind11
find_package(pybind11 REQUIRED)
include_directories("${pybind11_INCLUDE_DIRS}")
message("pybind11_DIR:${pybind11_DIR}")

########################################
# build brush library

# gather list of source files
file(GLOB_RECURSE sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE bindings ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/*.cpp)
list(REMOVE_ITEM sources ${bindings})
message(STATUS "Sources: '${sources}")

# the pure cpp brush library (no python bindings)
add_library(cbrush SHARED ${sources})
# target_link_libraries(cbrush PUBLIC ${Python_LIBRARIES} fmt::fmt)
# target_link_libraries(cbrush PUBLIC fmt::fmt)
target_link_libraries(cbrush PUBLIC fmt::fmt)

########################################
# generate python bindings

message("add module")
pybind11_add_module(_brush ${bindings})
# pybind11_add_module(_brush ${sources})
message("target link libraries")
target_link_libraries(_brush PRIVATE cbrush)
# target_link_libraries(_brush PRIVATE ${PYTHON_LIBRARIES} fmt::fmt)
target_compile_definitions(_brush PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})


########################################
# brush cpp tests
option(GTEST "build cpp tests" ON)
if (GTEST)
    # Locate GTest
    include(FetchContent)
    FetchContent_Declare(
    googletest
    # GIT_REPOSITORY https://github.com/google/googletest.git
    # GIT_TAG release-1.12.1
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    # Find tests
    file(GLOB_RECURSE testsSrc "tests/*.cpp")
    message(STATUS "testsSrc: '${testsSrc}")
    enable_testing()
    add_executable(tests ${testsSrc})
    # Link runTests with what we want to test and the GTest and pthread library
    target_link_libraries(tests cbrush GTest::gtest_main pthread fmt::fmt)
    # Google tests
    include(GoogleTest)
    gtest_discover_tests(tests)
endif()

########################################
# Handle stuff for docs (Sphinx/Doxygen)
option(DOCS "build docs" OFF)
if (DOCS)
    message("set")
    set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
    message("docs")
    add_subdirectory("docs")
    message("done")
endif()
