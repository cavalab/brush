# Brush
# copyright 2020 William La Cava 
# license: GNU/GPL v3

######### Options ###########################
# option( CORE_USE_CUDA   "Use CUDA to speed up certain parts of the code."  OFF )
#############################################

######### CUDA decisions ####################
#if (CORE_USE_CUDA)
#    MESSAGE( STATUS ">> --------------  USING CUDA --------------" )
#    set( CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-8.0")
#    if (APPLE OR UNIX)
#        set(CMAKE_C_COMPILER   /usr/bin/gcc)
#        set(CMAKE_CXX_COMPILER /usr/bin/g++)
#    elseif (WIN32)
#        ### By default we use VS
#        MESSAGE( STATUS ">> User compiler:             MSVC" )
#        MESSAGE( STATUS ">> Choosing a different compiler is not yet implemented for Windows" )
#    endif()
#endif()
#############################################
link_directories( $ENV{LD_LIBRARY_PATH} )
# set(CMAKE_CXX_COMPILER /usr/bin/g++-7)
# version requirement for Eigen
cmake_minimum_required (VERSION 3.15...3.19)
# use c++ 17
set (CMAKE_CXX_STANDARD 17)

#set (CMAKE_CXX_FLAGS "--std=c++11")

#set (CUDA_PROPAGATE_HOST_FLAGS ON)

  
# project name
project(brush)

# add Eigen include paths
if (DEFINED ENV{EIGEN3_INCLUDE_DIR})
    set(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
    include_directories($ENV{EIGEN3_INCLUDE_DIR})
elseif(DEFINED EIGEN3_INCLUDE_DIR)
    include_directories("${EIGEN3_INCLUDE_DIR}")
else()
    message(STATUS "No EIGEN3_INCLUDE_DIR found, using find_package...")
    find_package(Eigen3 REQUIRED NO_MODULE)
endif()
message(STATUS "EIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR}")


# Locate GTest
# if defined in environment
if (DEFINED ENV{GTEST_INCLUDE_DIRS} AND DEFINED ENV{GTEST_LIBRARIES})
    include_directories("$ENV{GTEST_INCLUDE_DIRS}")
    set(GTEST_INCLUDE_DIRS $ENV{GTEST_INCLUDE_DIRS})
    set(GTEST_LIBRARIES $ENV{GTEST_LIBRARIES})
# defined at command line
elseif(DEFINED GTEST_INCLUDE_DIRS AND DEFINED GTEST_LIBRARIES)
    set(GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIRS})
    set(GTEST_LIBRARIES ${GTEST_LIBRARIES})
    include_directories("${GTEST_INCLUDE_DIRS}")
# goin fishin
else()
    message(STATUS "looking for GTest")
    find_package(GTest REQUIRED)
endif()

include_directories(${GTEST_INCLUDE_DIRS})
message(STATUS "GTest dir=${GTEST_INCLUDE_DIRS}")
message(STATUS "GTest library=${GTEST_LIBRARIES}")

# endif()

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes")

# gather list of source files

file(GLOB_RECURSE sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
# ignore main for now, just build tests
list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

message(STATUS "Sources: '${sources}")
# build brush library
add_library(brush SHARED ${sources})

#brush executable
# add_executable(brush src/main.cpp)

# Google tests
# Link runTests with what we want to test and the GTest and pthread library
file(GLOB_RECURSE testsSrc "tests/*.cpp")
message(STATUS "testsSrc: '${testsSrc}")

add_executable(tests ${testsSrc})
target_link_libraries(tests brush ${GTEST_LIBRARIES} pthread)

# generate python bindings
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)
include_directories("${pybind11_INCLUDE_DIRS}")

message("Python_FOUND:${Python_FOUND}")
message("Python_VERSION:${Python_VERSION}")
message("Python_Development_FOUND:${Python_Development_FOUND}")
message("Python_LIBRARIES:${Python_LIBRARIES}")
message("pybind11_DIR:${pybind11_DIR}")

find_package(PythonLibs REQUIRED)
include_directories("${PYTHON_INCLUDE_DIRS}")
message("Python_INCLUDE_DIRS:${PYTHON_INCLUDE_DIRS}")
target_link_libraries(brush ${PYTHON_LIBRARIES})

pybind11_add_module(brushgp ${CMAKE_CURRENT_SOURCE_DIR}/src/brushgp.cpp)
target_link_libraries(brushgp PRIVATE brush)
target_compile_definitions(brushgp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})


# Handle stuff for docs (Sphinx/Doxygen)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
add_subdirectory("docs")