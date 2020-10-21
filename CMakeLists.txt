# Brush
# copyright 2020 William La Cava 
# license: GNU/GPL v3

######### Options ###########################
option( CORE_USE_CUDA   "Use CUDA to speed up certain parts of the code."  OFF )
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

set(CMAKE_CXX_COMPILER /usr/bin/g++-7)
# version requirement for Eigen
cmake_minimum_required (VERSION 3.0)
# use c++ 17
set (CMAKE_CXX_STANDARD 17)

#set (CMAKE_CXX_FLAGS "--std=c++11")

#set (CUDA_PROPAGATE_HOST_FLAGS ON)

  
# project name
project(brush)

# add Eigen include paths
option(EIGEN_DIR "use Eigen environment variable" OFF)
if (EIGEN_DIR)
    include_directories("$ENV{EIGEN3_INCLUDE_DIR}")
else()
    find_package(Eigen3 REQUIRED NO_MODULE)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Locate GTest
option(TEST "build tests" OFF)
if (TEST)
    option(GTEST_DIR "use GTest environment path" OFF)
    if (GTEST_DIR)
        include_directories("$ENV{GTEST_DIR}/include")

    else()
        find_package(GTest REQUIRED)
        include_directories(${GTEST_INCLUDE_DIRS})
    endif()
endif()

# autocomplete for YouCompleteMe needs commands to be exported
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set a default build type if none was specified
set(default_build_type "Release")
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

######### CUDA decisions ####################
if (CORE_USE_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DUSE_CUDA)
    #enable_language(CUDA)
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11 --expt-relaxed-constexpr --default-stream per-thread -Wno-deprecated-gpu-targets")

endif()
#############################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes")

# gather list of source files
if (CORE_USE_CUDA)   
    file(GLOB_RECURSE cuda_files "*.cu")
endif()    

file(GLOB_RECURSE sources "src/*.cc")

list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/parser.cc)
list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc)
list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/template_nodes.cc)
list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/arithmetic/n_add.cc)
list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/n_dx.cc)

# executable

if (CORE_USE_CUDA)
    cuda_add_library(brush_lib SHARED ${sources} ${cuda_files})
else()
    add_library(brush_lib SHARED ${sources})
endif()

# executables

#brush executable
add_executable(brush src/main.cc)

#link libraries for different executables

if (CORE_USE_CUDA)
    target_link_libraries(brush ${CUDA_LIBRARIES})
else()
    target_link_libraries(brush brush_lib )
endif()

if (TEST)
    # Link runTests with what we want to test and the GTest and pthread library
    file(GLOB_RECURSE testsSrc "tests/*.cc")
    
    add_executable(tests ${testsSrc})
    
    if (CORE_USE_CUDA)
        target_link_libraries(tests brush_lib ${CUDA_LIBRARIES} ${GTEST_LIBRARIES} pthread)
    else()
        target_link_libraries(tests brush_lib ${GTEST_LIBRARIES} pthread)
    endif()
endif()


