project(LSM)

SET(CMAKE_BUILD_TYPE Debug CACHE STRING "Debug configuration" FORCE)
SET(CMAKE_CXX_FLAGS -std=gnu++0x)

set (LSM_VERSION_MAJOR 1)
set (LSM_VERSION_MINOR 0)

cmake_minimum_required(VERSION 2.8)

set(NEMO_SOURCE_DIR  "NeMo-0.7.2" CACHE PATH "Install dir" )
IF ( NEMO_SOURCE_DIR )
    MESSAGE(STATUS "-- Looking for Nemo-0.7.2 - found")
ELSE ( NEMO_SOURCE_DIR )
    MESSAGE(FATAL_ERROR "-- Looking for Nemo-0.7.2 - not found")
ENDIF ( NEMO_SOURCE_DIR )       

include_directories(${NEMO_SOURCE_DIR}/build/src/)
include_directories(${NEMO_SOURCE_DIR}/src/)

link_directories(${NEMO_SOURCE_DIR}/build/src/nemo/cuda/)
link_directories(${NEMO_SOURCE_DIR}/build/src/nemo/)

include(FindEigen3.cmake)
include_directories(${EIGEN3_INCLUDE_DIR})
## SET HERE BOOST DIR!
SET(BOOST_DIR "boost_1_55_0/" CACHE PATH "Install dir" )
include(MyFindBoost.cmake)

include_directories("${PROJECT_BINARY_DIR}")

add_library(LSMlib Trainings.cpp Readouts.cpp Constructs.cpp NemoLSM.cpp)
set (EXTRA_LIBS LSMlib nemo_base nemo boost_system pthread blas m )
add_executable(main main.cpp )
target_link_libraries (main ${EXTRA_LIBS})

