cmake_minimum_required(VERSION 3.27)
project(MDAA CXX)

include(CMake/compiler.cmake)
include(CMake/standard.cmake)
include(CMake/compiler_flags.cmake)
include(CMake/ccache.cmake)
include(CMake/build_type.cmake)

include(CMake/logging.cmake)
include(CMake/mdaa_executable.cmake)
include(CMake/mdaa_module.cmake)

include(Dependencies/Eigen.cmake)

add_subdirectory(Source)
add_subdirectory(Labs)

mdaa_log_summary()
