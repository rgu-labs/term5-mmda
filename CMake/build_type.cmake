if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Development CACHE STRING "Build type" FORCE)
endif()

# Supported: Debug, Development
# Debug       - -O0, asserts on, this is what you build when you debug
# Development - -O3, asserts on, this is what you measure lab results with
if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi" CACHE STRING "" FORCE)

    set(CMAKE_CXX_FLAGS_DEVELOPMENT "/O2 /Zi" CACHE STRING "" FORCE)
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3" CACHE STRING "" FORCE)

    set(CMAKE_CXX_FLAGS_DEVELOPMENT "-O3 -g" CACHE STRING "" FORCE)
endif()

# Options derived on CMAKE_BUILD_TYPE with opportunity to override them with -D*
if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "Development")
    set(MDAA_ENABLE_ASSERTS ON CACHE BOOL "Enables asserts")
else()
    set(MDAA_ENABLE_ASSERTS OFF CACHE BOOL "Enables asserts")
endif()

# Extra instructions for the local cpu. Note that binaries built this way are
# not portable to other machines, keep it off if you submit results elsewhere
option(MDAA_NATIVE_ARCH "Build with instructions of the local cpu" OFF)

# Pass in definitions to compiler
if (MDAA_ENABLE_ASSERTS)
    list(APPEND MDAA_COMPILE_DEFINITIONS MDAA_ENABLE_ASSERTS)
endif()

if (MDAA_NATIVE_ARCH AND NOT MSVC)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" MDAA_HAS_MARCH_NATIVE)
    if (MDAA_HAS_MARCH_NATIVE)
        list(APPEND MDAA_COMPILE_FLAGS -march=native)
        message(STATUS "[arch] -march=native is enabled")
    endif()
endif()
