# Sets compiler agnostic compile flags
#
# NOTE: never enable -ffast-math or -Ofast here, labs are about exact numerical
# behaviour and those flags break IEEE 754 semantics

if (MSVC)
    # cmake auto adds EHsc into flags, remove it
    string(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

    set(MDAA_COMPILE_FLAGS /GR- /W4 /MP /permissive /EHs-c-
    /pathmap:${CMAKE_SOURCE_DIR}/=./)
    set(MDAA_COMPILE_DEFINITIONS _HAS_EXCEPTIONS=0)
else()
    set(MDAA_COMPILE_FLAGS -fno-rtti -fno-exceptions -Wall
        -Wextra -Wpedantic
    -ffile-prefix-map=${CMAKE_SOURCE_DIR}/=./)
endif()
