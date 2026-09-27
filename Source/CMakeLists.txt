# Shared code, header only, that every lab may use
#
# MDAA::Common puts this folder on the include path, so a public header is
# included canonically full, e.g. #include "MDAA/Types.h"

add_library(MDAACommon INTERFACE)
add_library(MDAA::Common ALIAS MDAACommon)

target_include_directories(MDAACommon
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/Public)

# Defines are interface level, a header only library has no compilation unit
# of its own to attach them to
target_compile_definitions(MDAACommon
    INTERFACE ${MDAA_COMPILE_DEFINITIONS})

# Add a header here and every lab picks it up through MDAA.h, no need to touch
# a single lab CMakeLists
target_precompile_headers(MDAACommon
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/Public/MDAA/MDAA.h)

set_property(GLOBAL APPEND PROPERTY MDAA_LIBRARIES Common)
