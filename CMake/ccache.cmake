# Manages ccache option and state

option(USE_CCACHE "Enables cache to speed up recompilation" ON)

if (USE_CCACHE)
    find_program(CCACHE_EXE ccache)
    if (CCACHE_EXE)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_EXE}" CACHE INTERNAL "")
        message(STATUS "[ccache] ccache found: ${CCACHE_EXE}")
    else()
        message(WARNING
            "[ccache] ccache executable not found. Proceeding without it.")
    endif()
else()
    message(STATUS "[ccache] ccache disabled.")
endif()
