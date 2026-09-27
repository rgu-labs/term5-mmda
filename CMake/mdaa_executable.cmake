# Declares a standalone executable, every lab is built with this
function(mdaa_executable)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES DEPS)

    cmake_parse_arguments(EXE
        "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT EXE_NAME)
        message(FATAL_ERROR "mdaa_executable: NAME is required.")
    endif()

    if (NOT EXE_SOURCES)
        message(FATAL_ERROR "mdaa_executable: SOURCES is required.")
    endif()

    set_property(GLOBAL APPEND PROPERTY MDAA_EXECUTABLES ${EXE_NAME})

    message(STATUS "[mdaa_executable] ${EXE_NAME}.")

    add_executable(${EXE_NAME} ${EXE_SOURCES})
    target_compile_options(${EXE_NAME} PRIVATE ${MDAA_COMPILE_FLAGS})
    target_compile_definitions(${EXE_NAME}
        PRIVATE ${MDAA_COMPILE_DEFINITIONS})

    set_target_properties(
        ${EXE_NAME}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Target"
        EXPORT_COMPILE_COMMANDS ON
    )

    if (EXE_DEPS)
        target_link_libraries(${EXE_NAME} PRIVATE ${EXE_DEPS})
    endif()

endfunction()
