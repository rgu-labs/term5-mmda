function(mdaa_module)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES PRIVATE_DEPS PUBLIC_DEPS)

    cmake_parse_arguments(MODULE
        "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT MODULE_NAME)
        message(FATAL_ERROR "mdaa_module: NAME is required.")
    endif ()

    foreach (source IN LISTS MODULE_SOURCES)
        if (NOT source MATCHES "^Private/.*\\.cpp$")
            message(FATAL_ERROR
                "mdaa_module ${MODULE_NAME}: a source must live in Private/, got ${source}.")
        endif ()
    endforeach ()

    set(MODULE_TARGET MDAA${MODULE_NAME})
    set(MODULE_ALIAS MDAA::${MODULE_NAME})

    set_property(GLOBAL APPEND PROPERTY MDAA_LIBRARIES ${MODULE_ALIAS})

    if (MODULE_SOURCES)
        message(STATUS "[mdaa_module] ${MODULE_NAME}, compiled.")
        add_library(${MODULE_TARGET} STATIC ${MODULE_SOURCES})

        target_include_directories(
            ${MODULE_TARGET}
            PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Public
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Private
        )
        target_compile_options(${MODULE_TARGET} PRIVATE ${MDAA_COMPILE_FLAGS})
        target_compile_definitions(${MODULE_TARGET}
            PRIVATE ${MDAA_COMPILE_DEFINITIONS})

        if (MODULE_PUBLIC_DEPS)
            target_link_libraries(${MODULE_TARGET} PUBLIC ${MODULE_PUBLIC_DEPS})
        endif ()
        if (MODULE_PRIVATE_DEPS)
            target_link_libraries(${MODULE_TARGET} PRIVATE ${MODULE_PRIVATE_DEPS})
        endif ()
    else ()
        message(STATUS "[mdaa_module] ${MODULE_NAME}, header only.")
        add_library(${MODULE_TARGET} INTERFACE)

        target_include_directories(
            ${MODULE_TARGET}
            INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/Public
        )
        target_compile_definitions(${MODULE_TARGET}
            INTERFACE ${MDAA_COMPILE_DEFINITIONS})

        if (MODULE_PUBLIC_DEPS)
            target_link_libraries(${MODULE_TARGET} INTERFACE ${MODULE_PUBLIC_DEPS})
        endif ()
    endif ()

    add_library(${MODULE_ALIAS} ALIAS ${MODULE_TARGET})

endfunction()
