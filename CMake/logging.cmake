function(mdaa_log_summary)
    get_property(MDAA_EXECUTABLES
        GLOBAL PROPERTY
        MDAA_EXECUTABLES
    )
    get_property(
        MDAA_LIBRARIES
        GLOBAL PROPERTY
        MDAA_LIBRARIES
    )

    if (NOT MDAA_EXECUTABLES)
        set(MDAA_EXECUTABLES "<none>")
    endif()
    if (NOT MDAA_LIBRARIES)
        set(MDAA_LIBRARIES "<none>")
    endif()

    message(STATUS "Configuration done.
        Build type: ${CMAKE_BUILD_TYPE}
        Executables: ${MDAA_EXECUTABLES}
        Libraries: ${MDAA_LIBRARIES}"
    )
endfunction()
