# Defines platform specific compiler
# We use gcc for unix and msvc for windows, override with
# -DCMAKE_CXX_COMPILER=<c++> if you want something else

if (DEFINED CMAKE_CXX_COMPILER)
    message(STATUS "[compiler] CXX=${CMAKE_CXX_COMPILER}")
    return()
endif()

if (WIN32)
    # Visual Studio generator already uses cl under the hood
    if (NOT CMAKE_GENERATOR MATCHES "Visual Studio")
        set(CMAKE_CXX_COMPILER cl CACHE STRING "C++ Compiler" FORCE)
    endif()
else()
    set(CMAKE_CXX_COMPILER g++ CACHE STRING "C++ Compiler" FORCE)
endif()

message(STATUS "[compiler] CXX=${CMAKE_CXX_COMPILER}")
