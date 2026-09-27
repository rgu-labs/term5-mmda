# Eigen, header only, used by the labs. Fetched once per build tree, so Debug
# and Development each keep their own copy under <build>/_deps

include_guard(GLOBAL)

set(ELIB_EIGEN_VERSION 3.4.0)
set(ELIB_EIGEN_URL "https://gitlab.com/libeigen/eigen/-/archive/${ELIB_EIGEN_VERSION}/eigen-${ELIB_EIGEN_VERSION}.tar.gz")
set(ELIB_EIGEN_SHA256 8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72)

# Eigen would otherwise add its own tests, docs and install rules
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN3_INSTALL OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(eigen
    URL ${ELIB_EIGEN_URL}
    URL_HASH SHA256=${ELIB_EIGEN_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(eigen)

# Eigen brings an interface target of its own, MDAA::Eigen is the one to link.
# The headers go in as a system include, so neither the compiler nor clang-tidy
# reports anything from inside a third party header
set(elib_eigen_dir "${Eigen3_SOURCE_DIR}")
if (NOT EXISTS "${elib_eigen_dir}/Eigen/Core")
    message(FATAL_ERROR
        "Eigen ${ELIB_EIGEN_VERSION} is required, it is fetched from ${ELIB_EIGEN_URL}.")
endif()

add_library(MdaaEigen INTERFACE)
target_include_directories(MdaaEigen SYSTEM INTERFACE "${elib_eigen_dir}")
add_library(MDAA::Eigen ALIAS MdaaEigen)

unset(elib_eigen_dir)

set_property(GLOBAL APPEND PROPERTY MDAA_LIBRARIES MDAA::Eigen)
