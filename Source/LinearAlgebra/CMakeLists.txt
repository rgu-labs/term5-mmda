mdaa_module(
    NAME LinearAlgebra
    SOURCES
        Private/Tridiagonal.cpp
    PUBLIC_DEPS
        MDAA::Core
        MDAA::Eigen
)
