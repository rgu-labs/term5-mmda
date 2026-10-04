mdaa_module(
    NAME Optimization
    SOURCES
        Private/LineSearch.cpp
        Private/Norms.cpp
        Private/SteepestDescent.cpp
    PUBLIC_DEPS
        MDAA::Core
        MDAA::Eigen
)
