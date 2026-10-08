#pragma once

#include "MDAA/Core/Types.h"

namespace MDAA {

enum class StopReason : u8 {
    Accuracy,
    Iterations,
    Stationary,
};

} // namespace MDAA
