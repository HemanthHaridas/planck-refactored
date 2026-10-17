#pragma once

// Compatibility names for the standalone swap probe. Production and probe
// share one checked algorithm, so numerical settings cannot silently drift.
#include "response_gmres.h"

namespace DFT::Driver::Probe
{
    using Action = DFT::Response::Action;
    using Solve = DFT::Response::Solve;
    using DFT::Response::gmres;
}
