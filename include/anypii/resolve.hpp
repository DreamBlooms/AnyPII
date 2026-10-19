#pragma once

#include "anypii/engine.hpp"
#include "anypii/types.hpp"

namespace anypii {

struct ResolveOptions {
    int max_branches_per_decode = 32;
    int max_text_cp = 16000;
    int max_candidates = 4000;
    int max_boundary_options = 25;
    double active_floor = 0.12;
    double direct_prob = 0.80;
    double direct_margin = 0.20;
    double exact_threshold = 0.50;
    double verify_threshold = 0.50;
};

AnyPiiResult run_pipeline(Engine & engine, const AnyPiiRequest & request,
                          const ResolveOptions & options = {});

} // namespace anypii
