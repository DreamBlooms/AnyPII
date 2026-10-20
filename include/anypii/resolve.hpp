#pragma once

#include "anypii/engine.hpp"
#include "anypii/types.hpp"

namespace anypii {

class PcdDecoder;

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
    double temperature = 1.0;
    double none_margin = 0.65;
    int none_margin_max_len = 4;
    double none_margin_redact = 0.20;
    bool compact_branches = false;
    bool single_type_orientation = false;
    bool type_refine_dedup = true;
    bool component_shortlist = false;
    PcdDecoder * pcd = nullptr;
};

AnyPiiResult run_pipeline(Engine & engine, const AnyPiiRequest & request,
                          const ResolveOptions & options = {});

} // namespace anypii
