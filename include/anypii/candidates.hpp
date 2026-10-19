#pragma once

#include <vector>

#include "anypii/engine.hpp"
#include "anypii/types.hpp"
#include "anypii/unicode.hpp"

namespace anypii {

struct CandidateOptions {
    int max_span_cp = 24;
    int max_char_ngram = 6;
    int token_merge = 4;
    int max_candidates = 2000;
};

std::vector<Candidate> propose_candidates(const Utf8Text & text, Engine & engine,
                                          const CandidateOptions & options);

} // namespace anypii
