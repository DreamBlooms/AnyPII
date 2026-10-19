#pragma once

#include <string>
#include <vector>

#include "anypii/types.hpp"
#include "anypii/unicode.hpp"

namespace anypii {

// Replace redacted spans with the requested mask. Offsets are Unicode code
// point indices, so "length" masks are one mask character per code point.
// `group_index` is required for style "typed": it maps each hit to the index
// of its EntitySpan in the response (0-based).
std::string apply_mask(const Utf8Text & text, const std::vector<PhraseHit> & spans,
                       const std::string & style,
                       const std::vector<int> * group_index = nullptr);

} // namespace anypii
