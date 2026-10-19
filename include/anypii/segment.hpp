// CJK word segmentation and POS tags via cppjieba. Used only as a candidate
// proposal prior: the model still decides type and redaction.
#pragma once

#include <string>
#include <vector>

namespace anypii {

struct SegmentWord {
    int start = 0; // code point offset in the input string
    int end = 0;   // half-open code point offset
    std::string pos;
};

// Segment a UTF-8 string. Returns false when the jieba dictionary is not
// available, in which case callers should fall back to the model tokenizer.
bool segment_cjk(const std::string & text, std::vector<SegmentWord> & words);

// True for POS tags that are poor entity boundaries (verb, adjective, adverb,
// preposition, pronoun, particle, numeral, quantifier, time, conjunction...).
bool weak_pos(const std::string & pos);

} // namespace anypii
