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

// True for POS tags that are poor entity boundaries. Single-character tags
// match that tag and its subclasses by prefix ("v" also matches vd/vn/vg);
// multi-character tags must match exactly.
bool weak_pos(const std::string & pos);

// Replace or extend the weak-POS set. An empty override set keeps the defaults;
// the override set replaces them. Tags are raw jieba POS labels such as "v",
// "vn", "uj", or "b". Configure before serving requests.
void configure_weak_pos(const std::vector<std::string> & append,
                        const std::vector<std::string> & override_tags);

} // namespace anypii
