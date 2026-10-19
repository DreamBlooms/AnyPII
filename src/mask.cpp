#include "anypii/mask.hpp"

#include <stdexcept>

namespace anypii {

std::string apply_mask(const Utf8Text & text, const std::vector<PhraseHit> & spans,
                       const std::string & style,
                       const std::vector<int> * group_index) {
    if (style != "token" && style != "length" && style != "mask" &&
        style != "typed" && style != "type") {
        throw std::invalid_argument("unsupported mask_style: " + style);
    }
    if (style == "typed" && (group_index == nullptr ||
                             group_index->size() != spans.size())) {
        throw std::invalid_argument("typed mask requires a group index per span");
    }

    std::string out;
    out.reserve(text.bytes.size());
    int cursor = 0;
    for (size_t si = 0; si < spans.size(); ++si) {
        const auto & span = spans[si];
        int start = span.start;
        int end = span.end;
        if (start < cursor) start = cursor;
        if (end > text.size()) end = text.size();
        if (start >= end) continue;
        out += text.substr(cursor, start);
        if (span.redacted) {
            if (style == "token") {
                out += "[REDACTED]";
            } else if (style == "type") {
                out += "[" + span.type + "]";
            } else if (style == "typed") {
                out += "[" + span.type + ":" +
                       std::to_string((*group_index)[si]) + "]";
            } else {
                const int n = end - start;
                for (int i = 0; i < n; ++i) out += "\xE2\x96\x88"; // U+2588
            }
        } else {
            out += text.substr(start, end);
        }
        cursor = end;
    }
    out += text.substr(cursor, text.size());
    return out;
}

} // namespace anypii
