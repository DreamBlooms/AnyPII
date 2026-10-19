#include "anypii/segment.hpp"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <cppjieba/Jieba.hpp>

#include "anypii/unicode.hpp"

#ifndef ANYPPI_JIEBA_DICT
#define ANYPPI_JIEBA_DICT ""
#endif

namespace anypii {
namespace {

std::string dict_dir() {
    const char * env = std::getenv("ANYPPI_JIEBA_DICT");
    if (env && *env) return env;
    return ANYPPI_JIEBA_DICT;
}

struct Segmenter {
    std::unique_ptr<cppjieba::Jieba> jieba;
    bool ready = false;

    Segmenter() {
        const std::string dir = dict_dir();
        if (dir.empty()) return;
        try {
            jieba = std::make_unique<cppjieba::Jieba>(
                dir + "/jieba.dict.utf8",
                dir + "/hmm_model.utf8",
                dir + "/user.dict.utf8",
                dir + "/idf.utf8",
                dir + "/stop_words.utf8");
            ready = true;
        } catch (...) {
            ready = false;
        }
    }
};

Segmenter & segmenter() {
    static Segmenter instance;
    return instance;
}

} // namespace

bool weak_pos(const std::string & pos) {
    if (pos.empty()) return false;
    switch (pos[0]) {
        case 'v': // verb
        case 'a': // adjective
        case 'd': // adverb
        case 'p': // preposition
        case 'r': // pronoun
        case 'u': // particle
        case 'm': // numeral
        case 'q': // quantifier
        case 't': // time
        case 'f': // direction
        case 'c': // conjunction
        case 'y': // modal
        case 'e': // interjection
        case 'o': // onomatopoeia
        case 'z': // status
        case 'w': // punctuation
            return true;
        default:
            return false;
    }
}

bool segment_cjk(const std::string & text, std::vector<SegmentWord> & words) {
    Segmenter & seg = segmenter();
    if (!seg.ready || text.empty()) return false;

    std::vector<cppjieba::Word> cut;
    seg.jieba->Cut(text, cut, true);
    if (cut.empty()) return false;

    const Utf8Text ut = decode_utf8(text);
    words.clear();
    words.reserve(cut.size());
    for (const auto & w : cut) {
        if (w.word.empty()) continue;
        SegmentWord out;
        out.start = byte_to_cp(ut, w.offset);
        out.end = byte_to_cp(ut, w.offset + w.word.size());
        if (out.start >= out.end) continue;
        out.pos = seg.jieba->LookupTag(w.word);
        words.push_back(std::move(out));
    }
    return !words.empty();
}

} // namespace anypii
