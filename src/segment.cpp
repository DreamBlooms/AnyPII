#include "anypii/segment.hpp"

#include <cstdlib>
#include <memory>
#include <unordered_set>
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

namespace {

std::unordered_set<char> g_weak_prefix;
std::unordered_set<std::string> g_weak_exact;
bool g_weak_configured = false;

const std::vector<std::string> & default_weak_tags() {
    // Function/morpheme/affix classes only. Nouns and proper nouns stay strong:
    // n, ng, nr, nrfg, nrt, ns, nt, nz. Also kept: j (abbreviations may be
    // entities), s (locative words may be addresses), x (unknown, used for
    // names), eng (English).
    static const std::vector<std::string> tags = {
        "v", "a", "d", "p", "r", "u", "m", "q", "t", "f", "c", "y",
        "e", "o", "z", "l", "i", "w", "b", "g", "h", "k"};
    return tags;
}

void add_weak_tag(const std::string & tag) {
    if (tag.empty()) return;
    if (tag.size() == 1) {
        g_weak_prefix.insert(tag[0]);
    } else {
        g_weak_exact.insert(tag);
    }
}

void ensure_weak_configured() {
    if (g_weak_configured) return;
    g_weak_configured = true;
    for (const auto & tag : default_weak_tags()) add_weak_tag(tag);
}

} // namespace

void configure_weak_pos(const std::vector<std::string> & append,
                        const std::vector<std::string> & override_tags) {
    g_weak_prefix.clear();
    g_weak_exact.clear();
    g_weak_configured = true;
    const bool clear = override_tags.size() == 1 && override_tags[0] == "none";
    if (override_tags.empty()) {
        for (const auto & tag : default_weak_tags()) add_weak_tag(tag);
    } else if (!clear) {
        for (const auto & tag : override_tags) add_weak_tag(tag);
    }
    for (const auto & tag : append) add_weak_tag(tag);
}

bool weak_pos(const std::string & pos) {
    if (pos.empty()) return false;
    ensure_weak_configured();
    if (g_weak_prefix.count(pos[0])) return true;
    return g_weak_exact.count(pos) > 0;
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
