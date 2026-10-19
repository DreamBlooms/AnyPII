#include "anypii/candidates.hpp"

#include "anypii/nist.hpp"
#include "anypii/segment.hpp"

#include <algorithm>
#include <list>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

namespace anypii {

namespace {

enum class CharClass { Space, Punct, Cjk, Word, Other };

CharClass classify(char32_t cp) {
    if (is_space_cp(cp)) return CharClass::Space;
    if (is_cjk(cp)) return CharClass::Cjk;
    if (is_latin_letter(cp) || is_digit_cp(cp)) return CharClass::Word;
    if (is_punct_cp(cp)) return CharClass::Punct;
    return CharClass::Other;
}

struct Run {
    int start;
    int end;
    CharClass cls;
};

std::string strip_piece_prefix(const std::string & s) {
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char) s[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            ++i;
            continue;
        }
        if (i + 2 < s.size() && (unsigned char) s[i] == 0xE2 &&
            (unsigned char) s[i + 1] == 0x96 && (unsigned char) s[i + 2] == 0x81) {
            i += 3;
            continue;
        }
        if (i + 1 < s.size() && (unsigned char) s[i] == 0xC4 &&
            (unsigned char) s[i + 1] == 0xA0) {
            i += 2;
            continue;
        }
        break;
    }
    return s.substr(i);
}

void trim_span(const Utf8Text & text, int & start, int & end) {
    while (start < end) {
        const CharClass c = classify(text.cps[(size_t) start]);
        if (c == CharClass::Space || c == CharClass::Punct) {
            ++start;
        } else {
            break;
        }
    }
    while (end > start) {
        const CharClass c = classify(text.cps[(size_t) end - 1]);
        if (c == CharClass::Space || c == CharClass::Punct) {
            --end;
        } else {
            break;
        }
    }
}

struct CandidateCacheEntry {
    std::vector<Candidate> candidates;
    std::list<std::string>::iterator recency;
};

struct CandidateCache {
    std::mutex mutex;
    std::list<std::string> order;
    std::unordered_map<std::string, CandidateCacheEntry> entries;
};

CandidateCache & candidate_cache() {
    static CandidateCache cache;
    return cache;
}

} // namespace

std::vector<Candidate> propose_candidates(const Utf8Text & text, Engine & engine,
                                          const CandidateOptions & options) {
    const std::string cache_key =
        engine.model_name() + "|" + std::to_string(options.max_span_cp) + "|" +
        std::to_string(options.max_char_ngram) + "|" +
        std::to_string(options.token_merge) + "|" + text.bytes;
    {
        CandidateCache & cache = candidate_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        auto it = cache.entries.find(cache_key);
        if (it != cache.entries.end()) {
            cache.order.splice(cache.order.begin(), cache.order, it->second.recency);
            return it->second.candidates;
        }
    }

    std::map<std::pair<int, int>, Candidate> unique;

    auto add = [&](int start, int end, double prior, bool regex_hit, bool pos_ok = true) {
        if (start < 0) start = 0;
        if (end > text.size()) end = text.size();
        if (!regex_hit) trim_span(text, start, end);
        if (start >= end) return;
        const int limit = regex_hit ? 256 : options.max_span_cp;
        if (end - start > limit) return;
        const auto key = std::make_pair(start, end);
        auto it = unique.find(key);
        if (it == unique.end()) {
            Candidate c;
            c.start = start;
            c.end = end;
            c.text = text.substr(start, end);
            c.prior = prior;
            c.regex_hit = regex_hit;
            c.pos_ok = pos_ok;
            unique.emplace(key, std::move(c));
        } else {
            it->second.prior = std::max(it->second.prior, prior);
            it->second.regex_hit = it->second.regex_hit || regex_hit;
            it->second.pos_ok = it->second.pos_ok || pos_ok;
        }
    };

    // Character-class runs.
    std::vector<Run> runs;
    int i = 0;
    while (i < text.size()) {
        const CharClass cls = classify(text.cps[(size_t) i]);
        int j = i + 1;
        while (j < text.size() && classify(text.cps[(size_t) j]) == cls) ++j;
        if (cls != CharClass::Space) runs.push_back({i, j, cls});
        i = j;
    }

    for (size_t r = 0; r < runs.size(); ++r) {
        const Run & run = runs[r];
        const int len = run.end - run.start;
        if (run.cls == CharClass::Cjk) {
            // CJK candidates are word-aligned. cppjieba supplies the word
            // boundaries and POS tags; adjacent words are merged to recover
            // multi-word names. If the dictionary is unavailable, fall back to
            // the model tokenizer.
            const std::string run_text = text.substr(run.start, run.end);
            std::vector<SegmentWord> seg;
            const bool seg_ok = segment_cjk(run_text, seg);
            if (seg_ok) {
                const int nw = (int) seg.size();
                auto pos_ok = [&](int a, int b) {
                    if (a < 0 || b < a || b >= nw) return true;
                    if (weak_pos(seg[(size_t) a].pos) || weak_pos(seg[(size_t) b].pos)) {
                        return false;
                    }
                    for (int k = a + 1; k < b; ++k) {
                        if (weak_pos(seg[(size_t) k].pos)) return false;
                    }
                    return true;
                };
                for (int a = 0; a < nw; ++a) {
                    for (int b = a; b < nw && b < a + options.token_merge; ++b) {
                        add(run.start + seg[(size_t) a].start,
                            run.start + seg[(size_t) b].end,
                            1.04, false, pos_ok(a, b));
                    }
                }
                if (len <= options.max_span_cp) {
                    add(run.start, run.end, 1.0, false, pos_ok(0, nw - 1));
                }
            } else {
                const auto toks = engine.tokenize(run_text, false);
                bool token_ok = false;
                if (!toks.empty()) {
                    std::vector<int> tok_start;
                    std::vector<int> tok_end;
                    std::string rebuilt;
                    int cursor = run.start;
                    bool ok = true;
                    for (llama_token t : toks) {
                        std::string piece = strip_piece_prefix(engine.token_to_piece(t));
                        if (piece.empty()) {
                            ok = false;
                            break;
                        }
                        const Utf8Text pt = decode_utf8(piece);
                        if (text.substr(cursor, cursor + pt.size()) != piece) {
                            ok = false;
                            break;
                        }
                        tok_start.push_back(cursor);
                        cursor += pt.size();
                        tok_end.push_back(cursor);
                        rebuilt += piece;
                    }
                    if (ok && cursor == run.end && rebuilt == run_text) {
                        token_ok = true;
                        const int nt = (int) toks.size();
                        for (int a = 0; a < nt; ++a) {
                            for (int b = a; b < nt && b < a + options.token_merge; ++b) {
                                add(tok_start[(size_t) a], tok_end[(size_t) b], 1.03, false);
                            }
                        }
                    }
                }
                if (!token_ok) {
                    const int ngram = std::min(options.max_char_ngram, len);
                    const int min_len = len <= 3 ? 1 : 2;
                    for (int l = min_len; l <= ngram; ++l) {
                        for (int s = run.start; s + l <= run.end; ++s) {
                            add(s, s + l, 1.0, false);
                        }
                    }
                }
                if (len <= options.max_span_cp) add(run.start, run.end, 1.0, false);
            }
        } else if (run.cls == CharClass::Word || run.cls == CharClass::Other) {
            if (len <= options.max_span_cp) add(run.start, run.end, 1.0, false);
            // Merge adjacent Latin words separated by a single space (Latin names).
            if (r + 1 < runs.size()) {
                const Run & next = runs[r + 1];
                auto starts_upper = [&](int at) {
                    const char32_t cp = text.cps[(size_t) at];
                    return cp >= U'A' && cp <= U'Z';
                };
                if ((run.cls == CharClass::Word || next.cls == CharClass::Word) &&
                    next.start == run.end + 1 && text.cps[(size_t) run.end] == ' ' &&
                    starts_upper(run.start) && starts_upper(next.start)) {
                    const int merged_len = next.end - run.start;
                    if (merged_len <= options.max_span_cp) {
                        add(run.start, next.end, 1.05, false);
                    }
                }
            }
        }
    }

    // Case numbers: code-point scan avoids std::regex quirks with fullwidth
    // parentheses while keeping the exact byte range.
    for (int s = 0; s + 7 <= text.size(); ++s) {
        const char32_t open = text.cps[(size_t) s];
        if (open != U'(' && open != (char32_t) 0xFF08) continue;
        bool four = true;
        for (int k = 1; k <= 4; ++k) {
            const char32_t d = text.cps[(size_t) (s + k)];
            if (d < '0' || d > '9') { four = false; break; }
        }
        if (!four) continue;
        const char32_t close = text.cps[(size_t) (s + 5)];
        if (close != U')' && close != (char32_t) 0xFF09) continue;
        int e = s + 6;
        while (e < text.size() && !is_space_cp(text.cps[(size_t) e]) &&
               !is_punct_cp(text.cps[(size_t) e])) {
            ++e;
        }
        if (e - s >= 8) add(s, e, 1.25, true);
    }

    // Generic structural patterns from the NIST-oriented PII format set.
    static const std::vector<std::pair<std::regex, double>> compiled = [] {
        std::vector<std::pair<std::regex, double>> out;
        for (const auto & pat : pii_patterns()) {
            out.emplace_back(std::regex(pat.regex, std::regex::optimize), pat.prior);
        }
        return out;
    }();

    for (const auto & [re, prior] : compiled) {
        for (std::sregex_iterator it(text.bytes.begin(), text.bytes.end(), re), end;
             it != end; ++it) {
            const size_t b = (size_t) it->position();
            const size_t e = b + (size_t) it->length();
            const int start = byte_to_cp(text, b);
            const int stop = byte_to_cp(text, e);
            add(start, stop, prior, true);
        }
    }

    std::vector<Candidate> out;
    out.reserve(unique.size());
    for (auto & [key, c] : unique) out.push_back(std::move(c));
    std::sort(out.begin(), out.end(), [](const Candidate & a, const Candidate & b) {
        if (a.start != b.start) return a.start < b.start;
        return a.end < b.end;
    });
    {
        CandidateCache & cache = candidate_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (auto it = cache.entries.find(cache_key); it != cache.entries.end()) {
            cache.order.splice(cache.order.begin(), cache.order, it->second.recency);
            it->second.candidates = out;
        } else {
            constexpr std::size_t max_entries = 8;
            while (!cache.order.empty() && cache.entries.size() + 1 > max_entries) {
                cache.entries.erase(cache.order.back());
                cache.order.pop_back();
            }
            cache.order.push_front(cache_key);
            cache.entries.emplace(cache_key,
                                  CandidateCacheEntry{out, cache.order.begin()});
        }
    }
    return out;
}

} // namespace anypii
