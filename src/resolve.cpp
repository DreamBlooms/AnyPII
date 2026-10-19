#include "anypii/resolve.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <string>
#include <regex>

#include "anypii/candidates.hpp"
#include "anypii/mask.hpp"
#include "anypii/nist.hpp"
#include "anypii/prompt.hpp"
#include "anypii/unicode.hpp"

namespace anypii {

namespace {

struct CandScore {
    bool keep = false;
    int best = -1;
    double bestp = 0.0;
    int second = -1;
    double secondp = 0.0;
    double exactp = 0.0;
    double none_p = 0.0;
};

struct Component {
    std::vector<int> members;
    int start = 0;
    int end = 0;
};

std::string pair_signature(const PairChoice & pc) {
    std::string s = pc.criterion;
    for (size_t i = 0; i < pc.choices.size(); ++i) {
        s += '\x1f';
        s += pc.choices[i];
        if (i < pc.types.size()) {
            s += '\x1e';
            s += pc.types[i];
        }
    }
    return s;
}

std::vector<Component> make_components(const std::vector<int> & members,
                                       const std::vector<Candidate> & candidates) {
    std::vector<int> sorted = members;
    std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
        if (candidates[a].start != candidates[b].start)
            return candidates[a].start < candidates[b].start;
        return candidates[a].end < candidates[b].end;
    });
    std::vector<Component> out;
    for (int idx : sorted) {
        const Candidate & c = candidates[idx];
        if (out.empty() || c.start >= out.back().end) {
            Component comp;
            comp.members.push_back(idx);
            comp.start = c.start;
            comp.end = c.end;
            out.push_back(std::move(comp));
        } else {
            out.back().members.push_back(idx);
            out.back().end = std::max(out.back().end, c.end);
        }
    }
    return out;
}

void account_branches(AnyPiiResult & out, size_t n, int per_decode) {
    if (n == 0) return;
    if (per_decode <= 0) per_decode = 1;
    out.decodes += 1 + (int) ((n + (size_t) per_decode - 1) / (size_t) per_decode);
    out.slots += (int) n;
}

const TypeDef * find_type(const AnyPiiRequest & req, const std::string & id) {
    for (const auto & t : req.types) {
        if (t.id == id) return &t;
    }
    return nullptr;
}

std::string clean_line(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        if (i + 1 < s.size() && s[i] == '<' && s[i + 1] == '|') { out += "< |"; i += 2; }
        else if (i + 1 < s.size() && s[i] == '|' && s[i + 1] == '>') { out += "| >"; i += 2; }
        else {
            char c = s[i];
            if (c == '\n' || c == '\r') c = ' ';
            out.push_back(c);
            ++i;
        }
    }
    return out;
}

std::string context_of(const Utf8Text & text, int start, int end, int radius = 20) {
    const int left = std::max(0, start - radius);
    const int right = std::min(text.size(), end + radius);
    return clean_line(text.substr(left, start)) + "【" + clean_line(text.substr(start, end)) +
           "】" + clean_line(text.substr(end, right));
}

int longest_common_substring(const std::u32string & a, const std::u32string & b) {
    int best = 0;
    std::vector<int> prev(b.size() + 1, 0), cur(b.size() + 1, 0);
    for (size_t i = 0; i < a.size(); ++i) {
        std::fill(cur.begin(), cur.end(), 0);
        for (size_t j = 0; j < b.size(); ++j) {
            if (a[i] == b[j]) {
                cur[j + 1] = prev[j] + 1;
                if (cur[j + 1] > best) best = cur[j + 1];
            }
        }
        prev.swap(cur);
    }
    return best;
}

std::string ascii_lower_copy(const std::string & s) {
    std::string out = s;
    for (char & c : out) c = (char) std::tolower((unsigned char) c);
    return out;
}

// A candidate that matches the caller's own type vocabulary is a label, not
// the entity itself. Two generic signals are used: whole words/acronyms of the
// id/description (e.g. "SSN", "DOB") and any 2..6 code-point substring of the
// id/description (e.g. "手机" inside "联系人手机"). No fixed word list.
bool is_type_label(const Utf8Text & text, const std::vector<TypeDef> & types,
                   const std::unordered_set<std::u32string> & ngrams) {
    if (text.size() < 2) return false;
    const std::string needle = ascii_lower_copy(text.bytes);
    for (const auto & def : types) {
        for (const std::string & field : {def.id, def.description}) {
            std::string acronym;
            size_t start = 0;
            while (start < field.size()) {
                while (start < field.size() &&
                       !std::isalnum((unsigned char) field[start])) {
                    ++start;
                }
                size_t end = start;
                while (end < field.size() &&
                       std::isalnum((unsigned char) field[end])) {
                    ++end;
                }
                if (end == start) break;
                const std::string word = ascii_lower_copy(field.substr(start, end - start));
                if (word == needle) return true;
                acronym.push_back(word.front());
                start = end;
            }
            if (acronym.size() >= 2 && acronym == needle) return true;
        }
    }
    const int max_len = std::min(6, text.size());
    for (int len = 2; len <= max_len; ++len) {
        for (int start = 0; start + len <= text.size(); ++start) {
            if (ngrams.count(text.cps.substr((size_t) start, (size_t) len))) return true;
        }
    }
    return false;
}

std::unordered_set<std::u32string> build_type_label_ngrams(
        const std::vector<TypeDef> & types) {
    std::unordered_set<std::u32string> out;
    for (const auto & def : types) {
        for (const std::string & field : {def.id, def.description}) {
            const Utf8Text t = decode_utf8(field);
            const int max_len = std::min(6, t.size());
            for (int len = 2; len <= max_len; ++len) {
                for (int start = 0; start + len <= t.size(); ++start) {
                    out.insert(t.cps.substr((size_t) start, (size_t) len));
                }
            }
        }
    }
    return out;
}

double length_type_bonus(int len, const TypeDef & t) {
    const std::string l = std::to_string(len);
    std::string d = t.id + " " + t.description;
    size_t pos = d.find(l);
    while (pos != std::string::npos) {
        const bool left = pos == 0 || !std::isdigit((unsigned char) d[pos - 1]);
        const bool right = pos + l.size() == d.size() ||
                           !std::isdigit((unsigned char) d[pos + l.size()]);
        if (left && right) return 3.0;
        pos = d.find(l, pos + 1);
    }
    return 1.0;
}

double lexical_type_bonus(const Utf8Text & span, const TypeDef & t) {
    const int ov = std::max(
        longest_common_substring(span.cps, decode_utf8(t.description).cps),
        longest_common_substring(span.cps, decode_utf8(t.id).cps));
    if (ov >= 2) return 1.0 + 0.8 * (ov - 1);
    return 1.0;
}

double length_factor(int len) {
    if (len <= 1) return 0.80;
    return std::min(1.50, 1.0 + 0.10 * (len - 2));
}

double boundary_factor(const std::string & span) {
    const Utf8Text t = decode_utf8(span);
    if (t.size() == 0) return 1.0;
    double f = 1.0;
    if (is_punct_cp(t.cps.front()) || is_space_cp(t.cps.front())) f *= 0.60;
    if (is_punct_cp(t.cps.back()) || is_space_cp(t.cps.back())) f *= 0.60;
    for (size_t i = 1; i + 1 < t.cps.size(); ++i) {
        if (is_punct_cp(t.cps[i]) || is_space_cp(t.cps[i])) f *= 0.70;
    }
    return std::max(0.2, f);
}

} // namespace

AnyPiiResult run_pipeline(Engine & engine, const AnyPiiRequest & request,
                          const ResolveOptions & options) {
    AnyPiiResult out;
    out.model = engine.model_name();
    out.backend = engine.backend_name();

    const Utf8Text text = decode_utf8(request.text);
    if (text.size() > options.max_text_cp) {
        throw std::invalid_argument(
            "state.text is too long (max " + std::to_string(options.max_text_cp) +
            " code points)");
    }
    if (text.size() == 0) {
        out.redacted_text = request.text;
        return out;
    }

    const TypeTable table = build_type_table(request.types);
    const int type_count = (int) table.types.size();
    if (type_count == 0) {
        out.redacted_text = request.text;
        return out;
    }
    if (request.skip_keep_types) {
        bool any_active = false;
        for (const auto & t : request.types) {
            if (t.redact != RedactMode::Keep) { any_active = true; break; }
        }
        if (!any_active) {
            out.redacted_text = request.text;
            return out;
        }
    }

    const auto proposed = propose_candidates(text, engine, CandidateOptions{});
    if ((int) proposed.size() > options.max_candidates) {
        throw std::invalid_argument(
            "too many candidate spans (max " + std::to_string(options.max_candidates) +
            "); shorten the text");
    }
    if (std::getenv("ANYPPI_DEBUG")) {
        std::cerr << "[candidates] proposed=" << proposed.size() << "\n";
        for (const auto & c : proposed) {
            std::cerr << "  P " << c.start << "-" << c.end << " [" << c.text
                      << "] regex=" << c.regex_hit << " bf=" << boundary_factor(c.text)
                      << "\n";
        }
    }
    // Candidate prefilter: single characters and spans with a leading/trailing
    // role/function cue are rarely the entity itself. Regex matches are kept.
    std::vector<PhraseHit> hits;
    std::vector<std::pair<int, int>> consumed;
    std::vector<Candidate> candidates;
    candidates.reserve(proposed.size());
    const bool skip_keep = request.skip_keep_types;
    const auto type_label_ngrams = build_type_label_ngrams(request.types);

    for (const auto & c : proposed) {
        if (!c.regex_hit) {
            if (c.end - c.start <= 1) continue;
            if (is_type_label(decode_utf8(c.text), request.types, type_label_ngrams)) continue;
            if (boundary_factor(c.text) < 0.75) continue;
        }
        bool already_consumed = false;
        for (const auto & r : consumed) {
            if (c.start < r.second && r.first < c.end) { already_consumed = true; break; }
        }
        if (already_consumed) continue;

        bool handled = false;
        if (c.regex_hit) {
            const PiiFormat fmt = detect_format(c.text);
            std::vector<int> ftypes;
            for (int t = 0; t < type_count; ++t) {
                if (type_matches_format(fmt, table.types[(size_t) t])) {
                    ftypes.push_back(t);
                }
            }
            std::vector<int> len_types;
            if (is_ascii_digits(c.text)) {
                const int cp_len = c.end - c.start;
                for (int t = 0; t < type_count; ++t) {
                    if (length_type_bonus(cp_len, table.types[(size_t) t]) > 1.0) {
                        len_types.push_back(t);
                    }
                }
            }
            int forced = -1;
            if (ftypes.size() == 1) forced = ftypes[0];
            else if (ftypes.empty() && len_types.size() == 1) forced = len_types[0];
            if (forced >= 0) {
                if (skip_keep && table.types[(size_t) forced].redact == RedactMode::Keep) {
                    consumed.push_back({c.start, c.end});
                } else {
                    PhraseHit h;
                    h.start = c.start;
                    h.end = c.end;
                    h.text = c.text;
                    h.type = table.types[(size_t) forced].id;
                    h.score = 1.0;
                    hits.push_back(std::move(h));
                }
                handled = true;
            }
        }
        if (!handled) candidates.push_back(c);
    }
    if (!consumed.empty()) {
        std::vector<PhraseHit> filtered_hits;
        for (auto & h : hits) {
            bool ov = false;
            for (const auto & r : consumed) {
                if (h.start < r.second && r.first < h.end) { ov = true; break; }
            }
            if (!ov) filtered_hits.push_back(h);
        }
        hits.swap(filtered_hits);
    }
    if (!hits.empty() || !consumed.empty()) {
        std::sort(hits.begin(), hits.end(), [](const PhraseHit & a, const PhraseHit & b) {
            if (a.start != b.start) return a.start < b.start;
            return a.end > b.end;
        });
        std::vector<PhraseHit> dedup;
        for (auto & h : hits) {
            bool merged = false;
            for (auto & d : dedup) {
                if (h.start < d.end && d.start < h.end) {
                    if ((h.end - h.start) > (d.end - d.start)) d = h;
                    merged = true;
                    break;
                }
            }
            if (!merged) dedup.push_back(h);
        }
        hits.swap(dedup);
        std::vector<Candidate> remaining;
        for (const auto & c : candidates) {
            bool overlaps = false;
            for (const auto & h : hits) {
                if (c.start < h.end && h.start < c.end) { overlaps = true; break; }
            }
            if (!overlaps) {
                for (const auto & r : consumed) {
                    if (c.start < r.second && r.first < c.end) { overlaps = true; break; }
                }
            }
            if (!overlaps) remaining.push_back(c);
        }
        candidates.swap(remaining);
    }
    if (std::getenv("ANYPPI_DEBUG")) {
        std::cerr << "[candidates] filtered=" << candidates.size() << "\n";
        for (const auto & c : candidates) {
            std::cerr << "  " << c.start << "-" << c.end << " [" << c.text << "] bf="
                      << boundary_factor(c.text) << "\n";
        }
    }
    if (candidates.empty()) goto finalize;

    {
    // ---------- Pass A: one shared evidence prefix, one branch per unique text ----------
    std::vector<CandScore> scores(candidates.size());
    {
        std::map<std::string, int> text_count;
        for (const auto & c : candidates) ++text_count[c.text];

        std::vector<Candidate> pass_a;
        std::vector<int> cand_to_unique(candidates.size(), -1);
        std::map<std::string, int> text_unique;
        for (size_t i = 0; i < candidates.size(); ++i) {
            const auto & c = candidates[i];
            const bool repeated_short =
                text_count[c.text] >= 2 && decode_utf8(c.text).size() <= 12;
            if (repeated_short) {
                auto it = text_unique.find(c.text);
                if (it == text_unique.end()) {
                    const int u = (int) pass_a.size();
                    pass_a.push_back(c);
                    text_unique.emplace(c.text, u);
                    cand_to_unique[i] = u;
                } else {
                    cand_to_unique[i] = it->second;
                }
            } else {
                cand_to_unique[i] = (int) pass_a.size();
                pass_a.push_back(c);
            }
        }

        BranchSet set = build_type_branches(engine, table, request, text, pass_a);
        const auto probs = engine.decode_branches(set.prefix, set.branches,
                                                  options.max_branches_per_decode,
                                                  options.temperature);
        account_branches(out, set.branches.size(), options.max_branches_per_decode);

        std::vector<CandScore> unique_scores(pass_a.size());
        for (size_t u = 0; u < pass_a.size(); ++u) {
            const auto & cand = pass_a[u];
            std::vector<double> p = probs[u];
            const Utf8Text span_text = decode_utf8(cand.text);
            double norm = 0.0;
            for (int t = 0; t < type_count; ++t) {
                p[(size_t) t] *= lexical_type_bonus(span_text, table.types[(size_t) t]);
                p[(size_t) t] *= length_type_bonus(cand.end - cand.start,
                                                   table.types[(size_t) t]);
                norm += p[(size_t) t];
            }
            norm += p[(size_t) type_count];
            if (norm > 0.0) {
                for (double & v : p) v /= norm;
            }
            CandScore cs;
            cs.exactp = 1.0;
            for (int t = 0; t < type_count; ++t) {
                if (p[(size_t) t] > cs.bestp) {
                    cs.second = cs.best;
                    cs.secondp = cs.bestp;
                    cs.best = t;
                    cs.bestp = p[(size_t) t];
                } else if (p[(size_t) t] > cs.secondp) {
                    cs.second = t;
                    cs.secondp = p[(size_t) t];
                }
            }
            cs.none_p = p[(size_t) type_count];
            if (cs.best < 0 || cs.none_p >= cs.bestp ||
                cs.bestp - cs.none_p < options.none_margin) {
                cs.best = -1;
                cs.second = -1;
                cs.secondp = 0.0;
                cs.bestp = cs.none_p;
                cs.keep = false;
            } else {
                cs.keep = cs.bestp >= request.threshold;
            }
            if (std::getenv("ANYPPI_DEBUG")) {
                std::cerr << "[A] [" << cand.text << "] "
                          << (cs.best >= 0 ? table.types[(size_t) cs.best].id : "?")
                          << "=" << cs.bestp << " none=" << cs.none_p;
                if (cs.second >= 0) {
                    std::cerr << " 2nd=" << table.types[(size_t) cs.second].id
                              << "=" << cs.secondp;
                }
                std::cerr << (cs.keep ? " KEEP" : "") << "\n";
            }
            unique_scores[u] = cs;
        }
        for (size_t i = 0; i < candidates.size(); ++i) {
            scores[i] = unique_scores[(size_t) cand_to_unique[i]];
        }
        if (std::getenv("ANYPPI_DEBUG")) {
            std::cerr << "[A] unique=" << pass_a.size()
                      << " candidates=" << candidates.size() << "\n";
        }
    }

    // ---------- Build per-component span options and pairwise boundary votes ----------
    struct SpanOpt {
        int cand = -1;
        int type = -1;
        double weight = 0.0;
        std::string text;
    };
    std::vector<int> active;
    for (int i = 0; i < (int) candidates.size(); ++i) {
        if (scores[(size_t) i].best >= 0 &&
            scores[(size_t) i].bestp >= options.active_floor) {
            active.push_back(i);
        }
    }
    if (active.empty()) goto finalize;

    auto comps = make_components(active, candidates);
    if (skip_keep) {
        std::vector<Component> kept_comps;
        for (const auto & comp : comps) {
            bool has_redact = false;
            for (int m : comp.members) {
                const auto & cs = scores[(size_t) m];
                if (cs.best >= 0 &&
                    table.types[(size_t) cs.best].redact != RedactMode::Keep) {
                    has_redact = true;
                    break;
                }
            }
            if (has_redact) kept_comps.push_back(comp);
        }
        comps.swap(kept_comps);
    }
    std::vector<std::vector<SpanOpt>> comp_opts;
    for (const auto & comp : comps) {
        std::map<std::pair<int, int>, SpanOpt> by_span;
        for (int m : comp.members) {
            const auto & cs = scores[(size_t) m];
            if (cs.best < 0) continue;
            const int len = candidates[(size_t) m].end - candidates[(size_t) m].start;
            double w = cs.bestp * candidates[(size_t) m].prior *
                       boundary_factor(candidates[(size_t) m].text) * length_factor(len);
            if (candidates[(size_t) m].regex_hit) w *= 4.0;
            if (candidates[(size_t) m].end - candidates[(size_t) m].start <= 1) w *= 0.5;
            auto key = std::make_pair(candidates[(size_t) m].start,
                                      candidates[(size_t) m].end);
            auto it = by_span.find(key);
            if (it == by_span.end() || it->second.weight < w) {
                SpanOpt o;
                o.cand = m;
                o.type = cs.best;
                o.weight = w;
                o.text = candidates[(size_t) m].text;
                by_span[key] = o;
            }
        }
        std::vector<SpanOpt> opts;
        for (auto & [key, o] : by_span) opts.push_back(o);
        std::sort(opts.begin(), opts.end(),
                  [](const SpanOpt & a, const SpanOpt & b) { return a.weight > b.weight; });
        if (opts.size() > 1) {
            const double top_w = opts[0].weight;
            int longest = -1;
            size_t longest_len = opts[0].text.size();
            for (int j = 1; j < (int) opts.size(); ++j) {
                const size_t len = opts[(size_t) j].text.size();
                if (len <= longest_len) continue;
                if (opts[(size_t) j].weight < 0.50 * top_w) continue;
                if (boundary_factor(opts[(size_t) j].text) < 0.75) continue;
                longest = j;
                longest_len = len;
            }
            if (longest >= 1) {
                opts[1] = opts[(size_t) longest];
                opts.resize(2);
            } else {
                opts.resize(1);
            }
        }
        comp_opts.push_back(std::move(opts));
    }

    struct PairMeta {
        int comp = -1;
        int a = -1;
        int b = -1;
        int pair_index = -1;
    };
    std::vector<PairChoice> pairs;
    std::vector<PairMeta> pair_meta;
    std::map<std::string, int> pair_cache;
    for (int ci = 0; ci < (int) comp_opts.size(); ++ci) {
        const auto & opts = comp_opts[(size_t) ci];
        for (int a = 0; a < (int) opts.size(); ++a) {
            for (int b = a + 1; b < (int) opts.size(); ++b) {
                PairChoice pc;
                pc.choices = {"\"" + clean_line(opts[(size_t) a].text) + "\"",
                              "\"" + clean_line(opts[(size_t) b].text) + "\"",
                              "neither"};
                pc.types = {table.types[(size_t) opts[(size_t) a].type].id,
                            table.types[(size_t) opts[(size_t) b].type].id, ""};
                pc.criterion =
                    "Two candidate spans refer to the same mention in the Evidence. "
                    "Which one is the complete, accurate entity mention? Choose the span "
                    "that includes all and only the entity characters; do not include "
                    "titles, roles, category cues, or neighboring words. Choose neither "
                    "if neither is a real entity.";
                const std::string sig = pair_signature(pc);
                auto it = pair_cache.find(sig);
                int pi = -1;
                if (it == pair_cache.end()) {
                    pi = (int) pairs.size();
                    pair_cache.emplace(sig, pi);
                    pairs.push_back(std::move(pc));
                } else {
                    pi = it->second;
                }
                pair_meta.push_back({ci, a, b, pi});
            }
        }
    }

    std::vector<std::vector<int>> wins(comp_opts.size());
    for (size_t i = 0; i < comp_opts.size(); ++i) {
        wins[i].assign(comp_opts[i].size(), 0);
    }

    if (!pairs.empty()) {
        BranchSet set = build_pair_branches(engine, request, pairs);
        const auto probs = engine.decode_branches(set.prefix, set.branches,
                                                  options.max_branches_per_decode,
                                                  options.temperature);
        account_branches(out, set.branches.size(), options.max_branches_per_decode);
        for (size_t k = 0; k < pair_meta.size(); ++k) {
            const int pi = pair_meta[k].pair_index;
            if (pi < 0 || 2 * pi + 1 >= (int) probs.size()) continue;
            const auto & pf = probs[2 * pi];
            const auto & pr = probs[2 * pi + 1];
            if (pf.size() < 3 || pr.size() < 3) continue;
            const double avg0 = 0.5 * (pf[0] + pr[2]);
            const double avg1 = 0.5 * (pf[1] + pr[1]);
            const double avg2 = 0.5 * (pf[2] + pr[0]);
            int pick = 0;
            if (avg1 > avg0 && avg1 >= avg2) pick = 1;
            else if (avg2 > avg0 && avg2 > avg1) pick = 2;
            if (std::getenv("ANYPPI_DEBUG")) {
                std::cerr << "[P] comp=" << pair_meta[k].comp << " "
                          << pair_meta[k].a << "v" << pair_meta[k].b
                          << " pick=" << pick << " p=" << (pick == 0 ? avg0 : pick == 1 ? avg1 : avg2)
                          << "\n";
            }
            if (pick == 0) wins[(size_t) pair_meta[k].comp][(size_t) pair_meta[k].a]++;
            else if (pick == 1) wins[(size_t) pair_meta[k].comp][(size_t) pair_meta[k].b]++;
        }
    }

    struct Selected {
        int cand = -1;
        int type = -1;
        double weight = 0.0;
        std::string text;
    };
    std::vector<Selected> selected;
    for (int ci = 0; ci < (int) comp_opts.size(); ++ci) {
        const auto & opts = comp_opts[(size_t) ci];
        if (opts.empty()) continue;
        if (opts.size() == 1) {
            const auto & cs0 = scores[(size_t) opts[0].cand];
            const double eff = request.threshold -
                (table.types[(size_t) opts[0].type].redact == RedactMode::Redact ? 0.05 : 0.0);
            if (cs0.bestp >= eff) {
                selected.push_back({opts[0].cand, opts[0].type, opts[0].weight, opts[0].text});
            }
            continue;
        }
        int best = 0;
        for (int j = 1; j < (int) opts.size(); ++j) {
            if (wins[(size_t) ci][(size_t) j] > wins[(size_t) ci][(size_t) best] ||
                (wins[(size_t) ci][(size_t) j] == wins[(size_t) ci][(size_t) best] &&
                 opts[(size_t) j].weight > opts[(size_t) best].weight)) {
                best = j;
            }
        }
        if (wins[(size_t) ci][(size_t) best] == 0) continue;

        // Completeness prior: if the winning span is a proper substring of a
        // longer high-weight candidate, prefer the longer complete mention.
        int chosen = best;
        for (int j = 0; j < (int) opts.size(); ++j) {
            if (j == chosen) continue;
            if (opts[(size_t) j].text.size() <= opts[(size_t) chosen].text.size()) continue;
            if (opts[(size_t) j].text.find(opts[(size_t) chosen].text) == std::string::npos) continue;
            if (opts[(size_t) j].weight < 0.70 * opts[(size_t) chosen].weight) continue;
            if (boundary_factor(opts[(size_t) j].text) < 0.75) continue;
            chosen = j;
        }
        selected.push_back({opts[(size_t) chosen].cand, opts[(size_t) chosen].type,
                            opts[(size_t) chosen].weight, opts[(size_t) chosen].text});
    }

    if (selected.empty()) goto finalize;

    // ---------- Pairwise type refinement ----------
    struct TypeMeta {
        int selected = -1;
        std::vector<int> types;
        int pair_index = -1;
    };
    std::vector<PairChoice> type_pairs;
    std::vector<TypeMeta> type_meta;
    std::vector<int> pair_selected;
    std::map<std::string, int> type_pair_cache;

    auto type_overlap = [&](const Utf8Text & span, int t) {
        return std::max(
            longest_common_substring(span.cps,
                decode_utf8(table.types[(size_t) t].description).cps),
            longest_common_substring(span.cps,
                decode_utf8(table.types[(size_t) t].id).cps));
    };
    auto direct_hit = [&](int si, int type) {
        if (skip_keep && table.types[(size_t) type].redact == RedactMode::Keep) return;
        if (std::getenv("ANYPPI_DEBUG")) {
            std::cerr << "[D] [" << selected[(size_t) si].text << "] -> "
                      << table.types[(size_t) type].id << "\n";
        }
        PhraseHit h;
        h.start = candidates[(size_t) selected[(size_t) si].cand].start;
        h.end = candidates[(size_t) selected[(size_t) si].cand].end;
        h.text = selected[(size_t) si].text;
        h.type = table.types[(size_t) type].id;
        h.score = scores[(size_t) selected[(size_t) si].cand].bestp;
        hits.push_back(std::move(h));
    };

    // Skip the pairwise type pass when the answer is already determined by the
    // model margin or by a caller-description overlap.
    for (int si = 0; si < (int) selected.size(); ++si) {
        const auto & cs = scores[(size_t) selected[(size_t) si].cand];
        const Utf8Text span_text = decode_utf8(selected[(size_t) si].text);

        int ov_max = type_overlap(span_text, cs.best);
        int ov_t = -1;
        for (int t = 0; t < type_count; ++t) {
            if (t == cs.best) continue;
            const int ov = type_overlap(span_text, t);
            if (ov >= 2 && ov > ov_max) {
                ov_max = ov;
                ov_t = t;
            }
        }
        if (ov_t >= 0) {
            direct_hit(si, ov_t);
            continue;
        }

        if (cs.bestp >= 0.80 && cs.bestp - cs.secondp >= 0.30) {
            direct_hit(si, cs.best);
            continue;
        }
        pair_selected.push_back(si);
    }

    for (int si : pair_selected) {
        const auto & cs = scores[(size_t) selected[(size_t) si].cand];
        std::vector<int> cand_types;
        auto add_type = [&](int t) {
            if (t < 0 || t >= type_count) return;
            if (std::find(cand_types.begin(), cand_types.end(), t) != cand_types.end()) return;
            cand_types.push_back(t);
        };
        const Utf8Text span_text = decode_utf8(selected[(size_t) si].text);
        add_type(cs.best);
        if (cs.second >= 0 && cs.secondp >= 0.05) add_type(cs.second);
        for (int t = 0; t < type_count && (int) cand_types.size() < 5; ++t) {
            if (lexical_type_bonus(span_text, table.types[(size_t) t]) > 1.0) add_type(t);
        }

        PairChoice pc;
        for (int t : cand_types) {
            pc.choices.push_back(table.types[(size_t) t].id + ": " +
                                 clean_line(table.types[(size_t) t].description));
            pc.types.push_back(table.types[(size_t) t].id);
        }
        pc.choices.push_back("none: not an entity of these types (e.g., a disease, symptom, action, or ordinary word)");
        pc.types.push_back("");

        pc.criterion = "Mention: \"" + clean_line(selected[(size_t) si].text) +
                       "\"\nContext: \"" +
                       context_of(text, candidates[(size_t) selected[(size_t) si].cand].start,
                                  candidates[(size_t) selected[(size_t) si].cand].end) +
                       "\"\nWhich type best describes this mention in the Evidence? "
                       "Choose the option that matches the exact mention; choose none if "
                       "it is not a real entity of these types.";
        const std::string sig = pair_signature(pc);
        auto it = type_pair_cache.find(sig);
        int pi = -1;
        if (it == type_pair_cache.end()) {
            pi = (int) type_pairs.size();
            type_pair_cache.emplace(sig, pi);
            type_pairs.push_back(std::move(pc));
        } else {
            pi = it->second;
        }
        type_meta.push_back({si, cand_types, pi});
    }

    {
        BranchSet set = build_pair_branches(engine, request, type_pairs);
        const auto probs = engine.decode_branches(set.prefix, set.branches,
                                                  options.max_branches_per_decode,
                                                  options.temperature);
        account_branches(out, set.branches.size(), options.max_branches_per_decode);

        for (size_t k = 0; k < type_meta.size(); ++k) {
            const int pi = type_meta[k].pair_index;
            if (pi < 0 || 2 * pi + 1 >= (int) probs.size()) continue;
            const auto & pf = probs[2 * pi];
            const auto & pr = probs[2 * pi + 1];
            const size_t n = pf.size();
            if (n == 0 || pr.size() != n) continue;
            const int si = type_meta[k].selected;
            const std::vector<int> & ts = type_meta[k].types;
            int pick = 0;
            double best = -1.0;
            const Utf8Text span_text = decode_utf8(selected[(size_t) si].text);
            std::vector<double> vals(n, 0.0);
            double vsum = 0.0;
            for (size_t j = 0; j < n; ++j) {
                double v = 0.5 * (pf[j] + pr[n - 1 - j]);
                if (j < ts.size() && ts[j] >= 0) {
                    v *= lexical_type_bonus(span_text, table.types[(size_t) ts[j]]);
                    v *= length_type_bonus(
                        (int) span_text.size(), table.types[(size_t) ts[j]]);
                }
                vals[j] = v;
                vsum += v;
            }
            if (vsum > 0.0) {
                for (double & v : vals) v /= vsum;
            }
            for (size_t j = 0; j < n; ++j) {
                if (vals[j] > best) { best = vals[j]; pick = (int) j; }
            }
            int chosen = -1;
            if (pick < (int) ts.size()) chosen = ts[(size_t) pick];

            if (chosen >= 0) {
                const Utf8Text span_text = decode_utf8(selected[(size_t) si].text);
                int best_overlap = 0;
                for (int t = 0; t < type_count; ++t) {
                    if (t == chosen) {
                        best_overlap = std::max(
                            best_overlap,
                            longest_common_substring(span_text.cps,
                                decode_utf8(table.types[(size_t) t].description).cps));
                        best_overlap = std::max(
                            best_overlap,
                            longest_common_substring(span_text.cps,
                                decode_utf8(table.types[(size_t) t].id).cps));
                    }
                }
                int override_t = -1;
                int override_overlap = best_overlap;
                for (int t = 0; t < type_count; ++t) {
                    if (t == chosen) continue;
                    const int ov = std::max(
                        longest_common_substring(span_text.cps,
                            decode_utf8(table.types[(size_t) t].description).cps),
                        longest_common_substring(span_text.cps,
                            decode_utf8(table.types[(size_t) t].id).cps));
                    if (ov >= 2 && ov > override_overlap) {
                        override_overlap = ov;
                        override_t = t;
                    }
                }
                if (override_t >= 0) {
                    if (std::getenv("ANYPPI_DEBUG")) {
                        std::cerr << "[T-override] [" << selected[(size_t) si].text
                                  << "] -> " << table.types[(size_t) override_t].id
                                  << " overlap=" << override_overlap << "\n";
                    }
                    chosen = override_t;
                }
            }
            if (std::getenv("ANYPPI_DEBUG")) {
                std::cerr << "[T] [" << selected[(size_t) si].text << "] pick=" << pick
                          << "/" << n << " p=" << best
                          << " type=" << (chosen >= 0 ? table.types[(size_t) chosen].id : "none")
                          << "\n";
            }
            if (chosen < 0) continue;
            if (skip_keep && table.types[(size_t) chosen].redact == RedactMode::Keep) continue;
            PhraseHit h;
            h.start = candidates[(size_t) selected[(size_t) si].cand].start;
            h.end = candidates[(size_t) selected[(size_t) si].cand].end;
            h.text = selected[(size_t) si].text;
            h.type = table.types[(size_t) chosen].id;
            h.score = scores[(size_t) selected[(size_t) si].cand].bestp * best;
            hits.push_back(std::move(h));
        }
    }

    } // end model pipeline

finalize:
    if (std::getenv("ANYPPI_DEBUG")) {
        std::cerr << "[H] hits before finalize=" << hits.size() << "\n";
        for (const auto & h : hits) {
            std::cerr << "  [" << h.text << "] " << h.type << " " << h.score << "\n";
        }
    }
    // ---------- Enforce non-overlap ----------
    std::sort(hits.begin(), hits.end(), [](const PhraseHit & a, const PhraseHit & b) {
        if (a.start != b.start) return a.start < b.start;
        return a.end < b.end;
    });
    std::vector<PhraseHit> final_hits;
    for (auto & h : hits) {
        if (final_hits.empty() || h.start >= final_hits.back().end) {
            final_hits.push_back(h);
        } else if (h.score > final_hits.back().score) {
            final_hits.back() = h;
        }
    }

    // ---------- Redaction decisions ----------
    std::vector<RedactItem> auto_items;
    std::vector<int> auto_index;
    for (int i = 0; i < (int) final_hits.size(); ++i) {
        auto & h = final_hits[(size_t) i];
        const TypeDef * t = find_type(request, h.type);
        if (!t) continue;
        if (t->redact == RedactMode::Redact) {
            h.redacted = true;
        } else if (t->redact == RedactMode::Keep) {
            h.redacted = false;
        } else {
            auto_items.push_back(RedactItem{h.start, h.end, h.text, h.type});
            auto_index.push_back(i);
        }
    }

    if (!auto_items.empty()) {
        BranchSet set = build_redact_branches(engine, table, request, auto_items);
        const auto probs = engine.decode_branches(set.prefix, set.branches,
                                                  options.max_branches_per_decode,
                                                  options.temperature);
        account_branches(out, set.branches.size(), options.max_branches_per_decode);
        for (size_t i = 0; i < auto_items.size() && 2 * i + 1 < probs.size(); ++i) {
            const auto & pf = probs[2 * i];
            const auto & pr = probs[2 * i + 1];
            double p_redact = 0.0;
            if (pf.size() >= 2 && pr.size() >= 2) {
                p_redact = 0.5 * (pf[0] + pr[1]);
            }
            auto & h = final_hits[(size_t) auto_index[i]];
            h.redact_score = p_redact;
            h.redacted = p_redact >= request.threshold;
        }
    }

    // Group repeated occurrences of the same entity (type + text + redacted
    // flag) into one EntitySpan. Group order and occurrence order follow the
    // start position.
    std::vector<EntitySpan> grouped;
    std::vector<int> hit_group(final_hits.size());
    std::map<std::tuple<std::string, std::string, bool>, int> group_lookup;
    for (size_t i = 0; i < final_hits.size(); ++i) {
        const auto & h = final_hits[i];
        const auto key = std::make_tuple(h.type, h.text, h.redacted);
        auto it = group_lookup.find(key);
        int gi = -1;
        if (it == group_lookup.end()) {
            gi = (int) grouped.size();
            EntitySpan s;
            s.type = h.type;
            s.text = h.text;
            s.redacted = h.redacted;
            grouped.push_back(std::move(s));
            group_lookup.emplace(key, gi);
        } else {
            gi = it->second;
        }
        SpanOccurrence occ;
        occ.start = h.start;
        occ.end = h.end;
        occ.score = h.score;
        occ.redact_score = h.redact_score;
        grouped[(size_t) gi].occurrences.push_back(occ);
        hit_group[i] = gi;
    }

    out.spans = std::move(grouped);
    out.redacted_text = apply_mask(text, final_hits, request.mask_style, &hit_group);
    out.generated_tokens = 0;
    return out;
}

} // namespace anypii
