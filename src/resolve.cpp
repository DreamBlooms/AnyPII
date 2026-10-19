#include "anypii/resolve.hpp"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <regex>

#include "anypii/candidates.hpp"
#include "anypii/mask.hpp"
#include "anypii/nist.hpp"
#include "anypii/pcd.hpp"
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
    std::vector<double> probs;
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
    // A label is typically a prefix ("患者李小明", "商品名阿莫仙") or a very
    // short suffix ("的身份证号"). Do not reject a longer entity merely because
    // it contains a category word ("辉瑞制药", "北京星河科技有限公司").
    const int max_len = std::min(6, text.size());
    for (int len = 2; len <= max_len; ++len) {
        if (len == text.size() && ngrams.count(text.cps)) return true;
        if (len < text.size() && ngrams.count(text.cps.substr(0, (size_t) len))) {
            return true;
        }
    }
    for (int len = 2; len <= max_len; ++len) {
        const int prefix = text.size() - len;
        if (prefix > 1) continue;
        if (ngrams.count(text.cps.substr((size_t) prefix, (size_t) len))) return true;
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
        if (is_punct_cp(t.cps[i])) f *= 0.70;
    }
    return std::max(0.2, f);
}

} // namespace


namespace {

std::string sanitize_pcd_text(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        if (i + 1 < s.size() && s[i] == '<' && s[i + 1] == '|') {
            out += "< |";
            i += 2;
        } else if (i + 1 < s.size() && s[i] == '|' && s[i + 1] == '>') {
            out += "| >";
            i += 2;
        } else {
            out.push_back(s[i++]);
        }
    }
    return out;
}

std::string pcd_chat_prefix(const std::string & system, const std::string & evidence) {
    return "<|im_start|>system\n" + sanitize_pcd_text(system) +
           "<|im_end|>\n<|im_start|>user\n" + sanitize_pcd_text(evidence) +
           "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{\n";
}

std::string pcd_field_name(const std::string & hint, int index) {
    const Utf8Text t = decode_utf8(hint);
    std::string name = clean_line(t.substr(0, std::min(12, t.size())));
    for (char & c : name) {
        if (c == '"' || c == '\\') c = '_';
    }
    if (name.empty()) name = "field";
    return name + "#" + std::to_string(index);
}

std::string pcd_type_legend(const TypeTable & table) {
    std::string out = "Entity types:\n";
    for (const auto & t : table.types) {
        out += "- " + t.id + ": " + t.description + "\n";
    }
    return out;
}

void emit_grouped(AnyPiiResult & out, const Utf8Text & text,
                  const std::vector<PhraseHit> & final_hits,
                  const AnyPiiRequest & request);

void add_pcd_metrics(AnyPiiResult & out, const PcdMetrics & m, int fields) {
    out.pcd = true;
    out.decodes += m.forward_passes;
    out.slots += fields;
    out.pcd_metrics.forward_passes += m.forward_passes;
    out.pcd_metrics.elapsed_ms += m.elapsed_ms;
    out.pcd_metrics.checkpoint_bytes =
        std::max(out.pcd_metrics.checkpoint_bytes, m.checkpoint_bytes);
    out.pcd_metrics.cache_status = m.cache_status;
    out.pcd_metrics.phases.tokenize_ms += m.phases.tokenize_ms;
    out.pcd_metrics.phases.restore_or_prefill_ms += m.phases.restore_or_prefill_ms;
    out.pcd_metrics.phases.broadcast_ms += m.phases.broadcast_ms;
    out.pcd_metrics.phases.suffix_ms += m.phases.suffix_ms;
    out.pcd_metrics.phases.tree_ms += m.phases.tree_ms;
}

void emit_grouped(AnyPiiResult & out, const Utf8Text & text,
                  const std::vector<PhraseHit> & final_hits,
                  const AnyPiiRequest & request) {
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
}

} // namespace

namespace {

struct PriorCacheEntry {
    CandScore score;
    std::list<uint64_t>::iterator recency;
};

struct PriorCache {
    std::mutex mutex;
    std::list<uint64_t> order;
    std::unordered_map<uint64_t, PriorCacheEntry> entries;
};

PriorCache & prior_cache() {
    static PriorCache cache;
    return cache;
}

} // namespace

std::vector<CandScore> readout_type_scores(
        Engine & engine, const TypeTable & table, const AnyPiiRequest & request,
        const Utf8Text & text, const std::vector<Candidate> & candidates,
        const ResolveOptions & options, AnyPiiResult & out) {
    std::vector<CandScore> scores(candidates.size());
    if (candidates.empty()) return scores;
    const int type_count = (int) table.types.size();

    std::string base = "prior-v1|" + engine.model_name() + "|" +
                       std::to_string(options.temperature) + "|" +
                       std::to_string(options.none_margin) + "|" +
                       std::to_string(options.none_margin_max_len) + "|" +
                       std::to_string(options.none_margin_redact) + "|" +
                       std::to_string((int) options.compact_branches) + "|" +
                       std::to_string(options.max_branches_per_decode) + "|" +
                       request.policy + "|" + text.bytes + "|";
    for (const auto & type : table.types) {
        base += type.id + ":" + type.description + ":" +
                std::to_string((int) type.redact) + "|";
    }
    for (const auto & candidate : candidates) {
        base += candidate.text + "|" + std::to_string(candidate.start) + "|" +
                std::to_string(candidate.end) + "|";
    }

    std::vector<int> miss_indices;
    std::vector<Candidate> miss_candidates;
    {
        PriorCache & cache = prior_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        for (size_t i = 0; i < candidates.size(); ++i) {
            const std::string key_text = base + candidates[i].text + "|" +
                                         std::to_string(candidates[i].start) + "|" +
                                         std::to_string(candidates[i].end);
            const uint64_t key = (uint64_t) std::hash<std::string>{}(key_text);
            auto it = cache.entries.find(key);
            if (it != cache.entries.end()) {
                cache.order.splice(cache.order.begin(), cache.order, it->second.recency);
                scores[i] = it->second.score;
                ++out.pcd_metrics.prior_cache_hits;
            } else {
                miss_indices.push_back((int) i);
                miss_candidates.push_back(candidates[i]);
            }
        }
        out.pcd_metrics.prior_cache_entries = (int) cache.entries.size();
    }
    if (miss_candidates.empty()) return scores;

    BranchSet set = build_type_branches(engine, table, request, text, miss_candidates,
                                        options.compact_branches);
    const auto probs = engine.decode_branches(set.prefix, set.branches,
                                              options.max_branches_per_decode,
                                              options.temperature);
    account_branches(out, set.branches.size(), options.max_branches_per_decode);
    for (size_t j = 0; j < miss_candidates.size() && j < probs.size(); ++j) {
        std::vector<double> p = probs[j];
        if ((int) p.size() <= type_count) continue;
        const Utf8Text span_text = decode_utf8(miss_candidates[j].text);
        double norm = 0.0;
        for (int t = 0; t < type_count; ++t) {
            p[(size_t) t] *= lexical_type_bonus(span_text, table.types[(size_t) t]);
            p[(size_t) t] *= length_type_bonus(
                miss_candidates[j].end - miss_candidates[j].start,
                table.types[(size_t) t]);
            norm += p[(size_t) t];
        }
        norm += p[(size_t) type_count];
        if (norm > 0.0) {
            for (double & v : p) v /= norm;
        }
        CandScore cs;
        cs.exactp = 1.0;
        cs.probs = p;
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
        const int cand_len = miss_candidates[j].end - miss_candidates[j].start;
        const bool apply_margin = cand_len <= options.none_margin_max_len;
        const double margin =
            (cs.best >= 0 && table.types[(size_t) cs.best].redact == RedactMode::Redact)
                ? options.none_margin_redact
                : options.none_margin;
        if (cs.best < 0 ||
            (apply_margin && (cs.none_p >= cs.bestp || cs.bestp - cs.none_p < margin))) {
            cs.best = -1;
            cs.second = -1;
            cs.secondp = 0.0;
            cs.bestp = cs.none_p;
        }
        const int original = miss_indices[j];
        scores[(size_t) original] = cs;
        const std::string key_text = base + miss_candidates[j].text + "|" +
                                     std::to_string(miss_candidates[j].start) + "|" +
                                     std::to_string(miss_candidates[j].end);
        const uint64_t key = (uint64_t) std::hash<std::string>{}(key_text);
        PriorCache & cache = prior_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (auto old = cache.entries.find(key); old != cache.entries.end()) {
            cache.order.erase(old->second.recency);
            cache.entries.erase(old);
        }
        constexpr std::size_t max_entries = 2048;
        while (!cache.order.empty() && cache.entries.size() + 1 > max_entries) {
            cache.entries.erase(cache.order.back());
            cache.order.pop_back();
        }
        cache.order.push_front(key);
        cache.entries.emplace(key, PriorCacheEntry{cs, cache.order.begin()});
        out.pcd_metrics.prior_cache_entries = (int) cache.entries.size();
    }
    return scores;
}

struct ParrotToken {
    int start = 0;
    int end = 0;
};

std::vector<ParrotToken> parrot_tokens(Engine & engine, const std::string & text) {
    const auto ids = engine.tokenize(text, false);
    std::vector<ParrotToken> out;
    std::string rebuilt;
    int cp = 0;
    for (llama_token id : ids) {
        const std::string piece = engine.token_to_piece(id);
        if (piece.empty()) {
            throw std::invalid_argument("parrot mode requires a tokenizer-reconstructible text");
        }
        const Utf8Text pt = decode_utf8(piece);
        out.push_back({cp, cp + pt.size()});
        cp += pt.size();
        rebuilt += piece;
    }
    if (rebuilt != text) {
        throw std::invalid_argument("parrot mode requires a tokenizer-reconstructible text");
    }
    return out;
}

AnyPiiResult run_parrot_pipeline(Engine & engine, const AnyPiiRequest & request,
                                 const ResolveOptions & options) {
    AnyPiiResult out;
    out.model = engine.model_name();
    out.backend = engine.backend_name();

    const Utf8Text text = decode_utf8(request.text);
    if (text.size() == 0) {
        out.redacted_text = request.text;
        return out;
    }
    if (text.size() > options.max_text_cp) {
        throw std::invalid_argument("state.text is too long (max " +
                                    std::to_string(options.max_text_cp) + " code points)");
    }

    const auto tokens = parrot_tokens(engine, request.text);
    constexpr int max_parrot_tokens = 256;
    if ((int) tokens.size() > max_parrot_tokens) {
        throw std::invalid_argument("parrot mode supports up to " +
                                    std::to_string(max_parrot_tokens) +
                                    " tokens; use readout/pcd for longer text");
    }

    const bool typed = request.parrot_typed && !request.types.empty();
    TypeTable table;
    if (typed) table = build_type_table(request.types);

    PcdDecoder local_decoder(engine);
    PcdDecoder & decoder = options.pcd ? *options.pcd : local_decoder;
    std::vector<std::string> token_label(tokens.size());
    std::vector<double> token_p(tokens.size(), 0.0);

    const size_t chunk = 31;
    for (size_t base = 0; base < tokens.size(); base += chunk) {
        std::vector<PcdField> fields;
        for (size_t i = base; i < tokens.size() && i < base + chunk; ++i) {
            PcdField f;
            f.name = "p" + std::to_string(i);
            const std::string token_text =
                clean_line(text.substr(tokens[i].start, tokens[i].end));
            const std::string context =
                clean_line(context_of(text, tokens[i].start, tokens[i].end, 16));
            if (typed) {
                f.description = "Token " + std::to_string(i) + ": \"" + token_text +
                                "\". Context: \"" + context +
                                "\". Choose copy for an ordinary/non-PII token, or "
                                "the exact type id for a PII token.";
                f.choices.push_back("copy");
                for (const auto & t : table.types) f.choices.push_back(t.id);
            } else {
                f.description = "Token " + std::to_string(i) + ": \"" + token_text +
                                "\". Context: \"" + context +
                                "\". Copy is the default; choose REDACTED only when "
                                "this exact token is part of a PII mention covered by "
                                "the Policy.";
                f.choices = {"copy", "REDACTED"};
            }
            fields.push_back(std::move(f));
        }
        const std::string policy = request.policy.empty() ? "none" : request.policy;
        std::string system;
        if (typed) {
            system =
                "You are a token-level PII type classifier (parrot/repeater). For "
                "every field choose copy for ordinary/non-PII tokens, or the exact "
                "type id for a PII token. Copy is the safe default for punctuation, "
                "verbs, and neighboring context. Policy:\n" + policy + "\n" +
                pcd_type_legend(table) + "\n" + pcd_system_prompt(fields);
        } else {
            system =
                "You are a token-level PII redaction engine (parrot/repeater). For "
                "every field choose copy or REDACTED. Copy is the safe default for "
                "ordinary words, punctuation, verbs, and neighboring context. Choose "
                "REDACTED only when this exact token is part of a PII mention covered "
                "by the Policy. Policy:\n" + policy + "\n" + pcd_system_prompt(fields);
        }
        PcdResult decoded = decoder.decode(pcd_chat_prefix(system, request.text), fields);
        add_pcd_metrics(out, decoded.metrics, (int) fields.size());
        for (size_t j = 0; j < fields.size(); ++j) {
            const auto & res = decoded.fields[j];
            if (res.value.empty() || res.probability < request.threshold) continue;
            if (typed) {
                if (res.value == "copy") continue;
                bool valid = false;
                for (const auto & t : table.types) {
                    if (t.id == res.value) { valid = true; break; }
                }
                if (!valid) continue;
                token_label[base + j] = res.value;
                token_p[base + j] = res.probability;
            } else if (res.value == "REDACTED") {
                token_label[base + j] = "redacted";
                token_p[base + j] = res.probability;
            }
        }
    }

    std::vector<PhraseHit> hits;
    for (size_t i = 0; i < tokens.size();) {
        if (token_label[i].empty()) {
            ++i;
            continue;
        }
        size_t j = i;
        double sum = 0.0;
        while (j < tokens.size() && token_label[j] == token_label[i]) {
            sum += token_p[j];
            ++j;
        }
        PhraseHit h;
        h.start = tokens[i].start;
        h.end = tokens[j - 1].end;
        h.text = text.substr(h.start, h.end);
        h.type = token_label[i];
        h.score = sum / (double) (j - i);
        h.redact_score = h.score;
        h.redacted = true;
        hits.push_back(std::move(h));
        i = j;
    }

    emit_grouped(out, text, hits, request);
    out.generated_tokens = 0;
    return out;
}

AnyPiiResult run_pcd_pipeline(Engine & engine, const AnyPiiRequest & request,
                              const ResolveOptions & options) {
    AnyPiiResult out;
    out.model = engine.model_name();
    out.backend = engine.backend_name();

    const Utf8Text text = decode_utf8(request.text);
    if (text.size() > options.max_text_cp) {
        throw std::invalid_argument("state.text is too long (max " +
                                    std::to_string(options.max_text_cp) + " code points)");
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
        throw std::invalid_argument("too many candidate spans (max " +
                                    std::to_string(options.max_candidates) +
                                    "); shorten the text");
    }
    const auto type_label_ngrams = build_type_label_ngrams(request.types);
    std::vector<std::pair<int, int>> regex_spans;
    for (const auto & c : proposed) {
        if (c.regex_hit) regex_spans.push_back({c.start, c.end});
    }

    std::vector<PhraseHit> hits;
    std::vector<std::pair<int, int>> consumed;
    std::vector<Candidate> candidates;
    candidates.reserve(proposed.size());
    for (const auto & c : proposed) {
        bool already_consumed = false;
        for (const auto & r : consumed) {
            if (c.start < r.second && r.first < c.end) { already_consumed = true; break; }
        }
        if (already_consumed) continue;

        if (c.regex_hit) {
            const PiiFormat fmt = detect_format(c.text);
            std::vector<int> ftypes;
            for (int t = 0; t < type_count; ++t) {
                if (type_matches_format(fmt, table.types[(size_t) t])) ftypes.push_back(t);
            }
            std::vector<int> len_types;
            if (is_ascii_digits(c.text)) {
                for (int t = 0; t < type_count; ++t) {
                    if (length_type_bonus(c.end - c.start, table.types[(size_t) t]) > 1.0) {
                        len_types.push_back(t);
                    }
                }
            }
            int forced = -1;
            if (ftypes.size() == 1) forced = ftypes[0];
            else if (ftypes.empty() && len_types.size() == 1) forced = len_types[0];
            if (forced >= 0) {
                consumed.push_back({c.start, c.end});
                if (!(request.skip_keep_types &&
                      table.types[(size_t) forced].redact == RedactMode::Keep)) {
                    PhraseHit h;
                    h.start = c.start;
                    h.end = c.end;
                    h.text = c.text;
                    h.type = table.types[(size_t) forced].id;
                    h.score = 1.0;
                    hits.push_back(std::move(h));
                }
                continue;
            }
        }
        if (!c.regex_hit) {
            if (c.end - c.start <= 1) continue;
            if (!c.pos_ok) continue;
            if (is_type_label(decode_utf8(c.text), request.types, type_label_ngrams)) continue;
            bool adjacent_regex = false;
            if (c.end - c.start <= 6) {
                for (const auto & [rs, re] : regex_spans) {
                    const int gap_before = rs - c.end;
                    const int gap_after = c.start - re;
                    if ((gap_before >= 0 && gap_before <= 2) ||
                        (gap_after >= 0 && gap_after <= 2)) {
                        adjacent_regex = true;
                        break;
                    }
                }
            }
            if (adjacent_regex) continue;
            if (boundary_factor(c.text) < 0.75) continue;
        }
        candidates.push_back(c);
    }
    if (candidates.empty()) {
        out.redacted_text = request.text;
        return out;
    }

    PcdDecoder local_decoder(engine);
    PcdDecoder & decoder = options.pcd ? *options.pcd : local_decoder;

    auto decode_fields = [&](const std::string & system_prefix,
                             const std::vector<PcdField> & fields,
                             const std::function<void(const PcdFieldResult &, size_t)> & handle) {
        const size_t chunk = 31;
        for (size_t base = 0; base < fields.size(); base += chunk) {
            std::vector<PcdField> cf;
            for (size_t i = base; i < fields.size() && i < base + chunk; ++i) {
                cf.push_back(fields[i]);
            }
            const std::string system = system_prefix + "\n\n" + pcd_system_prompt(cf);
            PcdResult decoded = decoder.decode(pcd_chat_prefix(system, request.text), cf);
            add_pcd_metrics(out, decoded.metrics, (int) cf.size());
            for (size_t i = 0; i < cf.size(); ++i) {
                handle(decoded.fields[i], base + i);
            }
        }
    };

    if (!candidates.empty()) {

    // ---------- Boundary stage: one PCD field per overlapping component ----------
    std::vector<int> all(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) all[i] = (int) i;
    const auto comps = make_components(all, candidates);

    std::vector<int> selected_cands;
    struct BoundarySpec {
        std::vector<int> cands;
        std::vector<std::string> choices;
    };
    std::vector<BoundarySpec> bspecs;
    std::vector<PcdField> bfields;
    for (const auto & comp : comps) {
        std::map<std::pair<int, int>, int> by_span;
        for (int m : comp.members) {
            const auto key = std::make_pair(candidates[(size_t) m].start,
                                            candidates[(size_t) m].end);
            auto it = by_span.find(key);
            if (it == by_span.end() ||
                candidates[(size_t) it->second].prior < candidates[(size_t) m].prior) {
                by_span[key] = m;
            }
        }
        std::vector<int> cands;
        for (const auto & [key, m] : by_span) cands.push_back(m);
        if (cands.empty()) continue;
        if (cands.size() == 1) {
            selected_cands.push_back(cands[0]);
            continue;
        }
        BoundarySpec spec;
        spec.cands = cands;
        PcdField field;
        field.name = "b" + std::to_string(bspecs.size());
        field.description =
            "Choose the complete entity mention from the overlapping candidates. "
            "A disease, symptom, action, or role label is not an entity. Context: \"" +
            clean_line(context_of(text, comp.start, comp.end)) + "\"";
        for (int m : cands) spec.choices.push_back(candidates[(size_t) m].text);
        spec.choices.push_back("none");
        field.choices = spec.choices;
        bspecs.push_back(spec);
        bfields.push_back(std::move(field));
    }
    if (!bfields.empty()) {
        const std::string policy = request.policy.empty() ? "none" : request.policy;
        const std::string sys = "You are a PII entity-boundary arbiter. Policy:\n" + policy;
        decode_fields(sys, bfields, [&](const PcdFieldResult & r, size_t idx) {
            if (std::getenv("ANYPPI_DEBUG")) {
                std::cerr << "[pcd-boundary] idx=" << idx << " value=[" << r.value
                          << "] p=" << r.probability << "\n";
            }
            if (idx >= bspecs.size() || r.value == "none") return;
            const auto & spec = bspecs[idx];
            for (size_t i = 0; i < spec.cands.size(); ++i) {
                if (spec.choices[i] == r.value) {
                    selected_cands.push_back(spec.cands[i]);
                    return;
                }
            }
        });
    }
    if (selected_cands.empty()) {
        out.redacted_text = request.text;
        return out;
    }

    // Completeness prior: extend a short selected candidate to the longest
    // containing candidate (Alice -> Alice Johnson, 赵敏 -> 责任护士赵敏).
    {
        std::vector<int> extended;
        for (int c : selected_cands) {
            int best = c;
            int best_len = candidates[(size_t) c].end - candidates[(size_t) c].start;
            for (int m = 0; m < (int) candidates.size(); ++m) {
                if (m == c) continue;
                if (!(candidates[(size_t) m].start <= candidates[(size_t) c].start &&
                      candidates[(size_t) m].end >= candidates[(size_t) c].end)) {
                    continue;
                }
                if (candidates[(size_t) m].prior < 1.0) continue;
                if (boundary_factor(candidates[(size_t) m].text) < 0.60) continue;
                const int len = candidates[(size_t) m].end - candidates[(size_t) m].start;
                if (len > best_len) { best = m; best_len = len; }
            }
            bool duplicate = false;
            for (int e : extended) {
                if (candidates[(size_t) e].start == candidates[(size_t) best].start &&
                    candidates[(size_t) e].end == candidates[(size_t) best].end) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) extended.push_back(best);
        }
        selected_cands.swap(extended);
    }

    // ---------- Type stage: one PCD field per selected span ----------
    std::vector<int> t_cands;
    std::vector<PcdField> tfields;
    for (int c : selected_cands) {
        if (std::find(t_cands.begin(), t_cands.end(), c) != t_cands.end()) continue;
        t_cands.push_back(c);
        PcdField f;
        f.name = pcd_field_name(candidates[(size_t) c].text, (int) tfields.size());
        f.description = "Policy: " +
                        (request.policy.empty() ? std::string("none") : request.policy) +
                        ". Mention: \"" + clean_line(candidates[(size_t) c].text) +
                        "\". Context: \"" +
                        clean_line(context_of(text, candidates[(size_t) c].start,
                                              candidates[(size_t) c].end)) +
                        "\". Choose the exact entity type, or none if the mention is "
                        "not an entity or is only a neighboring word.";
        for (const auto & t : table.types) {
            f.choices.push_back(t.id + ": " + (t.description.empty() ? "entity" : t.description));
        }
        f.choices.push_back("none");
        tfields.push_back(std::move(f));
    }

    std::vector<Candidate> t_candidates;
    t_candidates.reserve(t_cands.size());
    for (int c : t_cands) t_candidates.push_back(candidates[(size_t) c]);
    const std::vector<CandScore> t_prior =
        readout_type_scores(engine, table, request, text, t_candidates, options, out);

    if (!tfields.empty()) {
        const std::string policy = request.policy.empty() ? "none" : request.policy;
        const std::string sys =
            "You are a PII type classifier. A disease, symptom, action, or ordinary "
            "word is not a drug, person, or organization unless a listed type "
            "explicitly covers it. Use the most specific matching type; a generic "
            "drug name and a brand name are different types when both are listed. "
            "Policy:\n" + policy + "\n" + pcd_type_legend(table);
        decode_fields(sys, tfields, [&](const PcdFieldResult & r, size_t idx) {
            if (std::getenv("ANYPPI_DEBUG")) {
                std::cerr << "[pcd-type] idx=" << idx << " mention=["
                          << candidates[(size_t) t_cands[idx]].text << "] value=["
                          << r.value << "] p=" << r.probability << "\n";
            }
            if (idx >= t_cands.size()) return;
            const CandScore & rd = t_prior[(size_t) idx];
            if (rd.probs.empty()) return;
            const int span_len = candidates[(size_t) t_cands[idx]].end -
                                 candidates[(size_t) t_cands[idx]].start;
            // The readout type pass is the confidence gate. Short redact:true
            // names may pass on the PCD/readout combination even when Pass A is
            // just under threshold (this preserves 李四 while suppressing
            // longer non-entity spans such as 肺炎入院).
            const bool short_redact =
                rd.best >= 0 && span_len <= 3 &&
                table.types[(size_t) rd.best].redact == RedactMode::Redact;
            if ((rd.best < 0 || rd.bestp < request.threshold) && !short_redact) return;

            const auto & field = tfields[idx];
            const auto & rp = r.probabilities;
            double sum = 0.0;
            double best = 0.0;
            int best_choice = -1;
            for (size_t j = 0; j < field.choices.size(); ++j) {
                double prior = 0.0;
                if (field.choices[j] == "none") {
                    prior = rd.probs.back();
                } else {
                    for (int t = 0; t < type_count; ++t) {
                        const std::string prefix = table.types[(size_t) t].id + ":";
                        if (field.choices[j].rfind(prefix, 0) == 0) {
                            prior = rd.probs[(size_t) t];
                            break;
                        }
                    }
                }
                const double v = (j < rp.size() ? rp[j] : 0.0) * prior;
                sum += v;
                if (v > best) { best = v; best_choice = (int) j; }
            }
            if (best_choice < 0 || sum <= 0.0) return;
            const std::string & chosen = field.choices[(size_t) best_choice];
            if (chosen == "none") return;
            int type = -1;
            for (int t = 0; t < type_count; ++t) {
                const std::string prefix = table.types[(size_t) t].id + ":";
                if (chosen.rfind(prefix, 0) == 0) { type = t; break; }
            }
            if (type < 0) return;
            if (request.skip_keep_types &&
                table.types[(size_t) type].redact == RedactMode::Keep) {
                return;
            }
            PhraseHit h;
            h.start = candidates[(size_t) t_cands[idx]].start;
            h.end = candidates[(size_t) t_cands[idx]].end;
            h.text = candidates[(size_t) t_cands[idx]].text;
            h.type = table.types[(size_t) type].id;
            h.score = rd.bestp;
            h.redacted = false;
            hits.push_back(std::move(h));
        });
    }
    }

    if (hits.empty()) {
        out.redacted_text = request.text;
        return out;
    }

    // ---------- Redaction stage: only auto types ----------
    std::vector<PcdField> rfields;
    std::vector<int> r_hits;
    for (size_t i = 0; i < hits.size(); ++i) {
        const TypeDef * def = find_type(request, hits[i].type);
        if (!def) continue;
        if (def->redact == RedactMode::Redact) {
            hits[i].redacted = true;
            continue;
        }
        if (def->redact == RedactMode::Keep) {
            hits[i].redacted = false;
            continue;
        }
        PcdField f;
        f.name = pcd_field_name(hits[i].text, (int) rfields.size());
        f.description = "Mention: \"" + clean_line(hits[i].text) + "\". Type: " +
                        hits[i].type + " (" + def->description +
                        "). Should this mention be redacted under the Policy?";
        f.choices = {"redact", "keep"};
        rfields.push_back(std::move(f));
        r_hits.push_back((int) i);
    }
    if (!rfields.empty()) {
        const std::string policy = request.policy.empty() ? "none" : request.policy;
        const std::string sys = "You are a PII redaction policy engine. Policy:\n" + policy;
        decode_fields(sys, rfields, [&](const PcdFieldResult & r, size_t idx) {
            if (std::getenv("ANYPPI_DEBUG")) {
                std::cerr << "[pcd-redact] idx=" << idx << " mention=["
                          << hits[(size_t) r_hits[idx]].text << "] value=[" << r.value
                          << "] p=" << r.probability << "\n";
            }
            if (idx >= r_hits.size()) return;
            double p_redact = 0.0;
            for (size_t j = 0; j < r.probabilities.size() && j < rfields[idx].choices.size(); ++j) {
                if (rfields[idx].choices[j] == "redact") {
                    p_redact = r.probabilities[j];
                    break;
                }
            }
            PhraseHit & h = hits[(size_t) r_hits[idx]];
            h.redact_score = p_redact;
            h.redacted = p_redact >= request.threshold;
        });
    }

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

    if (request.omit_keep_spans) {
        final_hits.erase(std::remove_if(final_hits.begin(), final_hits.end(),
                                        [](const PhraseHit & h) { return !h.redacted; }),
                         final_hits.end());
    }
    emit_grouped(out, text, final_hits, request);
    out.generated_tokens = 0;
    return out;
}

AnyPiiResult run_pipeline(Engine & engine, const AnyPiiRequest & request,
                          const ResolveOptions & options) {
    if (request.mode == "parrot") {
        return run_parrot_pipeline(engine, request, options);
    }
    if (request.mode == "pcd") {
        return run_pcd_pipeline(engine, request, options);
    }
    AnyPiiResult out;
    out.model = engine.model_name();
    out.backend = engine.backend_name();
    auto capture_readout_cache = [&]() {
        if (!out.readout_cache_status.empty()) return;
        const std::string status = engine.readout_cache_status();
        if (status.empty()) return;
        out.readout_cache_status = status;
        out.readout_cache_entries = engine.readout_cache_entries();
        out.readout_checkpoint_bytes = engine.readout_checkpoint_bytes();
    };

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
    std::vector<std::pair<int, int>> regex_spans;
    for (const auto & c : proposed) {
        if (c.regex_hit) regex_spans.push_back({c.start, c.end});
    }

    for (const auto & c : proposed) {
        if (!c.regex_hit) {
            if (c.end - c.start <= 1) continue;
            // A short word immediately before/after a structured value is a
            // label (server, address, email, ticket, "should"...), not the PII.
            bool adjacent_regex = false;
            if (c.end - c.start <= 6) {
                for (const auto & [rs, re] : regex_spans) {
                    const int gap_before = rs - c.end;
                    const int gap_after = c.start - re;
                    if ((gap_before >= 0 && gap_before <= 2) ||
                        (gap_after >= 0 && gap_after <= 2)) {
                        adjacent_regex = true;
                        break;
                    }
                }
            }
            if (adjacent_regex) continue;
            if (!c.pos_ok) continue;
            if (is_type_label(decode_utf8(c.text), request.types, type_label_ngrams)) continue;
            // Do not start in the middle of a caller type-label bigram
            // (for example "名阿莫仙" after "商品名").
            if (c.start > 0) {
                const std::u32string pair{
                    text.cps[(size_t) c.start - 1], text.cps[(size_t) c.start]};
                if (type_label_ngrams.count(pair)) continue;
            }
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

        BranchSet set = build_type_branches(engine, table, request, text, pass_a,
                                            options.compact_branches);
        const auto probs = engine.decode_branches(set.prefix, set.branches,
                                                  options.max_branches_per_decode,
                                                  options.temperature);
        account_branches(out, set.branches.size(), options.max_branches_per_decode);
        capture_readout_cache();

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
            cs.probs = p;
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
            const int cand_len = cand.end - cand.start;
            const bool apply_margin = cand_len <= options.none_margin_max_len;
            const double margin =
                (cs.best >= 0 && table.types[(size_t) cs.best].redact == RedactMode::Redact)
                    ? options.none_margin_redact
                    : options.none_margin;
            if (cs.best < 0 ||
                (apply_margin &&
                 (cs.none_p >= cs.bestp || cs.bestp - cs.none_p < margin))) {
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
        capture_readout_cache();
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

    // Completeness prior: a short selected span can be a word inside a longer
    // name (Alice / Alice Johnson, Acme / Acme Corp, 赵敏 / 责任护士赵敏).
    // Extend only when the longer candidate is close in type probability.
    for (auto & s : selected) {
        if (candidates[(size_t) s.cand].regex_hit) continue;
        int best = s.cand;
        int best_len = candidates[(size_t) s.cand].end - candidates[(size_t) s.cand].start;
        const double base_p = scores[(size_t) s.cand].bestp;
        for (int m = 0; m < (int) candidates.size(); ++m) {
            if (m == s.cand) continue;
            if (scores[(size_t) m].best < 0) continue;
            if (!(candidates[(size_t) m].start <= candidates[(size_t) s.cand].start &&
                  candidates[(size_t) m].end >= candidates[(size_t) s.cand].end)) {
                continue;
            }
            const int len = candidates[(size_t) m].end - candidates[(size_t) m].start;
            if (len <= best_len) continue;
            if (scores[(size_t) m].bestp < base_p - 0.15) continue;
            if (boundary_factor(candidates[(size_t) m].text) < 0.60) continue;
            best = m;
            best_len = len;
        }
        if (best != s.cand) {
            s.cand = best;
            s.type = scores[(size_t) best].best;
            s.weight = scores[(size_t) best].bestp;
            s.text = candidates[(size_t) best].text;
        }
    }

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
        capture_readout_cache();

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
        capture_readout_cache();
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

    if (request.omit_keep_spans) {
        final_hits.erase(std::remove_if(final_hits.begin(), final_hits.end(),
                                        [](const PhraseHit & h) { return !h.redacted; }),
                         final_hits.end());
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
