#include "anypii/pcd.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <memory>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace anypii {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string json_escape(const std::string & text) {
    const std::string dumped = nlohmann::json(text).dump();
    return dumped.substr(1, dumped.size() - 2);
}

std::string one_line(const std::string & text) {
    std::string out = text;
    for (char & c : out) {
        if (c == '\n' || c == '\r') c = ' ';
    }
    return out;
}

struct AdvanceResult {
    bool resolved = false;
    int32_t winning_token = -1;
};

// Collision tree over tokenized choices, adapted from pcdServer's scorer.
class CandidateState {
public:
    using LogitFn = std::function<double(int32_t)>;

    CandidateState(std::vector<std::vector<int32_t>> candidates,
                   std::vector<std::string> labels)
        : candidates_(std::move(candidates)),
          labels_(std::move(labels)),
          live_(candidates_.size(), true),
          probabilities_(candidates_.size(), 0.0),
          live_count_((int) candidates_.size()) {
        if (candidates_.empty() || candidates_.size() != labels_.size()) {
            throw std::invalid_argument("PCD candidate/label size mismatch");
        }
        for (const auto & tokens : candidates_) {
            if (tokens.empty()) throw std::invalid_argument("empty PCD candidate");
        }
        if (candidates_.size() == 1) finish(0);
    }

    AdvanceResult advance(const LogitFn & logit_of) {
        if (resolved()) throw std::logic_error("PCD candidate already resolved");
        const auto level = (std::size_t) levels_;
        std::map<int32_t, std::vector<int>> groups;
        for (std::size_t i = 0; i < candidates_.size(); ++i) {
            if (!live_[i]) continue;
            if (level >= candidates_[i].size()) {
                for (std::size_t j = 0; j < candidates_.size(); ++j) {
                    if (live_[j] && j != i) live_[j] = false;
                }
                finish((int) i);
                return {true, last_token_};
            }
            groups[candidates_[i][level]].push_back((int) i);
        }

        std::vector<std::pair<int32_t, double>> logits;
        logits.reserve(groups.size());
        double max_logit = -std::numeric_limits<double>::infinity();
        for (const auto & [token, _] : groups) {
            const double logit = logit_of(token);
            logits.emplace_back(token, logit);
            max_logit = std::max(max_logit, logit);
        }
        double denominator = 0.0;
        for (auto & [token, logit] : logits) {
            logit = std::exp(logit - max_logit);
            denominator += logit;
        }

        int32_t best_token = logits.front().first;
        double best_probability = -1.0;
        for (const auto & [token, weight] : logits) {
            const double probability = weight / denominator;
            if (probability > best_probability) {
                best_probability = probability;
                best_token = token;
            }
        }

        for (const auto & [token, weight] : logits) {
            if (token == best_token) continue;
            const double probability = weight / denominator;
            const auto & members = groups[token];
            const double share = live_mass_ * probability / (double) members.size();
            for (int member : members) {
                probabilities_[(std::size_t) member] = share;
                live_[(std::size_t) member] = false;
            }
        }

        live_mass_ *= best_probability;
        live_count_ = (int) groups[best_token].size();
        last_token_ = best_token;
        consumed_.push_back(best_token);
        ++levels_;
        if (live_count_ == 1) {
            finish(groups[best_token].front());
            return {true, best_token};
        }
        return {false, best_token};
    }

    void force_resolve() {
        if (resolved()) return;
        const double share = live_mass_ / (double) live_count_;
        int first = -1;
        for (std::size_t i = 0; i < live_.size(); ++i) {
            if (!live_[i]) continue;
            probabilities_[i] = share;
            if (first < 0) first = (int) i;
        }
        live_mass_ = share;
        finish(first);
    }

    bool resolved() const { return winner_ >= 0; }
    int winner() const { return winner_; }
    int32_t last_token() const { return last_token_; }
    int levels() const { return levels_; }
    const std::vector<double> & probabilities() const { return probabilities_; }
    const std::vector<std::string> & labels() const { return labels_; }

private:
    void finish(int index) {
        if (index < 0) return;
        probabilities_[(std::size_t) index] = live_mass_;
        live_mass_ = 0.0;
        std::fill(live_.begin(), live_.end(), false);
        live_count_ = 0;
        winner_ = index;
    }

    std::vector<std::vector<int32_t>> candidates_;
    std::vector<std::string> labels_;
    std::vector<bool> live_;
    std::vector<double> probabilities_;
    std::vector<int32_t> consumed_;
    double live_mass_ = 1.0;
    int live_count_ = 0;
    int levels_ = 0;
    int32_t last_token_ = -1;
    int winner_ = -1;
};

constexpr int kBatchLimit = 2048;

} // namespace

std::string pcd_system_prompt(const std::vector<PcdField> & fields) {
    std::string prompt =
        "You are a precise PII decision engine. Read the user's text and decide "
        "the value of every field below. Each field must be set to exactly one of "
        "its allowed values. Respond only with a JSON object containing every field.\n\n"
        "Fields:\n";
    for (const auto & field : fields) {
        prompt += "- \"" + json_escape(field.name) + "\"";
        if (!field.description.empty()) {
            prompt += ": " + one_line(field.description);
        }
        prompt += ". Allowed values: ";
        for (std::size_t i = 0; i < field.choices.size(); ++i) {
            if (i) prompt += ", ";
            prompt += nlohmann::json(field.choices[i]).dump();
        }
        prompt += "\n";
    }
    prompt += "\nOutput format:\n{\n";
    for (std::size_t i = 0; i < fields.size(); ++i) {
        prompt += "  \"" + json_escape(fields[i].name) + "\": <value>";
        prompt += (i + 1 < fields.size()) ? ",\n" : "\n";
    }
    prompt += "}";
    return prompt;
}

PcdDecoder::PcdDecoder(Engine & engine, PcdOptions options)
    : engine_(engine), options_(options) {}

void PcdDecoder::clear_cache() {
    cache_.clear();
    cache_order_.clear();
    cache_bytes_ = 0;
}

bool PcdDecoder::restore_prefix(const std::string & key, int prefix_len,
                                PcdMetrics & metrics) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        metrics.cache_status = "miss";
        return false;
    }
    cache_order_.splice(cache_order_.begin(), cache_order_, it->second.recency);
    engine_.clear_memory();
    SequenceCheckpoint checkpoint;
    checkpoint.bytes = it->second.bytes;
    checkpoint.max_position = it->second.max_position;
    if (engine_.restore_sequence(0, checkpoint) &&
        engine_.sequence_max_position(0) == prefix_len - 1) {
        metrics.cache_status = "hit";
        metrics.checkpoint_bytes = checkpoint.bytes.size();
        return true;
    }
    cache_bytes_ -= it->second.bytes.size();
    cache_order_.erase(it->second.recency);
    cache_.erase(it);
    metrics.cache_status = "fallback";
    return false;
}

void PcdDecoder::evict() {
    while (!cache_order_.empty() &&
           (cache_.size() + 1 > options_.cache_entries ||
            cache_bytes_ > options_.cache_bytes)) {
        const std::string & key = cache_order_.back();
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            cache_bytes_ -= it->second.bytes.size();
            cache_.erase(it);
        }
        cache_order_.pop_back();
    }
}

void PcdDecoder::save_prefix(const std::string & key, int max_position) {
    if (options_.cache_entries == 0 || options_.cache_bytes == 0) return;
    SequenceCheckpoint checkpoint;
    try {
        checkpoint = engine_.save_sequence(0);
    } catch (const std::exception & e) {
        if (std::getenv("ANYPPI_DEBUG")) {
            std::cerr << "[pcd] checkpoint failed: " << e.what() << "\n";
        }
        return;
    }
    checkpoint.max_position = max_position;
    last_checkpoint_bytes_ = checkpoint.bytes.size();
    if (checkpoint.bytes.size() > options_.cache_bytes) return;
    if (auto it = cache_.find(key); it != cache_.end()) {
        cache_bytes_ -= it->second.bytes.size();
        cache_order_.erase(it->second.recency);
        cache_.erase(it);
    }
    evict();
    cache_order_.push_front(key);
    cache_bytes_ += checkpoint.bytes.size();
    cache_[key] = CacheEntry{std::move(checkpoint.bytes), max_position,
                             cache_order_.begin()};
}

PcdResult PcdDecoder::decode(const std::string & prefix_text,
                             const std::vector<PcdField> & fields) {
    if (fields.empty()) throw std::invalid_argument("PCD requires at least one field");
    if (fields.size() + 1 > (std::size_t) engine_.n_seq_max()) {
        throw std::invalid_argument("too many PCD fields for this sequence budget");
    }

    PcdResult result;
    result.fields.resize(fields.size());
    auto & metrics = result.metrics;
    const auto started = Clock::now();

    auto phase = Clock::now();
    // Sequence 0 is shared with the readout path; make sure its prefix cache
    // bookkeeping is invalidated before PCD overwrites it.
    engine_.invalidate_prefix_cache();
    const auto prefix_tokens = engine_.tokenize(prefix_text, true);
    if (prefix_tokens.empty()) throw std::runtime_error("PCD prefix tokenized to nothing");
    const int prefix_len = (int) prefix_tokens.size();

    struct FieldPlan {
        std::vector<llama_token> suffix;
        std::vector<std::vector<int32_t>> choices;
        std::vector<std::string> labels;
    };
    std::vector<FieldPlan> plans(fields.size());
    for (std::size_t f = 0; f < fields.size(); ++f) {
        const auto & field = fields[f];
        if (field.choices.size() < 1) {
            throw std::invalid_argument("PCD field has no choices: " + field.name);
        }
        // No shared-character-prefix extraction: the collision tree resolves
        // shared token prefixes anyway, and escaping the whole choice is safer
        // for span text containing quotes or backslashes.
        const std::string suffix_text =
            "  \"" + json_escape(field.name) + "\": \"";
        plans[f].suffix = engine_.tokenize(suffix_text, false);
        plans[f].labels = field.choices;
        for (const auto & choice : field.choices) {
            auto toks = engine_.tokenize(json_escape(choice) + "\"", false);
            if (toks.empty()) {
                throw std::runtime_error("PCD choice tokenized to nothing: " + choice);
            }
            plans[f].choices.push_back(
                std::vector<int32_t>(toks.begin(), toks.end()));
        }
        if (plans[f].suffix.empty()) {
            throw std::runtime_error("PCD field suffix tokenized to nothing");
        }
    }
    metrics.phases.tokenize_ms = ms_since(phase);

    phase = Clock::now();
    const std::string cache_key = prefix_text;
    const bool restored = restore_prefix(cache_key, prefix_len, metrics);
    metrics.forward_passes = 0;
    if (!restored) {
        engine_.clear_memory();
        engine_.decode_single(prefix_tokens, 0, 0, false);
        engine_.synchronize();
        metrics.forward_passes +=
            (prefix_len + kBatchLimit - 1) / kBatchLimit;
        save_prefix(cache_key, prefix_len - 1);
        metrics.checkpoint_bytes = last_checkpoint_bytes_;
    }
    metrics.phases.restore_or_prefill_ms = ms_since(phase);

    phase = Clock::now();
    const int n = (int) fields.size();
    for (int f = 0; f < n; ++f) {
        engine_.copy_sequence(0, f + 1, -1, -1);
    }
    metrics.phases.broadcast_ms = ms_since(phase);

    phase = Clock::now();
    std::vector<std::unique_ptr<CandidateState>> states;
    std::vector<int> next_position((std::size_t) n, 0);
    states.reserve((std::size_t) n);
    for (int f = 0; f < n; ++f) {
        states.push_back(std::make_unique<CandidateState>(
            plans[(std::size_t) f].choices, plans[(std::size_t) f].labels));
        next_position[(std::size_t) f] =
            prefix_len + (int) plans[(std::size_t) f].suffix.size();
    }

    std::vector<EngineBatchToken> batch;
    std::vector<std::pair<int, int>> pending;

    auto flush = [&](bool score) {
        if (batch.empty()) return;
        engine_.decode_batch(batch);
        ++metrics.forward_passes;
        engine_.synchronize();
        if (score) {
            for (const auto & [field, index] : pending) {
                const float * row = engine_.logits_ith(index);
                states[(std::size_t) field]->advance(
                    [row](int32_t token) { return (double) row[token]; });
            }
        }
        batch.clear();
        pending.clear();
    };

    for (int f = 0; f < n; ++f) {
        const auto & plan = plans[(std::size_t) f];
        const auto seq = (llama_seq_id) (f + 1);
        for (std::size_t i = 0; i < plan.suffix.size(); ++i) {
            if (batch.size() >= (std::size_t) kBatchLimit) flush(true);
            const bool last = i + 1 == plan.suffix.size();
            batch.push_back({plan.suffix[i], (llama_pos) (prefix_len + (int) i),
                             {seq}, last});
            if (last) pending.emplace_back(f, (int) batch.size() - 1);
        }
    }
    flush(true);
    metrics.phases.suffix_ms = ms_since(phase);

    phase = Clock::now();
    for (int level = 0; level < options_.max_tree_levels; ++level) {
        bool any = false;
        for (int f = 0; f < n; ++f) {
            if (states[(std::size_t) f]->resolved()) continue;
            any = true;
            if (batch.size() >= (std::size_t) kBatchLimit) flush(true);
            batch.push_back({states[(std::size_t) f]->last_token(),
                             (llama_pos) next_position[(std::size_t) f],
                             {(llama_seq_id) (f + 1)}, true});
            pending.emplace_back(f, (int) batch.size() - 1);
            ++next_position[(std::size_t) f];
        }
        if (!any) break;
        flush(true);
    }
    for (auto & state : states) state->force_resolve();
    metrics.phases.tree_ms = ms_since(phase);

    for (int f = 0; f < n; ++f) {
        const auto & state = *states[(std::size_t) f];
        const int winner = state.winner();
        auto & out = result.fields[(std::size_t) f];
        if (winner >= 0) {
            out.value = state.labels()[(std::size_t) winner];
            out.probability = state.probabilities()[(std::size_t) winner];
        }
        out.probabilities = state.probabilities();
        out.levels = std::max(state.levels(), 1);
        engine_.remove_sequence(f + 1, -1, -1);
    }

    metrics.elapsed_ms = ms_since(started);
    return result;
}

} // namespace anypii
