#include "anypii/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <iostream>
#include <cstdlib>

namespace anypii {

namespace {

std::string strip_piece_prefix(const std::string & s) {
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char) s[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            ++i;
            continue;
        }
        // U+2581 LOWER ONE EIGHTH BLOCK, used by SentencePiece.
        if (i + 2 < s.size() && (unsigned char) s[i] == 0xE2 &&
            (unsigned char) s[i + 1] == 0x96 && (unsigned char) s[i + 2] == 0x81) {
            i += 3;
            continue;
        }
        // U+0120 LATIN CAPITAL LETTER G WITH DOT ABOVE, used by byte-level BPE
        // implementations to spell a leading space.
        if (i + 1 < s.size() && (unsigned char) s[i] == 0xC4 &&
            (unsigned char) s[i + 1] == 0xA0) {
            i += 2;
            continue;
        }
        break;
    }
    return s.substr(i);
}

} // namespace

struct Engine::Impl {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int n_vocab = 0;
    int n_ctx = 0;
    int n_batch = 0;
    int gpu_layers = 0;
    bool supports_gpu = false;
    std::string model_desc;
    std::unordered_map<std::string, std::vector<int>> group_cache;
    std::vector<llama_token> prefix_cache_tokens;
    std::vector<uint8_t> prefix_cache_state;

    ~Impl() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }

    std::vector<int> build_group(const std::string & code) {
        std::vector<int> ids;
        for (int id = 0; id < n_vocab; ++id) {
            const std::string piece = token_to_piece((llama_token) id);
            if (piece.empty()) continue;
            if (strip_piece_prefix(piece) == code) ids.push_back(id);
        }
        if (std::getenv("ANYPPI_DEBUG")) {
            std::cerr << "[label] code=" << code << " group_size=" << ids.size() << " pieces=";
            for (size_t i = 0; i < ids.size() && i < 6; ++i) {
                std::cerr << "[" << token_to_piece((llama_token) ids[i]) << "]";
            }
            std::cerr << "\n";
        }
        return ids;
    }

    std::string token_to_piece(llama_token token) const {
        char buf[128];
        int n = llama_token_to_piece(vocab, token, buf, (int) sizeof(buf), 0, false);
        if (n < 0) {
            std::vector<char> big((size_t) (-n) + 1);
            n = llama_token_to_piece(vocab, token, big.data(), (int) big.size(), 0, false);
            if (n <= 0) return {};
            return std::string(big.data(), (size_t) n);
        }
        return std::string(buf, (size_t) n);
    }
};

Engine::Engine(const EngineOptions & options) : p(new Impl) {
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = options.n_gpu_layers;

    p->model = llama_model_load_from_file(options.model_path.c_str(), mparams);
    if (!p->model) {
        throw std::runtime_error("failed to load model: " + options.model_path);
    }
    p->vocab = llama_model_get_vocab(p->model);
    p->n_vocab = llama_vocab_n_tokens(p->vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) options.n_ctx;
    cparams.n_batch = (uint32_t) options.n_batch;
    cparams.n_ubatch = std::min<uint32_t>((uint32_t) options.n_batch,
                                          (uint32_t) std::max(1, options.n_ubatch));
    cparams.n_seq_max = (uint32_t) std::max(2, options.max_branches + 1);
    cparams.kv_unified = true;
    cparams.n_threads = options.n_threads > 0 ? options.n_threads : 0;
    cparams.n_threads_batch = cparams.n_threads;
    cparams.no_perf = true;

    p->ctx = llama_init_from_model(p->model, cparams);
    if (!p->ctx) throw std::runtime_error("failed to create llama context");
    p->n_ctx = (int) llama_n_ctx(p->ctx);
    p->n_batch = (int) llama_n_batch(p->ctx);
    p->gpu_layers = options.n_gpu_layers;
    p->supports_gpu = llama_supports_gpu_offload();

    char desc[256] = {0};
    if (llama_model_desc(p->model, desc, sizeof(desc)) > 0) {
        p->model_desc = desc;
    } else {
        p->model_desc = "unknown";
    }
}

Engine::~Engine() = default;

std::vector<llama_token> Engine::tokenize(const std::string & text, bool add_special) const {
    int needed = -llama_tokenize(p->vocab, text.data(), (int32_t) text.size(), nullptr, 0,
                                 add_special, true);
    if (needed <= 0) return {};
    std::vector<llama_token> out((size_t) needed);
    int got = llama_tokenize(p->vocab, text.data(), (int32_t) text.size(), out.data(),
                             (int32_t) out.size(), add_special, true);
    if (got < 0) throw std::runtime_error("tokenization failed");
    out.resize((size_t) got);
    return out;
}

PromptTokens Engine::tokenize_segments(const std::vector<std::string> & segments) const {
    PromptTokens out;
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto part = tokenize(segments[i], i == 0);
        out.tokens.insert(out.tokens.end(), part.begin(), part.end());
        if (i + 1 < segments.size()) {
            if (out.tokens.empty()) {
                throw std::runtime_error("decision segment produced no tokens");
            }
            out.slot_indices.push_back((int) out.tokens.size() - 1);
        }
    }
    return out;
}

const std::vector<int> & Engine::label_group(const std::string & code) {
    auto it = p->group_cache.find(code);
    if (it == p->group_cache.end()) {
        std::vector<int> ids = p->build_group(code);
        if (ids.empty()) {
            throw std::runtime_error("label code is not a single vocabulary token: " + code);
        }
        it = p->group_cache.emplace(code, std::move(ids)).first;
    }
    return it->second;
}

namespace {

std::vector<double> restricted_softmax(const float * row, int n_vocab,
                                       const std::vector<std::vector<int>> & groups,
                                       double temperature) {
    if (temperature <= 0.0) temperature = 1.0;
    std::vector<double> scores(groups.size(), -std::numeric_limits<double>::infinity());
    double max_score = -std::numeric_limits<double>::infinity();
    for (size_t g = 0; g < groups.size(); ++g) {
        double m = -std::numeric_limits<double>::infinity();
        for (int id : groups[g]) {
            if (id < 0 || id >= n_vocab) continue;
            const double v = (double) row[id] / temperature;
            if (v > m) m = v;
        }
        if (!std::isfinite(m)) continue;
        double sum = 0.0;
        for (int id : groups[g]) {
            if (id < 0 || id >= n_vocab) continue;
            sum += std::exp(((double) row[id] / temperature) - m);
        }
        const double lse = m + std::log(sum);
        scores[g] = lse;
        if (lse > max_score) max_score = lse;
    }
    double total = 0.0;
    for (double & v : scores) {
        if (std::isfinite(v) && std::isfinite(max_score)) {
            v = std::exp(v - max_score);
            total += v;
        } else {
            v = 0.0;
        }
    }
    if (total <= 0.0) {
        const double u = 1.0 / (double) std::max<size_t>(1, scores.size());
        std::fill(scores.begin(), scores.end(), u);
    } else {
        for (double & v : scores) v /= total;
    }
    return scores;
}

} // namespace

std::vector<std::vector<double>> Engine::decode_slots(
    const std::vector<llama_token> & tokens,
    const std::vector<SlotSpec> & slots,
    double temperature) {
    if (tokens.empty()) return {};
    if (temperature <= 0.0) temperature = 1.0;

    for (const auto & s : slots) {
        if (s.token_index < 0 || (size_t) s.token_index >= tokens.size()) {
            throw std::runtime_error("slot token index out of range");
        }
    }

    for (auto & s : slots) {
        for (auto & g : s.label_groups) {
            if (g.empty()) throw std::runtime_error("empty label group");
        }
    }

    llama_memory_clear(llama_get_memory(p->ctx), true);

    std::vector<std::vector<double>> result(slots.size());
    const int chunk_max = p->n_batch > 0 ? p->n_batch : (int) tokens.size();
    const int total = (int) tokens.size();

    for (int chunk_start = 0; chunk_start < total; chunk_start += chunk_max) {
        const int chunk_len = std::min(chunk_max, total - chunk_start);
        llama_batch batch = llama_batch_init(chunk_len, 0, 1);
        batch.n_tokens = chunk_len;
        for (int i = 0; i < chunk_len; ++i) {
            batch.token[i] = tokens[(size_t) chunk_start + i];
            batch.pos[i] = (llama_pos) (chunk_start + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 0;
        }

        std::vector<int> slot_ids;
        for (size_t si = 0; si < slots.size(); ++si) {
            const int idx = slots[si].token_index;
            if (idx >= chunk_start && idx < chunk_start + chunk_len) {
                batch.logits[idx - chunk_start] = 1;
                slot_ids.push_back((int) si);
            }
        }

        const int rc = llama_decode(p->ctx, batch);
        if (rc != 0) {
            llama_batch_free(batch);
            if (rc == 1) {
                throw std::runtime_error("context is too small for this request");
            }
            throw std::runtime_error("llama_decode failed");
        }

        for (int si : slot_ids) {
            const int idx = slots[si].token_index;
            const float * row = llama_get_logits_ith(p->ctx, idx - chunk_start);
            if (!row) {
                llama_batch_free(batch);
                throw std::runtime_error("logits unavailable for slot");
            }
            const auto & groups = slots[si].label_groups;
            std::vector<double> scores = restricted_softmax(row, p->n_vocab, groups, temperature);
            if (std::getenv("ANYPPI_DEBUG")) {
                const int slot_idx = slots[si].token_index;
                std::cerr << "[slot] idx=" << slot_idx << " tok=" << tokens[(size_t) slot_idx]
                          << " piece=[" << p->token_to_piece(tokens[(size_t) slot_idx]) << "]";
                if (slot_idx + 1 < (int) tokens.size()) {
                    std::cerr << " next=[" << p->token_to_piece(tokens[(size_t) slot_idx + 1]) << "]";
                }
                std::cerr << " probs=";
                for (size_t g = 0; g < scores.size(); ++g) {
                    std::cerr << g << ":" << scores[g] << " ";
                }
                std::cerr << "\n";
            }
            result[si] = std::move(scores);
        }

        llama_batch_free(batch);
    }

    return result;
}

std::vector<std::vector<double>> Engine::decode_branches(
    const std::string & prefix_text,
    const std::vector<BranchPrompt> & branches,
    int max_branches_per_decode,
    double temperature) {
    if (branches.empty()) return {};
    if (max_branches_per_decode <= 0) max_branches_per_decode = 1;

    const auto prefix = tokenize(prefix_text, true);
    const int P = (int) prefix.size();
    llama_memory_t mem = llama_get_memory(p->ctx);

    // The evidence/policy/types prefix is identical across the type, boundary
    // and redact passes. Keep its decoded KV and restore it instead of
    // re-prefilling the same tokens for every pass.
    if (prefix == p->prefix_cache_tokens && !p->prefix_cache_state.empty()) {
        llama_memory_seq_rm(mem, 0, -1, -1);
        llama_state_seq_set_data(p->ctx, p->prefix_cache_state.data(),
                                 p->prefix_cache_state.size(), 0);
    } else {
        llama_memory_clear(mem, true);
        const int chunk_max = p->n_batch > 0 ? p->n_batch : std::max(1, P);
        for (int start = 0; start < P; start += chunk_max) {
            const int len = std::min(chunk_max, P - start);
            llama_batch batch = llama_batch_init(len, 0, 1);
            batch.n_tokens = len;
            for (int i = 0; i < len; ++i) {
                batch.token[i] = prefix[(size_t) (start + i)];
                batch.pos[i] = (llama_pos) (start + i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 0;
            }
            const int rc = llama_decode(p->ctx, batch);
            llama_batch_free(batch);
            if (rc != 0) {
                if (rc == 1) throw std::runtime_error("context is too small for the shared prefix");
                throw std::runtime_error("llama_decode failed while prefilling the shared prefix");
            }
        }
        const size_t size = llama_state_seq_get_size(p->ctx, 0);
        p->prefix_cache_state.resize(size);
        const size_t got = llama_state_seq_get_data(p->ctx, p->prefix_cache_state.data(),
                                                    p->prefix_cache_state.size(), 0);
        p->prefix_cache_state.resize(got);
        p->prefix_cache_tokens = prefix;
    }

    std::vector<std::vector<llama_token>> all_tails(branches.size());
    int max_tail = 0;
    for (size_t i = 0; i < branches.size(); ++i) {
        all_tails[i] = tokenize(branches[i].tail, false);
        if (all_tails[i].empty()) throw std::runtime_error("empty decision branch");
        max_tail = std::max(max_tail, (int) all_tails[i].size());
    }

    // Conservative KV budget: seq 0 holds the prefix; every branch sequence
    // holds a copy of the prefix plus its tail. If the context cannot hold the
    // requested batch, reduce the batch size rather than failing the request.
    const int per_branch = P + max_tail + 1;
    const int safe_branches = (p->n_ctx - P) / std::max(1, per_branch);
    if (safe_branches < 1) {
        throw std::runtime_error(
            "context is too small for the prefix plus one branch; increase --ctx");
    }
    if (max_branches_per_decode > safe_branches) {
        max_branches_per_decode = safe_branches;
    }

    std::vector<std::vector<double>> out(branches.size());

    for (int base = 0; base < (int) branches.size(); base += max_branches_per_decode) {
        const int n = std::min(max_branches_per_decode, (int) branches.size() - base);

        // Give every branch its own copy of the shared prefix KV.
        for (int r = 0; r < n; ++r) {
            llama_memory_seq_cp(mem, 0, (llama_seq_id) (r + 1), 0, (llama_pos) P);
        }

        int total_tokens = 0;
        for (int r = 0; r < n; ++r) {
            total_tokens += (int) all_tails[(size_t) (base + r)].size();
        }

        llama_batch batch = llama_batch_init(total_tokens, 0, 1);
        batch.n_tokens = total_tokens;
        std::vector<int> last_index((size_t) n, -1);
        int at = 0;
        for (int r = 0; r < n; ++r) {
            const auto & tail = all_tails[(size_t) (base + r)];
            for (size_t i = 0; i < tail.size(); ++i) {
                batch.token[at] = tail[i];
                batch.pos[at] = (llama_pos) (P + (int) i);
                batch.n_seq_id[at] = 1;
                batch.seq_id[at][0] = (llama_seq_id) (r + 1);
                batch.logits[at] = 0;
                ++at;
            }
            last_index[(size_t) r] = at - 1;
            batch.logits[at - 1] = 1;
        }

        const int rc = llama_decode(p->ctx, batch);
        if (rc != 0) {
            llama_batch_free(batch);
            for (int r = 0; r < n; ++r) llama_memory_seq_rm(mem, (llama_seq_id) (r + 1), -1, -1);
            if (rc == 1) throw std::runtime_error("context is too small for this branch batch");
            throw std::runtime_error("llama_decode failed");
        }

        for (int r = 0; r < n; ++r) {
            const float * row = llama_get_logits_ith(p->ctx, last_index[(size_t) r]);
            if (!row) {
                llama_batch_free(batch);
                for (int k = 0; k < n; ++k) llama_memory_seq_rm(mem, (llama_seq_id) (k + 1), -1, -1);
                throw std::runtime_error("logits unavailable for branch");
            }
            out[(size_t) (base + r)] =
                restricted_softmax(row, p->n_vocab, branches[(size_t) (base + r)].label_groups,
                                   temperature);
        }

        llama_batch_free(batch);
        for (int r = 0; r < n; ++r) {
            llama_memory_seq_rm(mem, (llama_seq_id) (r + 1), -1, -1);
        }
    }

    return out;
}

std::string Engine::token_to_piece(llama_token token) const { return p->token_to_piece(token); }
std::string Engine::model_name() const { return p->model_desc; }
std::string Engine::backend_name() const {
    return p->model_desc +
           (p->gpu_layers != 0 && p->supports_gpu
                ? " / llama.cpp GPU"
                : " / llama.cpp CPU");
}
int Engine::n_ctx() const { return p->n_ctx; }
int Engine::n_vocab() const { return p->n_vocab; }

} // namespace anypii
