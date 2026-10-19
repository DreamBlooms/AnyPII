#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "llama.h"

namespace anypii {

struct EngineOptions {
    std::string model_path;
    int n_ctx = 8192;
    int n_batch = 2048;
    int n_ubatch = 1024;
    int n_threads = 0;
    int n_gpu_layers = 0;
    int max_branches = 32;
    llama_flash_attn_type flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
};

// One decision position. label_groups[i] is the set of vocabulary token ids
// that may express label i at this position (surface variants such as "A" and
// " A" are all placed in the same group).
struct SlotSpec {
    int token_index = 0;
    std::vector<std::vector<int>> label_groups;
};

struct PromptTokens {
    std::vector<llama_token> tokens;
    std::vector<int> slot_indices;
};

// One independent decision branch: a question appended to the shared prefix,
// ending at the assistant decision position. label_groups[i] are the vocabulary
// tokens that may express option i.
struct BranchPrompt {
    std::string tail;
    std::vector<std::vector<int>> label_groups;
};

class Engine {
public:
    explicit Engine(const EngineOptions & options);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;

    std::vector<llama_token> tokenize(const std::string & text, bool add_special) const;
    // Tokenize a sequence of segments; each non-final segment ends at a decision
    // slot. Returns the flattened token stream and the slot token positions.
    PromptTokens tokenize_segments(const std::vector<std::string> & segments) const;

    const std::vector<int> & label_group(const std::string & code);

    // Run exactly ONE decode for the given token stream and read restricted
    // softmax distributions at every slot. Chunked only to respect n_batch.
    std::vector<std::vector<double>> decode_slots(
        const std::vector<llama_token> & tokens,
        const std::vector<SlotSpec> & slots,
        double temperature = 1.0);

    // Decode a shared prefix once, then score many independent branch tails in
    // batches using multi-sequence decode. This is the high-performance path:
    // one forward per batch, not one per branch/token.
    std::vector<std::vector<double>> decode_branches(
        const std::string & prefix_text,
        const std::vector<BranchPrompt> & branches,
        int max_branches_per_decode,
        double temperature = 1.0);

    std::string token_to_piece(llama_token token) const;
    std::string model_name() const;
    std::string backend_name() const;
    int n_ctx() const;
    int n_vocab() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p;
};

} // namespace anypii
