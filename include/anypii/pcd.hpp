// Parallel Constrained Decoding (PCD) core.
//
// Adapted from the architecture of https://github.com/stephanj/pcdServer
// (MIT): a shared prompt prefix is broadcast to one sequence per bounded field,
// each field scores only its allowed choice tokens, and choices that share a
// token prefix are resolved level by level in a collision tree.
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "anypii/engine.hpp"
#include "anypii/types.hpp"

namespace anypii {

struct PcdField {
    std::string name;
    std::string description;
    std::vector<std::string> choices;
};

struct PcdFieldResult {
    std::string value;
    double probability = 0.0;
    std::vector<double> probabilities;
    int levels = 1;
};

struct PcdResult {
    std::vector<PcdFieldResult> fields;
    PcdMetrics metrics;
};

struct PcdOptions {
    std::size_t max_fields = 63;
    std::size_t cache_entries = 4;
    std::size_t cache_bytes = 256ull * 1024 * 1024;
    int max_tree_levels = 24;
    double temperature = 1.0;
};

// System prompt listing every field, its description and allowed values.
// Callers wrap it (plus evidence) in the model's chat template and append "{\n".
std::string pcd_system_prompt(const std::vector<PcdField> & fields);

// Scores bounded fields against one shared prefix. Not thread-safe; the owning
// server serializes calls together with the Engine.
class PcdDecoder {
public:
    explicit PcdDecoder(Engine & engine, PcdOptions options = {});

    PcdResult decode(const std::string & prefix_text,
                     const std::vector<PcdField> & fields);

    void clear_cache();
    std::size_t cache_entries() const { return cache_.size(); }
    std::size_t cache_bytes() const { return cache_bytes_; }

private:
    struct CacheEntry {
        std::vector<uint8_t> bytes;
        int max_position = -1;
        std::list<std::string>::iterator recency;
    };

    bool restore_prefix(const std::string & key, int prefix_len, PcdMetrics & metrics);
    void save_prefix(const std::string & key, int max_position);
    void evict();

    Engine & engine_;
    PcdOptions options_;
    std::list<std::string> cache_order_;
    std::unordered_map<std::string, CacheEntry> cache_;
    std::size_t cache_bytes_ = 0;
    std::size_t last_checkpoint_bytes_ = 0;
};

} // namespace anypii
