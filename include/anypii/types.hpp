#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace anypii {

enum class RedactMode { Keep, Redact, Auto };

struct TypeDef {
    std::string id;
    std::string description;
    RedactMode redact = RedactMode::Auto;
};

// A candidate span. Offsets are Unicode code point indices into state.text,
// half-open [start, end).
struct Candidate {
    int start = 0;
    int end = 0;
    std::string text;
    double prior = 1.0;
    bool regex_hit = false;
    bool pos_ok = true;
};

struct PhraseHit {
    int start = 0;
    int end = 0;
    std::string text;
    std::string type;
    double score = 0.0;
    double redact_score = -1.0;
    bool redacted = false;
};

struct SpanOccurrence {
    int start = 0;
    int end = 0;
    double score = 0.0;
    double redact_score = -1.0;
};

struct EntitySpan {
    std::string type;
    std::string text;
    bool redacted = false;
    std::vector<SpanOccurrence> occurrences;
};

struct AnyPiiRequest {
    std::string text;
    std::string policy;
    std::vector<TypeDef> types;
    double threshold = 0.55;
    std::string mask_style = "token";
    std::string mode = "readout";
    bool fallback = true;
    bool skip_keep_types = false;
};

struct PcdPhaseTimings {
    double tokenize_ms = 0.0;
    double restore_or_prefill_ms = 0.0;
    double broadcast_ms = 0.0;
    double suffix_ms = 0.0;
    double tree_ms = 0.0;
};

struct PcdMetrics {
    double elapsed_ms = 0.0;
    int forward_passes = 0;
    std::string cache_status;
    std::size_t checkpoint_bytes = 0;
    PcdPhaseTimings phases;
};

struct AnyPiiResult {
    std::string redacted_text;
    std::vector<EntitySpan> spans;
    std::string model;
    std::string backend;
    int decodes = 0;
    int slots = 0;
    long long prefill_tokens = 0;
    int generated_tokens = 0;
    bool fallback = false;
    bool pcd = false;
    PcdMetrics pcd_metrics;
    std::string readout_cache_status;
    std::size_t readout_cache_entries = 0;
    std::size_t readout_checkpoint_bytes = 0;
};

} // namespace anypii
