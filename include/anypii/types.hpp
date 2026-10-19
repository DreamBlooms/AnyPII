#pragma once

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
};

} // namespace anypii
