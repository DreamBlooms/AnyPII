#include "anypii/api.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <list>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "anypii/pcd.hpp"
#include "anypii/resolve.hpp"
#include "anypii/types.hpp"

namespace anypii {

namespace {

using json = nlohmann::ordered_json;

void send_error(httplib::Response & res, int status, const std::string & message,
                const std::string & type = "invalid_request_error") {
    json body;
    body["error"] = {{"message", message}, {"type", type}};
    res.status = status;
    res.set_content(body.dump(), "application/json; charset=utf-8");
}

RedactMode parse_redact(const json & value, const std::string & type_id) {
    if (value.is_null()) return RedactMode::Auto;
    if (value.is_boolean()) return value.get<bool>() ? RedactMode::Redact : RedactMode::Keep;
    if (value.is_string()) {
        const std::string s = value.get<std::string>();
        if (s == "auto") return RedactMode::Auto;
    }
    throw std::invalid_argument("types." + type_id +
                                ".redact must be true, false, or \"auto\"");
}

AnyPiiRequest parse_request(const std::string & body) {
    json j;
    try {
        j = json::parse(body);
    } catch (const std::exception & e) {
        throw std::invalid_argument(std::string("invalid JSON: ") + e.what());
    }
    if (!j.is_object()) throw std::invalid_argument("request body must be a JSON object");
    if (!j.contains("state") || !j["state"].is_object()) {
        throw std::invalid_argument("state is required and must be an object");
    }
    const auto & state = j["state"];
    if (!state.contains("text") || !state["text"].is_string()) {
        throw std::invalid_argument("state.text is required and must be a string");
    }
    if (state.contains("policy") && !state["policy"].is_string() &&
        !state["policy"].is_null()) {
        throw std::invalid_argument("state.policy must be a string");
    }
    AnyPiiRequest req;
    req.text = state["text"].get<std::string>();
    if (state.contains("policy") && state["policy"].is_string()) {
        req.policy = state["policy"].get<std::string>();
    }

    if (j.contains("mode")) {
        if (!j["mode"].is_string()) throw std::invalid_argument("mode must be a string");
        req.mode = j["mode"].get<std::string>();
        if (req.mode == "复读机") req.mode = "parrot";
        if (req.mode != "readout" && req.mode != "pcd" && req.mode != "parrot") {
            throw std::invalid_argument("mode must be \"readout\", \"pcd\", or \"parrot\"");
        }
    }
    const bool type_free = req.mode == "parrot";
    if (!type_free && (!j.contains("types") || !j["types"].is_object())) {
        throw std::invalid_argument("types is required and must be an object");
    }
    if (j.contains("types")) {
        if (!j["types"].is_object()) {
            throw std::invalid_argument("types must be an object");
        }
        for (auto it = j["types"].begin(); it != j["types"].end(); ++it) {
            const std::string id = it.key();
            const json & v = it.value();
            if (!v.is_object()) {
                throw std::invalid_argument("types." + id + " must be an object");
            }
            TypeDef t;
            t.id = id;
            if (v.contains("description")) {
                if (!v["description"].is_string()) {
                    throw std::invalid_argument("types." + id +
                                                ".description must be a string");
                }
                t.description = v["description"].get<std::string>();
            }
            if (v.contains("redact")) t.redact = parse_redact(v["redact"], id);
            req.types.push_back(std::move(t));
        }
    }

    if (j.contains("threshold")) {
        if (!j["threshold"].is_number()) {
            throw std::invalid_argument("threshold must be a number");
        }
        req.threshold = j["threshold"].get<double>();
        if (req.threshold < 0.0 || req.threshold > 1.0) {
            throw std::invalid_argument("threshold must be in [0, 1]");
        }
    }
    if (j.contains("mask_style")) {
        if (!j["mask_style"].is_string()) {
            throw std::invalid_argument("mask_style must be a string");
        }
        req.mask_style = j["mask_style"].get<std::string>();
        if (req.mask_style != "token" && req.mask_style != "length" &&
            req.mask_style != "mask" && req.mask_style != "typed" &&
            req.mask_style != "type") {
            throw std::invalid_argument(
                "mask_style must be \"token\", \"length\", \"mask\", \"typed\", or \"type\"");
        }
    }
    if (j.contains("fallback")) {
        if (!j["fallback"].is_boolean()) {
            throw std::invalid_argument("fallback must be a boolean");
        }
        req.fallback = j["fallback"].get<bool>();
    }
    if (j.contains("skip_keep_types")) {
        if (!j["skip_keep_types"].is_boolean()) {
            throw std::invalid_argument("skip_keep_types must be a boolean");
        }
        req.skip_keep_types = j["skip_keep_types"].get<bool>();
    }
    if (j.contains("omit_keep_spans")) {
        if (!j["omit_keep_spans"].is_boolean()) {
            throw std::invalid_argument("omit_keep_spans must be a boolean");
        }
        req.omit_keep_spans = j["omit_keep_spans"].get<bool>();
    }
    if (j.contains("parrot_typed")) {
        if (!j["parrot_typed"].is_boolean()) {
            throw std::invalid_argument("parrot_typed must be a boolean");
        }
        req.parrot_typed = j["parrot_typed"].get<bool>();
    }
    if (req.mode == "parrot" && req.parrot_typed && req.types.empty()) {
        throw std::invalid_argument("parrot_typed requires a non-empty types object");
    }
    return req;
}

json to_json(const AnyPiiResult & result) {
    json out;
    out["redacted_text"] = result.redacted_text;
    out["spans"] = json::array();
    for (const auto & s : result.spans) {
        json span;
        span["type"] = s.type;
        span["text"] = s.text;
        span["redacted"] = s.redacted;
        span["occurrences"] = json::array();
        for (const auto & o : s.occurrences) {
            json occ;
            occ["start"] = o.start;
            occ["end"] = o.end;
            occ["score"] = o.score;
            if (o.redact_score >= 0.0) occ["redact_score"] = o.redact_score;
            span["occurrences"].push_back(std::move(occ));
        }
        out["spans"].push_back(std::move(span));
    }
    out["model"] = result.model;
    out["backend"] = result.backend;
    out["usage"] = {
        {"decodes", result.decodes},
        {"branches", result.slots},
        {"generated_tokens", result.generated_tokens},
        {"fallback", result.fallback},
    };
    if (!result.readout_cache_status.empty()) {
        out["usage"]["readout"] = {
            {"cache_status", result.readout_cache_status},
            {"cache_entries", result.readout_cache_entries},
            {"checkpoint_bytes", result.readout_checkpoint_bytes},
        };
    }
    if (result.pcd) {
        out["usage"]["pcd"] = {
            {"forward_passes", result.pcd_metrics.forward_passes},
            {"elapsed_ms", result.pcd_metrics.elapsed_ms},
            {"cache_status", result.pcd_metrics.cache_status},
            {"checkpoint_bytes", result.pcd_metrics.checkpoint_bytes},
            {"phases", {
                {"tokenize_ms", result.pcd_metrics.phases.tokenize_ms},
                {"restore_or_prefill_ms", result.pcd_metrics.phases.restore_or_prefill_ms},
                {"broadcast_ms", result.pcd_metrics.phases.broadcast_ms},
                {"suffix_ms", result.pcd_metrics.phases.suffix_ms},
                {"tree_ms", result.pcd_metrics.phases.tree_ms},
            }},
        };
    }
    if (!result.result_cache_status.empty()) {
        out["usage"]["result_cache"] = {
            {"status", result.result_cache_status},
            {"entries", result.result_cache_entries},
        };
    }
    return out;
}

size_t result_bytes(const AnyPiiResult & result) {
    size_t bytes = sizeof(AnyPiiResult) + result.redacted_text.size() +
                   result.model.size() + result.backend.size();
    for (const auto & span : result.spans) {
        bytes += sizeof(EntitySpan) + span.type.size() + span.text.size();
        bytes += span.occurrences.size() * sizeof(SpanOccurrence);
    }
    return bytes;
}

struct CachedResult {
    std::string redacted_text;
    std::vector<EntitySpan> spans;
    std::string model;
    std::string backend;
    size_t bytes = 0;
    std::list<std::string>::iterator recency;
};

class ResultCache {
public:
    ResultCache(int max_entries, size_t max_bytes)
        : max_entries_(std::max(0, max_entries)), max_bytes_(max_bytes) {}

    bool get(const std::string & key, AnyPiiResult & out) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(key);
        if (it == entries_.end()) return false;
        order_.splice(order_.begin(), order_, it->second.recency);
        out = AnyPiiResult{};
        out.redacted_text = it->second.redacted_text;
        out.spans = it->second.spans;
        out.model = it->second.model;
        out.backend = it->second.backend;
        out.result_cache_status = "hit";
        out.result_cache_entries = entries_.size();
        return true;
    }

    void put(const std::string & key, const AnyPiiResult & result) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (max_entries_ <= 0 || max_bytes_ == 0) return;
        const size_t bytes = result_bytes(result);
        if (bytes > max_bytes_) return;
        if (auto old = entries_.find(key); old != entries_.end()) {
            used_ -= old->second.bytes;
            order_.erase(old->second.recency);
            entries_.erase(old);
        }
        while (!order_.empty() &&
               (entries_.size() + 1 > (size_t) max_entries_ ||
                used_ + bytes > max_bytes_)) {
            const std::string & oldest = order_.back();
            auto it = entries_.find(oldest);
            if (it != entries_.end()) {
                used_ -= it->second.bytes;
                entries_.erase(it);
            }
            order_.pop_back();
        }
        order_.push_front(key);
        CachedResult entry;
        entry.redacted_text = result.redacted_text;
        entry.spans = result.spans;
        entry.model = result.model;
        entry.backend = result.backend;
        entry.bytes = bytes;
        entry.recency = order_.begin();
        used_ += bytes;
        entries_[key] = std::move(entry);
    }

    size_t entries() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.size();
    }

private:
    mutable std::mutex mutex_;
    int max_entries_;
    size_t max_bytes_;
    size_t used_ = 0;
    std::list<std::string> order_;
    std::unordered_map<std::string, CachedResult> entries_;
};

std::string result_cache_key(const Engine & engine, const AnyPiiRequest & request) {
    json key;
    key["cache_version"] = "anypii-result-v1";
    key["model"] = engine.model_name();
    key["text"] = request.text;
    key["policy"] = request.policy;
    key["threshold"] = request.threshold;
    key["mask_style"] = request.mask_style;
    key["mode"] = request.mode;
    key["skip_keep_types"] = request.skip_keep_types;
    key["omit_keep_spans"] = request.omit_keep_spans;
    key["parrot_typed"] = request.parrot_typed;
    key["fallback"] = request.fallback;
    key["types"] = json::array();
    for (const auto & type : request.types) {
        key["types"].push_back({
            {"id", type.id},
            {"description", type.description},
            {"redact", (int) type.redact},
        });
    }
    return key.dump();
}

} // namespace

void run_server(Engine & engine, const ServerOptions & options) {
    httplib::Server server;
    server.set_payload_max_length(2 * 1024 * 1024);
    std::mutex engine_mutex;
    PcdOptions pcd_options;
    pcd_options.cache_entries = (size_t) std::max(0, options.pcd_cache_entries);
    pcd_options.cache_bytes = options.pcd_cache_bytes;
    pcd_options.temperature = options.temperature;
    PcdDecoder pcd(engine, pcd_options);
    ResultCache result_cache(options.result_cache_entries, options.result_cache_bytes);

    server.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        json body;
        body["ok"] = true;
        body["status"] = "healthy";
        body["model"] = engine.model_name();
        body["backend"] = engine.backend_name();
        body["ctx"] = engine.n_ctx();
        body["n_vocab"] = engine.n_vocab();
        body["readout"] = "shared_prefix_multi_branch";
        res.set_content(body.dump(), "application/json; charset=utf-8");
    });

    server.Post("/v1/anypii", [&](const httplib::Request & req,
                                  httplib::Response & res) {
        if (!options.api_key.empty()) {
            const std::string auth = req.get_header_value("Authorization");
            if (auth != "Bearer " + options.api_key) {
                send_error(res, 401, "missing or invalid API key", "authentication_error");
                return;
            }
        }

        AnyPiiRequest request;
        try {
            request = parse_request(req.body);
        } catch (const std::exception & e) {
            send_error(res, 400, e.what());
            return;
        }

        const std::string cache_key = result_cache_key(engine, request);
        AnyPiiResult cached_result;
        if (result_cache.get(cache_key, cached_result)) {
            res.set_content(to_json(cached_result).dump(), "application/json; charset=utf-8");
            return;
        }
        try {
            std::lock_guard<std::mutex> lock(engine_mutex);
            if (result_cache.get(cache_key, cached_result)) {
                res.set_content(to_json(cached_result).dump(), "application/json; charset=utf-8");
                return;
            }
            ResolveOptions ropts;
            ropts.max_branches_per_decode = options.max_branches;
            ropts.max_text_cp = options.max_text_cp;
            ropts.max_candidates = options.max_candidates;
            ropts.temperature = options.temperature;
            ropts.none_margin = options.none_margin;
            ropts.none_margin_max_len = options.none_margin_max_len;
            ropts.none_margin_redact = options.none_margin_redact;
            ropts.pcd = &pcd;
            AnyPiiResult result = run_pipeline(engine, request, ropts);
            result.result_cache_status = "miss";
            result_cache.put(cache_key, result);
            result.result_cache_entries = result_cache.entries();
            res.set_content(to_json(result).dump(), "application/json; charset=utf-8");
        } catch (const std::invalid_argument & e) {
            send_error(res, 400, e.what());
        } catch (const std::exception & e) {
            send_error(res, 500, e.what(), "inference_error");
        }
    });

    std::cerr << "AnyPII listening on http://" << options.host << ":" << options.port
              << "/v1/anypii\n";
    if (!server.listen(options.host.c_str(), options.port)) {
        throw std::runtime_error("failed to bind HTTP server");
    }
}

} // namespace anypii
