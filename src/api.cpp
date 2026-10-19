#include "anypii/api.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include <httplib.h>
#include <nlohmann/json.hpp>

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
    if (!j.contains("types") || !j["types"].is_object()) {
        throw std::invalid_argument("types is required and must be an object");
    }

    AnyPiiRequest req;
    req.text = state["text"].get<std::string>();
    if (state.contains("policy") && state["policy"].is_string()) {
        req.policy = state["policy"].get<std::string>();
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
            req.mask_style != "mask" && req.mask_style != "typed") {
            throw std::invalid_argument(
                "mask_style must be \"token\", \"length\", \"mask\", or \"typed\"");
        }
    }
    if (j.contains("mode")) {
        if (!j["mode"].is_string()) throw std::invalid_argument("mode must be a string");
        req.mode = j["mode"].get<std::string>();
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
    return out;
}

} // namespace

void run_server(Engine & engine, const ServerOptions & options) {
    httplib::Server server;
    server.set_payload_max_length(2 * 1024 * 1024);
    std::mutex engine_mutex;

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

        try {
            std::lock_guard<std::mutex> lock(engine_mutex);
            ResolveOptions ropts;
            ropts.max_branches_per_decode = options.max_branches;
            ropts.max_text_cp = options.max_text_cp;
            ropts.max_candidates = options.max_candidates;
            AnyPiiResult result = run_pipeline(engine, request, ropts);
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
