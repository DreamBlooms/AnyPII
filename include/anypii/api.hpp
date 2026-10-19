#pragma once

#include <cstddef>
#include <string>

#include "anypii/engine.hpp"

namespace anypii {

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;
    int max_branches = 32;
    int max_text_cp = 16000;
    int max_candidates = 4000;
    double temperature = 1.0;
    double none_margin = 0.65;
    int none_margin_max_len = 4;
    double none_margin_redact = 0.20;
    int pcd_cache_entries = 4;
    size_t pcd_cache_bytes = 256ull * 1024 * 1024;
};

void run_server(Engine & engine, const ServerOptions & options);

} // namespace anypii
