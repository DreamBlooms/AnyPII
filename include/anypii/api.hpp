#pragma once

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
};

void run_server(Engine & engine, const ServerOptions & options);

} // namespace anypii
