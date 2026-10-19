#include <cstdlib>
#include <iostream>
#include <string>

#include "anypii/api.hpp"
#include "anypii/engine.hpp"

namespace {

struct Args {
    std::string model;
    std::string host = "127.0.0.1";
    int port = 8080;
    int ctx = 65536;
    int threads = 0;
    int n_batch = 8192;
    int n_ubatch = 1024;
    int gpu_layers = 0;
    llama_flash_attn_type flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
    std::string api_key;
    int max_branches = 32;
    int max_text_cp = 16000;
    int max_candidates = 4000;
    double temperature = 1.0;
    double none_margin = 0.65;
};

void usage(const char * argv0) {
    std::cout
        << "AnyPII - policy-driven text redaction with logit readout\n\n"
        << "usage: " << argv0 << " --model MODEL.gguf [options]\n\n"
        << "options:\n"
        << "  --model PATH        GGUF model (default: $ANYPPI_MODEL or the local Qwen3.5 0.8B)\n"
        << "  --host HOST         bind host (default 127.0.0.1)\n"
        << "  --port N            bind port (default 8080)\n"
        << "  --ctx N             context size (default 8192)\n"
        << "  --threads N         CPU threads (default: physical CPU cores)\n"
        << "  --n-batch N         logical batch size (default 8192)\n"
        << "  --n-ubatch N        physical batch size (default 1024)\n"
        << "  --gpu-layers N      layers to offload (0 = CPU, negative = all; GPU builds)\n"
        << "  --flash-attn MODE   auto|on|off (default auto)\n"
        << "  --max-branches N    max decision branches per decode (default 32)\n"
        << "  --max-text-cp N     max input text length in code points (default 16000)\n"
        << "  --max-candidates N  max candidate spans (default 4000)\n"
        << "  --temperature T     softmax temperature (default 1.0)\n"
        << "  --none-margin M     minimum P(type)-P(none) margin (default 0.0)\n"
        << "  --api-key KEY       require Authorization: Bearer KEY\n"
        << "  --help              show this help\n";
}

Args parse_args(int argc, char ** argv) {
    Args a;
    const char * env = std::getenv("ANYPPI_MODEL");
    a.model = env ? env : "/home/dohnuts/models/qwen35-0.8b-gguf/Qwen3.5-0.8B-Q8_0.gguf";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char * name) -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (arg == "--model") {
            a.model = need("--model");
        } else if (arg == "--host") {
            a.host = need("--host");
        } else if (arg == "--port") {
            a.port = std::stoi(need("--port"));
        } else if (arg == "--ctx") {
            a.ctx = std::stoi(need("--ctx"));
        } else if (arg == "--threads") {
            a.threads = std::stoi(need("--threads"));
        } else if (arg == "--n-batch") {
            a.n_batch = std::stoi(need("--n-batch"));
        } else if (arg == "--n-ubatch") {
            a.n_ubatch = std::stoi(need("--n-ubatch"));
        } else if (arg == "--gpu-layers") {
            a.gpu_layers = std::stoi(need("--gpu-layers"));
        } else if (arg == "--flash-attn") {
            const std::string v = need("--flash-attn");
            if (v == "auto") {
                a.flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
            } else if (v == "on" || v == "true" || v == "1") {
                a.flash_attn = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            } else if (v == "off" || v == "false" || v == "0") {
                a.flash_attn = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            } else {
                throw std::invalid_argument("--flash-attn must be auto, on, or off");
            }
        } else if (arg == "--max-branches") {
            a.max_branches = std::stoi(need("--max-branches"));
        } else if (arg == "--max-text-cp") {
            a.max_text_cp = std::stoi(need("--max-text-cp"));
        } else if (arg == "--max-candidates") {
            a.max_candidates = std::stoi(need("--max-candidates"));
        } else if (arg == "--temperature") {
            a.temperature = std::stod(need("--temperature"));
        } else if (arg == "--none-margin") {
            a.none_margin = std::stod(need("--none-margin"));
        } else if (arg == "--api-key") {
            a.api_key = need("--api-key");
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    return a;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Args args = parse_args(argc, argv);
        anypii::EngineOptions eopts;
        eopts.model_path = args.model;
        eopts.n_ctx = args.ctx;
        eopts.n_threads = args.threads;
        eopts.n_batch = args.n_batch;
        eopts.n_ubatch = args.n_ubatch;
        eopts.n_gpu_layers = args.gpu_layers;
        eopts.max_branches = args.max_branches;
        eopts.flash_attn = args.flash_attn;

        std::cerr << "loading model: " << eopts.model_path << "\n";
        anypii::Engine engine(eopts);
        std::cerr << "model: " << engine.model_name() << "\n"
                  << "backend: " << engine.backend_name() << "\n"
                  << "ctx: " << engine.n_ctx() << ", vocab: " << engine.n_vocab() << "\n";

        anypii::ServerOptions sopts;
        sopts.host = args.host;
        sopts.port = args.port;
        sopts.api_key = args.api_key;
        sopts.max_branches = args.max_branches;
        sopts.max_text_cp = args.max_text_cp;
        sopts.max_candidates = args.max_candidates;
        sopts.temperature = args.temperature;
        sopts.none_margin = args.none_margin;
        anypii::run_server(engine, sopts);
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
