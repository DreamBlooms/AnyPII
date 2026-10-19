#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f third_party/llama.cpp/CMakeLists.txt ]; then
    git submodule update --init --recursive
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build -j"$(nproc)"
