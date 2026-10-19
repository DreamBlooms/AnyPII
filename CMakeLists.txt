cmake_minimum_required(VERSION 3.20)
project(anypii CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Official llama.cpp v0.6.0, pinned as the third_party/llama.cpp submodule.
# AnyPII only uses the public llama.h API, so a different checkout can be
# supplied with -DLLAMA_DIR=... when needed.
set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp"
    CACHE PATH "llama.cpp checkout")

if(NOT EXISTS "${LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp not found at ${LLAMA_DIR}. "
        "Run: git submodule update --init --recursive (or pass -DLLAMA_DIR=...)")
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)

# Optional GPU backends. The default build is CPU only.
option(ANYPPI_CUDA   "Build with the CUDA backend"   OFF)
option(ANYPPI_VULKAN "Build with the Vulkan backend" OFF)
option(ANYPPI_HIP    "Build with the ROCm/HIP backend" OFF)
option(ANYPPI_METAL  "Build with the Metal backend"  OFF)
if(ANYPPI_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(ANYPPI_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(ANYPPI_HIP)
    set(GGML_HIP ON CACHE BOOL "" FORCE)
endif()
if(ANYPPI_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

add_subdirectory("${LLAMA_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp")

find_package(Threads REQUIRED)

add_executable(anypii
    src/main.cpp
    src/engine.cpp
    src/nist.cpp
    src/segment.cpp
    src/sys.cpp
    src/unicode.cpp
    src/candidates.cpp
    src/prompt.cpp
    src/resolve.cpp
    src/mask.cpp
    src/api.cpp
)
target_include_directories(anypii PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cppjieba/include"
    "${LLAMA_DIR}/include"
)
target_compile_definitions(anypii PRIVATE
    ANYPPI_JIEBA_DICT="${CMAKE_CURRENT_SOURCE_DIR}/third_party/cppjieba/dict")
target_compile_options(anypii PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(anypii PRIVATE llama Threads::Threads)
if(WIN32)
    target_link_libraries(anypii PRIVATE ws2_32)
endif()

install(TARGETS anypii RUNTIME DESTINATION bin)
