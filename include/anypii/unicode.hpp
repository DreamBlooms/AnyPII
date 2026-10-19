#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace anypii {

struct Utf8Text {
    std::string bytes;
    std::u32string cps;
    // off[i] is the byte offset of code point i; off[cps.size()] == bytes.size().
    std::vector<size_t> off;

    int size() const { return (int) cps.size(); }
    std::string substr(int start, int end) const;
};

Utf8Text decode_utf8(const std::string & s);
std::string encode_utf8(const std::u32string & cps);
std::string encode_cp(char32_t cp);

bool is_cjk(char32_t cp);
bool is_digit_cp(char32_t cp);
bool is_latin_letter(char32_t cp);
bool is_space_cp(char32_t cp);
bool is_punct_cp(char32_t cp);
bool is_word_cp(char32_t cp);

// Convert a byte offset into a code point index (clamped).
int byte_to_cp(const Utf8Text & t, size_t byte_off);

} // namespace anypii
