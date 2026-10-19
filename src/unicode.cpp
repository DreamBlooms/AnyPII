#include "anypii/unicode.hpp"

#include <algorithm>

namespace anypii {

std::string Utf8Text::substr(int start, int end) const {
    if (start < 0) start = 0;
    if (end > size()) end = size();
    if (start >= end) return {};
    return bytes.substr(off[start], off[end] - off[start]);
}

namespace {

void append_utf8(std::string & out, char32_t cp) {
    if (cp <= 0x7F) {
        out.push_back((char) cp);
    } else if (cp <= 0x7FF) {
        out.push_back((char) (0xC0 | (cp >> 6)));
        out.push_back((char) (0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back((char) (0xE0 | (cp >> 12)));
        out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char) (0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char) (0xF0 | (cp >> 18)));
        out.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char) (0x80 | (cp & 0x3F)));
    }
}

} // namespace

Utf8Text decode_utf8(const std::string & s) {
    Utf8Text t;
    t.bytes = s;
    t.off.reserve(s.size() + 1);
    t.off.push_back(0);
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char) s[i];
        char32_t cp = 0xFFFD;
        size_t len = 1;
        if (c < 0x80) {
            cp = c;
            len = 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
            cp = ((char32_t) (c & 0x1F) << 6) | ((unsigned char) s[i + 1] & 0x3F);
            len = 2;
            if (cp < 0x80) cp = 0xFFFD;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
            cp = ((char32_t) (c & 0x0F) << 12) |
                 ((char32_t) ((unsigned char) s[i + 1] & 0x3F) << 6) |
                 ((unsigned char) s[i + 2] & 0x3F);
            len = 3;
            if (cp < 0x800) cp = 0xFFFD;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
            cp = ((char32_t) (c & 0x07) << 18) |
                 ((char32_t) ((unsigned char) s[i + 1] & 0x3F) << 12) |
                 ((char32_t) ((unsigned char) s[i + 2] & 0x3F) << 6) |
                 ((unsigned char) s[i + 3] & 0x3F);
            len = 4;
            if (cp < 0x10000 || cp > 0x10FFFF) cp = 0xFFFD;
        }
        t.cps.push_back(cp);
        i += len;
        t.off.push_back(i);
    }
    return t;
}

std::string encode_utf8(const std::u32string & cps) {
    std::string out;
    out.reserve(cps.size() * 3);
    for (char32_t cp : cps) append_utf8(out, cp);
    return out;
}

std::string encode_cp(char32_t cp) {
    std::string out;
    append_utf8(out, cp);
    return out;
}

bool is_cjk(char32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) ||
           (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0x2F800 && cp <= 0x2FA1F);
}

bool is_digit_cp(char32_t cp) {
    return (cp >= '0' && cp <= '9') || (cp >= 0xFF10 && cp <= 0xFF19);
}

bool is_latin_letter(char32_t cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    if (cp >= 0xC0 && cp <= 0x24F && cp != 0xD7 && cp != 0xF7) return true;
    if (cp >= 0xFF21 && cp <= 0xFF3A) return true;
    if (cp >= 0xFF41 && cp <= 0xFF5A) return true;
    return false;
}

bool is_space_cp(char32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == 0x00A0 || cp == 0x3000;
}

bool is_punct_cp(char32_t cp) {
    if (cp >= 0x21 && cp <= 0x2F) return true;
    if (cp >= 0x3A && cp <= 0x40) return true;
    if (cp >= 0x5B && cp <= 0x60) return true;
    if (cp >= 0x7B && cp <= 0x7E) return true;
    if (cp >= 0x3000 && cp <= 0x303F) return true;
    if (cp >= 0xFF01 && cp <= 0xFF0F) return true;
    if (cp >= 0xFF1A && cp <= 0xFF20) return true;
    if (cp >= 0xFF3B && cp <= 0xFF40) return true;
    if (cp >= 0xFF5B && cp <= 0xFF65) return true;
    if (cp == 0x2018 || cp == 0x2019 || cp == 0x201C || cp == 0x201D) return true;
    if (cp == 0x2013 || cp == 0x2014 || cp == 0x2026 || cp == 0x00B7) return true;
    return false;
}

bool is_word_cp(char32_t cp) {
    return is_latin_letter(cp) || is_digit_cp(cp) || is_cjk(cp);
}

int byte_to_cp(const Utf8Text & t, size_t byte_off) {
    if (byte_off >= t.bytes.size()) return t.size();
    auto it = std::upper_bound(t.off.begin(), t.off.end(), byte_off);
    int idx = (int) (it - t.off.begin()) - 1;
    if (idx < 0) idx = 0;
    return idx;
}

} // namespace anypii
