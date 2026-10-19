#include "anypii/nist.hpp"

#include <algorithm>
#include <cctype>
#include <regex>
#include <string>
#include <vector>

#include "anypii/unicode.hpp"

namespace anypii {
namespace {

struct CompiledPattern {
    PiiPattern spec;
    std::regex re;
};

const std::vector<CompiledPattern> & compiled_patterns() {
    static const std::vector<CompiledPattern> patterns = {
        {{R"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})", 1.30, PiiFormat::Email},
         std::regex(R"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})", std::regex::optimize)},
        {{R"(https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+)", 1.30, PiiFormat::Url},
         std::regex(R"(https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+)", std::regex::optimize)},
        {{R"((?:sk|pk|api|key)[-_][A-Za-z0-9_-]{12,})", 1.30, PiiFormat::ApiKey},
         std::regex(R"((?:sk|pk|api|key)[-_][A-Za-z0-9_-]{12,})", std::regex::optimize)},
        {{R"((?:[0-9]{1,3}\.){3}[0-9]{1,3})", 1.25, PiiFormat::Ip},
         std::regex(R"((?:[0-9]{1,3}\.){3}[0-9]{1,3})", std::regex::optimize)},
        {{R"((?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4})", 1.25, PiiFormat::Ip},
         std::regex(R"((?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4})", std::regex::optimize)},
        {{R"([(（][0-9]{4}[)）][^\s，。；,;]{2,})", 1.25, PiiFormat::CaseNumber},
         std::regex(R"([(（][0-9]{4}[)）][^\s，。；,;]{2,})", std::regex::optimize)},
        {{R"([0-9]{3}-[0-9]{2}-[0-9]{4})", 1.35, PiiFormat::Ssn},
         std::regex(R"([0-9]{3}-[0-9]{2}-[0-9]{4})", std::regex::optimize)},
        {{R"(9[0-9]{2}-[7-9][0-9]-[0-9]{4})", 1.35, PiiFormat::Itin},
         std::regex(R"(9[0-9]{2}-[7-9][0-9]-[0-9]{4})", std::regex::optimize)},
        {{R"([0-9]{2}-[0-9]{7})", 1.25, PiiFormat::Ein},
         std::regex(R"([0-9]{2}-[0-9]{7})", std::regex::optimize)},
        {{R"((?:[0-9][ -]?){12,18}[0-9])", 1.30, PiiFormat::CreditCard},
         std::regex(R"((?:[0-9][ -]?){12,18}[0-9])", std::regex::optimize)},
        {{R"([A-Z]{2}[0-9]{2}(?: ?[A-Z0-9]){11,30})", 1.30, PiiFormat::Iban},
         std::regex(R"([A-Z]{2}[0-9]{2}(?: ?[A-Z0-9]){11,30})", std::regex::optimize)},
        {{R"((?:19|20)[0-9]{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12][0-9]|3[01]))",
          1.20, PiiFormat::Dob},
         std::regex(R"((?:19|20)[0-9]{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12][0-9]|3[01]))",
                    std::regex::optimize)},
        {{R"((?:\+?86[- ]?)?1[3-9][0-9]{9})", 1.30, PiiFormat::Phone},
         std::regex(R"((?:\+?86[- ]?)?1[3-9][0-9]{9})", std::regex::optimize)},
        {{R"(\+?[0-9][0-9 ()\-]{7,}[0-9])", 1.25, PiiFormat::Phone},
         std::regex(R"(\+?[0-9][0-9 ()\-]{7,}[0-9])", std::regex::optimize)},
        {{R"([A-Za-z]{1,6}[-_][A-Za-z0-9]{2,}(?:[-_][A-Za-z0-9]+)*)", 1.25, PiiFormat::Ticket},
         std::regex(R"([A-Za-z]{1,6}[-_][A-Za-z0-9]{2,}(?:[-_][A-Za-z0-9]+)*)",
                    std::regex::optimize)},
        {{R"([0-9]{5,19})", 1.20, PiiFormat::Digits},
         std::regex(R"([0-9]{5,19})", std::regex::optimize)},
        {{R"([A-Za-z0-9_-]{24,})", 1.10, PiiFormat::None},
         std::regex(R"([A-Za-z0-9_-]{24,})", std::regex::optimize)},
    };
    return patterns;
}

std::string compact_digits(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c >= '0' && c <= '9') out.push_back(c);
    }
    return out;
}

bool luhn_ok(const std::string & s) {
    int sum = 0;
    bool dbl = false;
    for (auto it = s.rbegin(); it != s.rend(); ++it) {
        int d = *it - '0';
        if (dbl) {
            d *= 2;
            if (d > 9) d -= 9;
        }
        sum += d;
        dbl = !dbl;
    }
    return sum % 10 == 0;
}

bool valid_ssn(const std::string & s) {
    if (s.size() != 11 || s[3] != '-' || s[6] != '-') return false;
    const int area = std::stoi(s.substr(0, 3));
    const int group = std::stoi(s.substr(4, 2));
    const int serial = std::stoi(s.substr(7, 4));
    return area != 0 && area != 666 && area < 900 && group != 0 && serial != 0;
}

bool valid_itin(const std::string & s) {
    if (s.size() != 11 || s[3] != '-' || s[6] != '-') return false;
    const int group = s[4] - '0';
    return s[0] == '9' && group >= 7 && group <= 9;
}

bool valid_ein(const std::string & s) {
    return s.size() == 10 && s[2] == '-';
}

bool valid_ipv6(const std::string & s) {
    if (s.find(':') == std::string::npos) return false;
    int groups = 0;
    int empties = 0;
    size_t start = 0;
    while (start <= s.size()) {
        const size_t end = s.find(':', start);
        const std::string part = s.substr(
            start, end == std::string::npos ? std::string::npos : end - start);
        if (part.empty()) {
            ++empties;
        } else {
            if (part.size() > 4) return false;
            for (char c : part) {
                if (!std::isxdigit((unsigned char) c)) return false;
            }
            ++groups;
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    if (empties > 1) return false;
    return empties == 1 ? groups <= 7 : groups == 8;
}

bool valid_ipv4(const std::string & s) {
    int part = 0;
    int value = 0;
    int digits = 0;
    for (char c : s) {
        if (c == '.') {
            if (digits == 0 || value > 255) return false;
            ++part;
            value = 0;
            digits = 0;
        } else {
            value = value * 10 + (c - '0');
            ++digits;
            if (digits > 3) return false;
        }
    }
    return part == 3 && digits > 0 && value <= 255;
}

bool valid_iban(const std::string & s) {
    std::string compact;
    for (char c : s) {
        if (c != ' ') compact.push_back(c);
    }
    if (compact.size() < 15) return false;
    const std::string rearranged = compact.substr(4) + compact.substr(0, 4);
    int mod = 0;
    for (char c : rearranged) {
        if (c >= '0' && c <= '9') {
            mod = (mod * 10 + (c - '0')) % 97;
        } else if (c >= 'A' && c <= 'Z') {
            const int v = c - 'A' + 10;
            mod = (mod * 100 + v) % 97;
        } else {
            return false;
        }
    }
    return mod == 1;
}

bool format_valid(PiiFormat fmt, const std::string & s) {
    switch (fmt) {
        case PiiFormat::Ssn:
            return valid_ssn(s);
        case PiiFormat::Itin:
            return valid_itin(s);
        case PiiFormat::Ein:
            return valid_ein(s);
        case PiiFormat::CreditCard: {
            const std::string digits = compact_digits(s);
            return digits.size() >= 13 && digits.size() <= 19 && luhn_ok(digits);
        }
        case PiiFormat::Iban:
            return valid_iban(s);
        case PiiFormat::Ip:
            return s.find(':') != std::string::npos ? valid_ipv6(s) : valid_ipv4(s);
        default:
            return true;
    }
}

bool fullwidth_case_number(const std::string & s) {
    const Utf8Text t = decode_utf8(s);
    if (t.size() < 8) return false;
    if (t.cps[0] != U'(' && t.cps[0] != (char32_t) 0xFF08) return false;
    for (int k = 1; k <= 4; ++k) {
        if (t.cps[(size_t) k] < '0' || t.cps[(size_t) k] > '9') return false;
    }
    return t.cps[5] == U')' || t.cps[5] == (char32_t) 0xFF09;
}

std::string normalized_words(const std::string & s) {
    std::string out;
    bool space = true;
    for (unsigned char c : s) {
        if (std::isalnum(c)) {
            out.push_back((char) std::tolower(c));
            space = false;
        } else if (!space && !out.empty()) {
            out.push_back(' ');
            space = true;
        }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

bool has_phrase(const std::string & hay, const char * phrase) {
    return (" " + hay + " ").find(std::string(" ") + phrase + " ") != std::string::npos;
}

} // namespace

const std::vector<PiiPattern> & pii_patterns() {
    static const std::vector<PiiPattern> patterns = [] {
        std::vector<PiiPattern> out;
        out.reserve(compiled_patterns().size());
        for (const auto & p : compiled_patterns()) out.push_back(p.spec);
        return out;
    }();
    return patterns;
}

bool is_ascii_digits(const std::string & text) {
    if (text.empty()) return false;
    for (char c : text) {
        if (c < '0' || c > '9') return false;
    }
    return true;
}

PiiFormat detect_format(const std::string & text) {
    if (fullwidth_case_number(text)) return PiiFormat::CaseNumber;
    for (const auto & p : compiled_patterns()) {
        if (!std::regex_match(text, p.re)) continue;
        if (p.spec.format == PiiFormat::None) return PiiFormat::None;
        if (format_valid(p.spec.format, text)) return p.spec.format;
    }
    return PiiFormat::None;
}

bool type_matches_format(PiiFormat format, const TypeDef & type) {
    const std::string id = normalized_words(type.id);
    const std::string desc = normalized_words(type.description);
    auto has = [&](const char * phrase) {
        return has_phrase(id, phrase) || has_phrase(desc, phrase);
    };
    switch (format) {
        case PiiFormat::Email:
            return has("email");
        case PiiFormat::Phone:
            return has("phone") || has("telephone") || has("mobile");
        case PiiFormat::Url:
            return has("url") || has("website") || has("link");
        case PiiFormat::Ssn:
            return has("ssn") || has("social security number");
        case PiiFormat::Itin:
            return has("itin") || has("individual taxpayer identification number");
        case PiiFormat::Ein:
            return has("ein") || has("employer identification number");
        case PiiFormat::CreditCard:
            return has("credit card") || has("card number") || has("primary account number") ||
                   has("pan");
        case PiiFormat::Iban:
            return has("iban") || has("international bank account number");
        case PiiFormat::Ip:
            return has("ip") || has("ip address") || has("ipv4") || has("ipv6");
        case PiiFormat::ApiKey:
            return has("api key") || has("secret") || has("token");
        case PiiFormat::Ticket:
            return has("ticket") || has("order");
        case PiiFormat::CaseNumber:
            return has("case number") || has("case") || has("docket");
        case PiiFormat::Dob:
            return has("date of birth") || has("dob") || has("birthdate");
        default:
            return false;
    }
}

} // namespace anypii
