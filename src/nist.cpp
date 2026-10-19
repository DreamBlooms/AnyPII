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
        {{R"([0-9]{3}[- ][0-9]{2}[- ][0-9]{4})", 1.35, PiiFormat::Ssn},
         std::regex(R"([0-9]{3}[- ][0-9]{2}[- ][0-9]{4})", std::regex::optimize)},
        {{R"(9[0-9]{2}[- ][7-9][0-9][- ][0-9]{4})", 1.35, PiiFormat::Itin},
         std::regex(R"(9[0-9]{2}[- ][7-9][0-9][- ][0-9]{4})", std::regex::optimize)},
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
        {{R"(\b[A-Z][0-9]{8}\b)", 1.25, PiiFormat::Passport},
         std::regex(R"(\b[A-Z][0-9]{8}\b)", std::regex::optimize)},
        {{R"((?:AKIA|ASIA)[0-9A-Z]{16})", 1.35, PiiFormat::AwsKey},
         std::regex(R"((?:AKIA|ASIA)[0-9A-Z]{16})", std::regex::optimize)},
        {{R"([0-9]{10})", 1.25, PiiFormat::Npi},
         std::regex(R"([0-9]{10})", std::regex::optimize)},
        {{R"((?:\+?86[- ]?)?1[3-9][0-9]{9})", 1.30, PiiFormat::Phone},
         std::regex(R"((?:\+?86[- ]?)?1[3-9][0-9]{9})", std::regex::optimize)},
        {{R"(\b(?:\+?1[-. ]?)?(?:\([0-9]{3}\)|[0-9]{3})[-. ]?[0-9]{3}[-. ]?[0-9]{4}\b)",
          1.30, PiiFormat::Phone},
         std::regex(
             R"(\b(?:\+?1[-. ]?)?(?:\([0-9]{3}\)|[0-9]{3})[-. ]?[0-9]{3}[-. ]?[0-9]{4}\b)",
             std::regex::optimize)},
        {{R"(\+?[0-9][0-9 ()\-]{7,}[0-9])", 1.25, PiiFormat::Phone},
         std::regex(R"(\+?[0-9][0-9 ()\-]{7,}[0-9])", std::regex::optimize)},
        {{R"([A-Za-z]{1,6}[-_][A-Za-z0-9]{2,}(?:[-_][A-Za-z0-9]+)*)", 1.25, PiiFormat::Ticket},
         std::regex(R"([A-Za-z]{1,6}[-_][A-Za-z0-9]{2,}(?:[-_][A-Za-z0-9]+)*)",
                    std::regex::optimize)},
        {{R"([1-9][A-Za-z][A-Za-z0-9][0-9][A-Za-z][A-Za-z0-9][0-9][A-Za-z]{2}[0-9]{2})",
          1.30, PiiFormat::Mbi},
         std::regex(
             R"([1-9][A-Za-z][A-Za-z0-9][0-9][A-Za-z][A-Za-z0-9][0-9][A-Za-z]{2}[0-9]{2})",
             std::regex::optimize)},
        {{R"([A-Za-z]{2}[0-9]{7})", 1.25, PiiFormat::Dea},
         std::regex(R"([A-Za-z]{2}[0-9]{7})", std::regex::optimize)},
        {{R"([A-HJ-NPR-Z0-9]{17})", 1.25, PiiFormat::Vin},
         std::regex(R"([A-HJ-NPR-Z0-9]{17})", std::regex::optimize)},
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

std::string compact_digits_separators(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c >= '0' && c <= '9') out.push_back(c);
    }
    return out;
}

bool valid_ssn(const std::string & s) {
    const std::string d = compact_digits_separators(s);
    if (d.size() != 9) return false;
    const int area = std::stoi(d.substr(0, 3));
    const int group = std::stoi(d.substr(3, 2));
    const int serial = std::stoi(d.substr(5, 4));
    return area != 0 && area != 666 && area < 900 && group != 0 && serial != 0;
}

bool valid_itin(const std::string & s) {
    const std::string d = compact_digits_separators(s);
    if (d.size() != 9 || d[0] != '9') return false;
    const int group = std::stoi(d.substr(3, 2));
    return (group >= 50 && group <= 65) || (group >= 70 && group <= 88) ||
           (group >= 90 && group <= 92) || (group >= 94 && group <= 99);
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

bool valid_npi(const std::string & s) {
    if (s.size() != 10 || !is_ascii_digits(s)) return false;
    const std::string prefixed = "80840" + s;
    int sum = 0;
    bool dbl = false;
    for (auto it = prefixed.rbegin(); it != prefixed.rend(); ++it) {
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

bool valid_mbi(const std::string & s) {
    if (s.size() != 11) return false;
    auto is_digit = [](char c) { return c >= '0' && c <= '9'; };
    auto is_alpha = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    };
    auto allowed_alpha = [](char c) {
        const char u = (char) std::toupper((unsigned char) c);
        return u >= 'A' && u <= 'Z' && u != 'S' && u != 'L' && u != 'O' &&
               u != 'I' && u != 'B' && u != 'Z';
    };
    for (int i : {0, 3, 6, 9, 10}) {
        if (!is_digit(s[(size_t) i])) return false;
    }
    for (int i : {1, 4, 7, 8}) {
        if (!is_alpha(s[(size_t) i]) || !allowed_alpha(s[(size_t) i])) return false;
    }
    for (int i : {2, 5}) {
        if (!is_digit(s[(size_t) i]) &&
            !(is_alpha(s[(size_t) i]) && allowed_alpha(s[(size_t) i]))) {
            return false;
        }
    }
    return true;
}

bool valid_dea(const std::string & s) {
    if (s.size() != 9) return false;
    for (int i = 0; i < 2; ++i) {
        if (!std::isalpha((unsigned char) s[(size_t) i])) return false;
    }
    for (int i = 2; i < 9; ++i) {
        if (!std::isdigit((unsigned char) s[(size_t) i])) return false;
    }
    const int check = (s[2] - '0') + (s[4] - '0') + (s[6] - '0') +
                      2 * ((s[3] - '0') + (s[5] - '0') + (s[7] - '0'));
    return check % 10 == (s[8] - '0');
}

bool valid_vin(const std::string & s) {
    if (s.size() != 17) return false;
    static const int weights[17] = {8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2};
    int sum = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = (char) std::toupper((unsigned char) s[i]);
        int value = -1;
        if (c >= '0' && c <= '9') value = c - '0';
        else {
            switch (c) {
                case 'A': value = 1; break;
                case 'B': value = 2; break;
                case 'C': value = 3; break;
                case 'D': value = 4; break;
                case 'E': value = 5; break;
                case 'F': value = 6; break;
                case 'G': value = 7; break;
                case 'H': value = 8; break;
                case 'J': value = 1; break;
                case 'K': value = 2; break;
                case 'L': value = 3; break;
                case 'M': value = 4; break;
                case 'N': value = 5; break;
                case 'P': value = 7; break;
                case 'R': value = 9; break;
                case 'S': value = 2; break;
                case 'T': value = 3; break;
                case 'U': value = 4; break;
                case 'V': value = 5; break;
                case 'W': value = 6; break;
                case 'X': value = 7; break;
                case 'Y': value = 8; break;
                case 'Z': value = 9; break;
                default: return false;
            }
        }
        sum += value * weights[i];
    }
    const int remainder = sum % 11;
    const char expected = remainder == 10 ? 'X' : (char) ('0' + remainder);
    return s[8] == expected;
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
        case PiiFormat::Npi:
            return valid_npi(s);
        case PiiFormat::Mbi:
            return valid_mbi(s);
        case PiiFormat::Dea:
            return valid_dea(s);
        case PiiFormat::Vin:
            return valid_vin(s);
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
        case PiiFormat::Npi:
            return has("npi") || has("national provider identifier");
        case PiiFormat::Mbi:
            return has("mbi") || has("medicare beneficiary");
        case PiiFormat::Dea:
            return has("dea") || has("drug enforcement administration");
        case PiiFormat::Vin:
            return has("vin") || has("vehicle identification number");
        case PiiFormat::Passport:
            return has("passport") || has("passport number");
        case PiiFormat::AwsKey:
            return has("aws") || has("access key id") || has("access key");
        default:
            return false;
    }
}

} // namespace anypii
