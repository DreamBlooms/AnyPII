// Regex formats for common PII categories, following the NIST SP 800-122
// taxonomy where a stable surface format exists (email, phone, SSN, ITIN, EIN,
// payment card, IBAN, IP, URL, date of birth, etc.). These are structural
// patterns, not language-specific word lists.
#pragma once

#include <string>
#include <vector>

#include "anypii/types.hpp"

namespace anypii {

enum class PiiFormat {
    None,
    Email,
    Phone,
    Url,
    Ssn,
    Itin,
    Ein,
    CreditCard,
    Iban,
    Ip,
    ApiKey,
    Ticket,
    CaseNumber,
    Dob,
    Npi,
    Mbi,
    Dea,
    Vin,
    Passport,
    AwsKey,
    Digits,
};

struct PiiPattern {
    const char * regex;
    double prior;
    PiiFormat format;
};

// Structural candidate patterns shared by candidate proposal and the fast path.
const std::vector<PiiPattern> & pii_patterns();

// Classify a complete string. Returns None when the string is not a complete
// match or fails the format's checksum/validation (Luhn, IBAN mod-97, SSN/ITIN
// ranges, IPv4 octets).
PiiFormat detect_format(const std::string & text);

// True when the caller's type id or description names this NIST format. The
// match is on the canonical format name only (for example type id "ssn" or
// "social_security_number"), so the fast path stays driven by the request.
bool type_matches_format(PiiFormat format, const TypeDef & type);

bool is_ascii_digits(const std::string & text);

} // namespace anypii
