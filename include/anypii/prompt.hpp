#pragma once

#include <string>
#include <vector>

#include "anypii/engine.hpp"
#include "anypii/types.hpp"
#include "anypii/unicode.hpp"

namespace anypii {

struct TypeTable {
    std::vector<TypeDef> types;
    std::vector<std::string> codes; // one code per type, in request order
    std::string none_code;
};

struct BranchSet {
    std::string prefix;
    std::vector<BranchPrompt> branches;
};

struct PromptSegments {
    std::vector<std::string> segments;
    // slot_groups[s] is the label groups for slot s.
    std::vector<std::vector<std::vector<int>>> slot_groups;
};

struct BoundaryOption {
    int start = 0;
    int end = 0;
    std::string text;
    std::string type;
    double weight = 0.0;
};

struct BoundaryComponent {
    std::vector<BoundaryOption> options; // does not include "none"
};

struct PairChoice {
    std::vector<std::string> choices; // option text, e.g. spans or type descriptions
    std::vector<std::string> types;   // type id per option; empty means none/neither
    std::string criterion;
};

struct VerifyItem {
    int start = 0;
    int end = 0;
    std::string text;
    std::string type;
};

struct RedactItem {
    int start = 0;
    int end = 0;
    std::string text;
    std::string type;
};

TypeTable build_type_table(const std::vector<TypeDef> & types);
std::vector<std::vector<int>> groups_for_codes(Engine & engine,
                                               const std::vector<std::string> & codes);

PromptSegments build_type_pass(Engine & engine, const TypeTable & table,
                               const AnyPiiRequest & request,
                               const std::vector<Candidate> & candidates,
                               int shift = 0);

PromptSegments build_boundary_pass(Engine & engine, const TypeTable & table,
                                   const AnyPiiRequest & request,
                                   const std::vector<BoundaryComponent> & components);

PromptSegments build_redact_pass(Engine & engine, const TypeTable & table,
                                 const AnyPiiRequest & request,
                                 const std::vector<RedactItem> & items);

BranchSet build_type_branches(Engine & engine, const TypeTable & table,
                              const AnyPiiRequest & request, const Utf8Text & text,
                              const std::vector<Candidate> & candidates,
                              bool compact = false);

BranchSet build_boundary_branches(Engine & engine, const TypeTable & table,
                                  const AnyPiiRequest & request,
                                  const std::vector<BoundaryComponent> & components);

BranchSet build_redact_branches(Engine & engine, const TypeTable & table,
                                const AnyPiiRequest & request,
                                const std::vector<RedactItem> & items);

BranchSet build_verify_branches(Engine & engine, const TypeTable & table,
                                const AnyPiiRequest & request, const Utf8Text & text,
                                const std::vector<VerifyItem> & items);

BranchSet build_pair_branches(Engine & engine, const AnyPiiRequest & request,
                              const std::vector<PairChoice> & pairs);

} // namespace anypii
