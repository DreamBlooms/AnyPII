#include "anypii/prompt.hpp"

#include <algorithm>
#include <cstdlib>

#include <sstream>
#include <stdexcept>

#include "anypii/unicode.hpp"

namespace anypii {

namespace {

std::string redact_label(RedactMode mode) {
    switch (mode) {
        case RedactMode::Redact: return "打码";
        case RedactMode::Keep: return "保留";
        default: return "按政策自动判断";
    }
}

std::string sanitize_text(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        if (i + 1 < s.size() && s[i] == '<' && s[i + 1] == '|') {
            out += "< |";
            i += 2;
        } else if (i + 1 < s.size() && s[i] == '|' && s[i + 1] == '>') {
            out += "| >";
            i += 2;
        } else {
            out.push_back(s[i]);
            ++i;
        }
    }
    return out;
}

std::string sanitize_line(const std::string & s) {
    std::string out = sanitize_text(s);
    for (char & c : out) {
        if (c == '\n' || c == '\r') c = ' ';
    }
    return out;
}

std::string chatml_head(const std::string & system, const std::string & user) {
    // Matches the Qwen3.5 chat template with thinking disabled: the assistant
    // turn starts with an already-closed thought block.
    return "<|im_start|>system\n" + system +
           "<|im_end|>\n<|im_start|>user\n" + user +
           "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

std::string type_legend(const TypeTable & table) {
    std::ostringstream os;
    for (size_t i = 0; i < table.types.size(); ++i) {
        const auto & t = table.types[i];
        os << table.codes[i] << " = " << t.id << "：" << sanitize_text(t.description)
           << "（" << redact_label(t.redact) << "）\n";
    }
    os << table.none_code << " = none：该候选不是上述任何类型的实体\n";
    return os.str();
}

std::string common_system(const TypeTable & table, const AnyPiiRequest & request,
                          const std::string & task_rules) {
    std::ostringstream os;
    os << "你是 AnyPII，一个严格的信息抽取与脱敏判定引擎。"
          "你不写解释、不生成改写文本；你只根据原文、政策和类型说明填写短代码。\n\n";
    os << "政策：\n" << (request.policy.empty() ? "（未提供，按类型说明判断）" : sanitize_text(request.policy))
       << "\n\n";
    os << "类型（代码 = 类型 id）：\n" << type_legend(table) << "\n";
    os << task_rules;
    return os.str();
}

PromptSegments finish(Engine & engine, const std::string & head,
                      const std::vector<std::string> & line_prefixes,
                      const std::vector<std::vector<std::string>> & slot_codes,
                      const std::string & tail, const std::string & filler = "\n") {
    PromptSegments out;
    std::string text = head;
    for (size_t i = 0; i < line_prefixes.size(); ++i) {
        text += line_prefixes[i];
        out.segments.push_back(text);
        out.slot_groups.push_back(groups_for_codes(engine, slot_codes[i]));
        text = filler;
    }
    text += tail;
    out.segments.push_back(text);
    return out;
}

} // namespace

TypeTable build_type_table(const std::vector<TypeDef> & types) {
    static const std::string pool =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    if (types.size() + 1 > pool.size()) {
        throw std::invalid_argument("too many types: at most 51 are supported");
    }
    TypeTable table;
    table.types = types;
    for (size_t i = 0; i < types.size(); ++i) {
        table.codes.push_back(pool.substr(i, 1));
    }
    table.none_code = pool.substr(types.size(), 1);
    return table;
}

std::vector<std::vector<int>> groups_for_codes(Engine & engine,
                                               const std::vector<std::string> & codes) {
    std::vector<std::vector<int>> groups;
    groups.reserve(codes.size());
    for (const auto & code : codes) groups.push_back(engine.label_group(code));
    return groups;
}

PromptSegments build_type_pass(Engine & engine, const TypeTable & table,
                               const AnyPiiRequest & request,
                               const std::vector<Candidate> & candidates,
                               int shift) {
    const int T = (int) table.types.size();
    if (T <= 0) throw std::invalid_argument("no types");
    shift = ((shift % T) + T) % T;

    TypeTable shifted;
    shifted.codes = table.codes;
    shifted.none_code = table.none_code;
    shifted.types.resize((size_t) T);
    for (int j = 0; j < T; ++j) {
        shifted.types[(size_t) j] = table.types[(size_t) (((j - shift) % T + T) % T)];
    }

    std::string rules =
        "判定规则：\n"
        "1. 每个候选都有局部上下文和完整类型选项；选择最符合的一项。\n"
        "2. 必须选择“完整且准确”的实体提及：不要包含职务、头衔、角色、量词、动词、"
        "类别提示词（例如“患者”“医师”“护士”“商品名”“通用名”等）或相邻词语。\n"
        "3. 如果候选片段不是一个真实实体，或政策/类型没有描述它，选择 none。\n"
        "4. 同一类型可能有多个实体；每个候选独立判断。\n"
        "5. assistant 中只填写每个候选对应的类型代码，每行一个，不要解释。\n";

    const std::string system = common_system(shifted, request, rules);
    const Utf8Text ut = decode_utf8(request.text);

    std::string user = "原文：\n" + sanitize_text(request.text) + "\n\n候选与选项：\n";
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto & c = candidates[i];
        const int left = std::max(0, c.start - 12);
        const int right = std::min(ut.size(), c.end + 12);
        const std::string context = sanitize_line(ut.substr(left, c.start)) + "【" +
                                    sanitize_line(c.text) + "】" +
                                    sanitize_line(ut.substr(c.end, right));
        user += "候选" + std::to_string(i) + "：" + sanitize_line(c.text) + "\n上下文：";
        user += context;
        user += "\n";
        for (int j = 0; j < T; ++j) {
            user += shifted.codes[(size_t) j] + ". " + shifted.types[(size_t) j].id + "\n";
        }
        user += shifted.none_code + ". none\n\n";
    }

    std::vector<std::string> lines;
    std::vector<std::vector<std::string>> slots;
    std::vector<std::string> label_codes = shifted.codes;
    label_codes.push_back(shifted.none_code);
    lines.reserve(candidates.size());
    slots.reserve(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        lines.push_back("候选" + std::to_string(i) + "：");
        slots.push_back(label_codes);
    }
    return finish(engine, chatml_head(system, user), lines, slots, "\n<|im_end|>\n", "\n");
}

PromptSegments build_boundary_pass(Engine & engine, const TypeTable & table,
                                   const AnyPiiRequest & request,
                                   const std::vector<BoundaryComponent> & components) {
    std::string rules =
        "边界裁决规则：\n"
        "1. 每个 COMPONENT 是原文中同一处文字的若干重叠候选（片段 + 类型）。\n"
        "2. 选择唯一一个最完整、最准确的实体提及；如果都不符合政策/类型，选择 none。\n"
        "3. 不要把职务、头衔、角色、类别提示词或相邻词语算进去。\n"
        "4. 每个 COMPONENT 的选项使用小写字母，从 a 重新开始；只在 CHOICE:: 后面填写选项字母。\n";

    const std::string system = common_system(table, request, rules);
    const std::string user = "原文：\n" + sanitize_text(request.text) + "\n";
    const std::string head = chatml_head(system, user);

    static const std::string pool = "abcdefghijklmnopqrstuvwxyz";
    std::vector<std::string> lines;
    std::vector<std::vector<std::string>> slots;

    for (size_t ci = 0; ci < components.size(); ++ci) {
        const auto & comp = components[ci];
        if (comp.options.empty()) continue;
        if (comp.options.size() + 1 > 26) {
            throw std::invalid_argument("too many boundary options");
        }
        std::string block = "COMPONENT " + std::to_string(ci + 1) + ":\n";
        std::vector<std::string> codes;
        for (size_t oi = 0; oi < comp.options.size(); ++oi) {
            const std::string code = pool.substr(oi, 1);
            codes.push_back(code);
            block += code + ") " + sanitize_line(comp.options[oi].text) +
                     "  [" + comp.options[oi].type + "]\n";
        }
        const std::string none_code = pool.substr(comp.options.size(), 1);
        codes.push_back(none_code);
        block += none_code + ") 都不是\nCHOICE::";
        lines.push_back(block);
        slots.push_back(codes);
    }

    return finish(engine, head, lines, slots, "\n<|im_end|>\n");
}

PromptSegments build_redact_pass(Engine & engine, const TypeTable & table,
                                 const AnyPiiRequest & request,
                                 const std::vector<RedactItem> & items) {
    std::string rules =
        "脱敏裁决规则：\n"
        "1. 根据政策与类型说明，判断每个提及是否应该被打码。\n"
        "2. 只填写 Y（打码）或 N（保留）。\n";

    const std::string system = common_system(table, request, rules);
    const std::string user = "原文：\n" + sanitize_text(request.text) + "\n";

    std::vector<std::string> lines;
    std::vector<std::vector<std::string>> slots;
    for (size_t i = 0; i < items.size(); ++i) {
        lines.push_back("R" + std::to_string(i) + ": " +
                        sanitize_line(items[i].text) + "  [" + items[i].type +
                        "] | REDACT::");
        slots.push_back({"Y", "N"});
    }
    return finish(engine, chatml_head(system, user), lines, slots, "\n<|im_end|>\n");
}

namespace {

const char * kSystemPrompt =
    "You are a System One decision model. You read the State and answer each "
    "Question by choosing exactly one of the listed options. You never explain. "
    "You answer with the single option label only. Judge the mention itself: a "
    "disease, symptom, action, or ordinary noun is not a drug, person, or "
    "organization unless a listed type explicitly covers it. For example, 肺炎 "
    "or 感染 is not a drug name.";

std::string chatml_prefix(const std::string & body) {
    return std::string("<|im_start|>system\n") + kSystemPrompt +
           "<|im_end|>\n<|im_start|>user\n" + body;
}

std::string assistant_tail() {
    return "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

std::string branch_tail(const std::string & criterion, const std::string & options) {
    std::string out = "# Criterion\n" + criterion + "\n\n# Options\n" + options;
    out += "\n# Answer\n";
    out += assistant_tail();
    return out;
}

std::string shared_body(const AnyPiiRequest & request, const TypeTable & table) {
    std::string body = "# Evidence\n" + sanitize_text(request.text) + "\n\n";
    body += "# Policy\n";
    body += request.policy.empty() ? "none" : sanitize_text(request.policy);
    body += "\n\n# Entity types\n";
    for (size_t i = 0; i < table.types.size(); ++i) {
        body += table.codes[i] + ". " + table.types[i].id + ": " +
                sanitize_text(table.types[i].description) + "\n";
    }
    body += table.none_code + ". none: not an entity of the listed types "
            "(e.g., a disease, symptom, action, or ordinary word)\n";
    return body;
}

std::string local_context(const Utf8Text & text, const Candidate & c, int radius = 20) {
    const int left = std::max(0, c.start - radius);
    const int right = std::min(text.size(), c.end + radius);
    return sanitize_line(text.substr(left, c.start)) + "【" + sanitize_line(c.text) + "】" +
           sanitize_line(text.substr(c.end, right));
}

} // namespace

BranchSet build_type_branches(Engine & engine, const TypeTable & table,
                              const AnyPiiRequest & request, const Utf8Text & text,
                              const std::vector<Candidate> & candidates,
                              bool compact) {
    if (table.types.size() + 1 > 26) {
        throw std::invalid_argument("at most 25 types are supported in one request");
    }
    BranchSet set;
    set.prefix = chatml_prefix(shared_body(request, table));

    std::vector<std::string> label_codes = table.codes;
    label_codes.push_back(table.none_code);
    const auto groups = groups_for_codes(engine, label_codes);

    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto & c = candidates[i];
        const std::string ctx = local_context(text, c);

        std::string criterion = "Mention: \"" + sanitize_line(c.text) + "\"\n";
        criterion += "Context: \"" + ctx + "\"\n";
        if (compact) {
            criterion += "Which type code best describes this mention? Choose none if "
                         "it is not an entity.";
        } else {
            criterion += "Which entity type best describes this exact mention? Choose the "
                         "complete mention only; do not include titles, roles, category cues, "
                         "or neighboring words. If none applies, choose the none option.";
        }

        std::string options;
        for (size_t t = 0; t < table.types.size(); ++t) {
            options += table.codes[t] + ". " + table.types[t].id + "\n";
        }
        options += table.none_code + ". none\n";

        BranchPrompt type_branch;
        type_branch.tail = branch_tail(criterion, options);
        type_branch.label_groups = groups;
        set.branches.push_back(std::move(type_branch));
    }
    return set;
}

BranchSet build_boundary_branches(Engine & engine, const TypeTable & table,
                                  const AnyPiiRequest & request,
                                  const std::vector<BoundaryComponent> & components) {
    BranchSet set;
    set.prefix = chatml_prefix(shared_body(request, table));
    static const std::string pool = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    auto add_branch = [&](const std::vector<BoundaryOption> & opts, bool none_first) {
        std::string options;
        std::vector<std::string> codes;
        std::string none_line = "none";
        if (none_first) {
            codes.push_back(pool.substr(0, 1));
            options += codes.back() + ". none\n";
        }
        for (size_t i = 0; i < opts.size(); ++i) {
            const size_t pos = none_first ? i + 1 : i;
            const std::string code = pool.substr(pos, 1);
            codes.push_back(code);
            options += code + ". \"" + sanitize_line(opts[i].text) + "\" — " +
                       opts[i].type + "\n";
        }
        if (!none_first) {
            codes.push_back(pool.substr(opts.size(), 1));
            options += codes.back() + ". none\n";
        }
        std::string criterion =
            "The Evidence contains overlapping candidate spans for one mention. "
            "Choose the single most accurate complete mention and its type. Prefer the "
            "longest complete mention; do not choose a single-character fragment unless "
            "it is the whole entity. Do not include titles, roles, category cues, or "
            "neighboring words. If none is a real entity under the policy, choose none.";
        BranchPrompt br;
        br.tail = branch_tail(criterion, options);
        br.label_groups = groups_for_codes(engine, codes);
        set.branches.push_back(std::move(br));
    };

    for (const auto & comp : components) {
        if (comp.options.empty()) continue;
        if (comp.options.size() + 1 > 26) {
            throw std::invalid_argument("too many boundary options");
        }
        std::vector<BoundaryOption> rev(comp.options.rbegin(), comp.options.rend());
        add_branch(comp.options, false);
        add_branch(rev, true);
    }
    return set;
}

BranchSet build_redact_branches(Engine & engine, const TypeTable & table,
                                const AnyPiiRequest & request,
                                const std::vector<RedactItem> & items) {
    BranchSet set;
    set.prefix = chatml_prefix(shared_body(request, table));
    const auto groups = groups_for_codes(engine, {"A", "B"});

    auto add_branch = [&](const RedactItem & item, bool redact_first) {
        const TypeDef * def = nullptr;
        for (const auto & t : table.types) {
            if (t.id == item.type) { def = &t; break; }
        }
        std::string criterion = "Mention: \"" + sanitize_line(item.text) + "\"\n";
        criterion += "Type: " + item.type;
        if (def && !def->description.empty()) {
            criterion += " (" + sanitize_text(def->description) + ")";
        }
        criterion += "\nShould this mention be redacted under the Policy? Choose "
                     "redact if the policy asks to obscure it; choose keep if the "
                     "policy asks to preserve it.";
        const std::string options = redact_first ? "A. redact\nB. keep\n"
                                                 : "A. keep\nB. redact\n";
        BranchPrompt br;
        br.tail = branch_tail(criterion, options);
        br.label_groups = groups;
        set.branches.push_back(std::move(br));
    };

    for (const auto & item : items) {
        add_branch(item, true);
        add_branch(item, false);
    }
    return set;
}

BranchSet build_verify_branches(Engine & engine, const TypeTable & table,
                                const AnyPiiRequest & request, const Utf8Text & text,
                                const std::vector<VerifyItem> & items) {
    BranchSet set;
    set.prefix = chatml_prefix(shared_body(request, table));
    const auto yes_no = groups_for_codes(engine, {"A", "B"});

    for (const auto & item : items) {
        const TypeDef * def = nullptr;
        for (const auto & t : table.types) {
            if (t.id == item.type) { def = &t; break; }
        }
        Candidate c;
        c.start = item.start;
        c.end = item.end;
        c.text = item.text;
        std::string criterion = "Mention: \"" + sanitize_line(item.text) + "\"\n";
        criterion += "Context: \"" + local_context(text, c) + "\"\n";
        criterion += "Type: " + item.type;
        if (def && !def->description.empty()) {
            criterion += " (" + sanitize_text(def->description) + ")";
        }
        criterion += "\nIs this mention an instance of this type in the Evidence? "
                     "Answer yes or no.";
        BranchPrompt br;
        br.tail = branch_tail(criterion, "A. yes\nB. no\n");
        br.label_groups = yes_no;
        set.branches.push_back(std::move(br));
    }
    return set;
}

BranchSet build_pair_branches(Engine & engine, const AnyPiiRequest & request,
                              const std::vector<PairChoice> & pairs,
                              bool reverse) {
    BranchSet set;
    set.prefix = chatml_prefix(shared_body(request, build_type_table(request.types)));
    static const std::string pool = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    auto add = [&](const PairChoice & pair, bool reversed) {
        const size_t n = pair.choices.size();
        if (n < 2 || n > 26) throw std::invalid_argument("pair choice needs 2..26 options");
        std::string options;
        std::vector<std::string> codes;
        for (size_t j = 0; j < n; ++j) {
            const size_t src = reversed ? (n - 1 - j) : j;
            const std::string code = pool.substr(j, 1);
            codes.push_back(code);
            options += code + ". " + sanitize_line(pair.choices[src]) + "\n";
        }
        BranchPrompt br;
        br.tail = branch_tail(pair.criterion, options);
        br.label_groups = groups_for_codes(engine, codes);
        set.branches.push_back(std::move(br));
    };

    for (const auto & pair : pairs) {
        add(pair, false);
        if (reverse) add(pair, true);
    }
    return set;
}

} // namespace anypii