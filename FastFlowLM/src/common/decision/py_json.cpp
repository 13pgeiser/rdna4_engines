/// \file py_json.cpp
/// \brief Python json-module semantics for the d3 contract (see py_json.hpp).
#include "decision/py_json.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace flm::d3 {

// ---------------------------------------------------------------------------------------------
// PyValue
// ---------------------------------------------------------------------------------------------
PyValue PyValue::integer(std::string digits) {
    PyValue p;
    p.kind = Kind::Int;
    // int("-0") == 0, and JSON forbids leading zeros, so only the sign of zero needs care
    if (digits == "-0") digits = "0";
    p.text = std::move(digits);
    return p;
}

const PyValue* PyValue::get(const std::string& key) const {
    for (const auto& m : members) {
        if (m.first == key) return &m.second;
    }
    return nullptr;
}

void PyValue::set(const std::string& key, PyValue value) {
    for (auto& m : members) {
        if (m.first == key) {
            m.second = std::move(value);
            return;
        }
    }
    members.emplace_back(key, std::move(value));
}

bool PyValue::truthy() const {
    switch (kind) {
        case Kind::Null: return false;
        case Kind::Bool: return b;
        case Kind::Int: return text != "0";
        case Kind::Float: return f != 0.0;   // also false for -0.0; NaN is true (unreachable from JSON)
        case Kind::String: return !text.empty();
        case Kind::Array: return !items.empty();
        case Kind::Object: return !members.empty();
    }
    return false;
}

// ---------------------------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------------------------
namespace {

struct Parser {
    const std::string& s;
    size_t i = 0;
    int depth = 0;

    /// json.JSONDecodeError's text: "<msg>: line L column C (char N)" (N counts code points in
    /// Python; here bytes, which agree for ASCII bodies)
    [[noreturn]] void fail(const std::string& what, size_t at = std::string::npos) const {
        size_t pos = at == std::string::npos ? i : at;
        size_t line = 1, col = 1;
        for (size_t k = 0; k < pos && k < s.size(); k++) {
            if (s[k] == '\n') { line++; col = 1; } else { col++; }
        }
        throw JsonError(what + ": line " + std::to_string(line) + " column " + std::to_string(col) +
                        " (char " + std::to_string(pos) + ")");
    }

    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
    }

    static void put_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    uint32_t hex4() {
        if (i + 4 > s.size()) fail("Invalid \\uXXXX escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; k++) {
            char c = s[i + k];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else fail("Invalid \\uXXXX escape");
        }
        i += 4;
        return v;
    }

    std::string str() {
        // at the opening quote
        const size_t opened = i;
        i++;
        std::string out;
        while (true) {
            if (i >= s.size()) fail("Unterminated string starting at", opened);
            unsigned char c = static_cast<unsigned char>(s[i]);
            if (c == '"') { i++; return out; }
            if (c < 0x20) fail("Invalid control character at");
            if (c == '\\') {
                i++;
                if (i >= s.size()) fail("Unterminated string starting at", opened);
                char e = s[i++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        uint32_t cp = hex4();
                        if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                            size_t save = i;
                            i += 2;
                            uint32_t lo = hex4();
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            } else {
                                i = save;
                            }
                        }
                        if (cp >= 0xD800 && cp <= 0xDFFF) fail("Lone surrogate escape (not encodable as UTF-8)");
                        put_utf8(out, cp);
                        break;
                    }
                    default: i--; fail("Invalid \\escape");
                }
                continue;
            }
            // raw UTF-8: validate and copy one code point
            int n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
            if (n == 0 || i + n > s.size()) fail("Invalid UTF-8");
            uint32_t cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
            for (int k = 1; k < n; k++) {
                unsigned char cc = static_cast<unsigned char>(s[i + k]);
                if ((cc >> 6) != 0x2) fail("Invalid UTF-8");
                cp = (cp << 6) | (cc & 0x3F);
            }
            static const uint32_t min_cp[5] = {0, 0, 0x80, 0x800, 0x10000};
            if (cp < min_cp[n] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) fail("Invalid UTF-8");
            out.append(s, i, n);
            i += n;
        }
    }

    PyValue number() {
        size_t start = i;
        if (s[i] == '-') i++;
        if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) { i = start; fail("Expecting value"); }
        if (s[i] == '0') i++;
        else while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) i++;
        bool is_float = false;
        if (i + 1 < s.size() && s[i] == '.' && std::isdigit(static_cast<unsigned char>(s[i + 1]))) {
            is_float = true;
            i++;
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) i++;
        }
        if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
            size_t save = i;
            i++;
            if (i < s.size() && (s[i] == '+' || s[i] == '-')) i++;
            if (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
                is_float = true;
                while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) i++;
            } else {
                i = save;   // Python's number regex stops before a bare 'e'; what follows is then extra data
            }
        }
        std::string lex = s.substr(start, i - start);
        if (!is_float) return PyValue::integer(lex);
        double v = 0.0;
        auto r = std::from_chars(lex.data(), lex.data() + lex.size(), v);
        if (r.ec == std::errc::result_out_of_range) {
            // float(lexeme) in Python: overflow -> +-inf, underflow -> +-0.0
            size_t e = lex.find_first_of("eE");
            bool neg_exp = e != std::string::npos && lex[e + 1] == '-';
            bool neg = lex[0] == '-';
            v = neg_exp ? (neg ? -0.0 : 0.0) : (neg ? -INFINITY : INFINITY);
        } else if (r.ec != std::errc()) {
            fail("Expecting value");
        }
        return PyValue::number(v);
    }

    PyValue value() {
        ws();
        if (i >= s.size()) fail("Expecting value");
        if (++depth > 900) fail("Too deeply nested");   // Python raises RecursionError near 1000
        PyValue out;
        char c = s[i];
        if (c == '{') {
            i++;
            out = PyValue::object();
            ws();
            if (i < s.size() && s[i] == '}') { i++; depth--; return out; }
            // key -> position in members: a repeated key keeps its first position and takes the
            // last value (Python's dict), in O(1) per key. PyValue::set() scans and would make a
            // large object quadratic (200k keys: 25 s, under the NPU queue).
            std::unordered_map<std::string, size_t> index;
            while (true) {
                ws();
                if (i >= s.size() || s[i] != '"') fail("Expecting property name enclosed in double quotes");
                std::string key = str();
                ws();
                if (i >= s.size() || s[i] != ':') fail("Expecting ':' delimiter");
                i++;
                PyValue v = value();
                auto slot = index.emplace(key, out.members.size());
                if (slot.second) out.members.emplace_back(std::move(key), std::move(v));
                else out.members[slot.first->second].second = std::move(v);
                ws();
                if (i < s.size() && s[i] == ',') { i++; continue; }
                if (i < s.size() && s[i] == '}') { i++; break; }
                fail("Expecting ',' delimiter");
            }
        } else if (c == '[') {
            i++;
            out = PyValue::array();
            ws();
            if (i < s.size() && s[i] == ']') { i++; depth--; return out; }
            while (true) {
                out.items.push_back(value());
                ws();
                if (i < s.size() && s[i] == ',') { i++; continue; }
                if (i < s.size() && s[i] == ']') { i++; break; }
                fail("Expecting ',' delimiter");
            }
        } else if (c == '"') {
            out = PyValue::string(str());
        } else if (s.compare(i, 4, "null") == 0) {
            i += 4;
        } else if (s.compare(i, 4, "true") == 0) {
            i += 4;
            out = PyValue::boolean(true);
        } else if (s.compare(i, 5, "false") == 0) {
            i += 5;
            out = PyValue::boolean(false);
        } else if (s.compare(i, 3, "NaN") == 0 || s.compare(i, 8, "Infinity") == 0 || s.compare(i, 9, "-Infinity") == 0) {
            std::string name = s[i] == 'N' ? "NaN" : s[i] == 'I' ? "Infinity" : "-Infinity";
            throw JsonError(name + " is not valid JSON");
        } else if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            out = number();
        } else {
            fail("Expecting value");
        }
        depth--;
        return out;
    }
};

}  // namespace

namespace {

void put_cp(std::string& out, uint32_t cp) {
    Parser::put_utf8(out, cp);
}

/// UTF-16 / UTF-32 bodies (json.detect_encoding found no UTF-8) re-encoded as UTF-8
std::string from_wide(const std::string& b, size_t skip, int width, bool big) {
    std::string out;
    size_t i = skip;
    auto unit = [&](size_t at) -> uint32_t {
        uint32_t v = 0;
        for (int k = 0; k < width; k++) {
            uint32_t byte = static_cast<unsigned char>(b[at + (big ? k : width - 1 - k)]);
            v = (v << 8) | byte;
        }
        return v;
    };
    while (i + width <= b.size()) {
        uint32_t cp = unit(i);
        i += width;
        if (width == 2 && cp >= 0xD800 && cp <= 0xDBFF && i + 2 <= b.size()) {
            uint32_t lo = unit(i);
            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                i += 2;
            }
        }
        if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
            throw JsonError(std::string("'utf-") + (width == 2 ? "16" : "32") + "' codec can't decode the body");
        }
        put_cp(out, cp);
    }
    if (i != b.size()) throw JsonError(std::string("'utf-") + (width == 2 ? "16" : "32") + "' codec can't decode the body: truncated data");
    return out;
}

/// The byte-level step of `json.loads(bytes)`: detect the encoding the way json.detect_encoding does
/// (UTF-8 with or without BOM, UTF-16/32 by BOM or by NUL pattern), then decode with
/// 'surrogatepass', raising Python's UnicodeDecodeError text on invalid UTF-8.
std::string decode_body(const std::string& b) {
    auto at = [&](size_t i) { return static_cast<unsigned char>(b[i]); };
    const size_t n = b.size();
    if (n >= 4 && ((at(0) == 0x00 && at(1) == 0x00 && at(2) == 0xFE && at(3) == 0xFF) ||
                   (at(0) == 0xFF && at(1) == 0xFE && at(2) == 0x00 && at(3) == 0x00))) {
        return from_wide(b, 4, 4, at(0) == 0x00);
    }
    if (n >= 2 && ((at(0) == 0xFE && at(1) == 0xFF) || (at(0) == 0xFF && at(1) == 0xFE))) {
        return from_wide(b, 2, 2, at(0) == 0xFE);
    }
    size_t start = 0;
    if (n >= 3 && at(0) == 0xEF && at(1) == 0xBB && at(2) == 0xBF) start = 3;
    else if (n >= 4 && at(0) == 0) return from_wide(b, 0, at(1) ? 2 : 4, true);
    else if (n >= 4 && at(1) == 0) return from_wide(b, 0, (at(2) || at(3)) ? 2 : 4, false);
    else if (n == 2 && at(0) == 0) return from_wide(b, 0, 2, true);
    else if (n == 2 && at(1) == 0) return from_wide(b, 0, 2, false);
    // UTF-8 validation with CPython's messages: the error covers the lead byte and the
    // continuation bytes that were valid ("byte 0xNN in position P" or "bytes in position P-Q")
    auto fail = [&](size_t s, size_t e, const char* why) {
        char buf[160];
        if (e - s <= 1) std::snprintf(buf, sizeof(buf), "'utf-8' codec can't decode byte 0x%02x in position %zu: %s", at(s), s - start, why);
        else std::snprintf(buf, sizeof(buf), "'utf-8' codec can't decode bytes in position %zu-%zu: %s", s - start, e - 1 - start, why);
        throw JsonError(buf);
    };
    for (size_t i = start; i < n;) {
        const unsigned char c = at(i);
        if (c < 0x80) { i++; continue; }
        int len;
        unsigned char lo = 0x80, hi = 0xBF;   // allowed range of the second byte
        if (c >= 0xC2 && c <= 0xDF) len = 2;
        else if (c >= 0xE0 && c <= 0xEF) { len = 3; if (c == 0xE0) lo = 0xA0; }   // ED: surrogatepass
        else if (c >= 0xF0 && c <= 0xF4) { len = 4; if (c == 0xF0) lo = 0x90; if (c == 0xF4) hi = 0x8F; }
        else fail(i, i + 1, "invalid start byte");
        size_t k = 1;
        for (; k < static_cast<size_t>(len); k++) {
            if (i + k >= n) fail(i, i + k, "unexpected end of data");
            const unsigned char cc = at(i + k);
            const bool ok = k == 1 ? (cc >= lo && cc <= hi) : (cc >= 0x80 && cc <= 0xBF);
            if (!ok) fail(i, i + k, "invalid continuation byte");
        }
        i += len;
    }
    return start ? b.substr(start) : b;
}

}  // namespace

PyValue parse_json(const std::string& raw) {
    const std::string text = decode_body(raw);
    Parser p{text};
    PyValue v = p.value();
    p.ws();
    if (p.i != text.size()) p.fail("Extra data");
    return v;
}

// ---------------------------------------------------------------------------------------------
// Writers
// ---------------------------------------------------------------------------------------------
std::string float_repr(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::scientific);
    std::string sci(buf, r.ptr);
    bool neg = sci[0] == '-';
    if (neg) sci.erase(0, 1);
    size_t e = sci.find('e');
    std::string mant = sci.substr(0, e);
    int exp10 = std::stoi(sci.substr(e + 1));
    std::string digits;
    for (char c : mant) if (c != '.') digits += c;
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int nd = static_cast<int>(digits.size());
    const int decpt = exp10 + 1;
    std::string out = neg ? "-" : "";
    if (decpt <= -4 || decpt > 16) {
        out += digits[0];
        if (nd > 1) out += "." + digits.substr(1);
        int x = decpt - 1;
        char eb[16];
        std::snprintf(eb, sizeof(eb), "e%c%02d", x < 0 ? '-' : '+', x < 0 ? -x : x);
        out += eb;
    } else if (decpt <= 0) {
        out += "0." + std::string(-decpt, '0') + digits;
    } else if (decpt >= nd) {
        out += digits + std::string(decpt - nd, '0') + ".0";
    } else {
        out += digits.substr(0, decpt) + "." + digits.substr(decpt);
    }
    return out;
}

namespace {

void dump_string(std::string& out, const std::string& s) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x", c);
                    out += b;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    out += '"';
}

void dump(std::string& out, const PyValue& v, bool compact, bool sort_keys) {
    const char* item_sep = compact ? "," : ", ";
    const char* key_sep = compact ? ":" : ": ";
    switch (v.kind) {
        case PyValue::Kind::Null: out += "null"; break;
        case PyValue::Kind::Bool: out += v.b ? "true" : "false"; break;
        case PyValue::Kind::Int: out += v.text; break;
        case PyValue::Kind::Float:
            if (!std::isfinite(v.f)) throw JsonError("Out of range float values are not JSON compliant: " + float_repr(v.f));
            out += float_repr(v.f);
            break;
        case PyValue::Kind::String: dump_string(out, v.text); break;
        case PyValue::Kind::Array:
            out += '[';
            for (size_t k = 0; k < v.items.size(); k++) {
                if (k) out += item_sep;
                dump(out, v.items[k], compact, sort_keys);
            }
            out += ']';
            break;
        case PyValue::Kind::Object: {
            std::vector<const std::pair<std::string, PyValue>*> ms;
            for (const auto& m : v.members) ms.push_back(&m);
            // Python sorts str keys by code point, which is UTF-8 byte order
            if (sort_keys) std::stable_sort(ms.begin(), ms.end(), [](auto* a, auto* b) { return a->first < b->first; });
            out += '{';
            for (size_t k = 0; k < ms.size(); k++) {
                if (k) out += item_sep;
                dump_string(out, ms[k]->first);
                out += key_sep;
                dump(out, ms[k]->second, compact, sort_keys);
            }
            out += '}';
            break;
        }
    }
}

uint32_t next_cp(const std::string& s, size_t& i) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    int n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
    uint32_t cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < n && i + k < s.size(); k++) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    i += n;
    return cp;
}

/// str.isprintable() for the code points a request can plausibly carry: controls, format
/// characters, separators other than ' ', private use and noncharacters are not printable.
/// (Unassigned code points are treated as printable; CPython's table says otherwise.)
bool printable(uint32_t cp) {
    if (cp < 0x20 || (cp >= 0x7F && cp <= 0xA0) || cp == 0xAD) return false;
    if ((cp >= 0x600 && cp <= 0x605) || cp == 0x61C || cp == 0x6DD || cp == 0x70F || cp == 0x180E) return false;
    if (cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200F) || (cp >= 0x2028 && cp <= 0x202F)) return false;
    if ((cp >= 0x205F && cp <= 0x2064) || (cp >= 0x2066 && cp <= 0x206F) || cp == 0x3000) return false;
    if ((cp >= 0xD800 && cp <= 0xF8FF) || cp == 0xFEFF || (cp >= 0xFFF9 && cp <= 0xFFFB)) return false;
    if ((cp & 0xFFFE) == 0xFFFE || (cp >= 0xFDD0 && cp <= 0xFDEF)) return false;
    if (cp >= 0xF0000 || cp == 0x110BD || cp == 0x110CD || (cp >= 0x1BCA0 && cp <= 0x1BCA3)) return false;
    if ((cp >= 0x1D173 && cp <= 0x1D17A) || cp == 0xE0001 || (cp >= 0xE0020 && cp <= 0xE007F)) return false;
    return true;
}

std::string repr_string(const std::string& s) {
    char quote = '\'';
    if (s.find('\'') != std::string::npos && s.find('"') == std::string::npos) quote = '"';
    std::string out(1, quote);
    size_t i = 0;
    while (i < s.size()) {
        size_t start = i;
        uint32_t cp = next_cp(s, i);
        if (cp == static_cast<uint32_t>(quote) || cp == '\\') { out += '\\'; out += static_cast<char>(cp); }
        else if (cp == '\t') out += "\\t";
        else if (cp == '\n') out += "\\n";
        else if (cp == '\r') out += "\\r";
        else if (cp < 0x7F && cp >= 0x20) out += static_cast<char>(cp);
        else if (printable(cp)) out.append(s, start, i - start);
        else {
            char b[16];
            if (cp <= 0xFF) std::snprintf(b, sizeof(b), "\\x%02x", cp);
            else if (cp <= 0xFFFF) std::snprintf(b, sizeof(b), "\\u%04x", cp);
            else std::snprintf(b, sizeof(b), "\\U%08x", cp);
            out += b;
        }
    }
    out += quote;
    return out;
}

}  // namespace

std::string dumps(const PyValue& v, bool compact, bool sort_keys) {
    std::string out;
    dump(out, v, compact, sort_keys);
    return out;
}

std::string repr(const PyValue& v) {
    switch (v.kind) {
        case PyValue::Kind::Null: return "None";
        case PyValue::Kind::Bool: return v.b ? "True" : "False";
        case PyValue::Kind::Int: return v.text;
        case PyValue::Kind::Float: return float_repr(v.f);
        case PyValue::Kind::String: return repr_string(v.text);
        case PyValue::Kind::Array: {
            std::string out = "[";
            for (size_t k = 0; k < v.items.size(); k++) {
                if (k) out += ", ";
                out += repr(v.items[k]);
            }
            return out + "]";
        }
        case PyValue::Kind::Object: {
            std::string out = "{";
            for (size_t k = 0; k < v.members.size(); k++) {
                if (k) out += ", ";
                out += repr_string(v.members[k].first) + ": " + repr(v.members[k].second);
            }
            return out + "}";
        }
    }
    return "";
}

void check_json_value(const PyValue& v) {
    switch (v.kind) {
        case PyValue::Kind::Float:
            if (!std::isfinite(v.f)) throw JsonError("Out of range float values are not JSON compliant: " + float_repr(v.f));
            break;
        case PyValue::Kind::Array:
            for (const auto& x : v.items) check_json_value(x);
            break;
        case PyValue::Kind::Object:
            for (const auto& m : v.members) check_json_value(m.second);
            break;
        default: break;
    }
}

}  // namespace flm::d3
