/// \file py_json.hpp
/// \brief JSON values with Python's json module semantics, for the d3 prompt contract.
/// \note  The d3 prompt shows non-string states, instructions and descriptions as Python's
///        `json.dumps(v, ensure_ascii=False)` text, so the bytes the model reads depend on details
///        nlohmann::json does not keep: integers of any size, the shortest-repr float format
///        (`1e-05`, `1e+16`, `100000.0`), `", "` / `": "` separators, and the key order and
///        duplicate-key rule of `json.loads` (first position, last value). PyValue keeps exactly
///        what `json.loads` would build, and the writers below reproduce `json.dumps` and `repr`.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flm::d3 {

class PyValue {
public:
    enum class Kind { Null, Bool, Int, Float, String, Array, Object };

    Kind kind = Kind::Null;
    bool b = false;
    std::string text;   //!< Int: canonical decimal digits with an optional '-'; String: UTF-8
    double f = 0.0;
    std::vector<PyValue> items;                          //!< Array
    std::vector<std::pair<std::string, PyValue>> members; //!< Object, in json.loads key order

    static PyValue null() { return PyValue(); }
    static PyValue boolean(bool v) { PyValue p; p.kind = Kind::Bool; p.b = v; return p; }
    static PyValue integer(std::string digits);
    static PyValue number(double v) { PyValue p; p.kind = Kind::Float; p.f = v; return p; }
    static PyValue string(std::string v) { PyValue p; p.kind = Kind::String; p.text = std::move(v); return p; }
    static PyValue array() { PyValue p; p.kind = Kind::Array; return p; }
    static PyValue object() { PyValue p; p.kind = Kind::Object; return p; }

    bool is_null() const { return kind == Kind::Null; }
    bool is_string() const { return kind == Kind::String; }
    bool is_object() const { return kind == Kind::Object; }
    bool is_array() const { return kind == Kind::Array; }

    /// Object member lookup (nullptr when absent); `dict.get(key)`.
    const PyValue* get(const std::string& key) const;
    /// Insert or overwrite, keeping the first position (`d[key] = value`).
    void set(const std::string& key, PyValue value);

    /// Python truthiness: None, False, 0, 0.0, "", [] and {} are false.
    bool truthy() const;
};

struct JsonError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// `json.loads(text, parse_constant=<reject>)`: strict JSON (no raw control characters in strings),
/// NaN / Infinity literals rejected, numbers with a fraction or exponent become floats (and may
/// overflow to +-inf, as in Python), integers keep every digit. Throws JsonError on bad input,
/// including invalid UTF-8 and lone surrogate escapes (which Python would carry along and then
/// fail to encode).
PyValue parse_json(const std::string& text);

/// `json.dumps(v, ensure_ascii=False)` with the default `", "` / `": "` separators, or compact
/// `(",", ":")` separators and sorted keys (score legends). Throws JsonError for a non-finite float
/// (`allow_nan=False`) with Python's message.
std::string dumps(const PyValue& v, bool compact = false, bool sort_keys = false);

/// `repr(v)` of the Python object `json.loads` builds (used in error messages: `'rank'`, `None`).
std::string repr(const PyValue& v);

/// `repr(float)`: the shortest round-trip digits in Python's layout.
std::string float_repr(double v);

/// `_json_value` of d3_runtime: raises (JsonError) when `json.dumps(..., allow_nan=False)` would.
void check_json_value(const PyValue& v);

}  // namespace flm::d3
