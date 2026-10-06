/* core/server/json_partial.h -- what may be sent, and what may be called a tool call.
 *
 * Two questions about text that is still arriving, and they are different questions:
 *
 *   1. "Is this a tool call?"  -- answered by `json_complete`. Nothing is ever reported to a
 *      client as `finish_reason: "tool_calls"` unless the arguments parse whole. A malformed call
 *      is returned as ordinary content, because a client that gets a truncated arguments object
 *      and is told it is a tool call will call the tool with the wrong arguments.
 *
 *   2. "How much of it can I put on the wire right now?" -- answered by `json_stable_prefix`.
 *      A streamed fragment is itself JSON-encoded into the SSE payload, so it must end on a UTF-8
 *      boundary: cutting a multi-byte code point in half turns one delta into mojibake and the
 *      next into a decode error.
 *
 * And one way to write it: `json_quote_append`, a string literal exactly as nlohmann writes one,
 * for the chunks that carry a delta.
 *
 * This is a scanner, not a parser: it allocates nothing and builds no DOM.
 */
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace rad {
namespace server {

/* Does `s` parse as exactly one complete JSON value, with only whitespace after it? */
bool json_complete(std::string_view s);

/* Largest n <= s.size() such that s[0,n) ends on a UTF-8 code point boundary. Bytes that are not
 * valid UTF-8 at all are left alone -- this backs out of a truncation, it does not validate. */
size_t utf8_trunc(std::string_view s);

/* Largest prefix of `s` that is safe to emit as one streaming fragment: a UTF-8 boundary, and not
 * in the middle of a "\uXXXX" escape or on a lone trailing backslash. The escape rule buys
 * nothing for a client that only concatenates, and costs four bytes of latency; it is here
 * because a client that pretty-prints each delta is a client that would otherwise print a broken
 * escape, and the class of bug that produces is unbisectable from the server side. */
size_t json_stable_prefix(std::string_view s);

/* Append `s` to `out` as a JSON string literal, quotes included, byte for byte as nlohmann's
 * dump() writes one: `\b \t \n \f \r \" \\` as two characters, every other control character
 * as `\u00xx` in lowercase hex, and everything else -- DEL and non-ASCII included -- as itself.
 * Returns false, with `out` as it was, when `s` is not well-formed UTF-8 (overlong forms,
 * surrogates and code points past U+10FFFF included), which nlohmann refuses rather than writes.
 *
 * A streamed chunk is written with this around a fixed frame instead of through a DOM: building
 * and dumping a tree per token cost more than every other part of the token's host work. */
bool json_quote_append(std::string& out, std::string_view s);

}  /* namespace server */
}  /* namespace rad */
