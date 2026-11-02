/// \file transcript_format.hpp
/// \brief Turns Whisper's timestamped output into OpenAI's transcription formats
/// \author FastFlowLM Team
/// \note verbose_json segments, srt and vtt, for /v1/audio/transcriptions
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "wstream_buf.hpp"

namespace transcript {

struct Segment {
    double start;
    double end;
    std::string text;
};

/// The first "<|seconds|>" at or after pos, or npos. Sets its length and value.
inline size_t find_time_stamp(const std::string& output, size_t pos, size_t& length, double& seconds) {
    for (size_t at = output.find("<|", pos); at != std::string::npos; at = output.find("<|", at + 1)) {
        size_t close = output.find("|>", at + 2);
        if (close == std::string::npos) {
            break;
        }
        const char* begin = output.c_str() + at + 2;
        char* end = nullptr;
        seconds = std::strtod(begin, &end);
        if (end != begin && end == output.c_str() + close) {
            length = close + 2 - at;
            return at;
        }
    }
    return std::string::npos;
}

/// Splits Whisper::generate's output with return_time_stamp, as in
/// "<|0.00|> Hello.<|1.50|><|1.50|> Bye.<|3.00|>", into segments. Text starts
/// at the timestamp before it and ends at the one after it; text with no
/// timestamp after it, which a window cut off, ends with the clip.
inline std::vector<Segment> split_segments(const std::string& output, double duration) {
    std::vector<Segment> segments;
    double time = 0.0;
    size_t pos = 0;
    std::string carry;  // tokens are bytes: a timestamp can split a character
    while (true) {
        size_t length = 0;
        double seconds = 0.0;
        size_t at = find_time_stamp(output, pos, length, seconds);
        std::string text = carry + output.substr(pos, at == std::string::npos ? std::string::npos : at - pos);
        carry.clear();
        if (at != std::string::npos) {
            size_t tail = incomplete_utf8_tail(text);
            // Only spaces before the split character: they move with it.
            size_t keep = tail > 0 && text.find_first_not_of(" \t\r\n") >= text.size() - tail ? 0 : text.size() - tail;
            carry = text.substr(keep);
            text.resize(keep);
        }
        if (text.find_first_not_of(" \t\r\n") != std::string::npos) {
            segments.push_back({time, at == std::string::npos ? duration : seconds, text});
        }
        if (at == std::string::npos) {
            break;
        }
        time = seconds;
        pos = at + length;
    }
    // The last window is padded to 30 s, so its timestamps can pass the end of the clip.
    for (Segment& segment : segments) {
        segment.start = std::clamp(segment.start, 0.0, duration);
        segment.end = std::clamp(segment.end, segment.start, duration);
    }
    return segments;
}

inline std::string trim(const std::string& text) {
    size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    return text.substr(begin, text.find_last_not_of(" \t\r\n") - begin + 1);
}

/// HH:MM:SS followed by the separator and milliseconds: ',' for srt, '.' for vtt.
inline std::string clock(double seconds, char separator) {
    long long ms = std::llround(seconds * 1000);
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%02lld:%02lld:%02lld%c%03lld",
                  ms / 3600000, ms / 60000 % 60, ms / 1000 % 60, separator, ms % 1000);
    return buffer;
}

inline std::string to_srt(const std::vector<Segment>& segments) {
    std::string out;
    for (size_t i = 0; i < segments.size(); i++) {
        out += std::to_string(i + 1) + "\n"
             + clock(segments[i].start, ',') + " --> " + clock(segments[i].end, ',') + "\n"
             + trim(segments[i].text) + "\n\n";
    }
    return out;
}

inline std::string to_vtt(const std::vector<Segment>& segments) {
    std::string out = "WEBVTT\n\n";
    for (const Segment& segment : segments) {
        out += clock(segment.start, '.') + " --> " + clock(segment.end, '.') + "\n"
             + trim(segment.text) + "\n\n";
    }
    return out;
}

}  // namespace transcript
