/// \file wstream_buf.hpp
/// \brief wstream_buf class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// \note This class is used to write UTF-8 characters to a stream
#pragma once

#include <iostream>
#include <string>
#include <vector>

/// The number of bytes at the end of text that begin a UTF-8 character it does not finish.
/// \note Tokens are bytes, so a generation cut off by the length limit or a cancel
///       can stop partway through a character. Those bytes are invalid UTF-8, and
///       json::dump() throws on them.
inline size_t incomplete_utf8_tail(const std::string& text) {
    for (size_t back = 1; back <= 4 && back <= text.size(); back++) {
        unsigned char c = text[text.size() - back];
        if ((c & 0xC0) == 0x80) {
            continue;  // a continuation byte
        }
        size_t width = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return width > back ? back : 0;
    }
    return 0;
}

/// \brief wstream_buf class
/// \note This class is used to avoid partial UTF-8 characters from the decoder.
struct wstream_buf : std::streambuf {
    std::vector<char> utf8_buffer;  // Buffer for incomplete UTF-8 sequences
    std::ostream& os;
    wstream_buf(std::ostream& os) : os(os) {}

    /// \brief called when a sequence of chars is written
    /// \param s the string to write
    /// \param n the number of characters to write
    /// \return the number of characters written
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        // Append new data to buffer
        utf8_buffer.insert(utf8_buffer.end(), s, s + n);
        
        // Process complete UTF-8 sequences
        size_t pos = 0;
        while (pos < utf8_buffer.size()) {
            unsigned char first = utf8_buffer[pos];
            size_t seq_len = 1;
            
            // Determine sequence length
            if ((first & 0x80) == 0) {
                // Single byte sequence
                seq_len = 1;
            } else if ((first & 0xE0) == 0xC0) {
                // Two byte sequence
                seq_len = 2;
            } else if ((first & 0xF0) == 0xE0) {
                // Three byte sequence
                seq_len = 3;
            } else if ((first & 0xF8) == 0xF0) {
                // Four byte sequence
                seq_len = 4;
            } else {
                // Invalid UTF-8 start byte
                pos++;
                continue;
            }
            
            // Check if we have a complete sequence
            if (pos + seq_len > utf8_buffer.size()) {
                // Incomplete sequence, keep it in buffer
                break;
            }
            os.write(&utf8_buffer[pos], seq_len);
            pos += seq_len;
        }
     
        // Remove processed bytes from buffer
        if (pos > 0) {
            utf8_buffer.erase(utf8_buffer.begin(), utf8_buffer.begin() + pos);
        }
        
        return n;
    }

    /// \brief fallback for single-char writes
    /// \param ch the character to write
    /// \return the character written
    int_type overflow(int_type ch) override {
        if (ch == traits_type::eof()) return ch;
        char c = traits_type::to_char_type(ch);
        
        // Add single byte to buffer and process it
        utf8_buffer.push_back(c);
        xsputn(&c, 1);
        return ch;
    }
};

class nullbuf : public std::streambuf {
    protected:
        int overflow(int c) override { return c; }  // Simply ignore input
    };
    
    class nullstream : public std::ostream {
    public:
        nullstream() : std::ostream(&m_sb) {}
    private:
        nullbuf m_sb;
};