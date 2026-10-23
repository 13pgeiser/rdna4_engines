/* uocr_resample_golden.h -- loader for tests/data/golden/uocr_resample_golden.bin, the golden
 * data resample_golden.py generates (torch CPU, seed 20261019) from F.interpolate(bicubic,
 * antialias=True) and the Unlimited-OCR checkpoint. The data is a checked-in little-endian blob
 * read at test time, not C arrays: 40 MB of literals would add ~10 s of compile per TU.
 *
 * Blob layout (little-endian, no padding):
 *   u32 magic 'UOCR', u32 version=1, u32 n_cases, u32 pad
 *   n_cases * { i32 ih, iw, oh, ow; i64 in_n, out_n }     (32 B each)
 *   per case: f32 in[in_n], f32 out[out_n]
 *   u16 sam_src[64*64*768], sam_out[40*40*768], clip_src[257*1024], clip_out[101*1024]
 *
 * The bf16 payloads are stored as u16 bit patterns. */
#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "rad_test.h"

namespace uocr_golden {

struct Golden {
    int n_cases = 0;
    std::vector<std::array<int, 4>> geo;  /* in_h, in_w, out_h, out_w; C=3 */
    std::vector<std::vector<float>> in, out;
    std::vector<uint16_t> sam_src, sam_out, clip_src, clip_out;
};

/* Reads the blob from RAD_GOLDEN_FIXTURES (tests/CMakeLists.txt). Returns false, reporting in
 * `err`, on a missing, truncated, or mismatched blob; the caller skips via ARTIFACT_MISSING. */
inline bool load(Golden* g, std::string* err) {
#ifdef RAD_GOLDEN_FIXTURES
    const std::string path = std::string(RAD_GOLDEN_FIXTURES) + "/uocr_resample_golden.bin";
#else
    const std::string path = "uocr_resample_golden.bin";
#endif
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        *err = "cannot open " + path;
        return false;
    }
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const auto u32 = [&raw](size_t o) {
        uint32_t v;
        std::memcpy(&v, raw.data() + o, 4);
        return v;
    };
    const auto i64 = [&raw](size_t o) {
        int64_t v;
        std::memcpy(&v, raw.data() + o, 8);
        return v;
    };
    const auto f32 = [&raw](size_t o) {
        float v;
        std::memcpy(&v, raw.data() + o, 4);
        return v;
    };
    const auto u16 = [&raw](size_t o) {
        uint16_t v;
        std::memcpy(&v, raw.data() + o, 2);
        return v;
    };
    if (raw.size() < 16 || u32(0) != 0x52434F55u || u32(4) != 1u) {
        *err = path + ": not a uocr golden blob (magic/version)";
        return false;
    }
    const uint32_t n = u32(8);
    size_t off = 16;
    if (raw.size() < off + (size_t)n * 32) {
        *err = path + ": truncated case table";
        return false;
    }
    g->n_cases = (int)n;
    g->geo.assign(n, {});
    g->in.assign(n, {});
    g->out.assign(n, {});
    std::vector<std::pair<int64_t, int64_t>> ns(n);
    for (uint32_t k = 0; k < n; ++k) {
        g->geo[k] = { (int)(int32_t)u32(off), (int)(int32_t)u32(off + 4),
                      (int)(int32_t)u32(off + 8), (int)(int32_t)u32(off + 12) };
        ns[k] = { i64(off + 16), i64(off + 24) };
        off += 32;
    }
    for (uint32_t k = 0; k < n; ++k) {
        for (int which = 0; which < 2; ++which) {
            const int64_t cnt = which ? ns[k].second : ns[k].first;
            std::vector<float>& dst = which ? g->out[k] : g->in[k];
            if (cnt < 0 || raw.size() < off + (size_t)cnt * 4) {
                *err = path + ": truncated fp32 payload";
                return false;
            }
            dst.resize((size_t)cnt);
            for (int64_t i = 0; i < cnt; ++i) dst[(size_t)i] = f32(off + (size_t)i * 4);
            off += (size_t)cnt * 4;
        }
    }
    const size_t need = (size_t)(64 * 64 * 768 + 40 * 40 * 768 + 257 * 1024 + 101 * 1024) * 2;
    if (raw.size() != off + need) {
        *err = path + ": size mismatch (expected " + std::to_string(off + need) +
               " B, got " + std::to_string(raw.size()) + " B)";
        return false;
    }
    size_t p = 0;
    for (std::vector<uint16_t>* v : { &g->sam_src, &g->sam_out, &g->clip_src, &g->clip_out }) {
        const size_t cnt = v == &g->sam_src    ? (size_t)64 * 64 * 768
                           : v == &g->sam_out  ? (size_t)40 * 40 * 768
                           : v == &g->clip_src ? (size_t)257 * 1024
                                               : (size_t)101 * 1024;
        v->resize(cnt);
        for (size_t i = 0; i < cnt; ++i) (*v)[i] = u16(off + (p + i) * 2);
        p += cnt;
    }
    return true;
}

}  // namespace uocr_golden
