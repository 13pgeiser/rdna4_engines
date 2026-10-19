/* tests/r4d_emu: kda_update (r4d_kda.hip) on the host SIMT emulator. Every row and prefetch form
 * must give the old one-row kernel's output and whole state buffer bit for bit, in all four modes:
 * prefill per candidate and anchored, verify per candidate and anchored (after the replay of the
 * accepted prefix). Cold starts, an unseated sequence, an empty sequence and an unwritten slot
 * are mixed in. The GPU check of the same claim is r4d_selftest's kda_forms. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        t.shape[i] = shape[i];
        t.stride[i] = st;
        st *= shape[i];
    }
    return t;
}
static uint16_t f2b(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

struct Case {
    const char* name;
    const char* mode;
    bool anchor;
    int64_t width;
    std::vector<int> cu, idx, accepted, init;
    std::vector<std::pair<int64_t, float>> counts;
};

int main() {
    std::mt19937 rng(41);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    const int64_t H = 2, D = 128, slots = 16, tokens = 9;
    std::vector<float> state0((size_t)(slots * H * D * D)), alog((size_t)H), bias((size_t)(H * D));
    for (auto& v : state0) v = u(rng);
    for (auto& v : alog) v = 0.5f * u(rng);
    for (auto& v : bias) v = u(rng);
    std::vector<uint16_t> in[5];
    for (int i = 0; i < 5; ++i) {
        in[i].resize((size_t)(tokens * H * (i == 4 ? 1 : D)));
        for (auto& v : in[i]) v = f2b((i >= 3 ? 2.f : 1.f) * u(rng));
    }
    const std::vector<Case> cases = {
        { "prefill, per candidate",
          "prefill",
          false,
          1,
          { 0, 4, 4, 9 },
          { 3, 4, -1 },
          { 1, 1, 1 },
          { 0, 1, 1 },
          {} },
        { "prefill, anchored",
          "prefill",
          true,
          2,
          { 0, 5, 9 },
          { 1, 2, 3, 4 },
          { 1, 1 },
          { 1, 0 },
          {} },
        { "verify, per candidate",
          "verify",
          false,
          4,
          { 0, 4, 7, 9 },
          { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1 },
          { 3, 2, 1 },
          { 1, 0, 1 },
          {} },
        { "verify, anchored",
          "verify",
          true,
          4,
          { 0, 4, 7, 9 },
          { 1, 9, 5, 9, 2, 9, 6, 9, 3, 9, 7, 9 },
          { 3, 2, 1 },
          { 1, 1, 0 },
          { { 5, 4.f }, { 6, 3.f }, { 7, 0.f } } },
    };
    struct Form {
        const char* rows;
        const char* prefetch;
    };
    const Form forms[] = { { "1", "1" }, { "2", "1" }, { "4", "1" },
                           { "1", "0" }, { "2", "0" }, { "4", "0" } };
    int fails = 0, checks = 0;
    for (const Case& c : cases) {
        std::vector<float> start = state0;
        for (const auto& sc : c.counts)
            for (int64_t h = 0; h < H; ++h) start[(size_t)((sc.first * H + h) * D * D)] = sc.second;
        const int64_t seqs = (int64_t)c.cu.size() - 1;
        std::vector<int> idx = c.idx, cu = c.cu, acc = c.accepted, init = c.init;
        auto run = [&](std::vector<uint16_t>& out, std::vector<float>& state) {
            out.assign((size_t)(tokens * H * D), 0x7777);
            state = start;
            RadTensor t[13] = { tens(in[0].data(), RAD_BF16, { tokens, H, D }),
                                tens(in[1].data(), RAD_BF16, { tokens, H, D }),
                                tens(in[2].data(), RAD_BF16, { tokens, H, D }),
                                tens(in[3].data(), RAD_BF16, { tokens, H, D }),
                                tens(in[4].data(), RAD_BF16, { tokens, H }),
                                tens(alog.data(), RAD_F32, { H }),
                                tens(bias.data(), RAD_F32, { H * D }),
                                tens(state.data(), RAD_F32, { slots, H, D, D }),
                                tens(idx.data(), RAD_I32, { seqs, c.width }),
                                tens(cu.data(), RAD_I32, { seqs + 1 }),
                                tens(acc.data(), RAD_I32, { seqs }),
                                tens(init.data(), RAD_I32, { seqs }),
                                tens(out.data(), RAD_BF16, { tokens, H, D }) };
            RadParam p[10]{};
            p[0] = { "heads", RAD_P_INT, H };
            p[1] = { "head_k", RAD_P_INT, D };
            p[2] = { "head_v", RAD_P_INT, D };
            p[3].key = "lower_bound";
            p[3].kind = RAD_P_F64;
            p[3].dval = -5;
            p[4].key = "scale";
            p[4].kind = RAD_P_F64;
            p[4].dval = 1.0 / std::sqrt(128.0);
            p[5].key = "l2_eps";
            p[5].kind = RAD_P_F64;
            p[5].dval = 1e-6;
            p[6].key = "mode";
            p[6].kind = RAD_P_STR;
            p[6].sval = c.mode;
            p[7].key = "state_form";
            p[7].kind = RAD_P_STR;
            p[7].sval = c.anchor ? "anchor" : "per_candidate";
            p[8].key = "dtype";
            p[8].kind = RAD_P_STR;
            p[8].sval = "bf16";
            RadArgs a{};
            a.t = t;
            a.n_t = 13;
            a.p = p;
            a.n_p = 9;
            return r4d_rad_kda_update(&a, nullptr);
        };
        std::vector<uint16_t> ref_out, out;
        std::vector<float> ref_state, state;
        setenv("RADIANCE_KDA_FORM", "old", 1);
        if (run(ref_out, ref_state) != RAD_OK) {
            std::printf("FAIL %s: old form refused\n", c.name);
            ++fails;
            continue;
        }
        unsetenv("RADIANCE_KDA_FORM");
        bool written = false;
        for (size_t i = 0; i < ref_out.size(); ++i) written = written || ref_out[i] != 0x7777;
        if (!written) {
            std::printf("FAIL %s: the old form wrote no output\n", c.name);
            ++fails;
        }
        for (const Form& f : forms) {
            setenv("RADIANCE_KDA_ROWS", f.rows, 1);
            setenv("RADIANCE_KDA_PREFETCH", f.prefetch, 1);
            ++checks;
            const bool same = run(out, state) == RAD_OK && out == ref_out &&
                              std::memcmp(state.data(), ref_state.data(), state.size() * 4) == 0;
            if (!same) {
                ++fails;
                std::printf("FAIL %s, %s rows a wave, prefetch %s: differs from the old form\n",
                            c.name, f.rows, f.prefetch);
            }
        }
        unsetenv("RADIANCE_KDA_ROWS");
        unsetenv("RADIANCE_KDA_PREFETCH");
    }
    std::printf("kda_update forms against the old kernel: %d of %d bit-identical\n", checks - fails,
                checks);
    return fails ? 1 : 0;
}
