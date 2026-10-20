/* Drives libr4d's MLA matrix row (r4d_mla.hip, attend_matrix) through the SIMT emulator and checks
 * every output element against an f64 softmax over each row's own table. The cases put the paired
 * tile (eight heads of two neighbouring rows) on the tables it meets in serving: a prefill chunk
 * whose rows share one table, and rows past the DSA budget, where each row selects its own top
 * blocks and neighbours share most blocks but rarely a table position. The conflict counters
 * (RADIANCE_MLA_CONFLICT_STATS) show which of the two the pairing saw. Both pairings run on the
 * same operands: the union at the case's table width, the positional pairing on the table
 * widened past the union's lists (kUnionBlocks, 520 entries) with absent entries no row reaches. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" void r4d_mla_matrix_pin_form(int form);
extern "C" void r4d_mla_matrix_stats(unsigned long long* out);

namespace {

constexpr int D = 512, BS = 4, kUnionBlocks = 520;

enum class Fill { prefill, topk, ragged, repeat };

struct Case {
    std::string name;
    int rows, heads, blocks;
    Fill fill;
    int swaps = 0;            /* topk: blocks exchanged between neighbouring rows */
    int context = 0;          /* topk: tokens of the sequence the rows end */
    bool scratch = false;     /* lets the launcher split the history */
    int form = 0;             /* r4d_mla_matrix_pin_form */
    int expect_second = -1;   /* positional pairing: 0 no second pass, 1 some, -1 either */
    /* Unsplit only: with scratch the widened table gives the positional run more splits. */
    bool bitwise_a = true; /* the union's request r equals the positional pairing's bit for bit */
    bool bitwise_all = false; /* both requests do (the tables agree position by position) */
    bool union_falls_back = false; /* the table is wider than the union's lists */
};

/* Tables as qsa_select emits them past its budget: for the queries at positions context-rows ..
 * context-1 of one sequence, each row the top blocks-1 of its complete blocks, the strictly higher
 * ones in block order and a short tie run after them, then the partial page. Row r's choice is row
 * r-1's with `swaps` members exchanged for non-members. Pages are a fixed shuffle of the blocks. */
void topk_tables(int rows, int blocks, int context, int swaps, std::mt19937& rng,
                 std::vector<int>& bt, std::vector<int>& len, int& pages) {
    const int topk = blocks - 1, universe = context / BS + 1, ties = std::min(4, topk);
    pages = universe;
    std::vector<int> page(universe);
    for (int i = 0; i < universe; ++i) page[i] = i;
    std::shuffle(page.begin(), page.end(), rng);
    std::vector<char> in(universe, 0);
    std::vector<int> members;
    auto tie_rank = [](int b) { return (uint32_t)b * 2654435761u ^ 0x9e3779b9u; };
    bt.assign((size_t)rows * blocks, -1);
    len.assign(rows, 0);
    for (int r = 0; r < rows; ++r) {
        const int p = context - rows + r, n = (p + 1) / BS;
        if (n <= topk) {
            for (int b = 0; b < n; ++b)
                if (!in[b]) {
                    in[b] = 1;
                    members.push_back(b);
                }
        } else {
            while ((int)members.size() < topk) {
                const int b = (int)(rng() % (unsigned)n);
                if (!in[b]) {
                    in[b] = 1;
                    members.push_back(b);
                }
            }
            for (int s = 0; s < swaps; ++s) {
                int add;
                do add = (int)(rng() % (unsigned)n);
                while (in[add]);
                const size_t drop = rng() % members.size();
                in[members[drop]] = 0;
                members[drop] = add;
                in[add] = 1;
            }
        }
        std::vector<int> order(members);
        std::sort(order.begin(), order.end());
        if (n > topk) {
            std::vector<int> tied(order);
            std::sort(tied.begin(), tied.end(),
                      [&](int a, int b) { return tie_rank(a) < tie_rank(b); });
            tied.resize(ties);
            std::sort(tied.begin(), tied.end());
            order.erase(std::remove_if(
                            order.begin(), order.end(),
                            [&](int b) { return std::binary_search(tied.begin(), tied.end(), b); }),
                        order.end());
            order.insert(order.end(), tied.begin(), tied.end());
        }
        int out = 0;
        for (int b : order) bt[(size_t)r * blocks + out++] = page[b];
        if ((p + 1) % BS) bt[(size_t)r * blocks + out++] = page[n];
        len[r] = out ? (out - 1) * BS + p % BS + 1 : 0;
    }
}

int run(const Case& c, std::mt19937& rng) {
    const int M = c.rows, H = c.heads, B = c.blocks;
    std::vector<int> bt, len;
    int pages = 0;
    if (c.fill == Fill::topk) {
        topk_tables(M, B, c.context, c.swaps, rng, bt, len, pages);
    } else if (c.fill == Fill::prefill) {
        /* one sequence, its rows the last M positions of the capacity, one table */
        pages = B + 1;
        bt.assign((size_t)M * B, -1);
        len.assign(M, 0);
        for (int r = 0; r < M; ++r) {
            for (int b = 0; b < B; ++b) bt[(size_t)r * B + b] = (b * 7 + 3) % pages;
            len[r] = B * BS - M + r + 1;
        }
    } else if (c.fill == Fill::repeat) {
        /* every page twice in a row's table and shifted by one entry from row to row, so a page
         * repeats within a table and between the pair */
        pages = M + B;
        bt.assign((size_t)M * B, -1);
        len.assign(M, 0);
        for (int r = 0; r < M; ++r) {
            for (int b = 0; b < B; ++b) bt[(size_t)r * B + b] = ((b + r) / 2 * 5) % pages;
            len[r] = B * BS - r % 3;
        }
    } else {
        /* every row its own scattered table, with holes and out-of-range pages */
        pages = M * B;
        bt.assign((size_t)M * B, -1);
        len.assign(M, 0);
        for (int r = 0; r < M; ++r) {
            for (int b = 0; b < B; ++b)
                bt[(size_t)r * B + b] = b % 17 == 3   ? -1
                                        : b % 19 == 5 ? pages + 3
                                                      : (b * 7 + r * 11) % pages;
            len[r] = r % 5 == 4 ? 0 : B * BS - 1 - r % 7;
        }
    }
    std::normal_distribution<float> nrm(0.f, 1.f);
    std::vector<uint16_t> qb((size_t)M * H * D), kvb((size_t)pages * BS * D);
    for (auto& v : qb) v = rad_f32_to_bf16(0.25f * nrm(rng));
    for (auto& v : kvb) v = rad_f32_to_bf16(nrm(rng));
    std::vector<uint16_t> out((size_t)M * H * D, 0xffff);

    auto tens = [](void* p, uint32_t dt, std::vector<int64_t> shape) {
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
    };
    RadTensor t[5] = { tens(qb.data(), RAD_BF16, { M, H, D }),
                       tens(kvb.data(), RAD_BF16, { pages, 1, BS, D }),
                       tens(bt.data(), RAD_I32, { M, B }), tens(len.data(), RAD_I32, { M }),
                       tens(out.data(), RAD_BF16, { M, H, D }) };
    std::vector<RadParam> p;
    auto pi = [&](const char* k, long long v) {
        RadParam x{};
        x.key = k;
        x.kind = RAD_P_INT;
        x.ival = v;
        p.push_back(x);
    };
    auto pf = [&](const char* k, double v) {
        RadParam x{};
        x.key = k;
        x.kind = RAD_P_F64;
        x.dval = v;
        p.push_back(x);
    };
    pi("M", M);
    pi("latent_dim", D);
    pi("heads", H);
    pi("blocks", B);
    pi("block_size", BS);
    const double scale = 0.0625;
    pf("scale", scale);
    RadArgs a{};
    a.t = t;
    a.n_t = 5;
    a.p = p.data();
    a.n_p = (int)p.size();
    std::vector<char> scratch;
    if (c.scratch) {
        const int64_t need = r4d_rad_mla_attn_matrix_scratch(&a);
        scratch.assign((size_t)need + 16, 0);
        a.scratch = scratch.data() + ((16 - ((uintptr_t)scratch.data() & 15)) & 15);
        a.scratch_bytes = need;
    }

    /* the f64 reference */
    std::vector<double> want((size_t)M * H * D);
    std::vector<double> sc;
    for (int r = 0; r < M; ++r)
        for (int h = 0; h < H; ++h) {
            const int live = std::min(std::max(len[r], 0), B * BS);
            sc.assign(live, -INFINITY);
            double mx = -INFINITY;
            auto row_of = [&](int j) -> const uint16_t* {
                const int pg = bt[(size_t)r * B + j / BS];
                return pg >= 0 && pg < pages ? &kvb[((size_t)pg * BS + j % BS) * D] : nullptr;
            };
            const uint16_t* qr = &qb[((size_t)r * H + h) * D];
            for (int j = 0; j < live; ++j) {
                const uint16_t* k = row_of(j);
                if (!k) continue;
                double dot = 0;
                for (int d = 0; d < D; ++d)
                    dot += (double)rad_bf16_to_f32(qr[d]) * rad_bf16_to_f32(k[d]);
                sc[j] = dot * scale;
                mx = std::max(mx, sc[j]);
            }
            std::vector<double> acc(D, 0.0);
            double sum = 0;
            for (int j = 0; j < live; ++j) {
                if (sc[j] == -INFINITY) continue;
                const double w = std::exp(sc[j] - mx);
                sum += w;
                const uint16_t* k = row_of(j);
                for (int d = 0; d < D; ++d) acc[d] += w * rad_bf16_to_f32(k[d]);
            }
            for (int d = 0; d < D; ++d)
                want[((size_t)r * H + h) * D + d] = sum > 0 ? acc[d] / sum : 0.0;
        }

    /* Both pairings on the same operands: the positional one, on the table widened past the
     * union's lists, and at the case's own width the launcher's choice, the union, which must
     * never take a second pass. A tile of 16 heads has no pairing and runs once. */
    const bool paired = H % 16 == 8;
    const int wide = std::max(B, kUnionBlocks + 1);
    std::vector<int> bt_wide((size_t)M * wide, -1);
    for (int r = 0; r < M; ++r)
        std::copy(bt.begin() + (size_t)r * B, bt.begin() + (size_t)(r + 1) * B,
                  bt_wide.begin() + (size_t)r * wide);
    std::vector<uint16_t> by_pos;
    int fails = 0;
    for (const char* pairing : { "pos", "union" }) {
        const bool by_page = !std::strcmp(pairing, "union");
        if (!paired && by_page) break;
        const bool widened = paired && !by_page;
        t[2] = widened ? tens(bt_wide.data(), RAD_I32, { M, wide })
                       : tens(bt.data(), RAD_I32, { M, B });
        p[3].ival = widened ? wide : B;
        std::fill(out.begin(), out.end(), 0xffff);
        unsigned long long stats[3];
        r4d_mla_matrix_stats(stats);
        r4d_mla_matrix_pin_form(c.form);
        const int rc = r4d_rad_mla_attn_matrix(&a, nullptr);
        r4d_mla_matrix_pin_form(0);
        r4d_mla_matrix_stats(stats);
        if (rc != RAD_OK) {
            std::printf("%-40s %-5s launch returned %d\n", c.name.c_str(), pairing, rc);
            ++fails;
            continue;
        }
        double max_err = 0;
        long bad = 0;
        for (size_t i = 0; i < out.size(); ++i) {
            const float got = rad_bf16_to_f32(out[i]);
            const double e = std::fabs(got - want[i]);
            if (!std::isfinite(got) || e > 1e-5 + 0.008 * std::fabs(want[i])) ++bad;
            if (std::isfinite(got)) max_err = std::max(max_err, e);
        }
        bool ok = bad == 0;
        std::string note;
        if (!by_page) {
            by_pos = out;
            if (c.expect_second >= 0 && (stats[1] > 0) != (c.expect_second == 1)) {
                note +=
                    c.expect_second ? "  (expected a second pass)" : "  (expected no second pass)";
                ok = false;
            }
        } else {
            if ((stats[1] > 0) != c.union_falls_back) {
                note += c.union_falls_back ? "  (expected the positional fallback)"
                                           : "  (the union took a second pass)";
                ok = false;
            }
            /* Request r of each pair walks its own table in its own order, as the positional
             * pairing does, so at one split it must agree to the bit; r+1 sums in another order. */
            size_t first_diff_a = 0, diff_b = 0;
            double max_ab = 0;
            for (int r = 0; r < M; ++r)
                for (size_t i = (size_t)r * H * D; i < (size_t)(r + 1) * H * D; ++i) {
                    if (out[i] == by_pos[i]) continue;
                    if (r % 2 == 0)
                        ++first_diff_a;
                    else
                        ++diff_b;
                    max_ab = std::max(max_ab, (double)std::fabs(rad_bf16_to_f32(out[i]) -
                                                                rad_bf16_to_f32(by_pos[i])));
                }
            char buf[160];
            std::snprintf(buf, sizeof buf, "  vs pos: %zu/%zu differ (r/r+1), max abs %.2e",
                          first_diff_a, diff_b, max_ab);
            note += buf;
            if (c.bitwise_a && !c.scratch && first_diff_a) {
                note += "  (request r differs)";
                ok = false;
            }
            if (c.bitwise_all && !c.scratch && diff_b) {
                note += "  (request r+1 differs)";
                ok = false;
            }
        }
        std::printf("%-40s %-5s M %3d H %2d blocks %3d  passes %5llu raster %5llu second %5llu  "
                    "max_abs %.2e  bad %ld%s  %s\n",
                    c.name.c_str(), pairing, M, H, B, stats[0], stats[2], stats[1], max_err, bad,
                    note.c_str(), ok ? "PASS" : "FAIL");
        fails += !ok;
    }
    return fails;
}

/* The launcher prints the running totals every 1024th counted launch and must leave them
 * standing: 1024 one-chunk launches of a pair, then the hook reads exactly 1024 passes. */
int periodic_report() {
    std::vector<uint16_t> q(2 * 8 * D, 0x3f80), kv(4 * BS * D, 0x3f80), out(2 * 8 * D);
    std::vector<int> bt = { 0, 1 }, len = { 4, 3 };
    RadTensor t[5]{};
    auto set = [](RadTensor& x, void* p, uint32_t dt, std::vector<int64_t> shape) {
        x.data = p;
        x.dtype = dt;
        x.rank = (uint32_t)shape.size();
        int64_t st = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            x.shape[i] = shape[i];
            x.stride[i] = st;
            st *= shape[i];
        }
    };
    set(t[0], q.data(), RAD_BF16, { 2, 8, D });
    set(t[1], kv.data(), RAD_BF16, { 4, 1, BS, D });
    set(t[2], bt.data(), RAD_I32, { 2, 1 });
    set(t[3], len.data(), RAD_I32, { 2 });
    set(t[4], out.data(), RAD_BF16, { 2, 8, D });
    RadParam p[6]{};
    const char* keys[5] = { "M", "latent_dim", "heads", "blocks", "block_size" };
    const long long vals[5] = { 2, D, 8, 1, BS };
    for (int i = 0; i < 5; ++i) {
        p[i].key = keys[i];
        p[i].kind = RAD_P_INT;
        p[i].ival = vals[i];
    }
    p[5].key = "scale";
    p[5].kind = RAD_P_F64;
    p[5].dval = 0.0625;
    RadArgs a{};
    a.t = t;
    a.n_t = 5;
    a.p = p;
    a.n_p = 6;
    unsigned long long stats[3];
    r4d_mla_matrix_stats(stats);
    for (int i = 0; i < 1024; ++i)
        if (r4d_rad_mla_attn_matrix(&a, nullptr) != RAD_OK) {
            std::printf("periodic report: launch failed\n");
            return 1;
        }
    r4d_mla_matrix_stats(stats);
    const bool ok = stats[0] == 1024 && stats[2] == 1024 && stats[1] == 0;
    std::printf("periodic report: %llu passes over %llu raster chunks after 1024 launches  %s\n",
                stats[0], stats[2], ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

}   // namespace

int main(int argc, char** argv) {
    setenv("RADIANCE_MLA_CONFLICT_STATS", "1", 1);
    /* name, rows, heads, blocks, fill, swaps, context, scratch, form, expect_second, bitwise_a,
     * bitwise_all, union_falls_back */
    std::vector<Case> cases = {
        { "prefill, one table", 8, 8, 65, Fill::prefill, 0, 0, false, 0, 0, true, true },
        { "prefill, odd rows, split", 7, 8, 33, Fill::prefill, 0, 0, true, 0, 0, true, true },
        { "prefill, 16 heads", 4, 16, 33, Fill::prefill, 0, 0, false, 0, 0 },
        { "prefill, 24 heads (three pairs)", 5, 24, 17, Fill::prefill, 0, 0, false, 0, 0, true,
          true },
        { "topk d=0", 8, 8, 65, Fill::topk, 0, 600, false, 0, 0, true, true },
        { "topk d=4", 8, 8, 65, Fill::topk, 4, 600, false, 0, 1 },
        { "topk d=16", 8, 8, 65, Fill::topk, 16, 600, false, 0, 1 },
        { "topk d=64 (disjoint neighbours)", 6, 8, 65, Fill::topk, 64, 600, false, 0, 1 },
        { "topk d=16, 513 blocks", 4, 8, 513, Fill::topk, 16, 8192, false, 0, 1 },
        /* a split cuts the union's walk elsewhere than the positional raster */
        { "topk d=4, odd rows, split", 5, 8, 33, Fill::topk, 4, 400, true, 0, 1, false },
        { "topk d=4, element load", 4, 8, 33, Fill::topk, 4, 400, false, 1, 1 },
        { "topk d=4, 16 heads", 4, 16, 33, Fill::topk, 4, 400, false, 0, 0 },
        { "topk, table past the union's lists", 4, 8, 600, Fill::topk, 16, 4000, false, 0, 1, true,
          true, true },
        { "repeated pages", 6, 8, 33, Fill::repeat, 0, 0, false, 0, -1 },
        { "repeated pages, split", 3, 8, 33, Fill::repeat, 0, 0, true, 0, -1, false },
        { "ragged, holes and bad pages", 7, 8, 17, Fill::ragged, 0, 0, false, 0, -1 },
        { "ragged, split", 3, 8, 17, Fill::ragged, 0, 0, true, 0, -1, false },
    };
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int fails = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        if (filter && cases[i].name.find(filter) == std::string::npos) continue;
        std::mt19937 rng(20261018u + (unsigned)i);
        fails += run(cases[i], rng);
    }
    if (!filter) fails += periodic_report();
    std::printf("%d failure(s)\n", fails);
    return fails ? 1 : 0;
}
