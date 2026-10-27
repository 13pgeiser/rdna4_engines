/* quant_m2xfp_test.cpp -- the `m2xfp4` quantiser: the metadata plane's pack, the scheme's own
 * decode, and the search's quality against plain OCP MXFP4.
 *
 * Every case quantises through the quantiser's hooks as rad-convert does and reads the weight
 * back through enc_decode_rows, which asks libquant's decode hook for the scheme -- so the
 * round-trip checks the planes a reader that never saw the quantiser would unpack, against a
 * scalar decode of the spec's equation written out here by hand: the nibble stream, the E8M0
 * bytes and the metadata's low-bits-first subgroups are each unpacked independently. The quality
 * cases compare weight MSE against `rtn codes=fp4_e2m1 group=32 scale=e8m0`, the OCP MXFP4 the
 * scheme refines: the search's candidate set contains that quantiser's choice for every group,
 * so it can only tie or win, and the wins are where the format earns its half bit. The wide case
 * holds the same property one bias wider: wide=1 sweeps b in {-2, ..., 2} and can only tie or
 * beat adaptive, its planes decoding bit for bit as the equation does. The last two cases hand the
 * planes to libref's gemm_nt_q row at m2xfp4 (ref_m2xfp4.cpp) and hold its product to the same
 * hand decode, the second over a fused projection's parts.
 */
#include "quant_fixture.h"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using namespace rad;
using namespace qfix;

namespace {

constexpr const char* kName = "blk.2.attn_q.weight";

/* The thread count of the OpenMP runtime the plugin loaded, so a run can be repeated at another. */
void threads(int n) {
    using Fn = void (*)(int);
    if (Fn f = (Fn)dlsym(RTLD_DEFAULT, "omp_set_num_threads")) f(n);
}

bool same_planes(const Quantised& a, const Quantised& b) {
    if (a.enc.n_planes != b.enc.n_planes) return false;
    for (int i = 0; i < a.enc.n_planes; ++i)
        if (a.planes[(size_t)i] != b.planes[(size_t)i]) return false;
    return true;
}

/* The spec's decode, by hand: element j's nibble (element 2j in the low bits of byte j/2), its
 * group's E8M0 byte and metadata byte, subgroup s = (j%32)/8 in bits [2s, 2s+2). */
std::vector<float> ref_decode(const Quantised& q, int64_t rows, int64_t cols) {
    const int64_t ng = (cols + 31) / 32;
    const int64_t crow = (4 * cols + 7) / 8;
    const uint8_t* codes = q.planes[0].data();
    const uint8_t* scale = q.planes[1].data();
    const uint8_t* meta = q.planes[2].data();
    static const float mag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    std::vector<float> out((size_t)(rows * cols));
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t j = 0; j < cols; ++j) {
            const uint8_t nib = (codes[(size_t)(r * crow + (j >> 1))] >> ((j & 1) * 4)) & 15;
            const uint8_t sb = scale[(size_t)(r * ng + (j >> 5))];
            const uint8_t mb = meta[(size_t)(r * ng + (j >> 5))];
            const int k = (mb >> (2 * ((j & 31) >> 3))) & 3;
            const float s = std::ldexp(1.0f, (int)sb - 127);
            out[(size_t)(r * cols + j)] =
                (nib & 8 ? -mag[nib & 7] : mag[nib & 7]) * s * (1.0f + 0.25f * (float)k);
        }
    return out;
}

double mse(const std::vector<float>& w, const std::vector<float>& d) {
    double e = 0.0;
    for (size_t i = 0; i < w.size(); ++i) {
        const double x = (double)w[i] - d[i];
        e += x * x;
    }
    return e / (double)w.size();
}

/* Lognormal magnitudes, exp(sigma z) with z standard normal from a splitmix draw: the heavy tail
 * that makes one shared exponent a group a poor fit. */
std::vector<float> lognormal(int64_t n, uint64_t seed, double sigma) {
    uint64_t z = seed;
    auto next = [&]() {
        z += 0x9E3779B97F4A7C15ull;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return (double)((x ^ (x >> 31)) >> 11) / 9007199254740992.0;
    };
    std::vector<float> w((size_t)n);
    for (int64_t i = 0; i < n; i += 2) {
        const double u1 = next(), u2 = next();
        const double r = std::sqrt(-2.0 * std::log(u1 + 1e-12));
        const double a = sigma * r * std::cos(6.283185307179586 * u2);
        const double b = sigma * r * std::sin(6.283185307179586 * u2);
        w[(size_t)i] = (float)std::exp(a);
        if (i + 1 < n) w[(size_t)i + 1] = (float)std::exp(b);
    }
    return w;
}

/* Every group's amax on a different fraction of an octave, 2^u with u uniform in [0, 1): the
 * amax / 2^E0 quotient swept over [4, 8), where the OCP floor's choice of exponent is worst and
 * the bias search has most to say. */
std::vector<float> octave_groups(int64_t rows, int64_t cols, uint64_t seed) {
    const std::vector<float> z = draw(rows * cols, seed, 4.0f);
    std::vector<float> w(z);
    const int64_t ng = (cols + 31) / 32;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t g = 0; g < ng; ++g) {
            uint64_t h = seed ^ ((uint64_t)r * 0x9E3779B97F4A7C15ull) ^ (uint64_t)g;
            h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull;
            h = (h ^ (h >> 27)) * 0x94D049BB133111EBull;
            const double u = (double)(h >> 11) / 9007199254740992.0;
            const float lift = (float)std::exp2(u);       /* an octave, [1, 2) */
            for (int64_t c = g * 32; c < (g + 1) * 32 && c < cols; ++c)
                w[(size_t)(r * cols + c)] = z[(size_t)(r * cols + c)] * lift;
        }
    return w;
}

/* The fixture's quantise takes its options by reference, so each case builds its own. */
void m2xfp_opts(Opts& o, int adaptive, int wide = 0) {
    o.i("adaptive", adaptive);
    if (wide) o.i("wide", wide);
}

void mxfp4_opts(Opts& o) { o.s("codes", "fp4_e2m1").i("group", 32).s("scale", "e8m0"); }

}  // namespace

TEST(m2xfp4_is_three_planes_and_round_trips_through_its_own_decode) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* q = quantizer("m2xfp4");
    REQUIRE(q);
    const RadQuantWeight w0 = weight_of("x", 8, 64);
    RadEncoding e{};
    REQUIRE_EQ(q->encoding(nullptr, 0, &w0, &e), RAD_OK);
    CHECK_EQ(std::string(e.scheme), std::string("m2xfp4"));
    CHECK_EQ(e.transform[0], 0);
    REQUIRE_EQ(e.n_planes, 3);
    CHECK_EQ(std::string(e.plane[0].role), std::string("codes"));
    CHECK_EQ((int)e.plane[0].dtype, (int)RAD_FP4E2M1);
    CHECK_EQ(e.plane[0].block[0], 1);
    CHECK_EQ(e.plane[0].block[1], 1);
    CHECK_EQ(std::string(e.plane[1].role), std::string("scale"));
    CHECK_EQ((int)e.plane[1].dtype, (int)RAD_E8M0);
    CHECK_EQ(e.plane[1].block[1], 32);
    CHECK_EQ(std::string(e.plane[2].role), std::string("meta"));
    CHECK_EQ((int)e.plane[2].dtype, (int)RAD_U8);
    CHECK_EQ(e.plane[2].block[1], 32);
    CHECK_EQ(q->row_block(nullptr, 0, &w0), (int64_t)1);
    /* a rank-1 norm is declined, like every quantiser's, not an error */
    RadQuantWeight v{};
    v.name = "x";
    v.rank = 1;
    v.shape[0] = 64;
    v.rows = 1;
    v.cols = 64;
    CHECK_EQ(q->encoding(nullptr, 0, &v, &e), RAD_E_UNSUPPORTED);

    /* whole groups, a short final group, and an odd column count's nibble tail: the decode the
     * core reaches through the hook must equal the equation by hand, bit for bit. */
    for (int64_t C : { 256, 112, 101 }) {
        const int64_t R = 8;
        const std::vector<float> w = draw(R * C, 41, 8.0f);
        Opts o_adapt;
        m2xfp_opts(o_adapt, 1);
        Quantised a = quantise(q, o_adapt, kName, R, C, w);
        REQUIRE_EQ(a.status, RAD_OK);
        const std::vector<float> d = decode(a, R, C);
        REQUIRE_EQ(d.size(), w.size());
        const std::vector<float> r = ref_decode(a, R, C);
        for (size_t i = 0; i < d.size(); ++i) CHECK_EQ(d[(size_t)i], r[(size_t)i]);

        /* rows from the middle, as the oracle and rad-convert ask for them */
        std::vector<const void*> pd;
        for (auto& p : a.planes) pd.push_back(p.data());
        EncodedView v;
        std::string why;
        enc_view(a.enc, R, C, pd.data(), &v);
        std::vector<float> part((size_t)(3 * C));
        REQUIRE_EQ(enc_decode_rows(v, 3, 3, part.data(), &why), RAD_OK);
        for (int64_t i = 0; i < 3 * C; ++i) CHECK_EQ(part[(size_t)i], d[(size_t)(3 * C + i)]);
    }
}

TEST(m2xfp4_packs_the_metadata_low_bits_first_and_zero_groups_like_mxfp4) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* q = quantizer("m2xfp4");
    REQUIRE(q);
    /* One row, one group, its subgroups built to choose k = 0, 0, 1, 3 (the zeros keep k 0, the
     * 1.9s want 5/4 of the scale, the 5.5s want 7/4 of it). The group is built so the bias tie
     * is exact: every subgroup's best dequantised value is the same under b = -1 and b = 0
     * (0, 1.0, 1.875 and 5.25 -- the e2m1 grid is scale-invariant until a candidate saturates),
     * so the biases tie at 0.5082 and the search keeps the first of {-1, 0, 1}, the paper's
     * argmin. The exponent byte is 126, and the same codes under it decode to what b = 0's 127
     * would: the tie moves the byte, never the weight. */
    std::vector<float> w(32);
    for (int j = 0; j < 8; ++j) w[(size_t)j] = 0.02f;
    for (int j = 8; j < 16; ++j) w[(size_t)j] = 1.0f;
    for (int j = 16; j < 24; ++j) w[(size_t)j] = 1.9f;
    for (int j = 24; j < 32; ++j) w[(size_t)j] = 5.5f;
    Opts o_adapt;
    m2xfp_opts(o_adapt, 1);
    Quantised a = quantise(q, o_adapt, kName, 1, 32, w);
    REQUIRE_EQ(a.status, RAD_OK);
    REQUIRE_EQ(a.planes[1].size(), (size_t)1);
    REQUIRE_EQ(a.planes[2].size(), (size_t)1);
    CHECK_EQ((int)a.planes[1][0], 126);
    CHECK_EQ((int)a.planes[2][0], (0 << 0) | (0 << 2) | (1 << 4) | (3 << 6));
    const std::vector<float> d = decode(a, 1, 32);
    REQUIRE_EQ(d.size(), w.size());
    const std::vector<float> r = ref_decode(a, 1, 32);
    for (int j = 0; j < 32; ++j) CHECK_EQ(d[(size_t)j], r[(size_t)j]);

    /* A zero group: byte 0 and zero codes, the MXFP4 grid's exponent-floor clamp, not the
     * reference's clamped-amax exponent -- the same decode either way, radiance's bytes win. */
    const std::vector<float> z(2 * 64, 0.0f);
    Opts o_zero, o_mx;
    m2xfp_opts(o_zero, 1);
    mxfp4_opts(o_mx);
    Quantised m = quantise(q, o_zero, kName, 2, 64, z);
    Quantised g = quantise(quantizer("rtn"), o_mx, kName, 2, 64, z);
    REQUIRE_EQ(m.status, RAD_OK);
    REQUIRE_EQ(g.status, RAD_OK);
    CHECK(m.planes[1] == g.planes[1]);
    int bad = 0;
    for (uint8_t c : m.planes[0]) bad += c != 0;
    for (uint8_t b : m.planes[2]) bad += b != 0;
    CHECK_EQ(bad, 0);
}

TEST(m2xfp4_stores_the_same_bytes_one_band_or_four_threads_at_a_time) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* q = quantizer("m2xfp4");
    REQUIRE(q);
    const int64_t R = 32, C = 256;
    const std::vector<float> w = draw(R * C, 43, 6.0f);
    Opts o_adapt;
    m2xfp_opts(o_adapt, 1);
    Quantised whole = quantise(q, o_adapt, kName, R, C, w);
    threads(1);
    Quantised one = quantise(q, o_adapt, kName, R, C, w, nullptr, 4);
    threads(4);
    Quantised four = quantise(q, o_adapt, kName, R, C, w, nullptr, 4);
    REQUIRE_EQ(whole.status, RAD_OK);
    REQUIRE_EQ(one.status, RAD_OK);
    REQUIRE_EQ(four.status, RAD_OK);
    /* its output is a file: the bands and the thread count may not move a byte */
    CHECK(same_planes(whole, one));
    CHECK(same_planes(whole, four));
}

TEST(m2xfp4_beats_ocp_mxfp4_on_heavy_tails_and_its_search_never_hurts) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* q = quantizer("m2xfp4");
    REQUIRE(q);
    const int64_t R = 64, C = 256;
    struct Case { const char* what; std::vector<float> w; };
    std::vector<Case> cases;
    cases.push_back({ "lognormal", lognormal(R * C, 51, 1.5) });
    cases.push_back({ "octave", octave_groups(R, C, 52) });

    Opts o_mx, o_pin, o_adapt;
    mxfp4_opts(o_mx);
    m2xfp_opts(o_pin, 0);
    m2xfp_opts(o_adapt, 1);
    for (const Case& c : cases) {
        Quantised base = quantise(quantizer("rtn"), o_mx, kName, R, C, c.w);
        Quantised off = quantise(q, o_pin, kName, R, C, c.w);
        Quantised on = quantise(q, o_adapt, kName, R, C, c.w);
        REQUIRE_EQ(base.status, RAD_OK);
        REQUIRE_EQ(off.status, RAD_OK);
        REQUIRE_EQ(on.status, RAD_OK);
        const double eb = mse(c.w, decode(base, R, C));
        const double e0 = mse(c.w, decode(off, R, C));
        const double e1 = mse(c.w, decode(on, R, C));
        std::fprintf(stderr, "    %s: MXFP4 %.5f, m2xfp4 bias-pinned %.5f, adaptive %.5f "
                             "(%.1f%% of MXFP4)\n", c.what, eb, e0, e1, 100.0 * e1 / eb);
        /* every group's candidates contain plain MXFP4's choice, and the adaptive set contains
         * the pinned one, so each stage can only tie or improve */
        CHECK(e0 <= eb);
        CHECK(e1 <= e0);
        CHECK(e1 < eb * 0.98);
    }
}

TEST(m2xfp4_wide_bias_sweep_contains_the_narrow_one_and_round_trips_bit_exact) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* q = quantizer("m2xfp4");
    REQUIRE(q);
    const int64_t R = 64, C = 256;
    struct Case { const char* what; std::vector<float> w; };
    std::vector<Case> cases;
    cases.push_back({ "lognormal", lognormal(R * C, 61, 2.0) });
    cases.push_back({ "octave", octave_groups(R, C, 62) });

    Opts o_pin, o_adapt, o_wide;
    m2xfp_opts(o_pin, 0);
    m2xfp_opts(o_adapt, 1);
    m2xfp_opts(o_wide, 1, 1);
    for (const Case& c : cases) {
        Quantised off = quantise(q, o_pin, kName, R, C, c.w);
        Quantised on = quantise(q, o_adapt, kName, R, C, c.w);
        Quantised wide = quantise(q, o_wide, kName, R, C, c.w);
        REQUIRE_EQ(off.status, RAD_OK);
        REQUIRE_EQ(on.status, RAD_OK);
        REQUIRE_EQ(wide.status, RAD_OK);
        const double e0 = mse(c.w, decode(off, R, C));
        const double e1 = mse(c.w, decode(on, R, C));
        const double e2 = mse(c.w, decode(wide, R, C));
        std::fprintf(stderr, "    %s: pinned %.5f, adaptive %.5f, wide %.5f (%.1f%% of "
                             "pinned)\n", c.what, e0, e1, e2, 100.0 * e2 / e0);
        /* wide's candidates {-2, ..., 2} contain adaptive's {-1, 0, 1}, which contain the
         * pinned b = 0, so each stage can only tie or improve */
        CHECK(e1 <= e0);
        CHECK(e2 <= e1);

        /* the wide path's planes decode bit for bit as the equation unpacked by hand */
        const std::vector<float> d = decode(wide, R, C);
        const std::vector<float> r = ref_decode(wide, R, C);
        REQUIRE_EQ(d.size(), c.w.size());
        for (size_t i = 0; i < d.size(); ++i) CHECK_EQ(d[i], r[i]);
    }

    /* wide=1 at adaptive=0 is the pinned quantiser, plane for plane: there is no bias to widen,
     * so the option may not move a byte */
    const std::vector<float> w = draw(R * C, 55, 8.0f);
    Opts o_pin_wide;
    m2xfp_opts(o_pin_wide, 0, 1);
    Quantised off = quantise(q, o_pin, kName, R, C, w);
    Quantised both = quantise(q, o_pin_wide, kName, R, C, w);
    REQUIRE_EQ(off.status, RAD_OK);
    REQUIRE_EQ(both.status, RAD_OK);
    CHECK(same_planes(off, both));
}

/* libref's gemm_nt_q row at m2xfp4 over the quantiser's own planes: its layout hook takes codes and
 * meta for the weight and the exponent plane as stored, its relayout concatenates the two, and
 * the product is the hand decode's against an exact e4m3 activation and its per-128 scales,
 * summed in double. With parts (pb1/pb2), weight rows [0, pb1), [pb1, pb2) and [pb2, N) each read
 * their own copy of the activation, side by side in a row of `a`. */
static void libref_m2xfp4_gemm(int64_t M, int64_t N, int64_t K, int64_t pb1, int64_t pb2) {
    NEED_LIBQUANT();
    const std::string lib = home() + "/kernels/libref.so";
    void* so = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!so) ARTIFACT_MISSING(lib + " (does not load: " + dlerror() + ")");
    const auto count = (int (*)(void))dlsym(so, "rad_kernel_count");
    const auto at = (const RadKernelInfo* (*)(int))dlsym(so, "rad_kernel_at");
    REQUIRE(count && at);
    const RadKernelInfo* k = nullptr;
    for (int i = 0; i < count(); ++i)
        if (std::strcmp(at(i)->name, "ref_gemm_nt_q") == 0) k = at(i);
    REQUIRE(k && k->layout && k->relayout && k->launch);

    const int64_t kb = K / 32, akb = K / 128;
    const int64_t e1 = pb1 > 0 ? pb1 : N, e2 = pb2 > 0 ? pb2 : N;
    const int64_t parts = e2 < N ? 3 : (e1 < N ? 2 : 1);
    const int64_t PK = parts * K, PA = parts * akb;      /* a row of `a`, and of its scale */
    const RadQuantizerInfo* q = quantizer("m2xfp4");
    REQUIRE(q);
    Opts o;
    m2xfp_opts(o, 1);
    Quantised w = quantise(q, o, kName, N, K, draw(N * K, 61, 8.0f));
    REQUIRE_EQ(w.status, RAD_OK);
    const std::vector<float> wd = ref_decode(w, N, K);

    auto tensor = [](void* d, uint32_t dt, std::initializer_list<int64_t> s) {
        RadTensor t{};
        t.data = d;
        t.dtype = dt;
        for (int64_t e : s) t.shape[t.rank++] = e;
        int64_t st = 1;
        for (int i = (int)t.rank - 1; i >= 0; --i) {
            t.stride[i] = st;
            st *= t.shape[i];
        }
        return t;
    };
    Opts g;
    g.i("M", M).i("N", N).i("K", K).i("group", 32).s("dtype", "m2xfp4");
    if (pb1 > 0) g.i("pb1", pb1);
    if (pb2 > 0) g.i("pb2", pb2);
    const RadTensor pl[2] = { tensor(w.planes[0].data(), RAD_FP4E2M1, { N, K }),
                              tensor(w.planes[2].data(), RAD_U8, { N, kb }) };
    const RadTensor sp = tensor(w.planes[1].data(), RAD_E8M0, { N, kb });
    const int sel[2] = { 0, 2 }, ssel[1] = { 1 };
    RadLayout lay{};
    CHECK_EQ(k->layout(g.data(), g.n(), 3, &w.enc, ssel, &sp, 1, &lay), (int)RAD_E_UNSUPPORTED);
    REQUIRE_EQ(k->layout(g.data(), g.n(), 2, &w.enc, sel, pl, 2, &lay), (int)RAD_OK);
    REQUIRE_EQ(lay.bytes, N * (K / 2 + kb));
    std::vector<uint8_t> b((size_t)lay.bytes);
    REQUIRE_EQ(k->relayout(g.data(), g.n(), 2, &w.enc, sel, pl, 2, b.data(), lay.bytes),
               (int)RAD_OK);

    /* integer codes in [-13, 13], which e4m3 holds exactly */
    std::vector<uint8_t> a((size_t)(M * PK));
    std::vector<float> as((size_t)(M * PA)), y((size_t)(M * N), NAN);
    for (int64_t i = 0; i < M * PK; ++i)
        rad_store_f32(a.data(), RAD_F8E4M3, i, (float)((i * 7) % 27 - 13));
    for (int64_t i = 0; i < M * PA; ++i) as[(size_t)i] = 0.003f * (float)(1 + i % 5);
    RadTensor t[7] = { tensor(a.data(), RAD_F8E4M3, { M, PK }),
                       tensor(as.data(), RAD_F32, { M, PA }),
                       tensor(b.data(), RAD_U8, { lay.bytes }),
                       sp,
                       tensor(y.data(), RAD_F32, { M, N }),
                       RadTensor{},
                       sp };
    RadArgs args{};
    args.t = t;
    args.n_t = 7;
    args.p = g.data();
    args.n_p = g.n();
    REQUIRE_EQ(k->launch(&args, nullptr), (int)RAD_OK);

    double err = 0, mag = 0;
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            const int64_t pt = n < e1 ? 0 : (n < e2 ? 1 : 2);
            double r = 0;
            for (int64_t c = 0; c < K; ++c)
                r += (double)rad_load_f32(a.data(), RAD_F8E4M3, m * PK + pt * K + c) *
                     (double)as[(size_t)(m * PA + pt * akb + c / 128)] *
                     (double)wd[(size_t)(n * K + c)];
            err = std::max(err, std::fabs((double)y[(size_t)(m * N + n)] - r));
            mag = std::max(mag, std::fabs(r));
        }
    CHECK(mag > 0);
    CHECK(err <= 1e-5 * mag);
    dlclose(so);
}

/* N = 40 and M = 13 are not multiples of anything; K = 384 is three activation scale blocks. */
TEST(m2xfp4_libref_gemm_reads_the_planes_as_the_decode_does) {
    libref_m2xfp4_gemm(13, 40, 384, 0, 0);
}

/* A rotated fused projection's parts, two and three of them, at 16-row bounds. */
TEST(m2xfp4_libref_gemm_gives_each_part_its_own_activation) {
    libref_m2xfp4_gemm(13, 48, 384, 16, 0);
    libref_m2xfp4_gemm(13, 48, 384, 16, 32);
}

RAD_TEST_MAIN()
