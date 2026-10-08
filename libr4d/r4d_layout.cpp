/* r4d_layout.cpp -- what libr4d's integer and MXFP4 GEMMs store for a weight, and the relayouts
 * that produce it from the weight's canonical planes (spec.md §4.3). Pure host C++: nothing here
 * includes r4d.h or <hip/hip_runtime.h>, so it compiles and can be exercised on a box with no
 * ROCm, which is where a packing bug is cheapest to find.
 *
 * A container holds each weight's encoding -- codes, scales and zero points, canonical and
 * row-major -- and the kernel that reads it rearranges it at load. That is what lets this file
 * change a fragment order without anyone converting a model again; the numbers were chosen by a
 * quantiser (libquant), and nothing here chooses one.
 *
 * THE FRAGMENT ORDER IS THE WHOLE POINT. gfx1201's wave32 WMMA fragment layout is
 *
 *     idx = lane % 16 ,  k = 8*(e>>2) + 4*(lane>>4) + (e&3)
 *
 * so a B fragment maps lane l onto output row l&15: in canonical order the sixteen lanes of a
 * half-wave read sixteen rows K/2 bytes apart, which is a fully strided read on the operand that
 * dominates traffic. Rearranged, a wave's weight load for four k steps is one 512-byte
 * global_load_b128 with a constant stride and no address arithmetic in the loop.
 *
 * Every element->slot map below is transcribed from the kernel that reads it; the comment at the
 * top of each libr4d GEMM states it in closed form.
 */

#include "r4d_plugin.h"
#include "r4d_enc.h"

#include <cstdint>
#include <cstring>
#include <vector>

/* ------------------------------------------------------------------ half-precision
 *
 * Written out rather than pulled from a header because this file must build without ROCm and
 * without any float16 extension: __fp16 is not portable and _Float16 is not guaranteed on every
 * host compiler the converter might be built with. */

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFFu) == 0xFFu)                    /* inf / nan */
        return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);  /* overflow -> inf */
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;            /* underflow -> zero */
        man |= 0x800000u;                                /* subnormal: shift in the hidden bit */
        const int shift = 14 - exp;
        const uint32_t v = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        uint32_t out = v;
        if (rem > half || (rem == half && (v & 1u))) ++out;   /* round to nearest even */
        return (uint16_t)(sign | out);
    }
    uint32_t v = ((uint32_t)exp << 10) | (man >> 13);
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1u))) ++v;
    return (uint16_t)(sign | v);
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t man  = h & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) out = sign;
        else {
            int e = -1;
            uint32_t m = man;
            do { m <<= 1; ++e; } while (!(m & 0x400u));
            out = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1F) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

/* THE SCALE DWORD. libr4d packs the group's two quantiser parameters into one 32-bit word: the f16
 * scale in the low half and the f16 of -(1024 + zero) in the high half, ready to be broadcast into
 * a v_pk_add_f16. The kernel builds 1024 + code and adds that constant, which is what makes the
 * subtraction of an INTEGER zero point exact. A symmetric 4-bit grid is expressed in the same
 * encoding with zero == 8, which is why one packed weight can feed both the f16 and the int8
 * kernel byte for byte. */
static uint32_t scale_zero_dword(float scale, int zero) {
    const uint32_t lo = f32_to_f16(scale);
    const uint32_t hi = f32_to_f16(-(1024.0f + (float)zero));
    return lo | (hi << 16);
}

/* Geometry the layout hooks read out of the declared parameters. */
struct WGeom { long long N, K, group; };

static int wgeom(const RadParam* p, int n_p, long long dflt_group, WGeom* g) {
    g->N = r4d_pgeti_or(p, n_p, "N", 0);
    g->K = r4d_pgeti_or(p, n_p, "K", 0);
    g->group = r4d_pgeti_or(p, n_p, "group", dflt_group);
    if (g->N <= 0 || g->K <= 0 || g->group <= 0) return RAD_E_INVAL;
    if (g->N % R4D_NTILE) return RAD_E_SHAPE;
    if (g->K % g->group) return RAD_E_SHAPE;
    return RAD_OK;
}

/* Operand indices in `gemm_nt_q` (see oGemmNtQ in r4d_rows.cpp). */
enum { Q_B = 2, Q_BSCALE = 3, Q_BREF = 6 };

static void fill(RadLayout* o, const char* tag, uint32_t dtype, long long r0, long long r1,
                 long long bytes) {
    std::memset(o, 0, sizeof(*o));
    o->tag = tag;
    o->dtype = dtype;
    o->rank = 2;
    o->shape[0] = r0;
    o->shape[1] = r1;
    /* Every one of these streams is read with 128-bit loads; 256 is a cacheline on the
     * architectures in scope, and the .rad plane alignment. */
    o->align = 256;
    o->bytes = bytes;
}

/* THE ASYMMETRIC 4-BIT GRID IS A DIFFERENT WEIGHT, and `dtype` is what says so. `w4a8_asym`'s
 * kernels read the nibble UNSIGNED and subtract a per-group integer zero from the scale dword's
 * high half; the symmetric grid stores a two's-complement nibble with the zero pinned at 8. The
 * packing, the extents and the byte counts are identical, so a plane written for one and read by
 * the other is accepted everywhere and decodes to the wrong numbers -- a symmetric plane read
 * unsigned is eight levels off at every code. The test is the kernels' own (r4d_gemm_args.h's
 * is_asym), so the encoding the layout hook accepts -- i4 with no zero, or u4 with one -- is the
 * grid the launch selects, and the other is refused at declare. */
static bool w4_asym(const RadParam* p, int n_p) {
    const char* d = r4d_pgets(p, n_p, "dtype");
    return d && std::strstr(d, "_asym") != nullptr;
}

/* ================================================================== moe w4
 *
 * THE GROUPED MoE GEMM'S 4-BIT EXPERT WEIGHT, and it is a different plane from the dense w4 above
 * for one structural reason: the dense kernels read a WMMA fragment straight out of a coalesced
 * global load, so their weight is stored PERMUTED into fragment order; the grouped kernel stages
 * its weight through LDS -- it has to, because eight M tiles share one N tile of it -- so a
 * permutation would buy nothing and would have to be undone on the way in.
 *
 * WHAT THE FORMAT IS
 *
 *   weight  [N][K/2] bytes, row-major, TWO'S COMPLEMENT codes -8..7 -- an ordinary symmetric
 *           int4 grid, the one GPTQ and every other 4-bit method is written against.
 *   scale   [N][K/group] bf16, one per row per group of 128 along K. w ~= code * scale.
 *
 * THE DECODE IS TWO V_PERM_B32 AND A THIRD THAT CHOOSES BETWEEN THEM. A perm is a byte permute of
 * an EIGHT-byte source, so one table holds eight entries and a two's complement nibble needs
 * sixteen. Indexing both a positive and a negative table with the low three bits and selecting on
 * bit 3 -- by ADDING 4 to a selector byte, which cannot carry -- gets all sixteen levels for five
 * more VALU per eight weights than a fifteen-level sign-magnitude grid would cost. r4d_moe.hip's
 * w4_e4m3 is the five lines.
 *
 * THE INTEGERS -8..7 ARE ALL EXACT IN E4M3 (three mantissa bits reach 16), so expanding a code to an
 * E4M3 byte is lossless and the product the WMMA computes is code * activation with no rounding
 * the fp8 path did not already have. That is the whole reason this weight can ride the existing
 * f32_16x16x16_fp8_fp8 inner loop untouched: only the STAGING changes.
 *
 * THE NIBBLE ORDER IS THE ONE THE EXPANSION WANTS. Byte b of a packed dword holds element b in its
 * low nibble and element b+4 in its high nibble, so the two masks above produce elements 0..3 and
 * 4..7 already in order and nothing has to be interleaved afterwards. It is the same argument the
 * dense w4a8 kernel's header makes for the same pair of masks.
 *
 * IT IS NOT SHARDABLE AND THE ARCHITECTURE SAYS SO. A packed row is K/2 bytes and a column shard
 * would cut a byte in half; an expert under expert parallelism is whole on one rank, which is the
 * only configuration that declares this format. rad_block_moe_fp8.h refuses a width-sharded
 * declaration by name rather than leaving it to the loader to discover.
 */
/* THE ROTATED FORM IS THE SAME BYTES WITH DIFFERENT NUMBERS IN THEM, and that is exactly why the
 * encoding says so. `dtype=w4a8h` serves `i4*bf16[1x128]/fwht128`: every group of K was multiplied
 * by a `group`-wide Hadamard before it was quantised. Nothing in the packing, the extents or the
 * byte count would say so, and a weight quantised one way and served the other reads as noise --
 * so the layout hook accepts exactly the encoding the dtype names, and refuses the other at
 * declare.
 *
 * THE KERNEL DOES NOT BRANCH ON IT. <Hw, Ha> = group * <w, a>, and the residual factor is folded
 * into the stored weight scale by the quantiser (docs/MOE-W4.md), so the GEMM multiplies codes by
 * scales exactly as it does for the unrotated grid. What has to agree is the ACTIVATION: the
 * routed experts' input must be quantised by the rotated quantiser at the same width, which is
 * what had_quant_act_fp8 is for. That pairing is not checkable from here -- it is a property of the
 * graph, and the arch block that declares one declares the other. */
/* THE CODEBOOK FORM IS THE ROTATED FORM WITH OTHER NUMBERS FOR THE SAME CODES. `dtype=w4nla8h`
 * serves `u4*bf16[1x128]/fwht128` whose codes index a sixteen-entry `table` plane -- libquant's
 * `table=w4nl` -- and the kernel decodes that table and no other (r4d_args.h). An encoding cannot
 * say what its table holds, so the codes view selects the table plane beside the codes and the
 * relayout compares it with the kernel's, value for value, before it serves a byte: a container
 * quantised against another table is refused at load rather than served as other numbers. */
static int moe_w4_nl(const RadParam* p, int n_p) {
    const char* d = r4d_pgets(p, n_p, "dtype");
    return d && (std::strcmp(d, "w4nla8h") == 0 || std::strcmp(d, "w4nl64a8h") == 0 ||
                 std::strcmp(d, "w4nl32a8h") == 0 || std::strcmp(d, "w5nl64a8h") == 0);
}
/* `w5nl64a8h` IS w4nl64a8h AT FIVE BITS: `u5*fp8_e4m3[1x64]*f32[*x*]/fwht128` whose codes index
 * libquant's 32-entry `table=w5nl`. Its codes view relays each row into R4D_DT_MOEW5 -- the low
 * four bits of every code as w4nla8h's nibble plane, then the high bit of every code, the sign,
 * a dword a 32 elements (moew5_row says in which order) -- and compares the table with
 * kW5nlTable as w4nla8h's is; its scale view is w4nl64a8h's. */
static bool moe_w5(const RadParam* p, int n_p) {
    const char* d = r4d_pgets(p, n_p, "dtype");
    return d && std::strcmp(d, "w5nl64a8h") == 0;
}
/* `w4nl64a8h` IS THE CODEBOOK FORM WITH ITS SCALES A 64 OF K, as E4M3 under the fixed
 * kW4G64Scale -- `u4*fp8_e4m3[1x64]*f32[*x*]/fwht128` with the table, the f32 second level being
 * that constant, stated once a weight. The codes are stored as w4nla8h's; the scale view selects
 * the E4M3 plane and the second level, and its relayout compares the second with the kernel's
 * constant, as the codes view does the table, and stores the E4M3 plane as it is. */
/* And `w4nl32a8h` the same at one a (row, 32 of K). The value is the scale groups a 128 of K the
 * dtype has beside its bf16 forms' one: 2, 4, or 0 for neither. */
static int moe_w4_g64(const RadParam* p, int n_p) {
    const char* d = r4d_pgets(p, n_p, "dtype");
    if (!d) return 0;
    return (std::strcmp(d, "w4nl64a8h") == 0 || std::strcmp(d, "w5nl64a8h") == 0) ? 2
           : std::strcmp(d, "w4nl32a8h") == 0 ? 4 : 0;
}
static int moe_w4_rot(const RadParam* p, int n_p) {
    const char* d = r4d_pgets(p, n_p, "dtype");
    return d && (std::strcmp(d, "w4a8h") == 0 || moe_w4_nl(p, n_p));
}

/* ================================================================== moe_gemm_q's two classes
 *
 * `moe_gemm_q` takes its experts as two tables by parity: operands 2 and 3 hold the even experts
 * at [N, K], operands 6 and 7 the odd ones at [N_odd, K_odd] (r4d_rows.cpp's schema says why). A
 * hook asked about 6 or 7 answers exactly as it would for 2 or 3 at the odd geometry, so every
 * hook below is the plain one behind this view, and the two classes cannot come to disagree about
 * a format they share. */
namespace {
struct MoeView {
    RadParam p[32];
    int      n = 0;
    int      operand = 0;
};

int moe_view(const RadParam* p, int n_p, int operand, MoeView* v) {
    if (n_p < 0 || n_p > 32 || (n_p > 0 && !p)) return RAD_E_INVAL;
    v->n = n_p;
    v->operand = operand;
    for (int i = 0; i < n_p; ++i) v->p[i] = p[i];
    if (operand != 6 && operand != 7) return RAD_OK;
    v->operand = operand - 4;
    const long long n1 = r4d_pgeti_or(p, n_p, "N_odd", -1);
    const long long k1 = r4d_pgeti_or(p, n_p, "K_odd", -1);
    for (int i = 0; i < n_p; ++i) {
        if (!v->p[i].key) continue;
        if (n1 > 0 && std::strcmp(v->p[i].key, "N") == 0) v->p[i].ival = n1;
        if (k1 > 0 && std::strcmp(v->p[i].key, "K") == 0) v->p[i].ival = k1;
    }
    return RAD_OK;
}

/* `moe_gemm_q_mod4`'s four classes: operands 6/7, 8/9 and 10/11 are classes 1, 2 and 3 at
 * N_1/K_1, N_2/K_2 and N_3/K_3, answered as 2/3 at that geometry -- the same view, one class
 * further along. */
int moe_view_mod4(const RadParam* p, int n_p, int operand, MoeView* v) {
    if (n_p < 0 || n_p > 32 || (n_p > 0 && !p)) return RAD_E_INVAL;
    v->n = n_p;
    v->operand = operand;
    for (int i = 0; i < n_p; ++i) v->p[i] = p[i];
    if (operand < 6 || operand > 11) return RAD_OK;
    static const char* const kN[4] = { "N", "N_1", "N_2", "N_3" };
    static const char* const kK[4] = { "K", "K_1", "K_2", "K_3" };
    const int q = (operand - 4) / 2;                 /* 6/7 -> 1, 8/9 -> 2, 10/11 -> 3 */
    v->operand = 2 + (operand & 1);
    const long long nq = r4d_pgeti_or(p, n_p, kN[q], -1);
    const long long kq = r4d_pgeti_or(p, n_p, kK[q], -1);
    for (int i = 0; i < n_p; ++i) {
        if (!v->p[i].key) continue;
        if (nq > 0 && std::strcmp(v->p[i].key, "N") == 0) v->p[i].ival = nq;
        if (kq > 0 && std::strcmp(v->p[i].key, "K") == 0) v->p[i].ival = kq;
    }
    return RAD_OK;
}
}  /* namespace */

/* ================================================================== the hooks
 *
 * The codes and scales arrive chosen, by a quantiser, and what is left here is the arrangement the
 * kernel reads -- fragment order, the (scale, zero) dword, a tile-major scale plane, a nibble
 * order -- which moves bits and chooses none. Each refuses, with RAD_E_DTYPE, an encoding its
 * kernel does not read. */
namespace {

using r4d_enc::selects;
using r4d_enc::dims;

/* A selected plane's dense bytes. */
const uint8_t* pdata(const RadTensor& t) { return (const uint8_t*)t.data; }

/* The f16 scale of a canonical plane, as f32 -- the value scale_zero_dword re-narrows exactly. */
float f16_at(const RadTensor& t, long long i) { return f16_to_f32(((const uint16_t*)t.data)[i]); }

/* An integer zero point from a u4 or u8 plane. */
int zero_at(const RadTensor& t, long long i) { return (int)rad_load_f32(t.data, t.dtype, i); }

/* The dense w4/w8/w2 families' scale operand: (scale) for a symmetric grid, (scale, zero) for an
 * asymmetric one, tile-major [N/16][(K/group)*16] in dwords or, for w8a8, halves. */
int sz_layout(const WGeom& g, const RadEncoding* enc, const int* sel, const RadTensor* pl, int n,
              bool asym, const char* tag, uint32_t dt, int bytes_each, RadLayout* out) {
    const long long nsz = g.K / g.group;
    if (asym ? !selects(enc, sel, n, "scale", "zero") : !selects(enc, sel, n, "scale"))
        return RAD_E_SHAPE;
    for (int i = 0; i < n; ++i)
        if (!dims(pl[i], g.N, nsz)) return RAD_E_SHAPE;
    fill(out, tag, dt, g.N / R4D_NTILE, nsz * R4D_NTILE, g.N * nsz * bytes_each);
    return RAD_OK;
}

/* Write the tile-major (scale, zero) dwords. `zfixed` >= 0 is a symmetric grid's constant zero. */
void sz_write(const WGeom& g, const RadTensor* pl, int zfixed, uint32_t* sz) {
    const long long nt = g.N / R4D_NTILE, nsz = g.K / g.group;
    for (long long t = 0; t < nt; ++t)
        for (long long r = 0; r < R4D_NTILE; ++r)
            for (long long gi = 0; gi < nsz; ++gi) {
                const long long n = t * R4D_NTILE + r;
                const int z = zfixed >= 0 ? zfixed : zero_at(pl[1], n * nsz + gi);
                const float sc = f16_at(pl[0], n * nsz + gi);
                sz[(t * nsz + gi) * R4D_NTILE + r] = scale_zero_dword(sc, z);
            }
}

/* Every zero point a code of `bits` bits: what an asymmetric grid's zero is, and what the int8
 * prefill kernel reads back out of the scale dword (r4d_gemm_wpa8i8_tiled.hip takes five bits of it,
 * where the f16 kernel takes the whole value -- a larger zero would make the two disagree). */
bool sz_zeros_fit(const WGeom& g, const RadTensor& zp, int bits) {
    const long long nz = g.N * (g.K / g.group);
    for (long long i = 0; i < nz; ++i) {
        const int z = zero_at(zp, i);
        if (z < 0 || z >= (1 << bits)) return false;
    }
    return true;
}

void sz_read(const WGeom& g, const uint32_t* sz, const RadTensor* pl, bool asym) {
    const long long nt = g.N / R4D_NTILE, nsz = g.K / g.group;
    for (long long t = 0; t < nt; ++t)
        for (long long r = 0; r < R4D_NTILE; ++r)
            for (long long gi = 0; gi < nsz; ++gi) {
                const long long n = t * R4D_NTILE + r;
                const uint32_t d = sz[(t * nsz + gi) * R4D_NTILE + r];
                ((uint16_t*)pl[0].data)[n * nsz + gi] = (uint16_t)(d & 0xFFFFu);
                if (asym) {
                    const int z = (int)(-f16_to_f32((uint16_t)(d >> 16)) - 1024.0f + 0.5f);
                    rad_store_code(pl[1].data, pl[1].dtype, n * nsz + gi, z);
                }
            }
}

/* A ROTATED asymmetric grid (ParoQuant): codes of `codes` dtype, an f16 scale and a u8 zero point a
 * group, then a `givens` transform and its tables -- the same bytes as the unrotated grid, the
 * rotation being the activation's. */
static bool paro_asym_enc(const RadEncoding* e, uint32_t codes, long long group) {
    if (!e || !rad_enc_is(e, "affine") || e->n_planes < 3) return false;
    const bool giv = std::strncmp(e->transform, "givens", RAD_ENC_STR) == 0;
    if (!giv && e->transform[0]) return false;
    if (!giv && e->n_planes != 3) return false;
    if (giv && (!rad_enc_valid(e) || rad_enc_find(e, "t.pairs") < 0)) return false;
    if (std::strncmp(e->plane[0].role, "codes", RAD_ENC_STR) != 0 || e->plane[0].dtype != codes ||
        e->plane[0].block[0] != 1 || e->plane[0].block[1] != 1) return false;
    const int s = rad_enc_find(e, "scale"), z = rad_enc_find(e, "zero");
    if (s != 1 || z != 2) return false;
    return e->plane[s].dtype == RAD_F16 && e->plane[s].kind == RAD_PLANE_TILED &&
           e->plane[s].block[0] == 1 && e->plane[s].block[1] == group &&
           (e->plane[z].dtype == RAD_U8 || e->plane[z].dtype == codes) &&
           e->plane[z].kind == RAD_PLANE_TILED && e->plane[z].block[0] == 1 &&
           e->plane[z].block[1] == group;
}

/* ---- w4: a symmetric grid is i4 codes with an f16 scale; `_asym` is u4 codes with a zero. */
int w4_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, RadLayout* out, WGeom* gout) {
    if (!out || !pl) return RAD_E_INVAL;
    WGeom g;
    int rc = wgeom(p, n_p, R4D_W4_GROUP, &g);
    if (rc != RAD_OK) return rc;
    if (g.K % R4D_W4_KPB) return RAD_E_SHAPE;
    const bool asym = w4_asym(p, n_p);
    const bool ok = asym ? (r4d_enc::affine(enc, RAD_U4, RAD_F16, 1, g.group, RAD_U8) ||
                            r4d_enc::affine(enc, RAD_U4, RAD_F16, 1, g.group, RAD_U4) ||
                            paro_asym_enc(enc, RAD_U4, g.group))
                         : r4d_enc::affine(enc, RAD_I4, RAD_F16, 1, g.group);
    if (!ok) return RAD_E_DTYPE;
    if (gout) *gout = g;
    if (operand == Q_BSCALE)
        return sz_layout(g, enc, sel, pl, n, asym, asym ? "r4d.w4u.sz.g128" : "r4d.w4.sz.g128",
                         R4D_DT_W4_SZ, 4, out);
    if (operand != Q_B) return RAD_E_UNSUPPORTED;
    if (!selects(enc, sel, n, "codes") || !dims(pl[0], g.N, g.K)) return RAD_E_SHAPE;
    fill(out, asym ? "r4d.w4u.frag.g128" : "r4d.w4.frag.g128", R4D_DT_W4_FRAG, g.N, g.K,
         g.N * g.K / 2);
    return RAD_OK;
}

/* Where element (n, k) of a w4 plane lands: the 32-bit word and the nibble within it. */
void w4_slot(const WGeom& g, long long n, long long k, long long* word, int* nib) {
    const long long kpb = g.K / R4D_W4_KPB;
    const long long t = n / R4D_NTILE, r = n % R4D_NTILE;
    const long long kb = k / R4D_W4_KPB, kin = k % R4D_W4_KPB;
    const long long s = kin / 16, kk = kin % 16;
    const long long lh = (kk % 8) / 4, e = (kk / 8) * 4 + (kk % 4);
    *nib = (int)((e < 4) ? (2 * e) : (2 * (e - 4) + 1));
    *word = ((t * kpb + kb) * R4D_WAVE + lh * 16 + r) * 4 + s;
}

}  /* namespace */

extern "C" int r4d_rad_layout_w4(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                     int n, RadLayout* out) {
    return w4_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr);
}

extern "C" int r4d_rad_relayout_w4(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                   const int* sel, const RadTensor* pl, int n, void* dst,
                                   int64_t dst_bytes) {
    RadLayout L;
    WGeom g;
    const int rc = w4_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes) return RAD_E_SHAPE;
    std::memset(dst, 0, (size_t)dst_bytes);
    if (operand == Q_BSCALE) {
        /* a ROTATED (ParoQuant) grid's zero is a 4-bit code */
        if (w4_asym(p, n_p) && std::strncmp(enc->transform, "givens", RAD_ENC_STR) == 0 &&
            !sz_zeros_fit(g, pl[1], 4)) return RAD_E_FORMAT;
        sz_write(g, pl, w4_asym(p, n_p) ? -1 : 8, (uint32_t*)dst);
        return RAD_OK;
    }
    uint32_t* wq = (uint32_t*)dst;
    for (long long nn = 0; nn < g.N; ++nn) {
        const uint8_t* row = pdata(pl[0]) + nn * (g.K / 2);
        for (long long k = 0; k < g.K; ++k) {
            long long w;
            int nib;
            w4_slot(g, nn, k, &w, &nib);
            wq[w] |= (uint32_t)((row[k >> 1] >> (4 * (k & 1))) & 0xF) << (4 * nib);
        }
    }
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_w4(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel, const void* src,
                                     int64_t src_bytes, const RadTensor* pl, int n) {
    RadLayout L;
    WGeom g;
    const int rc = w4_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes) return RAD_E_SHAPE;
    if (operand == Q_BSCALE) {
        sz_read(g, (const uint32_t*)src, pl, w4_asym(p, n_p));
        return RAD_OK;
    }
    const uint32_t* wq = (const uint32_t*)src;
    uint8_t* d = (uint8_t*)pl[0].data;
    std::memset(d, 0, (size_t)(g.N * g.K / 2));
    for (long long nn = 0; nn < g.N; ++nn)
        for (long long k = 0; k < g.K; ++k) {
            long long w;
            int nib;
            w4_slot(g, nn, k, &w, &nib);
            const uint32_t q = (wq[w] >> (4 * nib)) & 0xF;
            d[nn * (g.K / 2) + (k >> 1)] |= (uint8_t)(q << (4 * (k & 1)));
        }
    return RAD_OK;
}

/* ---- w5: ParoQuant's int5, asymmetric: u5 codes, an f16 scale and a u8 zero point (0..31) a 128 of K.
 * The blob is w4's fragment order for the low four bits of every code (offset binary, no sign flip)
 * followed by a plane of 32-bit words holding the fifth bits, one a (n-tile, 64 of K, lane): byte s
 * is k step s of the block and bit e the fragment element e -- the element whose nibble is
 * 2e (e < 4) or 2(e-4)+1 in dword s of that lane's 16 bytes (r4d_gemm_w5a16_nt_m64.hip). The scale
 * operand is w4u's (scale, zero) dword plane. A ROTATED weight (a `givens` transform and its
 * tables beside the first three planes) is the same bytes: the rotation is the activation's. */
namespace {
bool w5_enc(const RadEncoding* e, long long group) { return paro_asym_enc(e, RAD_U5, group); }

int w5_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, RadLayout* out, WGeom* gout) {
    if (!out || !pl) return RAD_E_INVAL;
    WGeom g;
    int rc = wgeom(p, n_p, R4D_W4_GROUP, &g);
    if (rc != RAD_OK) return rc;
    if (g.K % R4D_W4_KPB) return RAD_E_SHAPE;
    if (!w5_enc(enc, g.group)) return RAD_E_DTYPE;
    if (gout) *gout = g;
    if (operand == Q_BSCALE)
        return sz_layout(g, enc, sel, pl, n, true, "r4d.w5.sz.g128", R4D_DT_W4_SZ, 4, out);
    if (operand != Q_B) return RAD_E_UNSUPPORTED;
    if (!selects(enc, sel, n, "codes") || !dims(pl[0], g.N, g.K)) return RAD_E_SHAPE;
    fill(out, "r4d.w5.frag.g128", R4D_DT_W4_FRAG, g.N, g.K, g.N * g.K * 5 / 8);
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_w5(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                 const int* sel, const RadTensor* pl, int n, RadLayout* out) {
    return w5_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr);
}

extern "C" int r4d_rad_relayout_w5(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                   const int* sel, const RadTensor* pl, int n, void* dst,
                                   int64_t dst_bytes) {
    RadLayout L;
    WGeom g;
    const int rc = w5_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes) return RAD_E_SHAPE;
    std::memset(dst, 0, (size_t)dst_bytes);
    if (operand == Q_BSCALE) {
        if (!sz_zeros_fit(g, pl[1], 5)) return RAD_E_FORMAT;
        sz_write(g, pl, -1, (uint32_t*)dst);
        return RAD_OK;
    }
    uint32_t* wq = (uint32_t*)dst;
    uint32_t* hi = wq + g.N * g.K / 8;                    /* after the N*K/2 bytes of nibbles */
    const long long rb = (g.K * 5 + 7) / 8;
    for (long long nn = 0; nn < g.N; ++nn) {
        const uint8_t* row = pdata(pl[0]) + nn * rb;
        for (long long k = 0; k < g.K; ++k) {
            const uint32_t q = rad_load_bits(row, 5, k);
            long long w;
            int nib;
            w4_slot(g, nn, k, &w, &nib);
            wq[w] |= (q & 15u) << (4 * nib);
            /* the fifth bit: word (t, kb, lane) = w / 4 (w4_slot's dword index is that * 4 + s);
             * byte s = w % 4, bit e from the nibble position */
            const long long hw = w / 4;
            const int s = (int)(w % 4);
            const int e = nib % 2 == 0 ? nib / 2 : 4 + nib / 2;     /* nibble 2e / 2(e-4)+1 -> e */
            hi[hw] |= (q >> 4) << (8 * s + e);
        }
    }
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_w5(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                     const int* sel, const void* src, int64_t src_bytes,
                                     const RadTensor* pl, int n) {
    RadLayout L;
    WGeom g;
    const int rc = w5_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes) return RAD_E_SHAPE;
    if (operand == Q_BSCALE) {
        sz_read(g, (const uint32_t*)src, pl, true);
        return RAD_OK;
    }
    const uint32_t* wq = (const uint32_t*)src;
    const uint32_t* hi = wq + g.N * g.K / 8;
    const long long rb = (g.K * 5 + 7) / 8;
    uint8_t* d = (uint8_t*)pl[0].data;
    std::memset(d, 0, (size_t)(g.N * rb));
    for (long long nn = 0; nn < g.N; ++nn)
        for (long long k = 0; k < g.K; ++k) {
            long long w;
            int nib;
            w4_slot(g, nn, k, &w, &nib);
            const int e = nib % 2 == 0 ? nib / 2 : 4 + nib / 2;     /* as r4d_rad_relayout_w5 */
            const uint32_t q = ((wq[w] >> (4 * nib)) & 15u) | (((hi[w / 4] >> (8 * (w % 4) + e)) & 1u) << 4);
            rad_store_bits(d + nn * rb, 5, k, q);
        }
    return RAD_OK;
}

/* ---- givens_dense: ParoQuant's rotation tables composed into one dense matrix a (part, group).
 *
 * A "givens" weight stores its rotation as pairs (i32 [P*S, K]) and angles (f32 [P*S, K/2]); the dense
 * kernel (r4d_givens_dense.hip) applies it as T (x / D), T being the composition of the group's S
 * stages.  The blob is [P * K/128][128 (j)][128 (i)] f16, T[j][i] = output j of the network fed the unit
 * vector e_i, so a (part, group) is 32 KiB the kernel stages whole.  Composed in f32 in the order the
 * stages are applied (stage 0 first), a' = a c + b s, b' = b c - a s -- libref's loop -- and rounded to
 * f16 once.  Operand 1 of givens_dense_quant; everything else stays as stored. */
namespace {
constexpr int GD_OPERAND = 1;
constexpr long long GD_GROUP = 128;

int gd_geom(const RadParam* p, int n_p, const RadEncoding* enc, const int* sel, const RadTensor* pl, int n,
            long long* P, long long* S, long long* K) {
    if (n != 2 || std::strncmp(r4d_enc::role(enc, sel, 0), "t.pairs", RAD_ENC_STR) != 0 ||
        std::strncmp(r4d_enc::role(enc, sel, 1), "t.angle", RAD_ENC_STR) != 0) return RAD_E_SHAPE;
    *K = r4d_pgeti_or(p, n_p, "n", 0);
    *P = r4d_pgeti_or(p, n_p, "parts", 1);
    if (*K <= 0 || *K % GD_GROUP || *P < 1) return RAD_E_SHAPE;
    if (pl[0].rank != 2 || pl[0].shape[1] != *K || pl[0].shape[0] % *P || pl[0].dtype != RAD_I32)
        return RAD_E_SHAPE;
    *S = pl[0].shape[0] / *P;
    if (*S < 1 || !dims(pl[1], *P * *S, *K / 2) || pl[1].dtype != RAD_F32) return RAD_E_SHAPE;
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_gdense(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                     const int* sel, const RadTensor* pl, int n, RadLayout* out) {
    if (!out || !pl) return RAD_E_INVAL;
    if (operand != GD_OPERAND) return RAD_E_UNSUPPORTED;
    long long P, S, K;
    const int rc = gd_geom(p, n_p, enc, sel, pl, n, &P, &S, &K);
    if (rc != RAD_OK) return rc;
    fill(out, "r4d.givens.dense.g128", RAD_F16, P * K, GD_GROUP, P * K * GD_GROUP * 2);
    return RAD_OK;
}

extern "C" int r4d_rad_relayout_gdense(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                       const int* sel, const RadTensor* pl, int n, void* dst,
                                       int64_t dst_bytes) {
    RadLayout L;
    int rc = r4d_rad_layout_gdense(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes) return RAD_E_SHAPE;
    long long P, S, K;
    rc = gd_geom(p, n_p, enc, sel, pl, n, &P, &S, &K);
    if (rc != RAD_OK) return rc;
    const int32_t* pairs = (const int32_t*)pdata(pl[0]);
    const float* ang = (const float*)pdata(pl[1]);
    uint16_t* t = (uint16_t*)dst;
    const long long nkb = K / GD_GROUP, half = GD_GROUP / 2;
    std::vector<int> ia((size_t)(S * half)), ib((size_t)(S * half));
    std::vector<float> cs((size_t)(S * half)), sn((size_t)(S * half));
    std::vector<float> v((size_t)GD_GROUP);
    for (long long pt = 0; pt < P; ++pt)
        for (long long g = 0; g < nkb; ++g) {
            for (long long st = 0; st < S; ++st)
                for (long long j = 0; j < half; ++j) {
                    const long long row = pt * S + st, o = (size_t)(st * half + j);
                    const long long a = pairs[row * K + g * GD_GROUP + 2 * j] - g * GD_GROUP;
                    const long long b = pairs[row * K + g * GD_GROUP + 2 * j + 1] - g * GD_GROUP;
                    if (a < 0 || a >= GD_GROUP || b < 0 || b >= GD_GROUP || a == b) return RAD_E_FORMAT;
                    ia[(size_t)o] = (int)a;
                    ib[(size_t)o] = (int)b;
                    const float th = ang[row * (K / 2) + g * half + j];
                    cs[(size_t)o] = std::cos(th);
                    sn[(size_t)o] = std::sin(th);
                }
            uint16_t* tg = t + (size_t)(pt * nkb + g) * GD_GROUP * GD_GROUP;
            for (long long i = 0; i < GD_GROUP; ++i) {
                std::fill(v.begin(), v.end(), 0.0f);
                v[(size_t)i] = 1.0f;
                for (long long o = 0; o < S * half; ++o) {
                    const float va = v[(size_t)ia[(size_t)o]], vb = v[(size_t)ib[(size_t)o]];
                    v[(size_t)ia[(size_t)o]] = va * cs[(size_t)o] + vb * sn[(size_t)o];
                    v[(size_t)ib[(size_t)o]] = vb * cs[(size_t)o] - va * sn[(size_t)o];
                }
                for (long long j = 0; j < GD_GROUP; ++j) tg[j * GD_GROUP + i] = f32_to_f16(v[(size_t)j]);
            }
        }
    return RAD_OK;
}

/* ---- w8: i8 codes, symmetric about zero, an f16 scale a group. The f16 kernel (a16) reads the
 * code offset binary with zero 128 in the shared dword; the int8 kernel (a8) reads it signed with a
 * bare f16 scale plane. */
namespace {
int w8_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, RadLayout* out, WGeom* gout, int a16) {
    if (!out || !pl) return RAD_E_INVAL;
    WGeom g;
    int rc = wgeom(p, n_p, R4D_W8_GROUP, &g);
    if (rc != RAD_OK) return rc;
    if (g.K % R4D_W8_KPB) return RAD_E_SHAPE;
    if (!r4d_enc::affine(enc, RAD_I8, RAD_F16, 1, g.group)) return RAD_E_DTYPE;
    if (gout) *gout = g;
    if (operand == Q_BSCALE)
        return a16 ? sz_layout(g, enc, sel, pl, n, false, "r4d.w8a16.sz.g128", R4D_DT_W4_SZ, 4, out)
                   : sz_layout(g, enc, sel, pl, n, false, "r4d.w8a8.s.g128", R4D_DT_W8_S, 2, out);
    if (operand != Q_B) return RAD_E_UNSUPPORTED;
    if (!selects(enc, sel, n, "codes") || !dims(pl[0], g.N, g.K)) return RAD_E_SHAPE;
    fill(out, a16 ? "r4d.w8a16.frag.g128" : "r4d.w8a8.frag.g128", R4D_DT_W8_FRAG, g.N, g.K,
         g.N * g.K);
    return RAD_OK;
}

long long w8_slot(const WGeom& g, long long n, long long k) {
    const long long kpb = g.K / R4D_W8_KPB;
    const long long t = n / R4D_NTILE, r = n % R4D_NTILE;
    const long long kb = k / R4D_W8_KPB, kin = k % R4D_W8_KPB;
    const long long s = kin / 16, kk = kin % 16;
    const long long lh = (kk % 8) / 4, e = (kk / 8) * 4 + (kk % 4);
    return ((t * kpb + kb) * R4D_WAVE + lh * 16 + r) * 16 + s * 8 + e;
}

int w8_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                const RadTensor* pl, int n, void* dst, int64_t dst_bytes, int a16) {
    RadLayout L;
    WGeom g;
    const int rc = w8_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g, a16);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes) return RAD_E_SHAPE;
    std::memset(dst, 0, (size_t)dst_bytes);
    if (operand == Q_BSCALE) {
        if (a16) {
            sz_write(g, pl, 128, (uint32_t*)dst);
        } else {
            const long long nt = g.N / R4D_NTILE, nsz = g.K / g.group;
            uint16_t* s16 = (uint16_t*)dst;
            for (long long t = 0; t < nt; ++t)
                for (long long r = 0; r < R4D_NTILE; ++r)
                    for (long long gi = 0; gi < nsz; ++gi)
                        s16[(t * nsz + gi) * R4D_NTILE + r] =
                            ((const uint16_t*)pl[0].data)[(t * R4D_NTILE + r) * nsz + gi];
        }
        return RAD_OK;
    }
    uint8_t* wq = (uint8_t*)dst;
    for (long long nn = 0; nn < g.N; ++nn)
        for (long long k = 0; k < g.K; ++k) {
            const int8_t q = ((const int8_t*)pl[0].data)[nn * g.K + k];
            wq[w8_slot(g, nn, k)] = a16 ? (uint8_t)(q + 128) : (uint8_t)q;
        }
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_w8a16(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel,
                                        const RadTensor* pl, int n, RadLayout* out) {
    return w8_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr, 1);
}
extern "C" int r4d_rad_layout_w8a8(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel,
                                       const RadTensor* pl, int n, RadLayout* out) {
    return w8_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr, 0);
}
extern "C" int r4d_rad_relayout_w8a16(const RadParam* p, int n_p, int operand,
                                      const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                      int n, void* dst, int64_t dst_bytes) {
    return w8_relayout(p, n_p, operand, enc, sel, pl, n, dst, dst_bytes, 1);
}
extern "C" int r4d_rad_relayout_w8a8(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                     int n, void* dst, int64_t dst_bytes) {
    return w8_relayout(p, n_p, operand, enc, sel, pl, n, dst, dst_bytes, 0);
}

/* ---- w2a8: u2 codes, asymmetric, an f16 scale and an integer zero a group, both streams
 * tile-major. */
namespace {
int w2_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, RadLayout* out, WGeom* gout) {
    if (!out || !pl) return RAD_E_INVAL;
    WGeom g;
    int rc = wgeom(p, n_p, R4D_W2_GROUP, &g);
    if (rc != RAD_OK) return rc;
    if (g.K % 128) return RAD_E_SHAPE;
    if (!(r4d_enc::affine(enc, RAD_U2, RAD_F16, 1, g.group, RAD_U8) ||
          r4d_enc::affine(enc, RAD_U2, RAD_F16, 1, g.group, RAD_U4))) return RAD_E_DTYPE;
    if (gout) *gout = g;
    if (operand == Q_BSCALE)
        return sz_layout(g, enc, sel, pl, n, true, "r4d.w2a8.sz.g128", R4D_DT_W4_SZ, 4, out);
    if (operand != Q_B) return RAD_E_UNSUPPORTED;
    if (!selects(enc, sel, n, "codes") || !dims(pl[0], g.N, g.K)) return RAD_E_SHAPE;
    fill(out, "r4d.w2a8.frag.g128", R4D_DT_W2_FRAG, g.N, g.K, g.N * g.K / 4);
    return RAD_OK;
}

void w2_slot(const WGeom& g, long long n, long long k, long long* word, int* field) {
    const long long kpb = g.K / 128;
    const long long t = n / R4D_NTILE, r = n % R4D_NTILE;
    const long long kb = k / 128, kin = k % 128;
    const long long step = kin / 16, d = step / 2, h = step % 2;
    const long long kk = kin % 16, lh = (kk % 8) / 4, e = (kk / 8) * 4 + (kk % 4);
    *field = (int)(4 * (e % 4) + (e / 4) + 2 * h);
    *word = ((t * kpb + kb) * R4D_WAVE + lh * 16 + r) * 4 + d;
}
}  /* namespace */

extern "C" int r4d_rad_layout_w2a8(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel,
                                       const RadTensor* pl, int n, RadLayout* out) {
    return w2_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr);
}

extern "C" int r4d_rad_relayout_w2a8(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                     int n, void* dst, int64_t dst_bytes) {
    RadLayout L;
    WGeom g;
    const int rc = w2_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes) return RAD_E_SHAPE;
    std::memset(dst, 0, (size_t)dst_bytes);
    if (operand == Q_BSCALE) {
        sz_write(g, pl, -1, (uint32_t*)dst);
        return RAD_OK;
    }
    uint32_t* wq = (uint32_t*)dst;
    for (long long nn = 0; nn < g.N; ++nn) {
        const uint8_t* row = pdata(pl[0]) + nn * (g.K / 4);
        for (long long k = 0; k < g.K; ++k) {
            long long w;
            int f;
            w2_slot(g, nn, k, &w, &f);
            wq[w] |= (uint32_t)((row[k >> 2] >> (2 * (k & 3))) & 3) << (2 * f);
        }
    }
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_w2a8(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel, const void* src,
                                       int64_t src_bytes, const RadTensor* pl, int n) {
    RadLayout L;
    WGeom g;
    const int rc = w2_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes) return RAD_E_SHAPE;
    if (operand == Q_BSCALE) {
        sz_read(g, (const uint32_t*)src, pl, true);
        return RAD_OK;
    }
    const uint32_t* wq = (const uint32_t*)src;
    uint8_t* d = (uint8_t*)pl[0].data;
    std::memset(d, 0, (size_t)(g.N * g.K / 4));
    for (long long nn = 0; nn < g.N; ++nn)
        for (long long k = 0; k < g.K; ++k) {
            long long w;
            int f;
            w2_slot(g, nn, k, &w, &f);
            d[nn * (g.K / 4) + (k >> 2)] |= (uint8_t)(((wq[w] >> (2 * f)) & 3) << (2 * (k & 3)));
        }
    return RAD_OK;
}

/* ---- givens pairs: ParoQuant's rotation pairs (i32 [P*S, K]), stored as they are but CHECKED.
 * The rotate kernels index a group's 128 LDS slots with them unchecked, so a stage that is not a
 * pairing of its own group -- an index outside the group, or one column twice -- is refused here,
 * at load, rather than read out of bounds. Operand 1 of the givens rows; the rest is as stored. */
namespace {
constexpr int GV_PAIRS_OPERAND = 1;
constexpr long long GV_GROUP = 128;

int gv_pairs_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                    const RadTensor* pl, int n, RadLayout* out) {
    if (!out || !pl) return RAD_E_INVAL;
    if (operand != GV_PAIRS_OPERAND) return RAD_E_UNSUPPORTED;
    const long long K = r4d_pgeti_or(p, n_p, "n", 0);
    if (K <= 0 || K % GV_GROUP || !selects(enc, sel, n, "t.pairs") || pl[0].rank != 2 ||
        pl[0].dtype != RAD_I32 || pl[0].shape[1] != K) return RAD_E_SHAPE;
    fill(out, "r4d.givens.pairs", RAD_I32, pl[0].shape[0], K, pl[0].shape[0] * K * 4);
    return RAD_OK;
}

/* Every row (a stage of a part) is, group by group, a perfect matching of the group's columns. */
bool gv_pairs_ok(const int32_t* pairs, long long rows, long long K) {
    for (long long r = 0; r < rows; ++r)
        for (long long g = 0; g < K; g += GV_GROUP) {
            bool seen[GV_GROUP] = {};
            for (long long i = 0; i < GV_GROUP; ++i) {
                const long long c = (long long)pairs[r * K + g + i] - g;
                if (c < 0 || c >= GV_GROUP || seen[c]) return false;
                seen[c] = true;
            }
        }
    return true;
}
}  /* namespace */

extern "C" int r4d_rad_layout_givens_pairs(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const RadTensor* pl, int n, RadLayout* out) {
    return gv_pairs_layout(p, n_p, operand, enc, sel, pl, n, out);
}

extern "C" int r4d_rad_relayout_givens_pairs(const RadParam* p, int n_p, int operand,
                                             const RadEncoding* enc, const int* sel,
                                             const RadTensor* pl, int n, void* dst,
                                             int64_t dst_bytes) {
    RadLayout L;
    const int rc = gv_pairs_layout(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    if (!gv_pairs_ok((const int32_t*)pl[0].data, L.shape[0], L.shape[1])) return RAD_E_FORMAT;
    std::memcpy(dst, pl[0].data, (size_t)L.bytes);
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_givens_pairs(const RadParam* p, int n_p, int operand,
                                               const RadEncoding* enc, const int* sel,
                                               const void* src, int64_t src_bytes,
                                               const RadTensor* pl, int n) {
    RadLayout L;
    const int rc = gv_pairs_layout(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    std::memcpy(pl[0].data, src, (size_t)L.bytes);
    return RAD_OK;
}

/* ---- MXFP4: fp4 e2m1 codes with an E8M0 exponent a 32. The codes go to fragment order, the
 * exponents to [K/32][N], and b_ref is each row's largest exponent -- a reduction over the plane,
 * which is a derivation any reader of the plane could make and so not a choice. */
namespace {
/* A ROTATED MXFP4 WEIGHT (ParoQuant): the same two planes -- e2m1 codes, an E8M0 exponent a 32 --
 * stored in the basis the "givens" transform names, with the rotation's tables beside them. The
 * GEMM reads the codes and exponents exactly as it reads an unrotated weight; the rotation is the
 * activation's (givens_quant_act_fp8), and the tables are separate weight operands of that op, so
 * nothing here touches them. What is checked is that they are there and are the kind of
 * rotation the quantiser applies: a transform this library does not read would hand the kernel
 * codes that decode to other numbers. */
bool mx_paro_enc(const RadEncoding* e, long long K) {
    if (!e || !rad_enc_is(e, "affine") || std::strncmp(e->transform, "givens", RAD_ENC_STR) != 0) return false;
    if (e->n_planes < 5 || !rad_enc_valid(e)) return false;
    if (std::strncmp(e->plane[0].role, "codes", RAD_ENC_STR) != 0 || e->plane[0].dtype != RAD_FP4E2M1 ||
        e->plane[0].block[0] != 1 || e->plane[0].block[1] != 1) return false;
    const int s = rad_enc_find(e, "scale");
    if (s != 1 || e->plane[s].dtype != RAD_E8M0 || e->plane[s].kind != RAD_PLANE_TILED ||
        e->plane[s].block[0] != 1 || e->plane[s].block[1] != R4D_MXFP4_GROUP) return false;
    const int p = rad_enc_find(e, "t.pairs"), an = rad_enc_find(e, "t.angle"), sc = rad_enc_find(e, "t.scale");
    if (p < 0 || an < 0 || sc < 0) return false;
    if (e->plane[p].dtype != RAD_I32 || e->plane[an].dtype != RAD_F32 || e->plane[sc].dtype != RAD_F32)
        return false;
    // the quantiser rotates whole groups of 128 columns, one scale per group
    return K % 128 == 0 && e->plane[p].extent[1] == K;
}

int mx_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, RadLayout* out, WGeom* gout) {
    if (!out || !pl) return RAD_E_INVAL;
    WGeom g;
    int rc = wgeom(p, n_p, R4D_MXFP4_GROUP, &g);
    if (rc != RAD_OK) return rc;
    if (g.group != R4D_MXFP4_GROUP || g.K % 16) return RAD_E_SHAPE;
    if (!r4d_enc::affine(enc, RAD_FP4E2M1, RAD_E8M0, 1, R4D_MXFP4_GROUP) &&
        !mx_paro_enc(enc, g.K)) return RAD_E_DTYPE;
    if (gout) *gout = g;
    const long long nblk = g.K / R4D_MXFP4_GROUP;
    if (operand == Q_BSCALE || operand == Q_BREF) {
        if (!selects(enc, sel, n, "scale") || !dims(pl[0], g.N, nblk)) return RAD_E_SHAPE;
        if (operand == Q_BSCALE) fill(out, "r4d.mxfp4.e8m0", R4D_DT_E8M0, nblk, g.N, nblk * g.N);
        else                     fill(out, "r4d.mxfp4.wref", R4D_DT_E8M0, g.N, 1, g.N);
        return RAD_OK;
    }
    if (operand != Q_B) return RAD_E_UNSUPPORTED;
    if (!selects(enc, sel, n, "codes") || !dims(pl[0], g.N, g.K)) return RAD_E_SHAPE;
    fill(out, "r4d.mxfp4.frag.g32", R4D_DT_MXFP4_FRAG, g.N, g.K, g.N * g.K / 2);
    return RAD_OK;
}

long long mx_slot(const WGeom& g, long long n, long long k) {
    const long long ks = g.K / 16;
    const long long t = n / R4D_NTILE, r = n % R4D_NTILE;
    const long long ksi = k / 16, kin = k % 16;
    const long long lh = kin / 8, bin = (kin % 8) / 2;
    return ((t * ks + ksi) * R4D_WAVE + lh * 16 + r) * 4 + bin;
}
}  /* namespace */

extern "C" int r4d_rad_layout_mxfp4(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel,
                                        const RadTensor* pl, int n, RadLayout* out) {
    return mx_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr);
}

extern "C" int r4d_rad_relayout_mxfp4(const RadParam* p, int n_p, int operand,
                                      const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                      int n, void* dst, int64_t dst_bytes) {
    RadLayout L;
    WGeom g;
    const int rc = mx_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes) return RAD_E_SHAPE;
    std::memset(dst, 0, (size_t)dst_bytes);
    const long long nblk = g.K / R4D_MXFP4_GROUP;
    uint8_t* d = (uint8_t*)dst;
    if (operand == Q_BSCALE || operand == Q_BREF) {
        const uint8_t* e = pdata(pl[0]);
        for (long long nn = 0; nn < g.N; ++nn) {
            uint8_t mx = 0;
            for (long long b = 0; b < nblk; ++b) {
                const uint8_t v = e[nn * nblk + b];
                if (operand == Q_BSCALE) d[b * g.N + nn] = v;
                if (v > mx) mx = v;
            }
            if (operand == Q_BREF) d[nn] = mx;
        }
        return RAD_OK;
    }
    for (long long nn = 0; nn < g.N; ++nn) {
        const uint8_t* row = pdata(pl[0]) + nn * (g.K / 2);
        for (long long k = 0; k < g.K; ++k) {
            const uint8_t code = (uint8_t)((row[k >> 1] >> (4 * (k & 1))) & 0xF);
            const long long idx = mx_slot(g, nn, k);
            if (k & 1) d[idx] = (uint8_t)((d[idx] & 0x0Fu) | (uint32_t)(code << 4));
            else       d[idx] = (uint8_t)((d[idx] & 0xF0u) | code);
        }
    }
    return RAD_OK;
}

/* ---- the MoE experts. Their weight is staged through LDS, so it is not permuted: the int4 plane
 * is the canonical one with each dword's nibbles reordered for the expansion (byte b holds element
 * b low and element b+4 high), and the fp8 plane is the canonical one exactly. `w4a8h` is the same
 * plane with the weight rotated by the group's Hadamard, and only an encoding that says so is
 * accepted for it. */
namespace {
int moew4_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                     const int* sel, const RadTensor* pl, int n, RadLayout* out, WGeom* gout) {
    if (!out || !pl) return RAD_E_INVAL;
    WGeom g;
    const int rc = wgeom(p, n_p, R4D_W4_GROUP, &g);
    if (rc != RAD_OK) return rc;
    if (g.K % 8) return RAD_E_SHAPE;
    char fw[RAD_ENC_STR] = "";
    if (moe_w4_rot(p, n_p)) std::snprintf(fw, sizeof fw, "fwht%lld", g.group);
    const bool nl = moe_w4_nl(p, n_p);
    const bool w5 = moe_w5(p, n_p);
    const int  g64 = moe_w4_g64(p, n_p);             /* scale groups a 128 of K, or 0 */
    if (w5 && g.K % 32) return RAD_E_SHAPE;           /* a sign word is 32 elements */
    if (g64 ? !r4d_enc::affine_table2(enc, w5 ? RAD_U5 : RAD_U4, RAD_F8E4M3, 1, g.group / g64,
                                      RAD_F32, RAD_F32, w5 ? 32 : 16, fw)
       : nl ? !r4d_enc::affine_table(enc, RAD_U4, RAD_BF16, 1, g.group, RAD_F32, 16, fw)
            : !r4d_enc::affine(enc, RAD_I4, RAD_BF16, 1, g.group, RAD_DT_INVALID, fw))
        return RAD_E_DTYPE;
    if (gout) *gout = g;
    if (operand == Q_BSCALE) {
        if (g64) {
            if (!selects(enc, sel, n, "scale", "scale.1") ||
                !dims(pl[0], g.N, g64 * g.K / g.group) || !dims(pl[1], 1, 1))
                return RAD_E_SHAPE;
            fill(out, g64 == 4 ? "r4d.moew4nh32.s" : "r4d.moew4nh64.s", RAD_F8E4M3, g.N,
                 g64 * g.K / g.group, g.N * g64 * g.K / g.group);
            return RAD_OK;
        }
        if (!selects(enc, sel, n, "scale") || !dims(pl[0], g.N, g.K / g.group)) return RAD_E_SHAPE;
        return RAD_E_UNSUPPORTED;                    /* bf16 a row a group, as it is */
    }
    if (operand != Q_B) return RAD_E_UNSUPPORTED;
    if (nl ? !selects(enc, sel, n, "codes", "table") || !dims(pl[1], 1, w5 ? 32 : 16)
           : !selects(enc, sel, n, "codes"))
        return RAD_E_SHAPE;
    if (!dims(pl[0], g.N, g.K)) return RAD_E_SHAPE;
    if (w5) {
        fill(out, "r4d.moew5nh64.w", R4D_DT_MOEW5, g.N, g.K, g.N * (g.K / 8) * 5);
        return RAD_OK;
    }
    fill(out, g64 == 4 ? "r4d.moew4nh32.w" : g64 ? "r4d.moew4nh64.w" : nl ? "r4d.moew4nh.w.g128"
                    : moe_w4_rot(p, n_p) ? "r4d.moew4h.w.g128" : "r4d.moew4.w.g128",
         R4D_DT_MOEW4, g.N, g.K, g.N * g.K / 2);
    return RAD_OK;
}

/* A w4nla8h weight's table plane against the one the kernel decodes, value for value. A mismatch
 * is RAD_E_DTYPE, which the loader reports with the weight's name, its encoding and the tag. */
bool moew4_table_is_kernels(const RadTensor& t) {
    const float* v = (const float*)t.data;
    for (int i = 0; i < 16; ++i)
        if (v[i] != kW4nlTable[i]) return false;
    return true;
}
/* And w5nl64a8h's against kW5nlTable. */
bool moew5_table_is_kernels(const RadTensor& t) {
    const float* v = (const float*)t.data;
    for (int i = 0; i < 32; ++i)
        if (v[i] != kW5nlTable[i]) return false;
    return true;
}

/* THE EXPANSION'S NIBBLE ORDER, two dwords a word. Canonical element 8g+i is nibble i of dword g;
 * stored, byte b holds element b low and b+4 high -- nibbles n0 n4 n1 n5 n2 n6 n3 n7, a perfect
 * shuffle of the dword's halves. Swapping the middle bytes and then each half's middle nibbles
 * makes it; each step exchanges two fields, so the same steps in the other order undo it. Done
 * a nibble at a time, with the read-modify-write that implies, the same permutation costs most of
 * a Flash-Next load's CPU: every routed expert goes through it. */
inline uint64_t moew4_bytes(uint64_t x) {
    return (x & 0xFF0000FFFF0000FFull) | ((x & 0x0000FF000000FF00ull) << 8) |
           ((x >> 8) & 0x0000FF000000FF00ull);
}
inline uint64_t moew4_nibbles(uint64_t x) {
    return (x & 0xF00FF00FF00FF00Full) | ((x & 0x00F000F000F000F0ull) << 4) |
           ((x >> 4) & 0x00F000F000F000F0ull);
}

/* `bytes` of codes, a multiple of four, from `s` to `d` -- forward or back. */
void moew4_permute(const uint8_t* s, uint8_t* d, int64_t bytes, bool forward) {
    int64_t i = 0;
    for (; i + 8 <= bytes; i += 8) {
        uint64_t x;
        std::memcpy(&x, s + i, 8);
        x = forward ? moew4_nibbles(moew4_bytes(x)) : moew4_bytes(moew4_nibbles(x));
        std::memcpy(d + i, &x, 8);
    }
    if (i < bytes) {                                 /* one dword left; no step crosses dwords */
        uint32_t w;
        std::memcpy(&w, s + i, 4);
        uint64_t x = w;
        x = forward ? moew4_nibbles(moew4_bytes(x)) : moew4_bytes(moew4_nibbles(x));
        w = (uint32_t)x;
        std::memcpy(d + i, &w, 4);
    }
}
/* ONE w5nl64a8h ROW BETWEEN THE CONTAINER AND THE KERNEL. Stored, a row is K five-bit codes,
 * LSB first (rad_plugin.h's packing), so every eight codes are five bytes. Served, it is the
 * codes' low nibbles in the canonical 4-bit order -- two a byte, element 2i low -- put through
 * moew4_permute as w4nla8h's are, then their high bits, one dword a 32 elements: K/2 + K/8 bytes,
 * the same count.
 *
 * THE SIGN DWORD IS TRANSPOSED for the kernel's merge: element 4m+b of a 32 -- byte b of the
 * expansion's output dword m -- is bit 7-m of byte b, so `sg << m` lines dword m's four signs up
 * on E4M3's sign bits at once (r4d_moe.hip, w5_expand). `nib` is a scratch row of K/2 bytes. */
inline uint32_t moew5_sign_bit(int64_t j) {        /* j: an element's place in its 32 */
    return 8u * (uint32_t)(j & 3) + 7u - (uint32_t)(j >> 2);
}
void moew5_row(const uint8_t* s, uint8_t* d, int64_t K, uint8_t* nib, bool forward) {
    if (forward) {
        uint8_t* sg = d + K / 2;
        for (int64_t c0 = 0; c0 < K; c0 += 32) {
            uint32_t w = 0;
            for (int64_t k = c0; k < c0 + 32; k += 8) {
                uint64_t x = 0;
                std::memcpy(&x, s + k / 8 * 5, 5);
                uint32_t lo = 0;
                for (int i = 0; i < 8; ++i) {
                    const uint32_t c = (uint32_t)(x >> (5 * i)) & 31u;
                    lo |= (c & 15u) << (4 * i);
                    w |= (c >> 4) << moew5_sign_bit(k - c0 + i);
                }
                std::memcpy(nib + k / 2, &lo, 4);
            }
            std::memcpy(sg + c0 / 8, &w, 4);
        }
        moew4_permute(nib, d, K / 2, true);
        return;
    }
    moew4_permute(s, nib, K / 2, false);
    const uint8_t* ss = s + K / 2;
    for (int64_t c0 = 0; c0 < K; c0 += 32) {
        uint32_t w = 0;
        std::memcpy(&w, ss + c0 / 8, 4);
        for (int64_t k = c0; k < c0 + 32; k += 8) {
            uint32_t lo = 0;
            std::memcpy(&lo, nib + k / 2, 4);
            uint64_t x = 0;
            for (int i = 0; i < 8; ++i) {
                const uint64_t c = ((lo >> (4 * i)) & 15u) |
                                   (uint64_t)(((w >> moew5_sign_bit(k - c0 + i)) & 1u) << 4);
                x |= c << (5 * i);
            }
            std::memcpy(d + k / 8 * 5, &x, 5);
        }
    }
}
}  /* namespace */

namespace {
using MoeViewFn = int (*)(const RadParam*, int, int, MoeView*);

int moew4_layout_by(MoeViewFn view, const RadParam* p, int n_p, int operand,
                    const RadEncoding* enc, const int* sel, const RadTensor* pl, int n,
                    RadLayout* out) {
    MoeView v;
    const int rc = view(p, n_p, operand, &v);
    return rc != RAD_OK ? rc : moew4_enc_layout(v.p, v.n, v.operand, enc, sel, pl, n, out, nullptr);
}

int moew4_relayout_by(MoeViewFn view, const RadParam* p, int n_p, int operand,
                      const RadEncoding* enc, const int* sel, const RadTensor* pl, int n,
                      void* dst, int64_t dst_bytes) {
    MoeView v;
    RadLayout L;
    WGeom g;
    int rc = view(p, n_p, operand, &v);
    if (rc == RAD_OK) rc = moew4_enc_layout(v.p, v.n, v.operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes) return RAD_E_SHAPE;
    if (v.operand == Q_BSCALE) {                     /* w4nl64a8h's: the second level is ours */
        float s2 = 0.0f;
        std::memcpy(&s2, pdata(pl[1]), sizeof s2);
        if (s2 != kW4G64Scale) return RAD_E_DTYPE;
        std::memcpy(dst, pdata(pl[0]), (size_t)L.bytes);
        return RAD_OK;
    }
    if (moe_w5(v.p, v.n)) {
        if (!moew5_table_is_kernels(pl[1])) return RAD_E_DTYPE;
        const int64_t rb = g.K / 8 * 5;
        std::vector<uint8_t> nib((size_t)(g.K / 2));
        for (long long nn = 0; nn < g.N; ++nn)
            moew5_row(pdata(pl[0]) + nn * rb, (uint8_t*)dst + nn * rb, g.K, nib.data(), true);
        return RAD_OK;
    }
    if (moe_w4_nl(v.p, v.n) && !moew4_table_is_kernels(pl[1])) return RAD_E_DTYPE;
    moew4_permute(pdata(pl[0]), (uint8_t*)dst, g.N * (g.K / 2), true);
    return RAD_OK;
}

int moew4_unrelayout_by(MoeViewFn view, const RadParam* p, int n_p, int operand,
                        const RadEncoding* enc, const int* sel, const void* src,
                        int64_t src_bytes, const RadTensor* pl, int n) {
    MoeView v;
    RadLayout L;
    WGeom g;
    int rc = view(p, n_p, operand, &v);
    if (rc == RAD_OK) rc = moew4_enc_layout(v.p, v.n, v.operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes) return RAD_E_SHAPE;
    if (v.operand == Q_BSCALE) {
        std::memcpy(pl[0].data, src, (size_t)L.bytes);
        std::memcpy(pl[1].data, &kW4G64Scale, sizeof kW4G64Scale);
        return RAD_OK;
    }
    if (moe_w5(v.p, v.n)) {
        const int64_t rb = g.K / 8 * 5;
        std::vector<uint8_t> nib((size_t)(g.K / 2));
        for (long long nn = 0; nn < g.N; ++nn)
            moew5_row((const uint8_t*)src + nn * rb, (uint8_t*)pl[0].data + nn * rb, g.K,
                      nib.data(), false);
        std::memcpy(pl[1].data, kW5nlTable, sizeof kW5nlTable);
        return RAD_OK;
    }
    moew4_permute((const uint8_t*)src, (uint8_t*)pl[0].data, g.N * (g.K / 2), false);
    /* The table the stored codes index is the kernel's: the relayout refused any other. */
    if (moe_w4_nl(v.p, v.n)) std::memcpy(pl[1].data, kW4nlTable, sizeof kW4nlTable);
    return RAD_OK;
}

}  /* namespace */

extern "C" int r4d_rad_layout_moe_w4(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel,
                                     const RadTensor* pl, int n, RadLayout* out) {
    return moew4_layout_by(moe_view, p, n_p, operand, enc, sel, pl, n, out);
}
extern "C" int r4d_rad_relayout_moe_w4(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel,
                                       const RadTensor* pl, int n, void* dst, int64_t dst_bytes) {
    return moew4_relayout_by(moe_view, p, n_p, operand, enc, sel, pl, n, dst, dst_bytes);
}
extern "C" int r4d_rad_unrelayout_moe_w4(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel, const void* src,
                                         int64_t src_bytes, const RadTensor* pl, int n) {
    return moew4_unrelayout_by(moe_view, p, n_p, operand, enc, sel, src, src_bytes, pl, n);
}
extern "C" int r4d_rad_layout_moe_w4_mod4(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, RadLayout* out) {
    return moew4_layout_by(moe_view_mod4, p, n_p, operand, enc, sel, pl, n, out);
}
extern "C" int r4d_rad_relayout_moe_w4_mod4(const RadParam* p, int n_p, int operand,
                                            const RadEncoding* enc, const int* sel,
                                            const RadTensor* pl, int n, void* dst,
                                            int64_t dst_bytes) {
    return moew4_relayout_by(moe_view_mod4, p, n_p, operand, enc, sel, pl, n, dst, dst_bytes);
}
extern "C" int r4d_rad_unrelayout_moe_w4_mod4(const RadParam* p, int n_p, int operand,
                                              const RadEncoding* enc, const int* sel,
                                              const void* src, int64_t src_bytes,
                                              const RadTensor* pl, int n) {
    return moew4_unrelayout_by(moe_view_mod4, p, n_p, operand, enc, sel, src, src_bytes, pl, n);
}

extern "C" int r4d_rad_layout_moe_fp8(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, RadLayout* out) {
    MoeView v;
    const int rc = moe_view(p, n_p, operand, &v);
    return rc != RAD_OK ? rc : r4d_rad_layout_fp8_rowmajor(v.p, v.n, v.operand, enc, sel, pl,
                                                               n, out);
}
extern "C" int r4d_rad_layout_moe_fp8_mod4(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const RadTensor* pl, int n, RadLayout* out) {
    MoeView v;
    const int rc = moe_view_mod4(p, n_p, operand, &v);
    return rc != RAD_OK ? rc : r4d_rad_layout_fp8_rowmajor(v.p, v.n, v.operand, enc, sel, pl,
                                                               n, out);
}
