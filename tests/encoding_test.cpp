/* encoding_test.cpp -- weight encodings: the struct, its spelling, its geometry, the core's decoder
 * and the plane cuts tensor parallelism takes. spec.md §4.1.
 *
 * The decoder is checked against hand-worked values, not a second implementation: a scale on the
 * wrong block, a zero point subtracted after the multiply, or a nibble read from the wrong half of
 * a byte all give plausible numbers.
 */
#include "rad_test.h"
#include "rad_types.h"
#include "rad_plugin.h"
#include "rad_planes.h"
#include "format/encoding.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

using namespace rad;

namespace {

/* Plain f32 values under one f32 scale for the whole weight: the base the transform cases add
 * their tables to. */
RadEncoding unit_affine() {
    RadEncoding e = rad_enc_plain(RAD_F32);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_add_plane(&e, "scale", RAD_F32, 0, 0);
    return e;
}

/* A quantiser's forward Givens rotation of K values, in place: each of `stages` stages rotates
 * its K/2 pairs, (pairs[s][2p], pairs[s][2p+1]) by angle[s][p]. */
void givens_forward(double* x, int K, int stages, const int32_t* pairs, const float* angle) {
    for (int s = 0; s < stages; ++s)
        for (int p = 0; p < K / 2; ++p) {
            const int a = pairs[s * K + 2 * p], b = pairs[s * K + 2 * p + 1];
            const double t = angle[s * (K / 2) + p], c = std::cos(t), sn = std::sin(t);
            const double xa = x[a], xb = x[b];
            x[a] = xa * c + xb * sn;
            x[b] = xb * c - xa * sn;
        }
}

}  /* namespace */

TEST(encodings_spell_themselves_canonically) {
    CHECK_EQ(enc_name(rad_enc_plain(RAD_BF16)), std::string("bf16"));

    RadEncoding fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    CHECK_EQ(enc_name(fp8), std::string("fp8_e4m3*bf16[128x128]"));

    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    CHECK_EQ(enc_name(w4), std::string("i4*bf16[1x128]/fwht128"));

    RadEncoding w2 = rad_enc_affine(RAD_U2, RAD_F16, 1, 128);
    rad_enc_add_plane(&w2, "zero", RAD_U8, 1, 128);
    CHECK_EQ(enc_name(w2), std::string("u2*f16[1x128]-u8[1x128]"));

    RadEncoding tensor = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 0, 0);
    CHECK_EQ(enc_name(tensor), std::string("fp8_e4m3*bf16[*x*]"));

    RadEncoding own;
    rad_enc_clear(&own);
    rad_enc_copy_str(own.scheme, "lattice");
    rad_enc_add_plane(&own, "codes", RAD_U8, 1, 1);
    rad_enc_add_plane(&own, "book", RAD_F16, 0, 0);
    CHECK_EQ(enc_name(own), std::string("lattice:codes=u8[1x1],book=f16[*x*]"));

    /* NVFP4: an e4m3 scale per 16 values and an f32 per tensor, in multiplication order */
    RadEncoding nv = rad_enc_affine(RAD_FP4E2M1, RAD_F8E4M3, 1, 16);
    rad_enc_add_plane(&nv, "scale.1", RAD_F32, 0, 0);
    CHECK(rad_enc_valid(&nv));
    CHECK_EQ(enc_name(nv), std::string("fp4_e2m1*fp8_e4m3[1x16]*f32[*x*]"));

    /* Q4_K's min chain has no short form */
    RadEncoding qk = rad_enc_affine(RAD_U4, RAD_U6, 1, 32);
    rad_enc_add_plane(&qk, "scale.1", RAD_F16, 1, 256);
    rad_enc_add_plane(&qk, "min", RAD_U6, 1, 32);
    rad_enc_add_plane(&qk, "min.1", RAD_F16, 1, 256);
    CHECK(rad_enc_valid(&qk));
    CHECK_EQ(enc_name(qk), std::string("affine:codes=u4[1x1],scale=u6[1x32],scale.1=f16[1x256],"
                                       "min=u6[1x32],min.1=f16[1x256]"));

    /* a table plane is written with its own extents */
    RadEncoding nl = rad_enc_affine(RAD_U4, RAD_F16, 1, 32);
    rad_enc_add_table(&nl, "table", RAD_I8, 1, 16);
    CHECK(rad_enc_valid(&nl));
    CHECK_EQ(enc_name(nl), std::string("affine:codes=u4[1x1],scale=f16[1x32],table=i8{1x16}"));
}

TEST(only_well_formed_encodings_are_valid) {
    RadEncoding e = rad_enc_plain(RAD_BF16);
    CHECK(rad_enc_valid(&e));
    e = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    CHECK(rad_enc_valid(&e));

    /* a rotation on a plain weight says nothing a reader could undo */
    e = rad_enc_plain(RAD_BF16);
    rad_enc_copy_str(e.transform, "fwht128");
    CHECK(!rad_enc_valid(&e));
    /* not a power of two */
    e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 96);
    rad_enc_copy_str(e.transform, "fwht96");
    CHECK(!rad_enc_valid(&e));
    /* affine with nothing beyond the codes is plain, and a weight has one spelling */
    e = rad_enc_plain(RAD_I4);
    rad_enc_copy_str(e.scheme, "affine");
    CHECK(!rad_enc_valid(&e));
    /* a plugin-private dtype in a container would be unreadable without that plugin */
    e = rad_enc_plain(RAD_DT_PLUGIN_BASE + 3);
    CHECK(!rad_enc_valid(&e));
    /* the codes come first and cover one value each -- unless a grid expands them */
    e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    e.plane[0].block[1] = 2;
    CHECK(!rad_enc_valid(&e));
    rad_enc_add_table(&e, "grid", RAD_I8, 16, 2);
    CHECK(rad_enc_valid(&e));
    e.plane[2].extent[1] = 4;                         /* a grid of 4-vectors for 2-wide codes */
    CHECK(!rad_enc_valid(&e));
    /* a scale.2 needs a scale.1 under it */
    e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 32);
    rad_enc_add_plane(&e, "scale.2", RAD_F32, 0, 0);
    CHECK(!rad_enc_valid(&e));
    /* a role the scheme does not read is a mistake, not an extension */
    e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 32);
    rad_enc_add_plane(&e, "scales", RAD_F32, 0, 0);
    CHECK(!rad_enc_valid(&e));
    /* a transform needs its tables */
    e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 32);
    rad_enc_copy_str(e.transform, "givens");
    CHECK(!rad_enc_valid(&e));
    /* signs are one bit a value */
    e = rad_enc_affine(RAD_U4, RAD_BF16, 1, 32);
    rad_enc_add_plane(&e, "signs", RAD_U2, 1, 1);
    CHECK(!rad_enc_valid(&e));
}

TEST(plane_geometry_rounds_blocks_up_and_rows_to_a_byte) {
    const RadEncoding e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    int64_t pr = 0, pc = 0;
    rad_enc_plane_dims(&e.plane[0], 10, 300, &pr, &pc);
    CHECK_EQ(pr, 10);
    CHECK_EQ(pc, 300);
    CHECK_EQ(rad_enc_plane_bytes(&e.plane[0], 10, 300), 10 * 150);
    rad_enc_plane_dims(&e.plane[1], 10, 300, &pr, &pc);
    CHECK_EQ(pr, 10);
    CHECK_EQ(pc, 3);                                  /* 128 + 128 + a short last block */
    CHECK_EQ(rad_enc_plane_bytes(&e.plane[1], 10, 300), 10 * 3 * 2);

    /* 2-bit codes, 7 per row: each row pads to two bytes */
    const RadEncoding u2 = rad_enc_plain(RAD_U2);
    CHECK_EQ(rad_enc_plane_bytes(&u2.plane[0], 3, 7), 3 * 2);

    const RadEncoding t = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 0, 0);
    rad_enc_plane_dims(&t.plane[1], 320001536, 160, &pr, &pc);
    CHECK_EQ(pr, 1);
    CHECK_EQ(pc, 1);

    int64_t rows = 0, cols = 0;
    const int64_t shape3[3] = { 4, 5, 6 };
    rad_enc_view(3, shape3, &rows, &cols);
    CHECK_EQ(rows, 20);
    CHECK_EQ(cols, 6);
    const int64_t shape1[1] = { 9 };
    rad_enc_view(1, shape1, &rows, &cols);
    CHECK_EQ(rows, 1);
    CHECK_EQ(cols, 9);
}

TEST(sub_byte_codes_pack_low_bits_first) {
    uint8_t b[4] = { 0, 0, 0, 0 };
    CHECK(rad_store_code(b, RAD_I4, 0, -3));          /* 0xd */
    CHECK(rad_store_code(b, RAD_I4, 1, 7));           /* 0x7 */
    CHECK_EQ((int)b[0], 0x7d);
    CHECK_EQ(rad_load_f32(b, RAD_I4, 0), -3.0f);
    CHECK_EQ(rad_load_f32(b, RAD_I4, 1), 7.0f);
    CHECK_EQ(rad_load_f32(b, RAD_U4, 0), 13.0f);

    std::memset(b, 0, sizeof b);
    for (int i = 0; i < 4; ++i) CHECK(rad_store_code(b, RAD_U2, i, i));
    CHECK_EQ((int)b[0], 0xe4);                        /* 3 2 1 0 */
    for (int i = 0; i < 4; ++i) CHECK_EQ(rad_load_f32(b, RAD_U2, i), (float)i);
    CHECK_EQ(rad_load_f32(b, RAD_I2, 3), -1.0f);
    CHECK_EQ(rad_load_f32(b, RAD_I2, 2), -2.0f);

    /* fp4 e2m1: 0x7 is 6, 0x9 is -0.5 */
    b[1] = 0x97;
    CHECK_EQ(rad_load_f32(b + 1, RAD_FP4E2M1, 0), 6.0f);
    CHECK_EQ(rad_load_f32(b + 1, RAD_FP4E2M1, 1), -0.5f);

    b[2] = 127;
    CHECK_EQ(rad_load_f32(b + 2, RAD_E8M0, 0), 1.0f);
    b[2] = 130;
    CHECK_EQ(rad_load_f32(b + 2, RAD_E8M0, 0), 8.0f);
    b[2] = 0xff;
    CHECK(std::isnan(rad_load_f32(b + 2, RAD_E8M0, 0)));

    CHECK(!rad_store_code(b, RAD_F32, 0, 1));         /* wider than 16 bits is not a code */
}

TEST(odd_widths_are_a_bit_stream_that_straddles_bytes) {
    /* 3-bit codes 0..7 in a row: 24 bits, element 2 straddling bytes 0 and 1 */
    uint8_t b[8] = {};
    for (int i = 0; i < 8; ++i) CHECK(rad_store_code(b, RAD_U3, i, i));
    CHECK_EQ((int)b[0], 0x88);                        /* 0b10 001 000: elements 0, 1, low of 2 */
    for (int i = 0; i < 8; ++i) CHECK_EQ(rad_load_f32(b, RAD_U3, i), (float)i);
    CHECK_EQ(rad_load_f32(b, RAD_I3, 7), -1.0f);
    CHECK_EQ(rad_load_f32(b, RAD_I3, 4), -4.0f);

    /* every width, with a pattern that cannot pass by accident */
    const uint32_t widths[] = { RAD_U1, RAD_U2, RAD_U3, RAD_U4, RAD_U5, RAD_U6, RAD_U7,
                                RAD_U9, RAD_U10, RAD_U11, RAD_U12 };
    for (uint32_t dt : widths) {
        const int bits = rad_dtype_bits(dt);
        std::vector<uint8_t> row((size_t)(37 * bits + 7) / 8 + 2, 0xa5);
        for (int i = 0; i < 37; ++i)
            rad_store_code(row.data(), dt, i, (int)((i * 2654435761u) & ((1u << bits) - 1)));
        int off = 0;
        for (int i = 0; i < 37; ++i)
            off += rad_load_f32(row.data(), dt, i) !=
                   (float)((i * 2654435761u) & ((1u << bits) - 1));
        CHECK_EQ(off, 0);
    }
    /* signed widths: the extremes */
    for (uint32_t dt : { RAD_I3, RAD_I5, RAD_I6, RAD_I7 }) {
        const int bits = rad_dtype_bits(dt), lo = -(1 << (bits - 1)), hi = (1 << (bits - 1)) - 1;
        uint8_t r[4] = {};
        rad_store_code(r, dt, 0, lo);
        rad_store_code(r, dt, 1, hi);
        CHECK_EQ(rad_load_f32(r, dt, 0), (float)lo);
        CHECK_EQ(rad_load_f32(r, dt, 1), (float)hi);
    }
}

TEST(fp6_decodes_both_forms) {
    uint8_t b[4] = {};
    rad_store_code(b, RAD_FP6E2M3, 0, 0x1f);          /* e=3 m=7: 1.875 * 4 */
    rad_store_code(b, RAD_FP6E2M3, 1, 0x21);          /* negative subnormal: -1/8 */
    CHECK_EQ(rad_load_f32(b, RAD_FP6E2M3, 0), 7.5f);
    CHECK_EQ(rad_load_f32(b, RAD_FP6E2M3, 1), -0.125f);
    std::memset(b, 0, sizeof b);
    rad_store_code(b, RAD_FP6E3M2, 0, 0x1f);          /* e=7 m=3: 1.75 * 16 */
    rad_store_code(b, RAD_FP6E3M2, 1, 0x0c);          /* e=3 m=0: 1 */
    CHECK_EQ(rad_load_f32(b, RAD_FP6E3M2, 0), 28.0f);
    CHECK_EQ(rad_load_f32(b, RAD_FP6E3M2, 1), 1.0f);
}

TEST(affine_decodes_through_the_right_block_and_subtracts_the_zero_first) {
    /* A 2 x 4 weight, u4 codes, a scale per [1 x 2] and a zero per [2 x 4]. */
    RadEncoding e = rad_enc_affine(RAD_U4, RAD_F32, 1, 2);
    rad_enc_add_plane(&e, "zero", RAD_U8, 2, 4);
    CHECK(rad_enc_valid(&e));
    uint8_t codes[4] = { 0, 0, 0, 0 };
    const int q[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 4; ++c) rad_store_code(codes + r * 2, RAD_U4, c, q[r * 4 + c]);
    const float scale[4] = { 1.0f, 10.0f, 100.0f, 1000.0f };   /* [2][2] */
    const uint8_t zero[1] = { 3 };
    const void* planes[3] = { codes, scale, zero };
    EncodedView v;
    enc_view(e, 2, 4, planes, &v);
    std::vector<float> out(8);
    CHECK_EQ(enc_decode_rows(v, 0, 2, out.data()), RAD_OK);
    const float want[8] = { -2, -1, 0, 10, 200, 300, 4000, 5000 };
    for (int i = 0; i < 8; ++i) CHECK_EQ(out[(size_t)i], want[i]);

    /* one row alone decodes the same */
    std::vector<float> one(4);
    CHECK_EQ(enc_decode_rows(v, 1, 1, one.data()), RAD_OK);
    for (int i = 0; i < 4; ++i) CHECK_EQ(one[(size_t)i], want[4 + i]);
}

TEST(a_rotation_decodes_back_to_the_weight) {
    /* Store v = H w / 4 for a 4-wide group; decoding applies H again and H H = 4 I, so the
     * weight comes back exactly. docs/MOE-W4.md folds the 1/4 into the stored scale. */
    const float w[8] = { 1, -2, 3, 0.5f, 0, 0, 2, -1 };
    float v[8];
    std::memcpy(v, w, sizeof v);
    fwht_inplace(v, 4);
    fwht_inplace(v + 4, 4);
    for (float& x : v) x /= 4.0f;
    RadEncoding e = unit_affine();
    rad_enc_copy_str(e.transform, "fwht4");
    CHECK(rad_enc_valid(&e));
    const float one = 1.0f;
    const void* planes[2] = { v, &one };
    EncodedView ev;
    enc_view(e, 1, 8, planes, &ev);
    float out[8];
    CHECK_EQ(enc_decode_rows(ev, 0, 1, out), RAD_OK);
    for (int i = 0; i < 8; ++i) CHECK_EQ(out[i], w[i]);

    /* the Sylvester order: H_4 e_1 is the second column, (1, -1, 1, -1) */
    float e1[4] = { 0, 1, 0, 0 };
    fwht_inplace(e1, 4);
    CHECK_EQ(e1[0], 1.0f);
    CHECK_EQ(e1[1], -1.0f);
    CHECK_EQ(e1[2], 1.0f);
    CHECK_EQ(e1[3], -1.0f);

    /* a group that does not divide the row is refused, not decoded short */
    enc_view(e, 1, 6, planes, &ev);
    std::string why;
    CHECK_EQ(enc_decode_rows(ev, 0, 1, out, &why), RAD_E_SHAPE);
}

TEST(multilevel_scales_and_minimums_decode_as_the_k_quants_do) {
    /* One row of 8: u4 codes, u6 scale and min per [1 x 4], f16 super-scale and super-min per row.
     * w = d * sc * q - dmin * m, Q4_K's formula. */
    RadEncoding e = rad_enc_affine(RAD_U4, RAD_U6, 1, 4);
    rad_enc_add_plane(&e, "scale.1", RAD_F16, 1, 0);
    rad_enc_add_plane(&e, "min", RAD_U6, 1, 4);
    rad_enc_add_plane(&e, "min.1", RAD_F16, 1, 0);
    REQUIRE(rad_enc_valid(&e));
    uint8_t q[4] = {}, sc[2] = {}, mn[2] = {};
    for (int i = 0; i < 8; ++i) rad_store_code(q, RAD_U4, i, i + 1);
    rad_store_code(sc, RAD_U6, 0, 3);
    rad_store_code(sc, RAD_U6, 1, 5);
    rad_store_code(mn, RAD_U6, 0, 2);
    rad_store_code(mn, RAD_U6, 1, 7);
    const uint16_t d = rad_f32_to_f16(0.5f), dm = rad_f32_to_f16(0.25f);
    const void* planes[5] = { q, sc, &d, mn, &dm };
    EncodedView v;
    enc_view(e, 1, 8, planes, &v);
    float out[8];
    REQUIRE_EQ(enc_decode_rows(v, 0, 1, out), RAD_OK);
    for (int i = 0; i < 8; ++i) {
        const float want = 0.5f * (i < 4 ? 3 : 5) * (float)(i + 1) - 0.25f * (i < 4 ? 2 : 7);
        CHECK_EQ(out[i], want);
    }
}

TEST(nvfp4_multiplies_both_scale_levels) {
    RadEncoding e = rad_enc_affine(RAD_FP4E2M1, RAD_F8E4M3, 1, 16);
    rad_enc_add_plane(&e, "scale.1", RAD_F32, 0, 0);
    uint8_t q[16] = {};
    for (int i = 0; i < 32; ++i) rad_store_code(q, RAD_FP4E2M1, i, i & 15);
    const uint8_t s[2] = { rad_f32_to_fp8e4m3(2.0f), rad_f32_to_fp8e4m3(0.5f) };
    const float t = 3.0f;
    const void* planes[3] = { q, s, &t };
    EncodedView v;
    enc_view(e, 1, 32, planes, &v);
    float out[32];
    REQUIRE_EQ(enc_decode_rows(v, 0, 1, out), RAD_OK);
    for (int i = 0; i < 32; ++i)
        CHECK_EQ(out[i], rad_fp4e2m1_to_f32((uint8_t)(i & 15)) * (i < 16 ? 2.0f : 0.5f) * 3.0f);
}

TEST(a_scalar_codebook_maps_each_code_before_the_scale) {
    /* IQ4_NL: the code indexes sixteen non-linear levels */
    static const int8_t kvalues[16] = { -127, -104, -83, -65, -49, -35, -22, -10,
                                        1, 13, 25, 38, 53, 69, 89, 113 };
    RadEncoding e = rad_enc_affine(RAD_U4, RAD_F16, 1, 4);
    rad_enc_add_table(&e, "table", RAD_I8, 1, 16);
    REQUIRE(rad_enc_valid(&e));
    uint8_t q[2] = {};
    const int codes[4] = { 0, 7, 8, 15 };
    for (int i = 0; i < 4; ++i) rad_store_code(q, RAD_U4, i, codes[i]);
    const uint16_t d = rad_f32_to_f16(0.25f);
    const void* planes[3] = { q, &d, kvalues };
    EncodedView v;
    enc_view(e, 1, 4, planes, &v);
    float out[4];
    REQUIRE_EQ(enc_decode_rows(v, 0, 1, out), RAD_OK);
    for (int i = 0; i < 4; ++i) CHECK_EQ(out[i], 0.25f * kvalues[codes[i]]);
}

TEST(a_vector_codebook_expands_a_code_into_its_grid_row_and_signs) {
    /* IQ2-shaped: a u8 code per 4 values into a grid of 4-vectors, a sign bit per value, a u4
     * scale per [1 x 8] and an f16 per row. */
    static const uint8_t grid[3][4] = { { 8, 8, 8, 8 }, { 8, 25, 43, 8 }, { 43, 43, 25, 8 } };
    RadEncoding e = rad_enc_affine(RAD_U8, RAD_U4, 1, 8);
    e.plane[0].block[1] = 4;
    rad_enc_add_plane(&e, "scale.1", RAD_F16, 1, 0);
    rad_enc_add_table(&e, "grid", RAD_U8, 3, 4);
    rad_enc_add_plane(&e, "signs", RAD_U1, 1, 1);
    REQUIRE(rad_enc_valid(&e));
    const uint8_t codes[2] = { 1, 2 };
    uint8_t sc[1] = {};
    rad_store_code(sc, RAD_U4, 0, 3);
    const uint16_t d = rad_f32_to_f16(0.125f);
    const uint8_t signs[1] = { 0x81 };                /* values 0 and 7 negative */
    const void* planes[5] = { codes, sc, &d, grid, signs };
    EncodedView v;
    enc_view(e, 1, 8, planes, &v);
    float out[8];
    REQUIRE_EQ(enc_decode_rows(v, 0, 1, out), RAD_OK);
    for (int i = 0; i < 8; ++i) {
        float g = (float)grid[codes[i / 4]][i % 4];
        if (i == 0 || i == 7) g = -g;
        CHECK_EQ(out[i], g * 3.0f * 0.125f);
    }
    /* a code past the grid's end is reported as a corrupt file */
    const uint8_t bad[2] = { 1, 9 };
    const void* bp[5] = { bad, sc, &d, grid, signs };
    enc_view(e, 1, 8, bp, &v);
    std::string why;
    CHECK_EQ(enc_decode_rows(v, 0, 1, out, &why), RAD_E_FORMAT);
}

TEST(an_act_order_permutation_puts_columns_back) {
    RadEncoding e = unit_affine();
    rad_enc_add_table(&e, "t.perm", RAD_I32, 1, 4);
    rad_enc_copy_str(e.transform, "perm");
    REQUIRE(rad_enc_valid(&e));
    const float w[4] = { 10, 20, 30, 40 };
    const int32_t perm[4] = { 2, 0, 3, 1 };
    float vq[4];
    for (int j = 0; j < 4; ++j) vq[j] = w[perm[j]];   /* stored in the order it was quantised */
    const float one = 1.0f;
    const void* planes[3] = { vq, &one, perm };
    EncodedView v;
    enc_view(e, 1, 4, planes, &v);
    float out[4];
    REQUIRE_EQ(enc_decode_rows(v, 0, 1, out), RAD_OK);
    for (int j = 0; j < 4; ++j) CHECK_EQ(out[j], w[j]);
}

TEST(a_givens_transform_with_channel_scales_inverts) {
    /* Two stages over 4 columns and a channel scale: build v = w . D . G1 . G2 the way a
     * quantiser would, store v exactly, and decode back to w. */
    const int K = 4;
    const float w[4] = { 1.0f, -2.0f, 0.5f, 3.0f };
    const float dsc[4] = { 2.0f, 0.5f, 1.0f, 4.0f };
    const int32_t pairs[2][4] = { { 0, 1, 2, 3 }, { 0, 2, 1, 3 } };
    const float ang[2][2] = { { 0.3f, -1.1f }, { 0.7f, 2.0f } };
    double x[4];
    for (int k = 0; k < K; ++k) x[k] = (double)w[k] * dsc[k];
    givens_forward(x, K, 2, pairs[0], ang[0]);
    float vq[4];
    for (int k = 0; k < K; ++k) vq[k] = (float)x[k];

    RadEncoding e = unit_affine();
    rad_enc_add_table(&e, "t.pairs", RAD_I32, 2, 4);
    rad_enc_add_table(&e, "t.angle", RAD_F32, 2, 2);
    rad_enc_add_table(&e, "t.scale", RAD_F32, 1, 4);
    rad_enc_copy_str(e.transform, "givens");
    REQUIRE(rad_enc_valid(&e));
    const float one = 1.0f;
    const void* planes[5] = { vq, &one, pairs, ang, dsc };
    EncodedView v;
    enc_view(e, 1, 4, planes, &v);
    float out[4];
    REQUIRE_EQ(enc_decode_rows(v, 0, 1, out), RAD_OK);
    for (int k = 0; k < K; ++k) CHECK(std::fabs(out[k] - w[k]) < 1e-5f);
}

TEST(a_givens_weight_with_two_parts_undoes_each_rows_own_rotation) {
    /* Two rows over 4 columns, each its own part (row 0 ends part 0, row 1 ends part 1). The
     * tables stack part-major, t.scale has one row per part, and row r decodes with its part's. */
    const int K = 4;
    const float w[2][4] = { { 1.0f, -2.0f, 0.5f, 3.0f }, { -0.25f, 1.5f, 2.0f, -1.0f } };
    const float dsc[2][4] = { { 2.0f, 0.5f, 1.0f, 4.0f }, { 1.0f, 3.0f, 0.25f, 2.0f } };
    const int32_t pairs[2][4] = { { 0, 1, 2, 3 }, { 0, 3, 1, 2 } };       /* one stage per part */
    const float ang[2][2] = { { 0.3f, -1.1f }, { 0.9f, 2.2f } };
    const int32_t end[2] = { 1, 2 };
    float vq[2][4];
    for (int r = 0; r < 2; ++r) {
        double x[4];
        for (int k = 0; k < K; ++k) x[k] = (double)w[r][k] * dsc[r][k];
        givens_forward(x, K, 1, pairs[r], ang[r]);
        for (int k = 0; k < K; ++k) vq[r][k] = (float)x[k];
    }
    RadEncoding e = unit_affine();
    rad_enc_add_table(&e, "t.pairs", RAD_I32, 2, 4);
    rad_enc_add_table(&e, "t.angle", RAD_F32, 2, 2);
    rad_enc_add_table(&e, "t.scale", RAD_F32, 2, 4);
    rad_enc_add_table(&e, "t.part", RAD_I32, 1, 2);
    rad_enc_copy_str(e.transform, "givens");
    REQUIRE(rad_enc_valid(&e));
    CHECK_EQ(rad_enc_givens_parts(&e), (int64_t)2);
    const float one = 1.0f;
    const void* planes[6] = { vq, &one, pairs, ang, dsc, end };
    EncodedView v;
    enc_view(e, 2, 4, planes, &v);
    float out[8];
    REQUIRE_EQ(enc_decode_rows(v, 0, 2, out), RAD_OK);
    for (int r = 0; r < 2; ++r)
        for (int k = 0; k < K; ++k) CHECK(std::fabs(out[r * K + k] - w[r][k]) < 1e-5f);
    /* a part table that does not stack to the pairs' rows is not well formed */
    RadEncoding bad = e;
    bad.plane[bad.n_planes - 1].extent[1] = 3;
    CHECK(!rad_enc_valid(&bad));
}

TEST(the_core_declines_a_scheme_it_did_not_define) {
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "lattice");
    rad_enc_add_plane(&e, "codes", RAD_U8, 1, 1);
    uint8_t c = 0;
    const void* planes[1] = { &c };
    EncodedView v;
    enc_view(e, 1, 1, planes, &v);
    float out = 0;
    std::string why;
    CHECK_EQ(enc_decode_rows(v, 0, 1, &out, &why), RAD_E_UNSUPPORTED);
    CHECK(why.find("lattice") != std::string::npos);
}

TEST(a_plane_is_cut_on_its_own_blocks) {
    const RadEncoding e = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    PlaneRect r;
    std::string why;
    /* a row shard of a [5120, 17408] weight at tp 2 */
    CHECK_EQ(enc_plane_rect(e.plane[0], 5120, 17408, 2560, 5120, 0, 17408, &r, &why), RAD_OK);
    CHECK_EQ(r.row0, 2560);
    CHECK_EQ(r.rows, 2560);
    CHECK_EQ(enc_plane_rect(e.plane[1], 5120, 17408, 2560, 5120, 0, 17408, &r, &why), RAD_OK);
    CHECK_EQ(r.row0, 20);
    CHECK_EQ(r.rows, 20);
    CHECK_EQ(r.cols, 136);
    /* a 640-wide weight split 3:2 in 128-column blocks */
    CHECK_EQ(enc_plane_rect(e.plane[1], 2560, 640, 0, 2560, 384, 640, &r, &why), RAD_OK);
    CHECK_EQ(r.col0, 3);
    CHECK_EQ(r.cols, 2);
    /* half of 640 splits a block */
    CHECK_EQ(enc_plane_rect(e.plane[1], 2560, 640, 0, 2560, 320, 640, &r, &why), RAD_E_SHAPE);
    CHECK(why.find("splits") != std::string::npos);
    /* the weight's own end is not a split: 300 columns end inside a block */
    CHECK_EQ(enc_plane_rect(e.plane[1], 4, 300, 0, 4, 256, 300, &r, &why), RAD_OK);
    CHECK_EQ(r.col0, 2);
    CHECK_EQ(r.cols, 1);

    /* a whole-extent scale belongs to every slice */
    const RadEncoding t = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 0, 0);
    CHECK_EQ(enc_plane_rect(t.plane[1], 1000, 160, 500, 1000, 0, 160, &r, &why), RAD_OK);
    CHECK_EQ(r.row0, 0);
    CHECK_EQ(r.rows, 1);

    /* a table goes whole into every slice */
    RadEncoding nl = rad_enc_affine(RAD_U4, RAD_F16, 1, 32);
    rad_enc_add_table(&nl, "table", RAD_I8, 1, 16);
    CHECK_EQ(enc_plane_rect(nl.plane[2], 64, 256, 32, 64, 128, 256, &r, &why), RAD_OK);
    CHECK_EQ(r.rows, 1);
    CHECK_EQ(r.cols, 16);

    /* a 4-bit column cut must land on a byte */
    const RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 1);
    CHECK_EQ(enc_plane_rect(w4.plane[0], 2, 10, 0, 2, 3, 10, &r, &why), RAD_E_SHAPE);
    CHECK_EQ(enc_plane_rect(w4.plane[0], 2, 10, 0, 2, 4, 10, &r, &why), RAD_OK);
}

/* ================================================== the ABI's narrowing conversions
 *
 * rad_plugin.h's f32 -> bf16 / f16 / fp8 conversions define what every kernel library's result
 * is, and libref and libavx both call them, so comparing the two libraries cannot find an error
 * in them. Here each is checked against a brute-force reference written from the formats'
 * definitions: every finite code decoded to a double, and an input's expected code the nearest
 * one, ties to the even code, overflowing to infinity (or saturating, for e4m3fn) past the top
 * code by half a step. The inputs are every midpoint between adjacent codes and the floats one
 * and two ulps either side of it, both signs, where a rounding error shows. */

namespace {

struct NarrowFmt {
    const char* name;
    int  ebits, mbits, bias;
    bool ieee;                    /* infinities and NaNs as IEEE; else e4m3fn: no inf, one NaN */
    uint32_t (*narrow)(float);
    float (*widen)(uint32_t);
};

uint32_t n_bf16(float f) { return rad_f32_to_bf16(f); }
uint32_t n_f16(float f)  { return rad_f32_to_f16(f); }
uint32_t n_e4m3(float f) { return rad_f32_to_fp8e4m3(f); }
uint32_t n_e5m2(float f) { return rad_f32_to_fp8e5m2(f); }
float w_bf16(uint32_t c) { return rad_bf16_to_f32((uint16_t)c); }
float w_f16(uint32_t c)  { return rad_f16_to_f32((uint16_t)c); }
float w_e4m3(uint32_t c) { return rad_fp8e4m3_to_f32((uint8_t)c); }
float w_e5m2(uint32_t c) { return rad_fp8e5m2_to_f32((uint8_t)c); }

const NarrowFmt kNarrow[] = {
    { "bf16",     8, 7, 127, true,  n_bf16, w_bf16 },
    { "f16",      5, 10, 15, true,  n_f16,  w_f16 },
    { "fp8_e4m3", 4, 3, 7,   false, n_e4m3, w_e4m3 },
    { "fp8_e5m2", 5, 2, 15,  true,  n_e5m2, w_e5m2 },
};

uint32_t sign_bit(const NarrowFmt& f) { return 1u << (f.ebits + f.mbits); }
bool is_nan(const NarrowFmt& f, uint32_t c) {
    const uint32_t e = (c >> f.mbits) & ((1u << f.ebits) - 1), m = c & ((1u << f.mbits) - 1);
    if (f.ieee) return e == (1u << f.ebits) - 1 && m != 0;
    return e == (1u << f.ebits) - 1 && m == (1u << f.mbits) - 1;
}
bool is_inf(const NarrowFmt& f, uint32_t c) {
    const uint32_t e = (c >> f.mbits) & ((1u << f.ebits) - 1), m = c & ((1u << f.mbits) - 1);
    return f.ieee && e == (1u << f.ebits) - 1 && m == 0;
}
/* A code's value from the format's definition alone. */
double decode(const NarrowFmt& f, uint32_t c) {
    const uint32_t e = (c >> f.mbits) & ((1u << f.ebits) - 1), m = c & ((1u << f.mbits) - 1);
    const double s = (c & sign_bit(f)) ? -1.0 : 1.0;
    if (is_nan(f, c)) return std::nan("");
    if (is_inf(f, c)) return s * INFINITY;
    if (e == 0) return s * std::ldexp((double)m, 1 - f.bias - f.mbits);
    return s * std::ldexp(1.0 + std::ldexp((double)m, -f.mbits), (int)e - f.bias);
}
/* The non-negative finite codes, in increasing value (code order is value order for them). */
std::vector<uint32_t> finite_codes(const NarrowFmt& f) {
    std::vector<uint32_t> v;
    for (uint32_t c = 0; c < sign_bit(f); ++c)
        if (!is_nan(f, c) && !is_inf(f, c)) v.push_back(c);
    return v;
}
/* The non-negative finite codes with their values, for expected(). */
struct Codes {
    std::vector<uint32_t> code;
    std::vector<double>   value;
};
Codes codes_of(const NarrowFmt& f) {
    Codes c;
    c.code = finite_codes(f);
    for (uint32_t x : c.code) c.value.push_back(decode(f, x));
    return c;
}

/* The code `x` must narrow to: nearest, ties to the even code, |x| at or past the top code plus
 * half a step to infinity (IEEE) or saturated to the top code (e4m3fn). */
uint32_t expected(const NarrowFmt& f, const Codes& c, double x) {
    const uint32_t sgn = std::signbit(x) ? sign_bit(f) : 0;
    const double a = std::fabs(x);
    const size_t n = c.value.size();
    const double vtop = c.value[n - 1], step = vtop - c.value[n - 2];
    if (a >= vtop) {
        if (f.ieee && a >= vtop + step / 2) return sgn | ((1u << f.ebits) - 1) << f.mbits;
        return sgn | c.code[n - 1];
    }
    /* value[hi - 1] <= a < value[hi] */
    const size_t hi =
        (size_t)(std::upper_bound(c.value.begin(), c.value.end(), a) - c.value.begin());
    const size_t lo = hi - 1;
    const double dl = a - c.value[lo], dh = c.value[hi] - a;
    if (dl < dh) return sgn | c.code[lo];
    if (dh < dl) return sgn | c.code[hi];
    return sgn | ((c.code[lo] & 1u) ? c.code[hi] : c.code[lo]);
}

}  /* namespace */

/* The joined form of abi/rad_planes.h, over a 2-bit asymmetric weight's three planes and a
 * codebook: each plane comes back byte for byte, at an offset a launch finds by role alone, with
 * its geometry, and every plane starts on RAD_PLANES_ALIGN. 2-bit rows of 130 codes are 33 bytes,
 * so a size computed from the element count instead of the row would be short. One plane is not
 * joined; a truncated destination or a missing plane is refused. */
TEST(the_joined_planes_form_round_trips_and_finds_each_plane_by_role) {
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_add_plane(&e, "codes", RAD_U2, 1, 1);
    rad_enc_add_plane(&e, "scale", RAD_F16, 1, 64);
    rad_enc_add_plane(&e, "zero", RAD_U8, 1, 64);
    rad_enc_add_table(&e, "table", RAD_F32, 1, 4);
    const int sel[3] = { 0, 3, 2 };                 /* codes, table, zero: not encoding order */
    const int64_t rows = 3, cols = 130;
    std::vector<uint8_t> codes((size_t)(rows * 33)), table(16), zero((size_t)(rows * 3));
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = (uint8_t)(i * 37 + 11);
    for (size_t i = 0; i < table.size(); ++i) table[i] = (uint8_t)(200 - i);
    for (size_t i = 0; i < zero.size(); ++i) zero[i] = (uint8_t)(i + 1);
    RadTensor t[3] = {};
    const int64_t ext[3][2] = { { rows, cols }, { 1, 4 }, { rows, 3 } };
    const uint32_t dt[3] = { RAD_U2, RAD_F32, RAD_U8 };
    uint8_t* src[3] = { codes.data(), table.data(), zero.data() };
    for (int k = 0; k < 3; ++k) {
        t[k].dtype = dt[k];
        t[k].rank = 2;
        t[k].shape[0] = ext[k][0];
        t[k].shape[1] = ext[k][1];
        t[k].stride[0] = ext[k][1];
        t[k].stride[1] = 1;
        t[k].data = src[k];
    }

    RadLayout L{};
    REQUIRE_EQ(rad_planes_layout(&e, sel, t, 3, &L), RAD_OK);
    CHECK_EQ(std::string(L.tag), std::string(RAD_PLANES_TAG));
    CHECK_EQ(L.dtype, (uint32_t)RAD_U2);
    CHECK_EQ((long long)L.align, (long long)RAD_PLANES_ALIGN);
    CHECK_EQ(L.bytes % RAD_PLANES_ALIGN, 0);
    CHECK_EQ(rad_planes_layout(&e, sel, t, 1, &L), RAD_E_UNSUPPORTED);
    REQUIRE_EQ(rad_planes_layout(&e, sel, t, 3, &L), RAD_OK);

    std::vector<uint8_t> joined((size_t)L.bytes, 0xAA);
    CHECK_EQ(rad_planes_relayout(&e, sel, t, 3, joined.data(), L.bytes - 1), RAD_E_SHAPE);
    REQUIRE_EQ(rad_planes_relayout(&e, sel, t, 3, joined.data(), L.bytes), RAD_OK);

    const RadPlanesHeader* hd = rad_planes_header(joined.data());
    REQUIRE(hd != nullptr);
    CHECK_EQ(hd->n_planes, 3);
    CHECK_EQ((long long)hd->bytes, (long long)L.bytes);
    const char* roles[3] = { "codes", "table", "zero" };
    for (int k = 0; k < 3; ++k) {
        const RadPlanesEntry* en = nullptr;
        const uint8_t* got = (const uint8_t*)rad_planes_find(joined.data(), roles[k], &en);
        REQUIRE(got != nullptr && en != nullptr);
        CHECK_EQ(en->dtype, dt[k]);
        CHECK_EQ((long long)en->rows, (long long)ext[k][0]);
        CHECK_EQ((long long)en->cols, (long long)ext[k][1]);
        CHECK_EQ(en->offset % RAD_PLANES_ALIGN, 0);
        CHECK_EQ((long long)en->row_bytes, (long long)rad_dtype_bytes(dt[k], ext[k][1]));
        CHECK(std::memcmp(got, src[k], (size_t)(en->rows * en->row_bytes)) == 0);
    }
    const RadPlanesEntry* tab = nullptr;
    rad_planes_find(joined.data(), "table", &tab);
    CHECK_EQ(tab->kind, (uint32_t)RAD_PLANE_TABLE);
    CHECK_EQ((long long)tab->block[1], 4LL);
    const RadPlanesEntry* zr = nullptr;
    rad_planes_find(joined.data(), "zero", &zr);
    CHECK_EQ((long long)zr->block[1], 64LL);
    CHECK(rad_planes_find(joined.data(), "scale", nullptr) == nullptr);
    CHECK(rad_planes_header(codes.data()) == nullptr);

    /* The inverse writes the planes back where `planes` points. */
    std::vector<uint8_t> c2(codes.size()), t2(table.size()), z2(zero.size());
    RadTensor back[3] = { t[0], t[1], t[2] };
    back[0].data = c2.data();
    back[1].data = t2.data();
    back[2].data = z2.data();
    REQUIRE_EQ(rad_planes_unrelayout(&e, sel, joined.data(), L.bytes, back, 3), RAD_OK);
    CHECK(c2 == codes && t2 == table && z2 == zero);

    back[1].data = nullptr;
    CHECK_EQ(rad_planes_unrelayout(&e, sel, joined.data(), L.bytes, back, 3), RAD_E_INVAL);
    const int bad[3] = { 0, 9, 2 };
    CHECK_EQ(rad_planes_layout(&e, bad, t, 3, &L), RAD_E_SHAPE);
}

/* Every finite midpoint between adjacent codes, and one and two floats either side, both signs:
 * one RTNE, with subnormals kept and the overflow edge exact. */
TEST(the_abi_narrowings_round_to_nearest_even_at_every_midpoint) {
    for (const NarrowFmt& f : kNarrow) {
        const Codes c = codes_of(f);
        std::vector<double> pts;
        for (size_t i = 0; i + 1 < c.value.size(); ++i) {
            pts.push_back(c.value[i]);
            pts.push_back((c.value[i] + c.value[i + 1]) / 2);
        }
        const double vtop = c.value.back(), step = vtop - c.value[c.value.size() - 2];
        pts.push_back(vtop);
        pts.push_back(vtop + step / 2);      /* the overflow edge */
        pts.push_back(vtop + step);
        pts.push_back(vtop * 4);
        int64_t n = 0, bad = 0;
        for (double p : pts) {
            const float m = (float)p;
            if (!std::isfinite(m)) continue;              /* past f32's range: nothing to narrow */
            if ((double)m != p) { ++bad; continue; }      /* every midpoint here is a float */
            float around[5] = { m, std::nextafter(m, 0.0f), std::nextafter(m, INFINITY), 0, 0 };
            around[3] = std::nextafter(around[1], 0.0f);
            around[4] = std::nextafter(around[2], INFINITY);
            for (float x : around)
                for (float s : { x, -x }) {
                    ++n;
                    const uint32_t got = f.narrow(s), want = expected(f, c, s);
                    if (got != want && bad++ < 8)
                        fprintf(stderr, "    %s(%a) = 0x%x, want 0x%x\n", f.name, (double)s, got,
                                want);
                }
        }
        CHECK(n > 1000);
        CHECK_EQ(bad, 0);
    }
}

/* Widening is exact for every code, NaN codes give NaN, and narrowing an f32 infinity gives the
 * format's infinity, or its top finite code where it has none. */
TEST(the_abi_widenings_are_exact_and_infinities_narrow_to_infinity_or_the_top_code) {
    for (const NarrowFmt& f : kNarrow) {
        int bad = 0;
        for (uint32_t c = 0; c < 2 * sign_bit(f); ++c) {
            const double want = decode(f, c), got = f.widen(c);
            if (std::isnan(want) ? !std::isnan(got) : got != want) ++bad;
        }
        CHECK_EQ(bad, 0);
        const uint32_t top = finite_codes(f).back();
        const uint32_t inf = ((1u << f.ebits) - 1) << f.mbits;
        CHECK_EQ(f.narrow(INFINITY), f.ieee ? inf : top);
        CHECK_EQ(f.narrow(-INFINITY), (f.ieee ? inf : top) | sign_bit(f));
    }
    /* e4m3fn saturates at 448 from its own top up: 464 is the midpoint to an absent 480. */
    for (float x : { 448.0f, 463.9f, 464.0f, 480.0f, 1e30f })
        CHECK_EQ(rad_f32_to_fp8e4m3(x), 0x7e);
    /* f16 and e5m2 share the top 57344 / 65504 edges: 61440 is e5m2's overflow tie, 65520
     * f16's. */
    CHECK_EQ(rad_f32_to_fp8e5m2(57344.0f), 0x7b);
    CHECK_EQ(rad_f32_to_fp8e5m2(61439.0f), 0x7b);
    CHECK_EQ(rad_f32_to_fp8e5m2(61440.0f), 0x7c);
    CHECK_EQ(rad_f32_to_f16(65504.0f), 0x7bff);
    CHECK_EQ(rad_f32_to_f16(65519.0f), 0x7bff);
    CHECK_EQ(rad_f32_to_f16(65520.0f), 0x7c00);
}

/* Every class of f32 NaN narrows to a NaN of each format: quiet and signalling, high and low
 * payload, both signs. A NaN that rounds to a number would let one bad activation pass for a
 * plausible value. */
TEST(every_nan_narrows_to_a_nan) {
    const uint32_t nans[] = { 0x7fc00000u, 0xffc00000u, 0x7fffffffu, 0xffffffffu, 0x7ff00000u,
                              0xfff00000u, 0x7f800001u, 0xff800001u, 0x7fbfffffu, 0x7f8000ffu };
    for (const NarrowFmt& f : kNarrow)
        for (uint32_t u : nans) {
            float x;
            std::memcpy(&x, &u, 4);
            const uint32_t c = f.narrow(x);
            if (!is_nan(f, c))
                fprintf(stderr, "    %s(0x%08x) = 0x%x, not a NaN\n", f.name, u, c);
            CHECK(is_nan(f, c));
        }
}

/* rad_store_f32 into an integer plane rounds to nearest with ties to even and saturates, NaN
 * storing 0, whatever the rounding mode: what a host kernel writes into an integer output and a
 * converter into an integer plane. Hand-worked values at every tie, bound and overflow. */
TEST(rad_store_f32_rounds_integers_to_nearest_even_and_saturates) {
    const float in[] = { 2.5f, 3.5f, -2.5f, 127.5f, -128.5f, 300.0f, 1e10f, -1e10f, NAN, 0.5f,
                         -0.5f, 1.5f, 3e9f };
    const int64_t i8[]  = { 2, 4, -2, 127, -128, 127, 127, -128, 0, 0, 0, 2, 127 };
    const int64_t u8[]  = { 2, 4, 0, 128, 0, 255, 255, 0, 0, 0, 0, 2, 255 };
    const int64_t i16[] = { 2, 4, -2, 128, -128, 300, 32767, -32768, 0, 0, 0, 2, 32767 };
    const int64_t u16[] = { 2, 4, 0, 128, 0, 300, 65535, 0, 0, 0, 0, 2, 65535 };
    const int64_t i32[] = { 2, 4, -2, 128, -128, 300, 2147483647, -2147483648LL, 0, 0, 0, 2,
                            2147483647 };
    const int64_t u32[] = { 2, 4, 0, 128, 0, 300, 4294967295LL, 0, 0, 0, 0, 2, 3000000000LL };
    const int64_t i64[] = { 2, 4, -2, 128, -128, 300, 10000000000LL, -10000000000LL, 0, 0, 0, 2,
                            3000000000LL };
    const struct { uint32_t dt; const int64_t* want; } cases[] = {
        { RAD_I8, i8 }, { RAD_U8, u8 }, { RAD_I16, i16 }, { RAD_U16, u16 },
        { RAD_I32, i32 }, { RAD_U32, u32 }, { RAD_I64, i64 } };
    for (const auto& k : cases)
        for (size_t j = 0; j < sizeof in / sizeof in[0]; ++j) {
            uint64_t raw = 0;
            rad_store_f32(&raw, k.dt, 0, in[j]);
            int64_t got = 0;
            switch (k.dt) {
                case RAD_I8:  got = (int8_t)raw; break;
                case RAD_U8:  got = (uint8_t)raw; break;
                case RAD_I16: got = (int16_t)raw; break;
                case RAD_U16: got = (uint16_t)raw; break;
                case RAD_I32: got = (int32_t)raw; break;
                case RAD_U32: got = (uint32_t)raw; break;
                default:      got = (int64_t)raw; break;
            }
            if (got != k.want[j])
                fprintf(stderr, "    %s(%g) = %lld, want %lld\n", rad_dtype_name(k.dt),
                        (double)in[j], (long long)got, (long long)k.want[j]);
            CHECK_EQ(got, k.want[j]);
        }
    /* The int64 edges: 2^63 and beyond saturate, as do their negatives. */
    int64_t r = 0;
    rad_store_f32(&r, RAD_I64, 0, 1e19f);
    CHECK_EQ(r, INT64_MAX);
    rad_store_f32(&r, RAD_I64, 0, -1e19f);
    CHECK_EQ(r, INT64_MIN);
    rad_store_f32(&r, RAD_I64, 0, 9223372036854775808.0f);
    CHECK_EQ(r, INT64_MAX);
    /* Integral values in range, the only ones the converter and quantisers store, are exact. */
    int bad = 0;
    for (int v = -128; v <= 127; ++v) {
        int8_t b = 0;
        rad_store_f32(&b, RAD_I8, 0, (float)v);
        bad += b != v;
    }
    for (int v = -4096; v <= 4096; ++v) {
        int32_t b = 0;
        rad_store_f32(&b, RAD_I32, 0, (float)v);
        bad += b != v;
    }
    CHECK_EQ(bad, 0);
}

RAD_TEST_MAIN()
