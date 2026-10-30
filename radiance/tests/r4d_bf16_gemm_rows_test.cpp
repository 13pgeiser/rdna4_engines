/* Host-only coverage of the BF16 row launch plan, without allocating multi-gigabyte matrices. */
#include "rad_test.h"
#include "../libr4d/r4d_bf16_gemm_rows.h"

TEST(bf16_rows_deployment_budget_is_not_a_kernel_limit) {
    for (int64_t m :
         {65536ll, 65537ll, 131072ll, 131073ll, 240000ll, 500000ll, 524288ll, (long long)INT_MAX}) {
        CHECK(r4d_bf16_gemm_chunk_rows(m, 128, 4096, 128, 128) > 0);
    }
    CHECK_EQ(r4d_bf16_gemm_chunk_rows(131072, 128, 4096, 128, 128), 131072);
    CHECK_EQ(r4d_bf16_gemm_chunk_rows(500000, 128, 4096, 128, 128), 500000);
}

TEST(bf16_rows_rebase_activation_spans_past_32_bits) {
    constexpr int64_t m = 500000, n = 128, k = 12288;
    CHECK(m * k > UINT32_MAX);
    for (int tile : {16, 64, 128}) {
        const int chunk = r4d_bf16_gemm_chunk_rows(m, n, k, tile, 128);
        CHECK(chunk > 0 && chunk < m);
        CHECK_EQ(chunk % tile, 0);
        CHECK((int64_t)chunk * k <= UINT32_MAX);
        CHECK((int64_t)(chunk + tile) * k > UINT32_MAX);
        CHECK_EQ((m + chunk - 1) / chunk, 2);
    }
}

TEST(bf16_rows_launch_grid_and_flattened_ids_are_bounded) {
    CHECK_EQ(r4d_bf16_gemm_chunk_rows(2000000, 1, 16, 16, 16), 65535 * 16);
    constexpr int64_t n = 67108800, k = 64;
    const int chunk = r4d_bf16_gemm_chunk_rows(INT_MAX, n, k, 64, 64);
    const int64_t gx = (n + 63) / 64, gy = (chunk + 63ll) / 64;
    CHECK(chunk > 0);
    CHECK(gx * gy <= UINT32_MAX);
    CHECK(gx * (gy + 1) > UINT32_MAX);
}

TEST(bf16_rows_keep_real_dimension_and_weight_limits) {
    for (int64_t bad : {-1ll, 0ll, (long long)INT_MAX + 1}) {
        CHECK_EQ(r4d_bf16_gemm_chunk_rows(bad, 128, 64, 64, 64), 0);
        CHECK_EQ(r4d_bf16_gemm_chunk_rows(1000, bad, 64, 64, 64), 0);
        CHECK_EQ(r4d_bf16_gemm_chunk_rows(1000, 128, bad, 64, 64), 0);
    }
    CHECK_EQ(r4d_bf16_gemm_chunk_rows(1, 1, 17, 16, 16), 0);
    CHECK_EQ(r4d_bf16_gemm_chunk_rows(1, 1, 16, 0, 16), 0);
    CHECK_EQ(r4d_bf16_gemm_chunk_rows(1, 1, 16, 16, 0), 0);
    // W is not row-chunked: the largest legal number of weight rows is shape-dependent.
    CHECK(r4d_bf16_gemm_chunk_rows(1, UINT32_MAX / 64, 64, 64, 64) > 0);
    CHECK_EQ(r4d_bf16_gemm_chunk_rows(1, UINT32_MAX / 64 + 1, 64, 64, 64), 0);
    // Even when one tile exceeds the offset span, clamping serves a smaller row chunk.
    CHECK_EQ(r4d_bf16_gemm_chunk_rows(100, 1, INT_MAX - 15, 64, 16), 2);
}

RAD_TEST_MAIN()
