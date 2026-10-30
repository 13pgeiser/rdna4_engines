/* Spark's declaration and step, without a GPU or checkpoint. */
#define RAD_ARCH_NO_EXPORTS 1
#include "arch_fakes.h"
#include "../arch/spark25_bf16/spark25_bf16.cpp"

using namespace rad_fake;

static MetaKV spark_meta() {
    RadModelMeta m{};
    m.arch_id = "spark25";
    m.n_layers = 4;
    m.n_embd = 2560;
    m.n_head = 16;
    m.n_head_kv = 4;
    m.head_dim = 256;
    m.n_ff = 10240;
    m.n_vocab = 131072;
    m.n_ctx_train = 1048576;
    m.rms_eps = 1e-6f;
    m.rope_theta = 10000;
    MetaKV k(m);
    k.set("tie_word_embeddings", "1").set("headwise_attn_output_gate", "1")
     .set("gate_attn_act_mode", "sigmoid").set("hidden_act", "gelu")
     .set("layer_types", "sliding_attention sliding_attention sliding_attention full_attention")
     .set("sliding_window", "512")
     .set("rope_parameters.sliding_attention.rope_theta", "10000")
     .set("rope_parameters.sliding_attention.partial_rotary_factor", "1")
     .set("rope_parameters.full_attention.rope_theta", "5000000")
     .set("rope_parameters.full_attention.partial_rotary_factor", "0.25");
    return k;
}

TEST(layer_schedule_rope_and_window_follow_the_checkpoint) {
    MetaKV k = spark_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(spark25_bf16::declare(&b, k.meta(), &ctx), RAD_OK);
    REQUIRE_EQ(count_op(b, "attn_paged"), 4);
    REQUIRE_EQ(count_op(b, "rope"), 8);
    REQUIRE_EQ(b.binds.size(), (size_t)4);
    for (int l = 0; l < 4; ++l) {
        const RecOp* a = find_op(b, "attn_paged", l);
        const RecOp* q = find_op(b, "rope", 2*l);
        const RecOp* key = find_op(b, "rope", 2*l+1);
        REQUIRE(a && q && key);
        CHECK_EQ(pint(a, "window"), l == 3 ? 0 : 512);
        CHECK_EQ(pint(a, "gqa"), 4);
        CHECK_EQ(pint(q, "rotary_dim"), l == 3 ? 64 : 256);
        CHECK_EQ(pint(key, "rotary_dim"), l == 3 ? 64 : 256);
        CHECK_EQ(pflt(q, "theta"), l == 3 ? 5000000.0 : 10000.0);
        CHECK_EQ(pflt(key, "theta"), pflt(q, "theta"));
    }
}

TEST(uses_fp32_residual_exact_gelu_and_one_sigmoid_per_head) {
    MetaKV k = spark_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(spark25_bf16::declare(&b, k.meta(), &ctx), RAD_OK);
    const RecBuf* x = find_b(b, "x");
    REQUIRE(x);
    CHECK_EQ(x->d.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(count_op(b, "sigmoid"), 4);
    CHECK_EQ(count_op(b, "scale_rows"), 4);
    CHECK_EQ(count_op(b, "act_mul"), 4);
    for (int l = 0; l < 4; ++l) {
        CHECK_EQ(pstr(find_op(b, "act_mul", l), "act"), std::string("gelu"));
        CHECK_EQ(pint(find_op(b, "scale_rows", l), "n"), 256);
        CHECK_EQ(pstr(find_op(b, "scale_rows", l), "act"), std::string("none"));
    }
    for (const auto& o : b.ops) if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights) if (!w.d.source) CHECK(find_map(b, w.name) != nullptr);
}

TEST(prefill_and_decode_issue_a_liveness_safe_graph) {
    MetaKV k = spark_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(spark25_bf16::declare(&b, k.meta(), &ctx), RAD_OK);
    for (int n : {1, 8}) {
        RadBatch batch = decode_batch();
        batch.n_tok = n;
        batch.max_q_len = n;
        RadCtx c{.b = &b, .batch = &batch};
        spark25_bf16::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
    }
}

TEST(refuses_unimplemented_variants) {
    RadBuildCtx ctx = build_ctx();
    for (const auto& change : {std::pair{"tie_word_embeddings", "0"},
                              std::pair{"headwise_attn_output_gate", "0"},
                              std::pair{"gate_attn_act_mode", "silu"},
                              std::pair{"hidden_act", "silu"},
                              std::pair{"attention_bias", "1"},
                              std::pair{"layer_types", "sliding_attention"},
                              std::pair{"rope_parameters.full_attention.partial_rotary_factor", "0.3"}}) {
        MetaKV k = spark_meta();
        k.set(change.first, change.second);
        RadBuilder b;
        CHECK(stderr_of_rc([&] { return spark25_bf16::declare(&b, k.meta(), &ctx); }) != RAD_OK);
    }
    MetaKV k = spark_meta();
    RadBuilder b;
    ctx.world_size = 2;
    CHECK(stderr_of_rc([&] { return spark25_bf16::declare(&b, k.meta(), &ctx); }) == RAD_E_UNSUPPORTED);
}

RAD_TEST_MAIN()
