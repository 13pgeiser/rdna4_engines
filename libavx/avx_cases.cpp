/* avx_cases.cpp -- the geometry table both harnesses run (one table, so the bench and the checker
 * cannot drift apart; tools/opshapes.h does the same for rad-kbench and rad-tune).
 *
 * Values are chosen to break hand-vectorised kernels, not for realism (a real n_embd would make the
 * checker take minutes and find nothing a 640-wide row does not):
 *
 *   Rows wider than any vector (16 floats at AVX-512), so the main body runs at every level.
 *   Rows that are not a multiple of it, where tails, masked stores and scalar loops can disagree:
 *     645 % 16 == 5, 645 % 8 == 5, 259 % 16 == 3.
 *   More than one row, to catch a kernel that computes row 0 and broadcasts it.
 *   M = 1, 7 and 65 on the GEMMs: decode, a speculative verify window, and one row past any
 *     register-block edge.
 *   Both sides of every option: `causal`, bounded and unbounded `window`, `norm`, both `a_order`
 *     spellings, both rope `mode`s.
 *
 * Ops run in f32 (tight tolerance: only summation order may differ) and in bf16 (narrowing).
 *
 * `bytes` and `flops` feed the bench's rate columns and stay 0 where the cost is not a simple
 * function of the geometry (in-place samplers, draw-dependent scatters).
 */
#include "avx_harness.h"

std::vector<Case> build_cases() {
    std::vector<Case> c;
    auto add = [&](const char* op, std::vector<P> p, double bytes = 0, double flops = 0) {
        c.push_back(Case{ op, std::move(p), bytes, flops });
    };
    /* Bytes moved by an elementwise op over `rows` x `n`, `w` bytes per element, `k` streams. */
    auto EB = [](double rows, double n, double w, double k) { return rows * n * w * k; };

    for (const char* dt : { "f32", "bf16", "f16" }) {
        const double w = std::strcmp(dt, "f32") == 0 ? 4.0 : 2.0;
        add("rmsnorm",     { HI("M", 7), HI("n", 640), HF("eps", 1e-6), HS("dtype", dt),
                             HF("wadd", 0) }, EB(7, 640, w, 2));
        add("rmsnorm",     { HI("M", 3), HI("n", 645), HF("eps", 1e-5), HS("dtype", dt),
                              HF("wadd", 1) }, EB(3, 645, w, 2));
        add("rmsnorm_p",   { HI("M", 7), HI("n", 640), HF("eps", 1e-6), HS("dtype", dt) },
                              EB(7, 640, w, 2));
        add("rmsnorm_p",   { HI("M", 3), HI("n", 645), HF("eps", 1e-5), HS("dtype", dt) },
                              EB(3, 645, w, 2));
        add("rmsnorm_add", { HI("M", 5), HI("n", 640), HF("eps", 1e-6), HS("dtype", dt),
                             HF("wadd", 0) }, EB(5, 640, w, 4));
        add("layernorm",   { HI("M", 5), HI("n", 645), HF("eps", 1e-5), HS("dtype", dt) },
                           EB(5, 645, w, 2));
        add("add",         { HI("M", 9), HI("n", 645), HS("dtype", dt) }, EB(9, 645, w, 3));
        add("mul",         { HI("M", 9), HI("n", 640), HS("dtype", dt) }, EB(9, 640, w, 3));
        add("silu_mul",    { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 3));
        add("gelu_mul",    { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 3));
        add("gelu",        { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 2));
        add("silu",        { HI("M", 6), HI("n", 640), HS("dtype", dt) }, EB(6, 640, w, 2));
        add("sigmoid",     { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 2));
        add("fill",        { HI("M", 6), HI("n", 645), HS("dtype", dt), HF("value", 1.3) },
            EB(6, 645, w, 1));
        add("logit_softcap", { HI("M", 6), HI("n", 645), HF("cap", 30.0), HS("dtype", dt) },
                              EB(6, 645, w, 2));
        add("scale",         { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 2));
        add("softmax",     { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 2));
        add("gather_rows", { HI("M", 6), HI("n", 640), HS("dtype", dt) }, EB(6, 640, w, 2));
        for (const char* act : { "silu", "gelu", "gelu_tanh", "sigmoid" })
            add("act_mul", { HI("M", 6), HI("n", 645), HS("act", act), HS("dtype", dt) },
                EB(6, 645, w, 3));
        add("mul_weight",  { HI("M", 9), HI("n", 645), HS("dtype", dt) }, EB(9, 645, w, 2));
        add("mul_weight",  { HI("M", 9), HI("n", 645), HS("dtype", dt), HI("scalar", 1) },
            EB(9, 645, w, 2));
        add("seg_mean",    { HI("M", 23), HI("n", 645), HS("dtype", dt) }, EB(23, 645, w, 1));
        add("relu",        { HI("M", 7), HI("n", 645), HS("dtype", dt) }, EB(7, 645, w, 2));
        add("clamp",       { HI("M", 7), HI("n", 645), HS("dtype", dt) }, EB(7, 645, w, 2));
        add("rmsnorm",     { HI("M", 5), HI("n", 645), HF("eps", 1e-6), HS("dtype", dt),
                             HF("wadd", 0), HF("mult", 0.5) }, EB(5, 645, w, 2));
        /* The audio subsampler's two geometries, scaled down in time: without and with the
         * frame mask and a segment boundary (cu_div 2: cu counts in units finer than rows). */
        add("conv2d", { HI("M", 24), HI("width", 16), HI("cin", 1), HI("cout", 24), HI("kh", 3),
                        HI("kw", 3), HI("stride", 2), HI("pad", 1), HS("dtype", dt) });
        add("conv2d", { HI("M", 24), HI("width", 8), HI("cin", 20), HI("cout", 12), HI("kh", 3),
                        HI("kw", 3), HI("stride", 2), HI("pad", 1), HI("cu_div", 2),
                        HI("valid_stride", 2), HS("dtype", dt) });
        /* Qwen3-ASR's: a [mel, time] kernel (time last), a bias and gelu after it; and its
         * output projection as a 1 x 16 convolution over the frequency axis. */
        add("conv2d", { HI("M", 24), HI("width", 16), HI("cin", 6), HI("cout", 10), HI("kh", 3),
                        HI("kw", 3), HI("stride", 2), HI("pad", 1), HS("dtype", dt),
                        HI("t_last", 1), HS("act", "gelu") });
        add("conv2d", { HI("M", 8), HI("width", 16), HI("cin", 6), HI("cout", 20), HI("kh", 1),
                        HI("kw", 16), HI("stride", 1), HI("pad", 0), HS("dtype", dt) });
        /* A kernel taller than sixteen rows. */
        add("conv2d_unfold", { HI("M", 40), HI("width", 4), HI("cin", 2), HI("kh", 17),
                               HI("kw", 1), HI("stride", 1), HI("pad", 8), HS("dtype", dt) });
        for (int tl : { 0, 1 })
            add("conv2d_unfold", { HI("M", 24), HI("width", 16), HI("cin", 6), HI("kh", 3),
                                   HI("kw", 3), HI("stride", 2), HI("pad", 1),
                                   HI("cu_div", 2), HI("valid_stride", 2), HI("t_last", tl),
                                   HS("dtype", dt) });
        add("conv1d_dw", { HI("M", 26), HI("n", 645), HI("k", 5), HS("dtype", dt) });
        add("conv1d_dw", { HI("M", 24), HI("n", 64), HI("k", 5), HI("cu_div", 4), HS("dtype", dt) });
        add("rel_sinusoid", { HI("M", 13), HI("n", 1024), HF("max_timescale", 10000.0),
                              HS("dtype", dt) });
        add("pool_rows",   { HI("M", 27), HI("n", 645), HI("k", 9), HF("mult", 27.712812921102035),
                             HS("dtype", dt) }, EB(27, 645, w, 1));
        add("scatter_rows", { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 2));
        add("scale_rows",  { HI("M", 6), HI("n", 645), HS("act", "sigmoid"), HS("dtype", dt) },
                           EB(6, 645, w, 2));
        add("scale_rows",  { HI("M", 6), HI("n", 640), HS("act", "none"), HS("dtype", dt) },
                           EB(6, 640, w, 2));
    }
    /* Every dtype pair: `cast` carries the narrowing conventions and is checked at 1e-6, i.e.
     * exactly. */
    for (const char* from : { "f32", "bf16", "f16", "fp8_e4m3", "fp8_e5m2", "i8" })
        for (const char* to : { "f32", "bf16", "f16", "fp8_e4m3", "fp8_e5m2", "i8" })
            add("cast", { HI("M", 5), HI("n", 259), HS("from", from), HS("to", to) });

    /* The wide residual stream: Qwen4-Exp's gated residual scaled down (hc 4, lowrank 8). `inject`
     * on is the trunk case, off the final mixer; they differ in the operand list. */
    for (const char* dt : { "f32", "bf16" }) {
        add("hc_enter",  { HI("M", 5), HI("n", 128), HI("hc", 4), HS("dtype", dt) });
        add("hc_write",  { HI("M", 5), HI("n", 128), HI("hc", 4), HS("dtype", dt) });
        add("hc_read",   { HI("M", 5), HI("n", 128), HI("hc", 4), HI("lowrank", 8),
                           HF("eps", 1e-6), HS("dtype", dt), HF("wadd", 1), HI("inject", 1),
                           HI("group", 0) });
        add("hc_read",   { HI("M", 5), HI("n", 128), HI("hc", 4), HI("lowrank", 8),
                           HF("eps", 1e-6), HS("dtype", dt), HF("wadd", 0), HI("inject", 0),
                           HI("group", 0) });
        /* The fp8 form and its Hadamard-rotated variant, 32-wide groups (four per stream). */
        add("hc_read",   { HI("M", 5), HI("n", 128), HI("hc", 4), HI("lowrank", 8),
                           HF("eps", 1e-6), HS("dtype", dt), HF("wadd", 1), HI("inject", 1),
                           HI("group", 32), HI("rotate", 1) });
        add("mtp_enter", { HI("M", 5), HI("n", 128), HI("hc", 4), HF("eps", 1e-6), HS("dtype", dt),
                           HF("wadd", 1), HI("ngroup", 4) });
        add("ple_gate",  { HI("M", 5), HI("n", 128), HI("hc", 4), HF("eps", 1e-6), HS("dtype", dt),
                           HF("wadd", 1), HF("gate_eps", 1e-6) });
        add("ple_conv",  { HI("M", 9), HI("n", 128), HI("width", 4), HI("dilation", 2),
                           HS("dtype", dt) });
        add("causal_conv1d", { HI("M", 9), HI("n", 128), HI("width", 3), HS("dtype", dt) });
    }
    add("ngram_ids", { HI("M", 9), HI("heads", 3), HI("ngram", 4), HI("eos", 2) });

    /* Quantisation. 128 is the real grid; 64 tests a second block edge. n = 256 is divisible by
     * both, which the had_* forms require (a rotation block may not straddle the row end). */
    for (int group : { 64, 128 }) {
        add("quant_act_i8",       { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16") }, EB(5, 256, 2, 1.5));
        add("quant_act_i8",       { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "f32") }, EB(5, 256, 4, 1.25));
        add("quant_act_i8g",      { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16") }, EB(5, 256, 2, 1.5));
        add("quant_act_i8g",      { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "f32") }, EB(5, 256, 4, 1.25));
        add("quant_act_fp8",      { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16") }, EB(5, 256, 2, 1.5));
        add("had_quant_act_fp8",  { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16") });
        add("had_quant_act_i8",   { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16") });
        add("dequant",            { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16") });
        add("gated_had_quant_i8", { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16"), HS("act", "silu"), HI("mode", 0) });
        add("rmsnorm_had_quant_i8", { HI("M", 5), HI("n", 256), HF("eps", 1e-6),
                                      HI("group", group), HS("dtype", "bf16"), HF("wadd", 0) });
        add("rmsnorm_quant_fp8",  { HI("M", 5), HI("n", 256), HF("eps", 1e-6),
                                    HI("group", group), HS("dtype", "bf16"), HF("wadd", 0) });
        add("gated_quant_fp8",    { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16"), HS("act", "silu") });
    }
    /* The fp8 fusions off the grid: a row the group does not divide (260 = 4 x 64 + 4, so the last
     * scale covers four values), f32 activations (still narrowed to bf16), and the other gate and
     * gain. */
    add("rmsnorm_quant_fp8", { HI("M", 5), HI("n", 260), HF("eps", 1e-6), HI("group", 64),
                               HS("dtype", "f32"), HF("wadd", 1) });
    add("gated_quant_fp8",   { HI("M", 5), HI("n", 260), HI("group", 64), HS("dtype", "f32"),
                               HS("act", "sigmoid") });
    add("gram_accum", { HI("M", 5), HI("n", 128), HI("group", 128), HS("dtype", "fp8_e4m3") },
        0, 2.0 * 5 * 128 * 128);
    /* The plain bf16 activation, with no scale. */
    add("gram_accum", { HI("M", 5), HI("n", 260), HI("group", 128), HS("dtype", "bf16") },
        0, 2.0 * 5 * 260 * 260);

    /* GEMM. K = 259 is not a multiple of any vector or block width.
     *
     * gemm_nt_bias's activations run at M = 7 only. The product is computed the same way whatever
     * `act` names, and the activation and residual are applied afterwards, element by element, by
     * an epilogue with no M blocking (avx_gemm.cpp: gemm_epilogue), so M = 1 and 65 add nothing
     * for them; those two keep the plain bias form for the product's own M edges. */
    for (const char* dt : { "f32", "bf16", "f16" })
        for (int M : { 1, 7, 65 }) {
            const double fl = 2.0 * M * 96 * 259;
            add("gemm_nt",      { HI("M", M), HI("N", 96), HI("K", 259), HS("dtype", dt) }, 0, fl);
            add("gemm_nt_bias", { HI("M", M), HI("N", 96), HI("K", 259), HS("dtype", dt) }, 0, fl);
            if (M == 7)
                for (const char* act : { "none", "gelu", "gelu_tanh" })
                    add("gemm_nt_bias", { HI("M", M), HI("N", 96), HI("K", 259), HS("dtype", dt),
                                          HS("act", act) }, 0, fl);
            add("logits_gemm",  { HI("M", M), HI("n_vocab", 96), HI("n_embd", 259),
                                  HS("dtype", dt) }, 0, fl);
        }
    /* Encoder-sized M with N not a multiple of the 48-column panel, K odd and even: 37 rows take
     * the bf16 dot-product path, 133 the outer-product one (column edge, zero-paired last k,
     * five-row tail). */
    for (const char* dt : { "f32", "bf16" })
        for (int M : { 37, 133 })
            for (int K : { 259, 64 }) {
                const double fl = 2.0 * M * 100 * K;
                add("gemm_nt",      { HI("M", M), HI("N", 100), HI("K", K), HS("dtype", dt) },
                    0, fl);
                add("gemm_nt_bias", { HI("M", M), HI("N", 100), HI("K", K), HS("dtype", dt),
                                      HS("act", "gelu_tanh") }, 0, fl);
            }
    /* The formats libref describes (ref_shapes.cpp): block fp8, and the int4 and int8 weights
     * against quant_act_i8's step-ordered codes. The bias form needs `a_scale`, which fp8a16 (a
     * bf16 activation) does not have, so libref declines it there. */
    for (const char* dt : { "fp8a8", "fp8a16", "w4a8", "w8a8" })
        for (int M : { 1, 65 }) {
            const double fl = 2.0 * M * 96 * 256;
            add("gemm_nt_q",      { HI("M", M), HI("N", 96), HI("K", 256), HI("group", 128),
                                    HS("dtype", dt) }, 0, fl);
            if (std::strcmp(dt, "fp8a16") != 0)
                add("gemm_nt_q_bias", { HI("M", M), HI("N", 96), HI("K", 256), HI("group", 128),
                                        HS("dtype", dt) }, 0, fl);
        }
    /* The int8 trunk (row-major codes, a scale per row per 128 of K on both sides) and OCP-MXFP4
     * (e2m1 codes with an E8M0 exponent per 32 of K against the E4M3 activation). */
    for (int M : { 1, 65 }) {
        add("gemm_nt_q", { HI("M", M), HI("N", 96), HI("K", 256), HI("group", 128),
                           HS("dtype", "i8a8") }, 0, 2.0 * M * 96 * 256);
        add("gemm_nt_q", { HI("M", M), HI("N", 96), HI("K", 256), HI("group", 32),
                           HS("dtype", "mxfp4a8") }, 0, 2.0 * M * 96 * 256);
    }
    /* Block fp8 off the 128 grid: N = 200 gives the weight's scale plane a second, partial block
     * row (rows 128..199) and K = 320 a partial last group (256..319). */
    for (const char* dt : { "fp8a8", "fp8a16" })
        for (int M : { 1, 65 })
            add("gemm_nt_q", { HI("M", M), HI("N", 200), HI("K", 320), HI("group", 128),
                               HS("dtype", dt) }, 0, 2.0 * M * 200 * 320);
    add("gemm_nt_q_bias", { HI("M", 7), HI("N", 200), HI("K", 320), HI("group", 128),
                            HS("dtype", "fp8a8") }, 0, 2.0 * 7 * 200 * 320);
    /* The tile drivers' edges (avx_gemm_tile.cpp): K = 1104 is two of the bf16 driver's K chunks
     * and three of the f32 one's, the last with a partial group; K = 17408 (the 27B's down) puts
     * 490 rows in two or three super-blocks, the last one short. */
    for (const char* dt : { "fp8a8", "fp8a16", "i8a8", "mxfp4a8", "w4a8" })
        add("gemm_nt_q", { HI("M", 40), HI("N", 100), HI("K", 1104),
                           HI("group", std::strcmp(dt, "mxfp4a8") ? 128 : 32), HS("dtype", dt) },
            0, 2.0 * 40 * 100 * 1104);
    add("gemm_nt_q", { HI("M", 490), HI("N", 20), HI("K", 17408), HI("group", 128),
                       HS("dtype", "fp8a8") }, 0, 2.0 * 490 * 20 * 17408);
    for (const char* dt : { "f32", "bf16" })
        add("gemm_nt", { HI("M", 40), HI("N", 100), HI("K", 1104), HS("dtype", dt) }, 0,
            2.0 * 40 * 100 * 1104);
    add("gemm_nt", { HI("M", 250), HI("N", 20), HI("K", 17408), HS("dtype", "bf16") }, 0,
        2.0 * 250 * 20 * 17408);
    /* The gated form, which libref describes at fp8a8 only. N is the weight's rows, so the output
     * is N/2 = 96 wide. */
    for (int M : { 1, 65 })
        for (const char* act : { "silu", "sigmoid" })
            add("gemm_nt_q_gated", { HI("M", M), HI("N", 192), HI("K", 256), HI("group", 128),
                                     HS("dtype", "fp8a8"), HS("act", act) }, 0,
                2.0 * M * 192 * 256);
    add("gemm_nt_q_gated", { HI("M", 7), HI("N", 512), HI("K", 256), HI("group", 128),
                             HS("dtype", "fp8a8"), HS("act", "silu") }, 0, 2.0 * 7 * 512 * 256);
    /* Runtime LoRA. libref's description draws `slot` in [0, S], so S itself -- a base row --
     * comes up in every case of more than one row and the comparison covers rows neither op may
     * touch (a single row draws an adapter, or the case would compare nothing). M = 1 takes the
     * shrink's team-over-columns arm and 9 and 65 its team-over-rows one; K = 1000
     * leaves a partial scale group and a vector tail at every level; R = 48 is not a power of two.
     * The dtype is `a`'s for shrink -- the coded ones with their scale stream -- and `y`'s for
     * expand. */
    for (int M : { 1, 9, 65 }) {
        for (const char* dt : { "f32", "bf16", "f16", "fp8_e4m3", "i8" })
            for (int K : { 256, 1000 })
                add("lora_shrink", { HI("M", M), HI("K", K), HI("R", 48), HI("S", 4),
                                     HI("group", 128), HS("dtype", dt) }, 0, 2.0 * M * 48 * K);
        for (const char* dt : { "f32", "bf16", "f16" })
            add("lora_expand", { HI("M", M), HI("N", 645), HI("R", 48), HI("S", 4),
                                 HS("dtype", dt) }, 0, 2.0 * M * 645 * 48);
    }
    for (const char* dt : { "f32", "bf16" }) {
        add("embed_lookup",   { HI("M", 9), HI("n_embd", 128), HI("n_vocab", 64), HS("dtype", dt),
                                HI("vocab_offset", 0), HF("mult", 1) });
        add("embed_lookup",   { HI("M", 9), HI("n_embd", 128), HI("n_vocab", 64), HS("dtype", dt),
                                HI("vocab_offset", 0), HF("mult", 22.625) });
        add("embed_lookup_q", { HI("M", 9), HI("n_embd", 128), HI("n_vocab", 64), HS("dtype", dt),
                                HI("vocab_offset", 0), HF("wscale", 0.01) });
    }

    /* Attention. q_len 1 is decode, 17 a ragged verify window, 64 the prefill panel form
     * (q_len >= 32, uniform). `window` 0 is unbounded, 8 sliding (a different loop bound). */
    for (const char* dt : { "f32", "bf16" })
        for (int qlen : { 1, 17, 64 })
            for (int window : { 0, 8 })
                add("attn_paged", { HI("q_len", qlen), HI("head_dim", 64), HI("gqa", 4),
                                    HI("block_size", 16), HI("causal", 1), HI("window", window),
                                    HS("q_dtype", dt), HS("kv_dtype", dt), HF("scale", 0.125),
                                    HI("max_ctx", 128), HI("n_head", 8), HI("max_seqs", 2) });
    for (const char* dt : { "f32", "bf16" })
        for (int causal : { 0, 1 })
            add("attn_dense", { HI("M", 24), HI("head_dim", 64), HI("gqa", 2),
                                HI("causal", causal), HS("dtype", dt), HF("scale", 0.125),
                                HI("max_seqlen", 24), HI("window", 0) });
    /* The sliding window both ways (bidirectional |i - j| < 5, causal i - j < 5), over segments
     * longer than the window. */
    for (const char* dt : { "f32", "bf16" })
        for (int causal : { 0, 1 })
            add("attn_dense", { HI("M", 24), HI("head_dim", 64), HI("gqa", 2),
                                HI("causal", causal), HS("dtype", dt), HF("scale", 0.125),
                                HI("max_seqlen", 24), HI("window", 5) });
    /* Blocks: each segment cut into blocks of 7 rows that attend within themselves (a chunked
     * encoder's windows), the last block of a segment short. */
    for (const char* dt : { "f32", "bf16" })
        add("attn_dense", { HI("M", 24), HI("head_dim", 64), HI("gqa", 1), HI("causal", 0),
                            HS("dtype", dt), HF("scale", 0.125), HI("max_seqlen", 24),
                            HI("block", 7) });
    for (const char* dt : { "f32", "bf16" })
        add("seg_sinusoid", { HI("M", 30), HI("n", 64), HI("period", 13), HS("dtype", dt) });
    /* Gemma 4's audio attention: left 12, over segments longer than the window. */
    for (const char* dt : { "f32", "bf16" })
        for (int cd : { 1, 4 })
            add("attn_local_rel", { HI("M", 40), HI("head_dim", 32), HI("n_head", 3),
                                    HI("left", 12), HI("right", 0), HF("softcap", 50.0),
                                    HI("cu_div", cd), HS("dtype", dt) });
    /* An fp8 cache under bf16 queries, as --kv-cache-dtype fp8 serves: decode and a ragged verify
     * window, unbounded and sliding. */
    for (int qlen : { 1, 17 })
        for (int window : { 0, 8 })
            add("attn_paged", { HI("q_len", qlen), HI("head_dim", 64), HI("gqa", 4),
                                HI("block_size", 16), HI("causal", 1), HI("window", window),
                                HS("q_dtype", "bf16"), HS("kv_dtype", "fp8_e4m3"),
                                HF("scale", 0.125), HI("max_ctx", 128), HI("n_head", 8),
                                HI("max_seqs", 2) });
    for (const char* dt : { "f32", "bf16" })
        add("kv_store", { HI("M", 12), HI("head_dim", 64), HI("n_head_kv", 2),
                          HI("block_size", 16), HS("kv_dtype", dt), HF("k_scale", 1.0),
                          HF("v_scale", 1.0) });
    /* The fp8 store divides by the reader's descales; one below 1 and one above. */
    for (double ks : { 0.5, 2.0 })
        add("kv_store", { HI("M", 12), HI("head_dim", 64), HI("n_head_kv", 2),
                          HI("block_size", 16), HS("kv_dtype", "fp8_e4m3"), HF("k_scale", ks),
                          HF("v_scale", 1.0 / ks) });
    for (const char* mode : { "neox", "gptj" }) {
        add("rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 2),
                      HF("theta", 10000.0), HF("scale", 1.0), HS("mode", mode),
                      HI("rotary_dim", 64) });
        add("qk_norm_rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 2),
                              HF("theta", 10000.0), HF("eps", 1e-6), HF("scale", 1.0),
                              HS("mode", mode), HI("rotary_dim", 64), HF("wadd", 0) });
        add("qk_norm_rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 2),
                              HF("theta", 10000.0), HF("eps", 1e-6), HF("scale", 1.0),
                              HS("mode", mode), HI("rotary_dim", 64), HF("wadd", 0),
                              HI("v_norm", 1) });
    }
    /* `blocks`, the two-component vision form: x over the first half, y over the second, with the
     * components listed in the opposite order to the position planes. */
    for (const char* dt : { "f32", "bf16" }) {
        add("qk_norm_rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 4),
                              HF("theta", 100.0), HF("eps", 1e-6), HS("mode", "blocks"),
                              HS("sections", "16 16"), HS("comps", "1 0"), HI("v_norm", 1),
                              HS("dtype", dt) });
        /* Qwen-Image's two forms over one position component (libref describes no other):
         * the DiT's adjacent pairs in one section, and Qwen3-VL's interleaved M-RoPE, which then
         * reduces to NeoX pairs. Three components are avx_check_hand.cpp's. */
        add("qk_norm_rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 4),
                              HF("theta", 10000.0), HF("eps", 1e-6), HS("mode", "gptj_blocks"),
                              HS("sections", "32"), HS("dtype", dt) });
        add("qk_norm_rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 2),
                              HF("theta", 5e6), HF("eps", 1e-6), HS("mode", "imrope"),
                              HI("rotary_dim", 64), HS("sections", "12 10 10"), HS("dtype", dt) });
    }
    add("rope_table", { HI("M", 12), HI("rot", 64), HF("theta", 10000.0), HF("scale", 1.0),
                        HS("dtype", "f32") });
    for (const char* dt : { "f32", "bf16" }) {
        for (const char* mode : { "neox", "gptj", "imrope" })
            add("rope", { HI("M", 12), HI("head_dim", 128), HI("n_head", 4), HI("n_head_kv", 2),
                          HF("theta", 1e7), HF("scale", 1), HS("mode", mode),
                          HI("rotary_dim", 64), HS("sections", "11 11 10"), HS("dtype", dt),
                          HF("yarn_factor", 2), HF("yarn_low", 14), HF("yarn_high", 22),
                          HF("yarn_magnitude", 1.069314718) });
        add("rope_table", { HI("M", 12), HI("rot", 64), HF("theta", 1e7), HF("scale", 1),
                            HS("dtype", dt), HF("yarn_factor", 2), HF("yarn_low", 14),
                            HF("yarn_high", 22), HF("yarn_magnitude", 1.069314718) });
    }
    for (double magnitude : {2.0, (double)NAN}) {
        add("rope", { HI("M", 12), HI("head_dim", 128), HI("n_head", 4), HI("n_head_kv", 2),
                      HF("theta", 1e7), HF("scale", 1), HS("mode", "neox"),
                      HI("rotary_dim", 64), HS("dtype", "bf16"), HF("yarn_factor", 0),
                      HF("yarn_magnitude", magnitude) });
        add("rope_table", { HI("M", 12), HI("rot", 64), HF("theta", 1e7), HF("scale", 1),
                            HS("dtype", "f32"), HF("yarn_factor", 0), HF("yarn_magnitude", magnitude) });
    }
    /* QSA's compressed key at Flash-Next's geometry (n 128, ratio 4, 64-wide rotary), plain and
     * with YaRN; positions are drawn up to block count times ratio. */
    for (const char* dt : { "f32", "bf16" }) {
        add("qsa_block_key", { HI("M", 12), HI("n", 128), HI("ratio", 4), HI("rotary_dim", 64),
                               HF("eps", 1e-6), HF("theta", 1e7), HF("scale", 1), HF("wadd", 1),
                               HS("dtype", dt) });
        add("qsa_block_key", { HI("M", 12), HI("n", 128), HI("ratio", 4), HI("rotary_dim", 64),
                               HF("eps", 1e-6), HF("theta", 1e7), HF("scale", 1), HF("wadd", 0),
                               HS("dtype", dt), HF("yarn_factor", 2), HF("yarn_low", 14),
                                HF("yarn_high", 22), HF("yarn_magnitude", 1.069314718) });
    }
    /* The rest of the indexer at Flash-Next's geometry (4 heads of 128, ratio 4, a 7-row ring for
     * depth 3) with the block count cut to the checker's budget, then the edges: block counts that
     * are no multiple of any vector width, one block, several sequences, more rows than a score
     * tile, a head width that pads, and a topk that keeps every live block or splits the zeros.
     * libref's descriptions draw page ids >= 0, cu from 0 and nc in [0, blocks]; the hand-built
     * cases (avx_check_hand.cpp) cover the rest. */
    for (const char* dt : { "bf16", "f32" }) {
        add("qsa_score", { HI("M", 4), HI("n", 128), HI("heads", 4), HI("blocks", 300),
                           HI("seqs", 1), HS("dtype", dt) });
        add("qsa_tail_store", { HI("M", 5), HI("n", 128), HI("ratio", 4), HI("ring", 7),
                                HI("seqs", 5), HS("dtype", dt) });
        add("qsa_work", { HI("M", 8), HI("n", 128), HI("ratio", 4), HI("ring", 7), HI("seqs", 4),
                          HI("work", 6), HS("dtype", dt) });
    }
    add("qsa_score", { HI("M", 7), HI("n", 128), HI("heads", 4), HI("blocks", 37), HI("seqs", 3),
                       HS("dtype", "bf16") });
    add("qsa_score", { HI("M", 19), HI("n", 72), HI("heads", 3), HI("blocks", 41), HI("seqs", 2),
                       HS("dtype", "f32") });
    add("qsa_score", { HI("M", 1), HI("n", 128), HI("heads", 4), HI("blocks", 1), HI("seqs", 1),
                       HS("dtype", "bf16") });
    add("qsa_select", { HI("M", 4), HI("blocks", 300), HI("topk", 32), HI("ratio", 4),
                        HI("seqs", 1) });
    add("qsa_select", { HI("M", 4), HI("blocks", 300), HI("topk", 200), HI("ratio", 4),
                        HI("seqs", 1) });
    add("qsa_select", { HI("M", 3), HI("blocks", 300), HI("topk", 512), HI("ratio", 4),
                        HI("seqs", 1) });
    add("qsa_select", { HI("M", 7), HI("blocks", 129), HI("topk", 16), HI("ratio", 4),
                        HI("seqs", 3) });
    add("qsa_select", { HI("M", 2), HI("blocks", 1), HI("topk", 1), HI("ratio", 3),
                        HI("seqs", 1) });
    add("qsa_tail_store", { HI("M", 3), HI("n", 37), HI("ratio", 2), HI("ring", 2), HI("seqs", 3),
                            HS("dtype", "f32") });
    add("qsa_work", { HI("M", 3), HI("n", 40), HI("ratio", 2), HI("ring", 3), HI("seqs", 3),
                      HI("work", 3), HS("dtype", "f32") });
    /* The proportional ladder: a partial rotary (rotary_dim < head_dim) with `ladder` head, so the
     * frequency denominator is head_dim. */
    for (const char* mode : { "neox", "gptj" }) {
        add("rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 2),
                      HF("theta", 10000.0), HF("scale", 1.0), HS("mode", mode),
                      HI("rotary_dim", 32), HS("ladder", "head") });
    }
    add("rope_table", { HI("M", 12), HI("rot", 32), HF("theta", 10000.0), HF("scale", 1.0),
                        HS("dtype", "f32"), HI("head_dim", 64), HS("ladder", "head") });
    /* A rotary wider than 1024 (libref takes up to 2048, i.e. 1024 frequencies), plain and with
     * contiguous sections, and QSA's key at the same width. */
    add("rope", { HI("M", 3), HI("head_dim", 1280), HI("n_head", 1), HI("n_head_kv", 1),
                  HF("theta", 10000.0), HF("scale", 1.0), HS("mode", "neox"),
                  HI("rotary_dim", 1280), HS("dtype", "f32") });
    add("rope", { HI("M", 3), HI("head_dim", 1280), HI("n_head", 1), HI("n_head_kv", 1),
                  HF("theta", 10000.0), HF("scale", 1.0), HS("mode", "mrope"),
                  HI("rotary_dim", 1280), HS("sections", "214 213 213"), HS("dtype", "f32") });
    add("qsa_block_key", { HI("M", 3), HI("n", 1280), HI("ratio", 2), HI("rotary_dim", 1280),
                           HF("eps", 1e-6), HF("theta", 10000.0), HF("scale", 1), HF("wadd", 1),
                           HS("dtype", "f32") });
    /* The multi-component forms: Qwen3-VL's interleaved M-RoPE over a partial rotary, and a vision
     * tower's 2-D axial rotary over a whole 72-wide head. */
    for (const char* dt : { "f32", "bf16" }) {
        add("rope", { HI("M", 12), HI("head_dim", 128), HI("n_head", 4), HI("n_head_kv", 2),
                      HF("theta", 1e7), HF("scale", 1.0), HS("mode", "imrope"),
                      HI("rotary_dim", 64), HS("sections", "11 11 10"), HS("dtype", dt) });
        add("rope", { HI("M", 12), HI("head_dim", 72), HI("n_head", 4), HI("n_head_kv", 4),
                      HF("theta", 10000.0), HF("scale", 1.0), HS("mode", "axial"),
                      HI("rotary_dim", 72), HS("sections", "18 18"), HS("dtype", dt) });
        add("grid_embed", { HI("M", 20), HI("n", 72), HI("side", 6), HS("dtype", dt) });
        /* Two table sides. A drawn grid never exceeds the table, so upscaling is covered by
         * avx_check's hand-built case. */
        add("grid_embed_aa", { HI("M", 20), HI("n", 72), HI("side", 16), HS("dtype", dt) });
        add("grid_embed_aa", { HI("M", 40), HI("n", 72), HI("side", 4), HS("dtype", dt) });
    }

    /* MoE. Both `a_order` spellings, since the two grouped GEMMs of a routed block use
     * different ones. */
    for (const char* dt : { "f32", "bf16" }) {
        add("router_topk", { HI("M", 9), HI("n_expert", 8), HI("top_k", 2), HI("norm", 1),
                             HS("dtype", dt) });
        add("router_topk", { HI("M", 9), HI("n_expert", 8), HI("top_k", 2), HI("norm", 0),
                             HS("dtype", dt) });
        add("moe_gather",  { HI("M", 9), HI("n", 128), HI("top_k", 2), HS("dtype", dt) });
        /* The shared-expert fold: `shared`/`shared_gate` are always described, so always
         * present; this case uses sigmoid, the one above "none". */
        add("moe_gather",  { HI("M", 9), HI("n", 128), HI("top_k", 2), HS("dtype", dt),
                             HS("act", "sigmoid") });
        add("row_topk",    { HI("M", 5), HI("N", 97), HI("R", 4), HI("vocab_off", 0),
                             HS("dtype", dt) });
        add("row_topk_merge", { HI("M", 5), HI("R", 4), HI("world_size", 2), HS("dtype", dt) });
        add("pair_argmax", { HI("M", 5), HI("n", 8), HS("dtype", dt) });
        /* "sorted", not "slot": libref refuses other spellings with RAD_E_UNSUPPORTED. */
        for (const char* order : { "token", "sorted" }) {
            add("moe_gemm",   { HI("M", 9), HI("N", 64), HI("K", 128), HI("n_expert", 8),
                                HI("top_k", 2), HS("a_order", order), HS("dtype", dt) });
            /* `fp8a8` (block fp8): libref describes a grouped GEMM's scale plane only for a
             * dtype that names its grid. */
            add("moe_gemm_q", { HI("M", 9), HI("N", 64), HI("K", 128), HI("n_expert", 8),
                                HI("top_k", 2), HI("group", 128), HS("a_order", order),
                                HS("dtype", "fp8a8") });
        }
        /* Two classes at different widths: a gate/up pair whose odd experts are half as wide
         * (the output's second part begins past the odd class's zeros), and a down projection
         * whose odd experts are half as long. */
        add("moe_gemm_q", { HI("M", 9), HI("N", 256), HI("K", 128), HI("N_odd", 128),
                            HI("parts", 2), HI("n_expert", 8), HI("top_k", 2), HI("group", 128),
                            HS("a_order", "token"), HS("dtype", "fp8a8") });
        add("moe_gemm_q", { HI("M", 9), HI("N", 64), HI("K", 256), HI("K_odd", 128),
                            HI("n_expert", 8), HI("top_k", 2), HI("group", 128),
                            HS("a_order", "sorted"), HS("dtype", "fp8a8") });
        /* Four classes at three widths (a four-rank slice of a five-block expert is two blocks of
         * one class and one of each other), over ten experts, so classes 2 and 3 hold one fewer
         * than 0 and 1; for the gate/up pair and the down projection. */
        add("moe_gemm_q_mod4", { HI("M", 9), HI("N", 128), HI("K", 128), HI("N_1", 256),
                                 HI("N_2", 128), HI("N_3", 128), HI("parts", 2),
                                 HI("n_expert", 10), HI("top_k", 2), HI("group", 128),
                                 HS("a_order", "token"), HS("dtype", "fp8a8") });
        add("moe_gemm_q_mod4", { HI("M", 9), HI("N", 64), HI("K", 128), HI("K_1", 128),
                                 HI("K_2", 256), HI("K_3", 128), HI("n_expert", 10),
                                 HI("top_k", 2), HI("group", 128), HS("a_order", "sorted"),
                                 HS("dtype", "fp8a8") });
    }
    add("moe_scatter", { HI("M", 9), HI("n_expert", 8), HI("top_k", 2), HI("expert_base", 0) });
    /* The draft head's rerank: f32 and bf16 activations over an f32 head, the whole vocabulary on
     * this rank and a shard of it (candidates below vocab_off become pads). */
    for (const char* dt : { "f32", "bf16" })
        add("logit_rerank", { HI("M", 5), HI("R", 8), HI("n_vocab", 300), HI("n_embd", 259),
                              HI("vocab_off", 0), HS("dtype", dt) });
    add("logit_rerank", { HI("M", 3), HI("R", 16), HI("n_vocab", 64), HI("n_embd", 128),
                          HI("vocab_off", 100), HS("dtype", "f32") });

    /* Gated delta net. head_k = head_v = 32 and chunk 8 keep the states small enough to inspect
     * while running every loop the 128/128 shape does.
     *
     * `n_head_k`, `n_head_v` and `n_seq` are not op parameters: the gdn ops read head counts from
     * their operands, so libref's shape hook cannot size them cold and declines without these
     * hints (tools/opshapes.h takes the same three, as `ShapeCtx`, from container metadata).
     * Without them every gdn case would be skipped.
     *
     * `n_seq` is 2 so that a wrong `cu` walk differs from a right one. */
    add("gdn_conv_prep", { HI("M", 16), HI("head_k", 32), HI("head_v", 32), HI("chunk", 8),
                           HI("conv_width", 4), HF("l2_eps", 1e-6), HF("softplus_thr", 20.0),
                           HI("n_head_k", 2), HI("n_head_v", 2), HI("n_seq", 2) });
    add("gdn_conv_update", { HI("q_len", 2), HI("head_k", 32), HI("head_v", 32),
                             HI("conv_width", 4), HI("max_query_len", 2),
                             HI("n_head_k", 2), HI("n_head_v", 2), HI("n_seq", 2) });
    add("gdn_kkt_solve", { HI("M", 16), HI("head_k", 32), HI("chunk", 8),
                           HI("n_head_k", 2), HI("n_head_v", 2), HI("n_seq", 2) });
    add("gdn_chunk_scan", { HI("M", 16), HI("head_k", 32), HI("head_v", 32), HI("chunk", 8),
                            HF("scale", 0.176), HI("state_fp16", 0),
                            HI("n_head_k", 2), HI("n_head_v", 2), HI("n_seq", 2) });
    add("gdn_recurrent_update", { HI("q_len", 2), HI("head_k", 32), HI("head_v", 32),
                                  HI("n_head_v", 2), HF("scale", 0.176), HF("eps", 1e-6),
                                  HF("l2_eps", 1e-6), HF("softplus_thr", 20.0), HS("act", "silu"),
                                  HI("state_fp16", 0), HI("n_head_k", 2), HI("n_seq", 2) });
    add("gdn_conv_recurrent_update", { HI("q_len", 2), HI("head_k", 32), HI("head_v", 32),
                                       HI("conv_width", 4), HI("n_head_v", 2), HF("scale", 0.176),
                                       HF("eps", 1e-6), HF("l2_eps", 1e-6),
                                       HF("softplus_thr", 20.0), HS("act", "silu"),
                                       HI("state_fp16", 0), HI("n_head_k", 2), HI("n_seq", 2) });
    for (const char* act : { "silu", "sigmoid" })
        add("gdn_gated_rmsnorm", { HI("M", 12), HI("channels", 128), HF("eps", 1e-6),
                                   HS("act", act) });

    /* Collectives at world_size 1, the only width libavx serves. */
    add("all_reduce", { HI("world_size", 1), HI("numel", 645), HS("dtype", "f32"), HI("exact", 1),
                        HI("min_bytes", 0), HI("hops", 1) });
    add("all_gather", { HI("world_size", 1), HI("numel", 645), HS("dtype", "f32"), HI("row", 0) });

    /* Sampling. n_vocab 517 exceeds every vector width and is not a multiple of any; n_cand 12 is
     * a realistic candidate set after top-k. */
    for (const char* op : { "sample_penalties", "sample_dry", "sample_temp", "sample_mask",
                            "sample_argmax" })
        add(op, { HI("M", 5), HI("n_vocab", 517) });
    add("sample_topk", { HI("M", 5), HI("n_vocab", 517), HI("vocab_off", 0) });
    /* k 7 and 20 (OpenAI's top_logprobs bound) against 517 columns; the offset case checks that
     * a shard reports global ids. */
    for (int64_t k : { 7, 20 })
        add("sample_logprobs", { HI("M", 5), HI("n_vocab", 517), HI("k", k), HI("vocab_off", 0) });
    add("sample_logprobs", { HI("M", 3), HI("n_vocab", 517), HI("k", 5), HI("vocab_off", 1000) });
    /* sample_merge_topk has no table case: libref cannot describe its packed (id, logit) words
     * (ref_shapes.cpp), so avx_check_hand.cpp builds them. */
    for (const char* op : { "sample_topp", "sample_minp", "sample_typical", "sample_xtc",
                            "sample_pick" })
        add(op, { HI("M", 5), HI("n_cand", 12) });

    /* The drafter's walk: every step after the first reads its predecessor from the walk. Rank 64
     * is whole vectors at every level, 37 is not; anchor stride 2 reads every other anchor. */
    add("dflash_select", { HI("M", 3), HI("steps", 4), HI("top_k", 8), HI("rank", 64),
                           HI("n_vocab", 300), HI("anchor_stride", 1), HS("dtype", "bf16") });
    add("dflash_select", { HI("M", 5), HI("steps", 3), HI("top_k", 12), HI("rank", 37),
                           HI("n_vocab", 97), HI("anchor_stride", 2), HS("dtype", "bf16") });
    /* DFlash2's convolution: the 27B drafter's form (2 taps, groups of 16) over a narrower row,
     * then groups that are no multiple of a vector, three taps and a one-row block. f16 rows take
     * the path through f32 copies, bf16 and f32 the fused one. */
    for (const char* dt : { "bf16", "f32" })
        add("dflash_conv", { HI("T", 16), HI("hidden_size", 640), HI("taps", 2),
                             HI("group_size", 16), HI("NG", 40), HI("block_size", 4),
                             HS("dtype", dt) });
    add("dflash_conv", { HI("T", 11), HI("hidden_size", 36), HI("taps", 3), HI("group_size", 4),
                         HI("NG", 9), HI("block_size", 5), HS("dtype", "f32") });
    add("dflash_conv", { HI("T", 6), HI("hidden_size", 48), HI("taps", 2), HI("group_size", 48),
                         HI("NG", 1), HI("block_size", 1), HS("dtype", "bf16") });
    add("dflash_conv", { HI("T", 11), HI("hidden_size", 36), HI("taps", 3), HI("group_size", 4),
                         HI("NG", 9), HI("block_size", 5), HS("dtype", "f16") });

    /* Image generation (Qwen-Image 2.1, arch/qwenimage21), scaled down: the DiT's norms over rows
     * that are not a multiple of any vector, the latent's 64 channels in lanes of 16 rows over two
     * entries, the decoder's convolutions and shortcuts on the eight-column pictures libref picks
     * (a channel count below and above the convolution's 48-column panel, every padding and
     * resampling form, and 256 pixels for the bf16 outer-product path), and the DiT's attention:
     * 40 rows (one query block and part of a second) against a 37-key prefix (two whole blocks
     * and part of a third), both score forms. Several entries, missing blocks and column slices
     * are avx_check_hand.cpp's. */
    static const std::string kMul = [] {
        std::string s;
        for (int i = 0; i < 67; ++i) s += (i ? " " : "") + std::to_string(0.9 + 0.013 * i);
        return s;
    }();
    static const std::string kAdd = [] {
        std::string s;
        for (int i = 0; i < 67; ++i) s += (i ? " " : "") + std::to_string(0.017 * i - 0.5);
        return s;
    }();
    for (const char* dt : { "f32", "bf16" }) {
        add("ln_mod",   { HI("M", 7), HI("n", 645), HF("eps", 1e-6), HS("dtype", dt) },
            EB(7, 645, dt[0] == 'f' ? 4 : 2, 2));
        add("gate_add", { HI("M", 7), HI("n", 645), HS("dtype", dt) },
            EB(7, 645, dt[0] == 'f' ? 4 : 2, 3));
        for (int gated : { 1, 0 })
            add("ln_mod_quant_fp8", { HI("M", 5), HI("n", 256), HF("eps", 1e-6), HI("group", 128),
                                      HI("gated", gated), HS("dtype", dt) });
        for (int round : { 1, 0 })
            add("timestep_embed", { HI("M", 3), HI("n", 256), HF("scale", 1000.0),
                                    HF("max_period", 10000.0), HI("round", round),
                                    HS("dtype", dt) });
        add("lane_gather", { HI("M", 24), HI("n", 64), HI("stride", 16), HI("n_seq", 2),
                             HS("dtype", dt) });
        add("lane_store",  { HI("M", 12), HI("n", 64), HI("stride", 16), HS("dtype", dt) });
        add("randn_lanes", { HI("M", 24), HI("n", 64), HI("stride", 16), HI("n_seq", 2),
                             HS("dtype", dt) });
        add("flow_euler",  { HI("M", 24), HI("n", 64), HI("stride", 16), HS("dtype", dt) });
        for (const char* mode : { "fma", "inv" })
            add("chan_affine", { HI("M", 9), HI("n", 67), HS("mul", kMul.c_str()),
                                 HS("add", kAdd.c_str()), HS("mode", mode), HS("dtype", dt) });
        for (const char* act : { "silu", "none" })
            add("l2norm_chan", { HI("M", 9), HI("n", 645), HS("act", act), HS("dtype", dt) });
        add("to_rgba8",  { HI("M", 9), HI("n", 4), HS("dtype", dt) });
        add("gelu_tanh", { HI("M", 6), HI("n", 645), HS("dtype", dt) });
        add("scatter_add_rows", { HI("M", 6), HI("n", 645), HS("dtype", dt) });
        struct Conv { int M, cin, cout, k, stride, up; const char* pad; };
        static const Conv kConv[] = {
            { 64, 20, 24, 3, 1, 1, "same" }, { 64, 20, 52, 3, 1, 1, "same" },
            { 64, 4, 52, 3, 1, 1, "same" },  { 64, 20, 52, 1, 1, 1, "same" },
            { 64, 20, 12, 1, 1, 1, "same" }, { 64, 20, 52, 3, 2, 1, "down" },
            { 64, 20, 52, 3, 1, 2, "same" }, { 64, 20, 4, 3, 1, 1, "same" },
            { 256, 20, 52, 3, 1, 1, "same" },
        };
        for (const Conv& v : kConv)
            add("conv2d_nhwc", { HI("M", v.M), HI("cin", v.cin), HI("cout", v.cout), HI("k", v.k),
                                 HI("stride", v.stride), HS("pad", v.pad), HI("up", v.up),
                                 HS("dtype", dt) },
                0, 2.0 * v.M * v.cout * v.cin * v.k * v.k);
        for (int ft : { 1, 2 }) {
            add("d2s_add",     { HI("M", 64), HI("cin", 24), HI("cout", 24), HI("ft", ft),
                                 HS("dtype", dt) });
            add("d2s_add",     { HI("M", 64), HI("cin", 48), HI("cout", 24), HI("ft", ft),
                                 HS("dtype", dt) });
            add("s2d_avg_add", { HI("M", 64), HI("cin", 24), HI("cout", 24), HI("ft", ft),
                                 HS("dtype", dt) });
            add("s2d_avg_add", { HI("M", 64), HI("cin", 12), HI("cout", 24), HI("ft", ft),
                                 HS("dtype", dt) });
        }
        for (const char* qk : { "bf16", "i8" })
            for (int causal : { 0, 1 })
                for (int hd : { 64, 72 })
                    add("attn_prefix_self", { HI("q_len", 40), HI("head_dim", hd), HI("gqa", 2),
                                              HI("block_size", 16), HI("causal", causal),
                                              HS("q_dtype", dt), HS("kv_dtype", dt),
                                              HF("scale", 1.0 / std::sqrt((double)hd)),
                                              HI("n_head", 4), HS("qk", qk) });
    }

    /* ================================================================== speed-only, at model size
     *
     * Timed by rad-avx-bench, skipped by rad-avx-check. The cases above are sized to break kernels;
     * speed depends on the cache hierarchy and needs real shapes.
     *
     * Geometry of a ~1.5B dense model (n_embd 2048, n_ff 5632, n_vocab 32000, 16 heads of 128).
     * M = 1 is decode, 64 a speculative verify window, 512 a prefill chunk.
     *
     * `logits_gemm` at 32000 x 2048 is memory-bound: its 128 MiB bf16 weight fits no cache, so
     * compare it with the DRAM triad, not the FMA ceiling. */
    auto speed = [&](const char* op, std::vector<P> p, double bytes, double flops) {
        c.push_back(Case{ op, std::move(p), bytes, flops, true });
    };
    for (int M : { 1, 64, 512 }) {
        speed("gemm_nt", { HI("M", M), HI("N", 2048), HI("K", 2048), HS("dtype", "bf16") },
              0, 2.0 * M * 2048 * 2048);
        speed("gemm_nt", { HI("M", M), HI("N", 5632), HI("K", 2048), HS("dtype", "bf16") },
              0, 2.0 * M * 5632 * 2048);
    }
    /* An encoder step (all tokens at once) at EmbeddingGemma 2's two FFN shapes. */
    for (int M : { 2048 }) {
        speed("gemm_nt", { HI("M", M), HI("N", 4096), HI("K", 512), HS("dtype", "bf16") },
              0, 2.0 * M * 4096 * 512);
        speed("gemm_nt", { HI("M", M), HI("N", 512), HI("K", 2048), HS("dtype", "bf16") },
              0, 2.0 * M * 512 * 2048);
        speed("rmsnorm", { HI("M", M), HI("n", 2048), HF("eps", 1e-6), HS("dtype", "bf16"),
                           HF("wadd", 0) }, (double)M * 2048 * 2 * 2, 0);
        speed("silu_mul", { HI("M", M), HI("n", 5632), HS("dtype", "bf16") },
              (double)M * 5632 * 3 * 2, 0);
        speed("softmax", { HI("M", M), HI("n", 2048), HS("dtype", "f32") },
              (double)M * 2048 * 4 * 2, 0);
        speed("quant_act_fp8", { HI("M", M), HI("n", 2048), HI("group", 128),
                                 HS("dtype", "bf16") }, (double)M * 2048 * 3, 0);
        speed("had_quant_act_fp8", { HI("M", M), HI("n", 2048), HI("group", 128),
                                     HS("dtype", "bf16") }, (double)M * 2048 * 3, 0);
        speed("quant_act_i8", { HI("M", M), HI("n", 2048), HI("group", 128),
                                HS("dtype", "bf16") }, (double)M * 2048 * 3, 0);
        speed("quant_act_i8g", { HI("M", M), HI("n", 2048), HI("group", 128),
                                 HS("dtype", "bf16") }, (double)M * 2048 * 3, 0);
        speed("had_quant_act_i8", { HI("M", M), HI("n", 2048), HI("group", 128),
                                    HS("dtype", "bf16") }, (double)M * 2048 * 3, 0);
    }
    /* GDN at qwen35 model shape (k128/v128, chunk 64, 16 key heads, 48 value heads): chunk_scan
     * over a 512 chunk (prefill) and recurrent_update at q_len 1 (decode). No rate: the traffic
     * depends on the state layout. libref runs these scalar, so keep --iters low. */
    speed("gdn_chunk_scan", { HI("M", 512), HI("head_k", 128), HI("head_v", 128), HI("chunk", 64),
                              HF("scale", 0.176), HI("state_fp16", 0),
                              HI("n_head_k", 16), HI("n_head_v", 48), HI("n_seq", 2) }, 0, 0);
    speed("gdn_chunk_scan", { HI("M", 128), HI("head_k", 128), HI("head_v", 128), HI("chunk", 64),
                              HF("scale", 0.176), HI("state_fp16", 0),
                              HI("n_head_k", 16), HI("n_head_v", 48), HI("n_seq", 2) }, 0, 0);
    speed("gdn_recurrent_update", { HI("q_len", 1), HI("head_k", 128), HI("head_v", 128),
                                    HI("n_head_v", 48), HF("scale", 0.176), HF("eps", 1e-6),
                                    HF("l2_eps", 1e-6), HF("softplus_thr", 20.0), HS("act", "silu"),
                                    HI("state_fp16", 0), HI("n_head_k", 16), HI("n_seq", 2) }, 0, 0);
    speed("gdn_kkt_solve", { HI("M", 512), HI("head_k", 128), HI("chunk", 64),
                             HI("n_head_k", 16), HI("n_head_v", 48), HI("n_seq", 2) }, 0, 0);
    /* Conv prep at model shape: conv_dim 10240 = 2*16*128 + 48*128, width 4, over a 512 chunk. */
    speed("gdn_conv_prep", { HI("M", 512), HI("head_k", 128), HI("head_v", 128), HI("chunk", 64),
                             HI("conv_width", 4), HF("l2_eps", 1e-6), HF("softplus_thr", 20.0),
                             HI("n_head_k", 16), HI("n_head_v", 48), HI("n_seq", 2) }, 0, 0);
    /* Conv update at decode shape: one new token per sequence, same widths. */
    speed("gdn_conv_update", { HI("q_len", 1), HI("head_k", 128), HI("head_v", 128),
                               HI("conv_width", 4), HI("max_query_len", 1),
                               HI("n_head_k", 16), HI("n_head_v", 48), HI("n_seq", 2) }, 0, 0);
    speed("logits_gemm", { HI("M", 1), HI("n_vocab", 32000), HI("n_embd", 2048),
                           HS("dtype", "bf16") }, (double)32000 * 2048 * 2, 2.0 * 32000 * 2048);
    speed("sample_argmax", { HI("M", 1), HI("n_vocab", 32000) }, (double)32000 * 4, 0);
    speed("sample_logprobs", { HI("M", 1), HI("n_vocab", 32000), HI("k", 20) }, (double)32000 * 4, 0);
    speed("sample_topk", { HI("M", 1), HI("n_vocab", 32000), HI("vocab_off", 0) }, 0, 0);
    /* Decode attention over a real context: 16 query heads in groups of 4, 2048 cached keys. */
    speed("attn_paged", { HI("q_len", 1), HI("head_dim", 128), HI("gqa", 4), HI("block_size", 16),
                          HI("causal", 1), HI("window", 0), HS("q_dtype", "bf16"),
                          HS("kv_dtype", "bf16"), HF("scale", 0.088), HI("max_ctx", 2048),
                          HI("n_head", 16), HI("max_seqs", 1) },
          /* No rate: the KV bytes read depend on the context length drawn into `seqused`,
             which this table does not control. */
          0, 0);
    /* Prefill attention: 512 queries over 512 cached keys, same heads. */
    speed("attn_paged", { HI("q_len", 512), HI("head_dim", 128), HI("gqa", 4),
                          HI("block_size", 16), HI("causal", 1), HI("window", 0),
                          HS("q_dtype", "bf16"), HS("kv_dtype", "bf16"), HF("scale", 0.088),
                          HI("max_ctx", 512), HI("n_head", 16), HI("max_seqs", 1) },
          0, 0);
    /* Qwen3.8-Flash-Next's indexer (arch/qwen4exp_fp8: 4 heads of 128, ratio 4, topk 512, a
     * 7-row ring at depth 3) over 8K, 64K and 200K tokens of context: decode at one query and at
     * one sequence's four verify rows, and a 2048-token prefill chunk at 64K. Decode streams the
     * bf16 block keys (256 bytes a block), so it is rated in GB/s; the prefill chunk is FMA work.
     * The bench sets every block complete and every query past all of them (avx_bench.cpp:
     * model_shape). Page ids are drawn at random, which a block table need not be in practice. */
    for (const int64_t blocks : { 2048, 16384, 50000 })
        for (const int64_t M : { 1, 4 })
            speed("qsa_score", { HI("M", M), HI("n", 128), HI("heads", 4), HI("blocks", blocks),
                                 HI("seqs", 1), HS("dtype", "bf16") },
                  (double)blocks * (128 * 2 + 4) + (double)M * blocks * 4, 0);
    speed("qsa_score", { HI("M", 2048), HI("n", 128), HI("heads", 4), HI("blocks", 16384),
                         HI("seqs", 1), HS("dtype", "bf16") },
          0, 2.0 * 2048 * 4 * 128 * 16384);
    for (const int64_t blocks : { 2048, 16384, 50000 })
        for (const int64_t M : { 1, 4 })
            speed("qsa_select", { HI("M", M), HI("blocks", blocks), HI("topk", 512),
                                  HI("ratio", 4), HI("seqs", 1) },
                  (double)M * blocks * 8, 0);
    speed("qsa_select", { HI("M", 2048), HI("blocks", 16384), HI("topk", 512), HI("ratio", 4),
                          HI("seqs", 1) }, 2048.0 * 16384 * 8, 0);
    /* The tail and the work list: four decoding sequences, and a prefill chunk's volume (512
     * sequences of four rows each completing a block, 2048 key rows staged). */
    speed("qsa_tail_store", { HI("M", 4), HI("n", 128), HI("ratio", 4), HI("ring", 7),
                              HI("seqs", 4), HS("dtype", "bf16") }, 4.0 * 128 * 2 * 2, 0);
    speed("qsa_work", { HI("M", 4), HI("n", 128), HI("ratio", 4), HI("ring", 7), HI("seqs", 4),
                        HI("work", 5), HS("dtype", "bf16") }, 4.0 * 4 * 128 * 2 * 2, 0);
    speed("qsa_work", { HI("M", 2048), HI("n", 128), HI("ratio", 4), HI("ring", 7),
                        HI("seqs", 512), HI("work", 513), HS("dtype", "bf16") },
          2048.0 * 128 * 2 * 2, 0);
    /* The 27B DFlash2 drafter's convolution (arch/common/rad_block_dflash2.h: n_embd 5120, 2
     * taps, groups of 16, a 16-row query block): one sequence's block and four. */
    for (const int64_t T : { 16, 64 })
        speed("dflash_conv", { HI("T", T), HI("hidden_size", 5120), HI("taps", 2),
                               HI("group_size", 16), HI("NG", 320), HI("block_size", 16),
                               HS("dtype", "bf16") },
              (double)T * 5120 * 2 * 2 + (double)T * 2 * 320 * 2 + 2.0 * 5120 * 2, 0);
    /* Rope at model shape: 512 tokens, 16+4 heads of 128. Time-only (libref scalar). */
    speed("rope", { HI("M", 512), HI("head_dim", 128), HI("n_head", 16), HI("n_head_kv", 4),
                    HF("theta", 10000.0), HF("scale", 1.0), HS("mode", "neox"),
                    HI("rotary_dim", 128) }, 0, 0);
    speed("qk_norm_rope", { HI("M", 512), HI("head_dim", 128), HI("n_head", 16),
                            HI("n_head_kv", 4), HF("theta", 10000.0), HF("eps", 1e-6),
                            HF("scale", 1.0), HS("mode", "neox"), HI("rotary_dim", 128),
                            HF("wadd", 0) }, 0, 0);

    /* Qwen-Image 2.1 (docs/ARCHITECTURES.md): the VAE decoder's widths 1152..144 and the DiT's
     * 32 heads of 128 over 4096 rows. Each case is about 25 GFLOP so a one-thread run stays
     * practical; libref is not timed past 2e10 (rad-avx-bench). A 512 x 512 picture's decoder
     * runs its 1152-wide convolutions at 32 x 32 pixels, its 576-wide ones at 128 x 128 and its
     * 144-wide ones at 512 x 512; the cases cut the pixels to keep the width (libref describes an
     * eight-column picture, the same work per pixel). */
    struct ConvSpeed { int64_t M, cin, cout, k, up; };
    static const ConvSpeed kConvSpeed[] = {
        { 1024, 1152, 1152, 3, 1 },    /* mid block and up block 0, at 32 x 32 */
        { 4096, 576, 576, 3, 1 },      /* up block 2 */
        { 16384, 288, 288, 3, 2 },     /* up block 3's upsampler, nearest 2x then 3 x 3 */
        { 65536, 144, 144, 3, 1 },     /* up block 4, at the picture's resolution */
        { 262144, 144, 4, 3, 1 },      /* conv_out at 512 x 512: four output channels */
        { 1024, 1152, 3456, 1, 1 },    /* the middle attention's qkv, 1 x 1 */
    };
    for (const ConvSpeed& v : kConvSpeed)
        speed("conv2d_nhwc", { HI("M", v.M), HI("cin", v.cin), HI("cout", v.cout), HI("k", v.k),
                               HI("stride", 1), HS("pad", "same"), HI("up", v.up),
                               HS("dtype", "bf16") },
              0, 2.0 * (double)v.M * v.cout * v.cin * v.k * v.k);
    /* The denoising attention: an image's 1024 latent rows (512 x 512) against a 256-key
     * conditioning, all 32 heads; 4096 rows (1024 x 1024) over four heads (the cost is per head,
     * and four heads of 128 query blocks fill the threads). Both score forms. */
    for (const char* qk : { "bf16", "i8" }) {
        speed("attn_prefix_self", { HI("q_len", 1024), HI("head_dim", 128), HI("gqa", 1),
                                    HI("block_size", 16), HI("causal", 0), HS("q_dtype", "bf16"),
                                    HS("kv_dtype", "bf16"), HF("scale", 0.08838834764831845),
                                    HI("n_head", 32), HI("ctx", 256), HS("qk", qk) },
              0, 4.0 * 1024 * (1024 + 256) * 128 * 32);
        speed("attn_prefix_self", { HI("q_len", 4096), HI("head_dim", 128), HI("gqa", 1),
                                    HI("block_size", 16), HI("causal", 0), HS("q_dtype", "bf16"),
                                    HS("kv_dtype", "bf16"), HF("scale", 0.08838834764831845),
                                    HI("n_head", 4), HI("ctx", 256), HS("qk", qk) },
              0, 4.0 * 4096 * (4096 + 256) * 128 * 4);
    }
    /* The DiT's row ops over a 512 x 512 picture's 1024 rows of 4096, and the decoder's at its
     * last level (512 x 512 pixels): bytes are the operands' reads and writes. */
    speed("ln_mod", { HI("M", 1024), HI("n", 4096), HF("eps", 1e-6), HS("dtype", "bf16") },
          1024.0 * 4096 * 2 * 2, 0);
    speed("gate_add", { HI("M", 1024), HI("n", 4096), HS("dtype", "bf16") },
          1024.0 * 4096 * 2 * 3, 0);
    speed("ln_mod_quant_fp8", { HI("M", 1024), HI("n", 4096), HF("eps", 1e-6), HI("group", 128),
                                HI("gated", 1), HS("dtype", "bf16") },
          1024.0 * 4096 * (2 * 3 + 1), 0);
    speed("qk_norm_rope", { HI("M", 1024), HI("head_dim", 128), HI("n_head", 32),
                            HI("n_head_kv", 32), HF("theta", 10000.0), HF("eps", 1e-6),
                            HS("mode", "gptj_blocks"), HS("sections", "64"),
                            HS("dtype", "bf16") }, 1024.0 * 64 * 128 * 2 * 2, 0);
    speed("l2norm_chan", { HI("M", 262144), HI("n", 144), HS("act", "silu"), HS("dtype", "bf16") },
          262144.0 * 144 * 2 * 2, 0);
    speed("d2s_add", { HI("M", 262144), HI("cin", 576), HI("cout", 288), HI("ft", 1),
                       HS("dtype", "bf16") }, 262144.0 * 288 * 2 * 2 + 65536.0 * 576 * 2, 0);
    /* Qwen3.6-35B-A3B's routed experts in block fp8 (256 experts of 512 at top 8 over n_embd
     * 2048): gate|up in two parts with the token-order activation, down with the sorted one. The
     * bytes are the experts a call routes to at most, top_k * M of them. */
    for (int M : { 1, 8, 512 }) {
        const double ex = std::min(256.0, 8.0 * M);
        speed("moe_gemm_q", { HI("M", M), HI("N", 1024), HI("K", 2048), HI("parts", 2),
                              HI("n_expert", 256), HI("top_k", 8), HI("group", 128),
                              HS("a_order", "token"), HS("dtype", "fp8a8") },
              ex * 1024 * 2048, 2.0 * M * 8 * 1024 * 2048);
        speed("moe_gemm_q", { HI("M", M), HI("N", 2048), HI("K", 512), HI("n_expert", 256),
                              HI("top_k", 8), HI("group", 128), HS("a_order", "sorted"),
                              HS("dtype", "fp8a8") },
              ex * 2048 * 512, 2.0 * M * 8 * 2048 * 512);
    }
    /* Flash-Next's gated residual read (n 2560, hc 4, lowrank 320) at decode, a verify window and
     * a prefill chunk; the bytes are the two mixing matrices in bf16, read once per launch. */
    for (int M : { 1, 4, 512 })
        speed("hc_read", { HI("M", M), HI("n", 2560), HI("hc", 4), HI("lowrank", 320),
                           HF("eps", 1e-6), HS("dtype", "bf16"), HF("wadd", 1), HI("inject", 1),
                           HI("group", 128), HI("rotate", 0) },
              2.0 * 2 * 320 * 10240, 2.0 * 2 * M * 320 * 10240);
    /* The DiT's feed-forward gate_up over 1024 rows, as the bf16 and the block-fp8 containers
     * hold it: 61% of a block's FLOPs are in its three feed-forward products. */
    speed("gemm_nt", { HI("M", 1024), HI("N", 12288), HI("K", 4096), HS("dtype", "bf16") }, 0,
          2.0 * 1024 * 12288 * 4096);
    speed("gemm_nt_q", { HI("M", 1024), HI("N", 12288), HI("K", 4096), HI("group", 128),
                         HS("dtype", "fp8a8") }, 0, 2.0 * 1024 * 12288 * 4096);

    return c;
}
