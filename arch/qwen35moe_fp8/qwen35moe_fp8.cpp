/* qwen35moe_fp8 -- Qwen3.5 / 3.6 MoE (35B-A3B), block-scaled FP8 weights and FP8 activations.
 *
 * This is arch/qwen35_fp8/qwen35_fp8.cpp with a routed feed-forward: three macros and an include.
 * Everything above the FFN (gated attention, gated delta net, layer schedule, rope, vocabulary
 * edges, MTP head) is identical between the dense and MoE configs. The places where a routed
 * feed-forward differs are behind `#if QWEN35_MOE` in that file.
 *
 * A second translation unit rather than a branch on `n_expert`, because the architecture id and
 * quantisation descriptor are the selection key (spec §2.4): a container converted from
 * `model_type: qwen3_5_moe` reports `qwen35moe`, the dense one `qwen35`, and one .so cannot claim
 * both ids.
 */
#define QWEN35_MOE     1
#define QWEN35_NS      qwen35moe_fp8
#define QWEN35_ARCH_ID "qwen35moe"

#include "../qwen35_fp8/qwen35_fp8.cpp"
