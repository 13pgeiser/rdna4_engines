#pragma once

#include <cstdint>

/* Shapes read from the local Tencent Hy-Memory-30B-A3B-BF16 model.safetensors headers and checked
 * against its config.json (48 layers, hidden=2048, 32 Q heads, 4 KV heads, 128 experts). Keep these
 * source tensor shapes separate from the architecture's declared, sometimes concatenated shapes. */
namespace hyv3_bf16_fixture {
constexpr char source[] = "Hy-Memory-30B-A3B-BF16 / model.safetensors headers";
constexpr int64_t q_proj[2] = { 4096, 2048 };
constexpr int64_t k_proj[2] = { 512, 2048 };
constexpr int64_t v_proj[2] = { 512, 2048 };
constexpr int64_t o_proj[2] = { 2048, 4096 };
constexpr int64_t q_norm[1] = { 128 };
constexpr int64_t k_norm[1] = { 128 };
constexpr int64_t dense_gate[2] = { 6912, 2048 };
constexpr int64_t dense_up[2] = { 6912, 2048 };
constexpr int64_t dense_down[2] = { 2048, 6912 };
constexpr int64_t expert_gate[2] = { 768, 2048 };
constexpr int64_t expert_up[2] = { 768, 2048 };
constexpr int64_t expert_down[2] = { 2048, 768 };
constexpr int64_t shared_gate[2] = { 768, 2048 };
constexpr int64_t shared_up[2] = { 768, 2048 };
constexpr int64_t shared_down[2] = { 2048, 768 };
constexpr int64_t router[2] = { 128, 2048 };
constexpr int64_t correction[1] = { 128 };
}  // namespace hyv3_bf16_fixture
