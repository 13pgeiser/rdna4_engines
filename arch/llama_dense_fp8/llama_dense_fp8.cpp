/* llama_dense_fp8 -- arch/llama_fp8 over a bf16 checkpoint quantised by rad-convert.
 *
 * For MiniCPM5-2B, which OpenBMB publishes only in bf16. The recipe
 * data/recipes/minicpm5-2b-dspark.recipe takes one amax per 128x128 block with scale = amax/448
 * rounded to bf16, so the container matches what a checkpoint shipped in fp8 would hold and every
 * kernel, layout and placement decision downstream is the same.
 *
 * A separate .so because the selection key is the (architecture, quantisation) pair (spec §2.4),
 * and a dense checkpoint's descriptor is empty. Everything else is arch/llama_fp8/llama_fp8.cpp;
 * `#if LLAMA_DENSE_SRC` there changes only the descriptor check and the graph dump text.
 */
#define LLAMA_DENSE_SRC 1
#define LLAMA_NS        llama_dense_fp8
#define LLAMA_QUANT     ""

#include "../llama_fp8/llama_fp8.cpp"
