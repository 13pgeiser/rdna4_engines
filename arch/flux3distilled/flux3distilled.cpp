/* flux3distilled -- arch/flux3 under the architecture name of FLUX.3 [klein]'s step-distilled
 * release.
 *
 * The released pipeline is a `Flux3DistilledModularPipeline` and its base a
 * `Flux3ModularPipeline`, so rad-convert names them `flux3distilled` and `flux3`. Both hold the same
 * networks; the transformer's `flow_map` flag is what tells them apart inside the plugin. A
 * separate .so because a plugin answers to one architecture name (spec §2.4). Everything else is
 * arch/flux3/flux3.cpp.
 */
#define FLUX3_NS flux3distilled
#define FLUX3_ID "flux3distilled"

#include "../flux3/flux3.cpp"
