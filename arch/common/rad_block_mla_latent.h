/* Absorbed GLM MLA core. The caller projects/norms q and stores the current
 * latent before step(). Complete heads shard as rows of the shared K/V weight. */
#ifndef RAD_BLOCK_MLA_LATENT_H
#define RAD_BLOCK_MLA_LATENT_H
#include "rad_arch.h"
#include <cmath>

namespace rad {
namespace arch {
struct MlaLatent {
    struct Config {
        int64_t heads = 16, head_dim = 256, latent_dim = 512, blocks = 513;
    };
    struct Wire {
        rad_buf q = 0, q_latent = 0, attn_latent = 0, out = 0, selected = 0, length = 0;
    };
    Geom g{};
    Config c{};
    Wire w{};
    int layer = 0;
    rad_kvgroup kv = 0;
    rad_weight projection = 0;
    rad_op op_absorb = 0, op_attn = 0, op_expand = 0;

    int declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                const Config& cfg, const Wire& wire, const char* source) {
        g = geom;
        c = cfg;
        w = wire;
        layer = l;
        kv = group;
        if (!b || !source || !group || c.heads < 1 || c.blocks < 1 || !w.q || !w.q_latent ||
            !w.attn_latent || !w.out || !w.selected || !w.length || w.q == w.q_latent ||
            w.q_latent == w.attn_latent || w.attn_latent == w.out || c.latent_dim != 512 ||
            c.head_dim != 256)
            return RAD_E_INVAL;
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, group));
        const char* name = nm.f("blk.%d.attn_kv_b.weight", l);
        RAD_ARCH_TRY(map_copy(b, name, source));
        // The conversion recipe decodes this small projection only.
        projection = decl_w(b, name, RAD_BF16, { c.heads * 2 * c.head_dim, c.latent_dim },
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
        if (!projection) return RAD_E_INVAL;
        op_absorb = rw(
            b,
            RAD_OP(b, "mla_project",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("heads", c.heads),
                              RAD_INT("head_dim", c.head_dim), RAD_INT("latent_dim", c.latent_dim),
                              RAD_STR("mode", "absorb"), RAD_STR("dtype", "bf16")),
                   RAD_WEIGHTS(projection)),
            { w.q }, { w.q_latent });
        op_attn =
            rw(b,
               RAD_OP(b, "mla_attn",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("latent_dim", c.latent_dim),
                                 RAD_INT("heads", c.heads), RAD_INT("blocks", c.blocks),
                                 RAD_INT("block_size", 4),
                                 RAD_F64("scale", 1.0 / std::sqrt((double)c.head_dim)),
                                 RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS),
               { w.q_latent, w.selected, w.length }, { w.attn_latent });
        if (!op_attn || rad_kv_block_size(b, group) != 4) return RAD_E_INVAL;
        op_expand = rw(
            b,
            RAD_OP(b, "mla_project",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("heads", c.heads),
                              RAD_INT("head_dim", c.head_dim), RAD_INT("latent_dim", c.latent_dim),
                              RAD_STR("mode", "expand"), RAD_STR("dtype", "bf16")),
                   RAD_WEIGHTS(projection)),
            { w.attn_latent }, { w.out });
        return op_absorb && op_attn && op_expand ? RAD_OK : RAD_E_INVAL;
    }
    void step(RadCtx* ctx, int64_t rows) const {
        RAD_ISSUE_N(ctx, op_absorb, rows, brows(w.q, rows), RAD_W(projection),
                    brows(w.q_latent, rows));
        RAD_ISSUE_N(ctx, op_attn, rows, brows(w.q_latent, rows), kv_cache(kv, layer),
                    brows(w.selected, rows), brows(w.length, rows), brows(w.attn_latent, rows));
        RAD_ISSUE_N(ctx, op_expand, rows, brows(w.attn_latent, rows), RAD_W(projection),
                    brows(w.out, rows));
    }
};
}
} // namespace rad::arch
#endif
