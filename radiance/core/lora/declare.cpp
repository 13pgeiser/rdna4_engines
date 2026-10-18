/* lora/declare.cpp -- see declare.h. */
#include "lora/declare.h"
#include "build/rad_build.h"
#include "rad_internal.h"

#include <cstring>
#include <map>
#include <string>

namespace rad {

namespace {

std::string join(const lora::TargetSet& t) {
    std::string s;
    for (const std::string& x : t) s += (s.empty() ? "" : ",") + x;
    return s.empty() ? std::string("(none)") : s;
}

int opd_index(const OpInfo& o, const char* name) {
    if (!o.schema || !o.schema->operands) return -1;
    for (int k = 0; k < o.schema->n_operands; ++k)
        if (o.schema->operands[k].name && std::strcmp(o.schema->operands[k].name, name) == 0) return k;
    return -1;
}

bool is_coded(uint32_t dt) {
    return dt == RAD_F8E4M3 || dt == RAD_F8E5M2 || dt == RAD_I8;
}

rad_buf decl_persist(Builder& b, const std::string& name, uint32_t dtype, uint32_t rank,
                     const int64_t* shape) {
    RadBufDecl d{};
    d.dtype = dtype;
    d.rank = rank;
    for (uint32_t i = 0; i < rank; ++i) d.shape[i] = shape[i];
    d.kind = RAD_BUF_PERSIST;
    d.domain = RAD_DOMAIN_DEVICE;
    return b.decl_buffer(name.c_str(), &d);
}

}  /* namespace */

int lora_declare(Builder& b, const LoraDeclare& d) {
    if (!d.cfg || !d.cfg->enabled()) return RAD_OK;
    Program& P = b.program();
    const int     S = d.cfg->slots();
    const int     max_rank = d.cfg->max_rank;
    const int64_t max_tok = P.ctx.max_tok;
    if (S <= 0 || max_rank <= 0 || max_tok <= 0) {
        RAD_ERR("lora: %d slots of rank %d over steps of %lld tokens is not a configuration", S,
                max_rank, (long long)max_tok);
        return RAD_E_INVAL;
    }

    /* EVERY READER OF A TARGETED WEIGHT CARRIES THE DELTA, OR STARTUP REFUSES. A targeted weight
     * that some op multiplies without the adapter is a model that is the base in one place and the
     * adapter in another, and the text it writes is wrong in a way no test of the targeted GEMM
     * would show. */
    std::vector<lora::UnsupportedUse> bad;
    int s = lora::find_unsupported_uses(P, d.targets, &bad);
    if (s < 0) return s;
    if (!bad.empty()) {
        for (const lora::UnsupportedUse& u : bad)
            RAD_ERR("lora: weight '%s' (%s) is read by op %u '%s', which cannot carry an adapter: %s",
                    u.weight.c_str(), u.module.c_str(), u.op_index, u.op.c_str(), u.why.c_str());
        RAD_ERR("lora: narrow --lora-target-modules (now %s) to modules every reader of which is a "
                "plain GEMM", join(d.targets).c_str());
        return RAD_E_UNSUPPORTED;
    }

    std::vector<lora::Site>  sites;
    std::vector<std::string> pool_weights;
    std::string              err;
    s = lora::build_sites(P, d.targets, max_rank, &sites, &err, &pool_weights);
    if (s < 0) {
        RAD_ERR("lora: %s", err.c_str());
        return s;
    }

    /* ---- the pools: one per targeted weight, [S, R, K] / [S, N, R] bf16 and [S, N] f32 */
    P.lora_pools.clear();
    P.lora_pools.resize(pool_weights.size());
    for (size_t p = 0; p < pool_weights.size(); ++p) {
        const lora::Site* first = nullptr;
        for (const lora::Site& x : sites)
            if (x.pool == (int32_t)p) { first = &x; break; }
        if (!first) {
            RAD_ERR("lora: pool %zu ('%s') has no site", p, pool_weights[p].c_str());
            return RAD_E_STATE;
        }
        for (const lora::Site& x : sites)
            if (x.pool == (int32_t)p && (x.R != first->R || x.K_r != first->K_r || x.N_r != first->N_r)) {
                RAD_ERR("lora: two GEMMs over '%s' disagree about its geometry (R %lld/%lld, K %lld/%lld, "
                        "N %lld/%lld)", pool_weights[p].c_str(), (long long)x.R, (long long)first->R,
                        (long long)x.K_r, (long long)first->K_r, (long long)x.N_r,
                        (long long)first->N_r);
                return RAD_E_STATE;
            }
        LoraPool& lp = P.lora_pools[p];
        lp.weight = pool_weights[p];
        lp.R = first->R; lp.K = first->K_r; lp.N = first->N_r; lp.S = S;
        const int64_t sa[3] = { S, lp.R, lp.K }, sb[3] = { S, lp.N, lp.R }, ss[2] = { S, lp.N };
        lp.a     = decl_persist(b, "lora." + lp.weight + ".a", RAD_BF16, 3, sa);
        lp.b     = decl_persist(b, "lora." + lp.weight + ".b", RAD_BF16, 3, sb);
        lp.scale = decl_persist(b, "lora." + lp.weight + ".scale", RAD_F32, 2, ss);
        if (!lp.a || !lp.b || !lp.scale) {
            RAD_ERR("lora: could not declare the pools of '%s'", lp.weight.c_str());
            return RAD_E_STATE;
        }
    }

    /* ---- t = A.x, one f32 plane per capacity R and lane. PERSIST and not TRANSIENT: the ops that
     * use it are issued inside the GEMM's issue, at the GEMM's position, while the buffer plan would
     * place a transient by the appended ops' own position at the end of the program -- and share
     * its bytes with whatever is live there. */
    std::map<int64_t, int32_t> t_by_r;
    P.lora_t.clear();
    for (const lora::Site& x : sites) {
        if (t_by_r.count(x.R)) continue;
        LoraScratch sc;
        sc.R = x.R;
        const int64_t sh[2] = { max_tok, x.R };
        for (int l = 0; l < 2; ++l) {
            sc.lane[l] = decl_persist(b, fmt("lora.t.r%lld.lane%d", (long long)x.R, l), RAD_F32, 2, sh);
            if (!sc.lane[l]) return RAD_E_STATE;
        }
        t_by_r[x.R] = (int32_t)P.lora_t.size();
        P.lora_t.push_back(sc);
    }

    /* ---- the two ops behind each targeted GEMM */
    P.lora_sites.clear();
    int n_refused = 0;
    for (lora::Site& x : sites) {
        const uint32_t gh = x.gemm;
        if (gh == 0 || gh >= P.ops.size()) return RAD_E_STATE;
        if (!b.op_resolved(gh)) continue;   /* never handed to the plugin, so never issued */
        /* Copied out first: decl_op below appends to P.ops and moves every OpInfo. */
        const OpInfo&  g = P.ops[gh];
        const std::string gop = g.op;
        const std::string rkey = g.ranged_key;
        const int64_t  lo = g.range_lo, hi = g.range_hi;
        long long gk = 0, gn = 0, gg = 0;
        const bool     has_k = g.base.get_i("K", &gk), has_n = g.base.get_i("N", &gn);
        const bool     has_group = g.base.get_i("group", &gg) && gg > 0;
        const int      oa = opd_index(g, "a"), oas = opd_index(g, "a_scale"), oy = opd_index(g, "y");
        const std::vector<rad_buf> reads = b.op_read_set(gh), writes = b.op_write_set(gh);

        auto refuse = [&](const std::string& why) {
            RAD_ERR("lora: op %u '%s' over '%s': %s", gh, gop.c_str(), x.weight.c_str(), why.c_str());
            ++n_refused;
        };
        if (rkey != "M") { refuse("its row count is not the ranged parameter M"); continue; }
        if (!has_k || !has_n || gk != x.K_r || gn != x.N_r) {
            refuse(fmt("declared K %lld, N %lld; the weight's per-rank slice is K %lld, N %lld", gk, gn,
                       (long long)x.K_r, (long long)x.N_r));
            continue;
        }
        if (oa < 0 || oy < 0) { refuse("its schema names no `a` or no `y` operand"); continue; }
        if (reads.empty() || writes.empty()) {
            refuse("it declared no read or write set (rad_op_reads / rad_op_writes), so the dtypes "
                   "of its activation and output are not known at declare");
            continue;
        }
        const uint32_t a_dt = P.buffers[reads[0]].decl.dtype;
        const uint32_t y_dt = P.buffers[writes[0]].decl.dtype;
        const bool coded = is_coded(a_dt);
        if (coded && oas < 0) { refuse("a coded activation and no `a_scale` operand"); continue; }

        RadParam sp[6];
        int n_sp = 0;
        sp[n_sp++] = RAD_RANGE("M", lo, hi);
        sp[n_sp++] = RAD_INT("K", x.K_r);
        sp[n_sp++] = RAD_INT("R", x.R);
        sp[n_sp++] = RAD_INT("S", S);
        if (coded && has_group) sp[n_sp++] = RAD_INT("group", gg);
        sp[n_sp++] = RAD_STR("dtype", rad_dtype_name(a_dt));
        const rad_op hs = b.decl_op("lora_shrink", sp, n_sp, nullptr, 0);

        const RadParam ep[5] = { RAD_RANGE("M", lo, hi), RAD_INT("N", x.N_r), RAD_INT("R", x.R),
                                 RAD_INT("S", S), RAD_STR("dtype", rad_dtype_name(y_dt)) };
        const rad_op he = b.decl_op("lora_expand", ep, 5, nullptr, 0);
        if (!hs || !he) {
            refuse(fmt("no kernel resolves lora_%s for an activation of %s and an output of %s",
                       !hs ? "shrink" : "expand", rad_dtype_name(a_dt), rad_dtype_name(y_dt)));
            continue;
        }

        LoraIssueSite is;
        is.gemm = gh; is.shrink = hs; is.expand = he;
        is.opd_a = (int8_t)oa;
        is.opd_a_scale = (int8_t)(coded ? oas : -1);
        is.opd_y = (int8_t)oy;
        is.a_dtype = a_dt; is.y_dtype = y_dt;
        is.pool = x.pool;
        is.t = t_by_r[x.R];
        is.R = x.R; is.K = x.K_r; is.N = x.N_r;
        P.ops[hs].anchor = (int32_t)gh;
        P.ops[he].anchor = (int32_t)gh;
        P.ops[gh].lora_site = (int32_t)P.lora_sites.size();
        P.lora_sites.push_back(is);

        x.lora_shrink = hs; x.lora_expand = he;
        const LoraPool& lp = P.lora_pools[(size_t)x.pool];
        x.pool_a = lp.a; x.pool_b = lp.b; x.pool_scale = lp.scale;
    }
    if (n_refused) return RAD_E_UNSUPPORTED;

    if (d.sites_out) *d.sites_out = std::move(sites);
    return RAD_OK;
}

}  /* namespace rad */
