/* mem_test.cpp -- core/mem: pools, the KV block manager, prefix caching, the idle tiers.
 *
 * The properties under test are the ones that are silent when they break: a free list that leaks
 * one block per fork looks fine for an hour and then the server stops admitting; a chained hash
 * that collides in the middle of a prefix serves one conversation another's context and produces
 * fluent wrong output with no way to bisect it (spec §17).
 */
#include "rad_test.h"

#include "mem/kv.h"
#include "mem/pools.h"
#include "mem/prefix.h"
#include "mem/vram_budget.h"
#include "mem/kvtier.h"
#include "device/device.h"
#include "plugin/registry.h"
#include "runtime/tierexec.h"
#include "sched/mem_iface.h"
#include "rad_internal.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unistd.h>

using namespace rad;

/* ------------------------------------------------------------------ fixtures */

static Config small_config() {
    Config c;
    c.vram_weights_mib = 1;
    c.vram_kv_mib = 1;
    c.gpu_headroom_mib = 1;
    c.host_pool_mib = 0;
    c.max_seqs = 8;
    c.checkpoint_interval = 64;
    c.prefix_cache = true;
    return c;
}

/* A hybrid model in miniature: full attention over 2 layers at a 16-token block, and GDN
 * recurrent state over 4 layers. Small numbers, real geometry -- the point is the arithmetic, not
 * the size. */
/* HANDLE 0 IS THE NULL GROUP. The builder reserves slot 0 of every handle table so a zero handle
 * means "none", and KVManager indexes its plans BY HANDLE -- so a group table whose first entry is
 * a real group gives that group handle 0, which every consumer reads as "no group". The sentinel
 * keeps these fixtures on the shape the engine actually produces. */
static KVGroupInfo null_group() { return KVGroupInfo{}; }

/* The HANDLES hybrid_groups() hands out, named because they are handles and not positions: with
 * the sentinel at 0 the paged group is 1 and the linear one is 2. */
enum : int32_t { G_NULL = 0, G_FULL = 1, G_GDN = 2 };

static std::vector<KVGroupInfo> hybrid_groups(bool with_linear = true) {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    {
        KVGroupInfo k;
        k.name = "full";
        k.decl.kind = RAD_KV_FULL;
        k.decl.dtype = RAD_BF16;
        k.decl.n_head_kv = 2;
        k.decl.head_dim = 8;
        k.block_size = 16;             /* off the resolved attention kernel, never a constant */
        k.layers = { 0, 1 };
        g.push_back(k);                /* page = 16*2*8*2*2*2 = 2048 B */
    }
    if (with_linear) {
        KVGroupInfo k;
        k.name = "gdn";
        k.decl.kind = RAD_KV_LINEAR;
        k.decl.dtype = RAD_F32;
        k.decl.n_head_kv = 4;
        k.decl.state_dim[0] = 16;
        k.decl.state_dim[1] = 16;
        k.layers = { 2, 3, 4, 5 };
        g.push_back(k);                /* state = 4*16*16*4*4 = 16 KiB */
    }
    return g;
}

struct Fixture {
    Pools pools;
    HostArena ck;              /* the engine's snapshot store, in miniature */
    KVManager kv;
    PrefixCache pc;
    Config cfg = small_config();

    int build(bool with_linear = true, int64_t ckpt_slots = 4) {
        cfg.checkpoint_slots = with_linear ? ckpt_slots : 0;   /* --checkpoint-slots */
        return build_groups(hybrid_groups(with_linear));
    }
    int build_groups(const std::vector<KVGroupInfo>& groups) {
        int s = pools.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator());
        if (s < 0) return s;
        s = kv.configure(groups, cfg, pools.kv());
        if (s < 0) return s;
        if (kv.checkpoint_bytes() > 0 && cfg.checkpoint_slots > 0) {
            s = ck.open(kv.checkpoint_bytes(), kv.checkpoint_bytes() * cfg.checkpoint_slots);
            if (s < 0) return s;
            kv.set_checkpoint_store(&ck, 0);
        }
        return pc.configure(cfg, &kv, cfg.checkpoint_interval);
    }
};

static std::vector<int32_t> tok_run(int32_t base, int n) {
    std::vector<int32_t> v;
    for (int i = 0; i < n; ++i) v.push_back(base + i);
    return v;
}

/* Give the sequence `id` enough blocks for `n_tok` and publish its whole blocks under `tokens`. */
static void publish(Fixture& f, uint64_t id, const std::vector<int32_t>& tokens, int64_t n_tok) {
    CHECK_OK(f.kv.add_sequence(id));
    CHECK_OK(f.kv.ensure(id, (int64_t)tokens.size()));
    std::vector<std::vector<int32_t>> bt;
    for (int32_t g : f.pc.cached_groups()) bt.push_back(f.kv.block_table(id, g));
    CHECK_OK(f.pc.insert(id, tokens, {}, bt, n_tok));
}

/* ================================================================== pools */

TEST(pools_refuse_a_split_the_resolver_never_made) {
    Config c;
    c.vram_weights_mib = 0;     /* the resolver never ran */
    c.vram_kv_mib = 0;
    Pools p;
    CHECK_EQ(p.configure(c, 0, &rad_malloc_allocator(), &rad_malloc_allocator()), RAD_E_INVAL);
    CHECK(p.fit_report().find("vram_weights resolved to 0 MiB") != std::string::npos);
    CHECK(p.fit_report().find("vram_kv resolved to 0 MiB") != std::string::npos);
    CHECK(!p.configured());
}

TEST(pool_oversubscription_is_refused_by_name) {
    Config c;
    c.vram_weights_mib  = 24 * 1024;
    c.vram_kv_mib       = 12 * 1024;
    c.gpu_headroom_mib  =  1 * 1024;
    Pools p;
    const int64_t capacity = 32ll * 1024 * 1024 * 1024;   /* a 32 GiB card */
    CHECK_EQ(p.configure(c, capacity, &rad_malloc_allocator(), &rad_malloc_allocator()),
             RAD_E_NOMEM);
    /* By name and by how much -- refusing to start beats silently shrinking something (spec §6). */
    const std::string& r = p.fit_report();
    CHECK(r.find("vram_weights") != std::string::npos);
    CHECK(r.find("vram_kv") != std::string::npos);
    CHECK(r.find("OVER BY") != std::string::npos);
    CHECK(r.find("4.00 GiB") != std::string::npos);  /* 36 GiB wanted against a 32 GiB card */
    CHECK(!p.configured());
}

TEST(pool_reservation_never_falls_through) {
    Config c = small_config();
    Pools p;
    CHECK_OK(p.configure(c, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));

    /* The KV pool is 1 MiB. Ask for 2 and it is refused, by name, and the pool is untouched --
     * there is no path from here to the weight slab, which is the structural half of §6: the
     * mover works inside its own slab and cannot evict a KV block to promote an expert. */
    void* q = p.kv().reserve("oversized block array", 2 << 20, RAD_ALIGN_UNIT);
    CHECK(q == nullptr);
    CHECK(!p.kv().ok());
    REQUIRE_EQ((long long)p.kv().shortfalls().size(), 1ll);
    CHECK_EQ(p.kv().shortfalls()[0].pool, std::string("vram_kv"));
    CHECK_EQ(p.kv().shortfalls()[0].missing(), (2ll << 20) - (1ll << 20));
    CHECK_EQ(p.kv().used(), 0ll);
    CHECK_EQ(p.weights().used(), 0ll);          /* nothing leaked into the weight slab */

    void* ok = p.kv().reserve("block array", 512 << 10, RAD_ALIGN_UNIT);
    CHECK(ok != nullptr);
    CHECK_EQ(p.kv().used(), 512ll << 10);
}

/* A BACKING ALLOCATION THE ALLOCATOR REFUSES IS REPORTED AS A REFUSAL, with the allocator's reason
 * where it has one -- not as "0 B available", which reads as a measurement of the card and sends
 * the reader looking for memory that was never measured. */
TEST(a_refused_backing_allocation_says_it_was_refused_and_why) {
    struct Refuses final : MemAllocator {
        void* alloc(int64_t, int) override { return nullptr; }
        void  free(void*, int) override {}
        const char* name() const override { return "refuses"; }
    } none;
    Config c = small_config();
    Pools p;
    CHECK_EQ(p.configure(c, 0, &none, &none), RAD_E_NOMEM);
    const std::string& r = p.fit_report();
    CHECK(r.find("backing allocation of") != std::string::npos);
    CHECK(r.find("refused: the allocator returned nothing") != std::string::npos);
    CHECK(r.find("0 B available") == std::string::npos);
    CHECK(!p.configured());
}

/* ================================================================== the derived budget */

TEST(a_device_with_no_capacity_sizes_its_pools_from_the_program) {
    /* A device that reports no capacity: the pools are still derived, from what the program
     * needs rather than from a share of anything. */
    Config c = small_config();
    c.vram_weights_mib = 0;
    c.vram_kv_mib      = 0;
    VramFacts f{};
    f.statics    = 0;                 /* every op on the host: its weights stay in the mapping */
    f.kv_ceiling = 300LL << 20;
    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK_EQ(c.vram_weights_mib, (int64_t)1);           /* a pool nothing is placed in */
    CHECK_EQ(c.vram_kv_mib, (int64_t)300);              /* every sequence at the full context */

    f.statics = 5LL << 20;                              /* a device-domain op's weights */
    Config c2 = small_config();
    c2.vram_weights_mib = 0;
    c2.vram_kv_mib      = 0;
    CHECK_OK(vram_budget_resolve(c2, { f }, &b));
    CHECK_EQ(c2.vram_weights_mib, (int64_t)5);
}

TEST(a_program_that_copies_no_weight_still_gets_a_weights_pool) {
    /* Every op on the host -- CPU-only serving -- reads its weights where the container's mapping
     * has them, so nothing is copied to the weights pool. It still has to exist: a zero budget is
     * what the pools refuse as a resolver that never ran. */
    Config c = small_config();
    VramFacts f{};
    f.capacity   = 16LL << 30;        /* the host backend presents half the machine as a card */
    f.free       = f.capacity;
    f.arena      = 64LL << 20;
    f.kv_ceiling = 300LL << 20;
    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK_EQ(c.vram_weights_mib, (int64_t)1);
    CHECK(c.vram_kv_mib >= 299);
}

TEST(derived_budget_takes_the_statics_off_the_top_then_divides_the_rest) {
    /* With no lender stated the experts come first; what each side gets is the subject of the
     * cases below. */
    Config c = small_config();
    c.vram_weights_mib = 0;              /* both unset: derive */
    c.vram_kv_mib      = 0;
    c.gpu_headroom_mib = 96;
    c.max_seqs = 8;
    c.max_ctx  = 2048;

    VramFacts f{};
    f.capacity   = 32LL << 30;
    f.free       = f.capacity - (512LL << 20);   /* driver + code objects already resident */
    f.arena      = 144LL << 20;
    f.statics    = 2LL << 30;                    /* this rank's shard of the non-expert weights */
    f.experts    = 200LL << 30;                  /* far more than fits: the plane is elastic */
    f.kv_ceiling = 100LL << 30;                  /* far above any share, so no cap fires */
    f.kv_one     = 1LL << 30;

    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));

    /* Held, the driver's drift on it, the headroom and the arena. */
    CHECK_EQ(b.drift, 16LL << 20);
    CHECK_EQ(b.claimable, f.capacity - (512LL << 20) - (16LL << 20) - (96LL << 20) - (144LL << 20));

    /* THE FLOOR IS NOT A SHARE. The statics come out whole before the split sees anything, so
     * nothing the cache is given can demote an attention projection to host memory. */
    CHECK_EQ(b.statics, f.statics);
    CHECK_EQ(b.elastic, b.claimable - f.statics);
    CHECK_EQ(b.experts + b.kv, b.elastic);
    CHECK_EQ(b.weights, b.statics + b.experts);

    const int64_t MiB = 1024 * 1024;
    const int64_t unclaimed = f.capacity - (512LL << 20) - f.arena
                            - c.vram_weights_mib * MiB - c.vram_kv_mib * MiB;
    CHECK(unclaimed >= 112 * MiB);                /* never less than the headroom and the drift */
    CHECK(unclaimed < 114 * MiB);                 /* and never more than MiB rounding can strand */
}

static Config derive_config() {
    Config c = small_config();
    c.vram_weights_mib = 0; c.vram_kv_mib = 0; c.gpu_headroom_mib = 96;
    return c;
}
static VramFacts big_moe_facts() {
    VramFacts f{};
    f.capacity   = 32LL << 30;
    f.free       = f.capacity - (512LL << 20);
    f.arena      = 144LL << 20;
    f.statics    = 4LL << 30;
    f.experts    = 40LL << 30;                   /* more than the card: some must stream */
    f.kv_ceiling = 100LL << 30;
    f.kv_one     = 2LL << 30;
    return f;
}

/* A LENDING CACHE TAKES EVERYTHING THE EXPERTS CAN DO WITHOUT. The experts keep on the card what
 * the host pool cannot hold (expert_vram_floor, which place_test holds to the planner) and not a
 * byte more: every block the cache does not use is lent back to them. */
TEST(a_lending_cache_takes_everything_the_experts_can_do_without) {
    VramFacts f = big_moe_facts();
    f.lends     = true;
    f.expert_floor = 16LL << 30;                 /* the host pool takes 24 of the 40 */
    Config c = derive_config();
    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK(b.lends);
    CHECK_EQ(b.expert_floor, 16LL << 30);
    CHECK_EQ(b.experts, b.expert_floor);
    CHECK_EQ(b.kv, b.elastic - b.expert_floor);
    CHECK_EQ(b.weights, b.statics + b.experts);

    /* AND THE HOST POOL IS WHAT MOVES IT: a larger pool takes experts off the card, and the cache
     * gets exactly that much more. */
    VramFacts g = f;
    g.expert_floor = 12LL << 30;
    Config c2 = derive_config();
    VramBudget b2;
    CHECK_OK(vram_budget_resolve(c2, { g }, &b2));
    CHECK_EQ(b2.experts, b.experts - (4LL << 30));
    CHECK_EQ(b2.kv, b.kv + (4LL << 30));
    CHECK_EQ(b2.statics, b.statics);
}

/* WITHOUT A LENDER a block the cache does not use sits empty, so the experts come first and the
 * cache keeps one --max-model-len sequence. */
TEST(without_a_lender_the_experts_come_first_and_the_cache_keeps_one_sequence) {
    VramFacts f = big_moe_facts();
    f.lends     = false;
    f.expert_floor = 16LL << 30;
    Config c = derive_config();
    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK(!b.lends);
    CHECK_EQ(b.kv, f.kv_one);
    CHECK_EQ(b.experts, b.elastic - f.kv_one);
}

/* THE CACHE NEVER GETS LESS THAN ONE FULL-LENGTH SEQUENCE: a context the API accepts and no pool
 * can hold is a configuration error, and the experts' shortfall is the planner's to name. */
TEST(the_cache_keeps_one_full_length_sequence_even_against_the_expert_floor) {
    VramFacts f = big_moe_facts();
    f.lends     = true;
    f.expert_floor = f.experts;                  /* every expert wants the card */
    Config c = derive_config();
    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK_EQ(b.kv, f.kv_one);
    CHECK_EQ(b.experts, b.elastic - f.kv_one);

    /* And a cache that could never use more than its ceiling gives the rest to the experts. */
    VramFacts g = big_moe_facts();
    g.lends      = true;
    g.expert_floor = 1LL << 30;
    g.kv_ceiling = 3LL << 30;
    Config c2 = derive_config();
    VramBudget b2;
    CHECK_OK(vram_budget_resolve(c2, { g }, &b2));
    CHECK_EQ(b2.kv, g.kv_ceiling);
    CHECK_EQ(b2.experts, b2.elastic - g.kv_ceiling);
}

TEST(neither_side_is_given_more_than_it_could_use) {
    /* A model that FITS. The expert plane is 4 GiB and the cache can address 1.2; between them
     * they cannot spend a 32 GiB card, and the remainder is reported as unusable rather than
     * allocated into a pool that would sit empty. */
    Config c = small_config();
    c.vram_weights_mib = 0; c.vram_kv_mib = 0; c.gpu_headroom_mib = 96;

    VramFacts f{};
    f.capacity   = 32LL << 30;
    f.free       = f.capacity - (512LL << 20);
    f.arena      = 144LL << 20;
    f.statics    = 2LL << 30;
    f.experts    = 4LL << 30;
    f.kv_ceiling = 1200LL << 20;

    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK_EQ(b.experts, f.experts);
    CHECK_EQ(b.kv, f.kv_ceiling);
    CHECK_EQ(b.weights, b.statics + b.experts);
    CHECK_EQ(b.experts + b.kv + b.unusable, b.elastic);
    CHECK(b.unusable > 20LL << 30);               /* a big card and a small model */
}

TEST(a_dense_model_has_no_expert_share_to_lose) {
    /* A dense model in miniature: 42 dense layers and not one expert. An expert share would have
     * nothing to claim it, and a share nothing can claim is not slack -- it is memory no pool will
     * ever touch. Handing it to the weights anyway reserves tens of gigabytes for 0 B of experts
     * and leaves the cache short of the worst case it was asked to back. */
    Config c = small_config();
    c.vram_weights_mib = 0; c.vram_kv_mib = 0; c.gpu_headroom_mib = 96;
    c.max_seqs = 8;
    c.max_ctx  = 131072;

    VramFacts f{};
    f.capacity   = 32LL << 30;
    f.free       = f.capacity - (512LL << 20);
    f.arena      = 144LL << 20;
    /* DELIBERATELY NOT A WHOLE MiB. With no expert share padding it, the weights budget is the
     * static footprint to the byte, and the pools take MiB and truncate -- so this is exactly
     * the case where a pool comes out short of the weights it was sized for. */
    f.statics    = (1331LL << 20) + 12345;
    f.experts    = 0;
    f.kv_ceiling = 10LL << 30;

    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK_EQ(b.experts, 0ll);
    CHECK_EQ(b.kv, f.kv_ceiling);                 /* everything a paged cache can address */
    CHECK_EQ(b.experts + b.kv + b.unusable, b.elastic);

    /* AND THE FLOOR IS NEVER SHORT. The budget the pool is actually built from is the MiB
     * figure, not the byte one. */
    const int64_t MiB = 1024 * 1024;
    CHECK_EQ(b.weights, f.statics);               /* no expert share left to pad it */
    CHECK(c.vram_weights_mib * MiB >= f.statics); /* and the MiB figure the pool is built from covers it */
    CHECK(c.vram_weights_mib * MiB - f.statics < MiB);
}

/* THE ENGINE'S HARDWARE QUEUES ARE A FIXED TERM. Each one costs the card VRAM the driver backs it
 * with, and the streams are made after the card is read, so a split that left them out spent
 * them out of the headroom: 67 MiB a card on a two-card Flash-Next server, leaving 9 MiB free
 * under load against 96. The rank that needs most sets the term, as every fold here does. */
TEST(the_engine_s_queues_come_off_the_card_before_the_split) {
    Config c = small_config();
    c.vram_weights_mib = 0;
    c.vram_kv_mib      = 0;
    c.gpu_headroom_mib = 96;
    VramFacts f{};
    f.capacity   = 32LL << 30;
    f.free       = f.capacity - (512LL << 20);
    f.arena      = 144LL << 20;
    f.staging    = 98LL << 20;
    f.statics    = 2LL << 30;
    f.experts    = 200LL << 30;
    f.kv_ceiling = 100LL << 30;
    VramBudget b0;
    CHECK_OK(vram_budget_resolve(c, { f }, &b0));

    /* A fresh config: the resolver writes its answer back into the one it is given. */
    Config c2 = small_config();
    c2.vram_weights_mib = 0;
    c2.vram_kv_mib      = 0;
    c2.gpu_headroom_mib = 96;
    VramFacts g = f, h = f;
    g.n_queues = 2; g.queue_bytes = 33LL << 20;
    h.n_queues = 3; h.queue_bytes = 33LL << 20;     /* a rank with a second compute lane */
    VramBudget b;
    CHECK_OK(vram_budget_resolve(c2, { g, h }, &b));
    CHECK_EQ(b.queues, 99LL << 20);
    CHECK_EQ(b.n_queues, 3);
    CHECK_EQ(b.claimable, b0.claimable - (99LL << 20));
    CHECK(b.report.find("hardware queues") != std::string::npos);

    /* What is left after the pools is still the headroom and the drift, with the queues and the
     * staging paid. */
    const int64_t MiB = 1024 * 1024;
    const int64_t unclaimed = f.capacity - (512LL << 20) - f.arena - f.staging - b.queues
                            - c2.vram_weights_mib * MiB - c2.vram_kv_mib * MiB;
    CHECK(unclaimed >= 112 * MiB);
    CHECK(unclaimed < 114 * MiB);
}

TEST(the_smallest_card_binds_and_a_short_one_refuses_by_name) {
    Config c = small_config();
    c.vram_weights_mib = 0;
    c.vram_kv_mib      = 0;
    c.gpu_headroom_mib = 96;

    VramFacts big{}, small{};
    big.capacity   = 32LL << 30;  big.free   = big.capacity - (512LL << 20);
    big.arena      = 144LL << 20; big.kv_ceiling = 100LL << 30;
    small = big;
    small.capacity = 16LL << 30;  small.free = small.capacity - (512LL << 20);

    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { big, small }, &b));
    CHECK_EQ(b.capacity, 16LL << 30);             /* one split has to fit every card it is used on */

    /* And a card the fixed terms alone overrun is a refusal with the shortfall named, not a
     * silently shrunk pool. */
    Config c2 = small_config();
    c2.vram_weights_mib = 0; c2.vram_kv_mib = 0;
    c2.gpu_headroom_mib = 96;
    VramFacts tiny{};
    tiny.capacity = 256LL << 20;
    tiny.free     = 64LL << 20;
    tiny.arena    = 144LL << 20;
    VramBudget b2;
    CHECK(vram_budget_resolve(c2, { tiny }, &b2) < 0);
    CHECK(b2.report.find("SHORT BY") != std::string::npos);
}


/* ================================================================== group sizing */

TEST(group_sizing_wastes_nothing) {
    /* The vllm-radiance case: 48 gated-delta-net + 16 full + 5 drafter layers. vLLM's rule picks
     * the smallest bundle, 5, which divides neither 48 nor 16, and allocates 75 slots for 69 real
     * layers across 15 groups. patch_kv_group_size's search recovers 72/9. Radiance allocates 69
     * in 3, because a KV group here is one state kind with its own pool and there is no uniform
     * page size to pad up to. */
    BundleComparison c = compare_bundling({ 48, 16, 5 });
    CHECK_EQ(c.stock_group_size, 5ll);
    CHECK_EQ(c.stock_groups, 15ll);
    CHECK_EQ(c.stock_slots, 75ll);
    CHECK_EQ(c.patched_group_size, 8ll);
    CHECK_EQ(c.patched_groups, 9ll);
    CHECK_EQ(c.patched_slots, 72ll);
    CHECK_EQ(c.our_groups, 3ll);
    CHECK_EQ(c.our_slots, 69ll);           /* == sum(bundles): zero padding, structurally */

    /* An n:1 model pads nothing under vLLM's rule, and must pad nothing here either. */
    BundleComparison n1 = compare_bundling({ 50, 10 });
    CHECK_EQ(n1.stock_slots, 60ll);
    CHECK_EQ(n1.our_slots, 60ll);
    CHECK_EQ(n1.our_groups, 2ll);
}

/* ================================================================== KV manager */

TEST(kv_configure_reports_and_sizes) {
    Fixture f;
    CHECK_OK(f.build());
    const KVGroupPlan* full = f.kv.plan(G_FULL);
    const KVGroupPlan* gdn  = f.kv.plan(G_GDN);
    CHECK(full && gdn);
    CHECK_EQ(full->block_size, 16ll);
    CHECK_EQ(full->bytes_per_block, 2048ll);      /* 16 tok * 2 kv-heads * 8 dim * 2 B * K/V * 2 layers */
    CHECK_EQ(gdn->bytes_per_state, 16384ll);      /* 4 heads * 16x16 * 4 B * 4 layers */
    CHECK_EQ(gdn->n_states, 8ll);                 /* exactly max_seqs; not a heuristic */
    CHECK_EQ(f.kv.checkpoint_bytes(), 16384ll);
    CHECK_EQ(f.kv.checkpoint_slots(), 4ll);
    CHECK(full->n_blocks > 0);
    /* 1 MiB minus 8 states and ONE ALIGNMENT UNIT, all of it in pages of 2 KiB. The unit is the
     * budget admitting what the pool actually spends: every reserve() aligns its offset up, so a
     * carve that hands the paged groups every remaining byte overruns by the padding. The four
     * snapshots cost the pool nothing -- they are in host memory. */
    CHECK_EQ(full->n_blocks, ((1ll << 20) - 8 * 16384 - RAD_ALIGN_UNIT) / 2048);
}

TEST(kv_ranks_with_unequal_pools_carve_the_least_ranks_blocks_under_a_cap) {
    /* One block table serves every rank (Engine::serve), so every rank's paged groups must hold
     * the same number of blocks. Under an uneven tensor-parallel split the pools are not equal --
     * here a rank with 1 MiB and one with 2 MiB -- and Engine::configure_kv plans each, asks what
     * it could page, and caps every rank at the least before any carves. */
    Fixture a, b;
    a.cfg.checkpoint_slots = b.cfg.checkpoint_slots = 0;
    b.cfg.vram_kv_mib = 2;
    CHECK_OK(a.pools.configure(a.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_OK(b.pools.configure(b.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_OK(a.kv.plan_groups(hybrid_groups(), a.cfg));
    CHECK_OK(b.kv.plan_groups(hybrid_groups(), b.cfg));
    const int64_t ta = a.kv.paged_tokens(a.pools.kv().avail());
    const int64_t tb = b.kv.paged_tokens(b.pools.kv().avail());
    CHECK(ta > 0);
    CHECK(tb > ta);                              /* the larger pool pages more on its own */
    const int64_t least = ta < tb ? ta : tb;
    a.kv.cap_paged_tokens(least);
    b.kv.cap_paged_tokens(least);
    CHECK_OK(a.kv.carve(a.pools.kv()));
    CHECK_OK(b.kv.carve(b.pools.kv()));
    CHECK_EQ(a.kv.plan(G_FULL)->n_blocks, b.kv.plan(G_FULL)->n_blocks);
    CHECK_EQ(a.kv.plan(G_FULL)->n_blocks, least / 16);
    /* The cap does not touch what is not paged: each rank's own state count stays its own. */
    CHECK_EQ(a.kv.plan(G_GDN)->n_states, b.kv.plan(G_GDN)->n_states);
    /* paged_tokens() carved nothing: the uncapped figure is what carve() spends without a cap. */
    Fixture c;
    c.cfg.checkpoint_slots = 0;
    CHECK_OK(c.pools.configure(c.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_OK(c.kv.plan_groups(hybrid_groups(), c.cfg));
    const int64_t tc = c.kv.paged_tokens(c.pools.kv().avail());
    CHECK_OK(c.kv.carve(c.pools.kv()));
    CHECK_EQ(c.kv.plan(G_FULL)->n_blocks, tc / 16);
}

/* A RANK WHOSE ONLY PAGED GROUP IS ZERO (rad_kv_group_zero) pages nothing of its own, so its pages
 * limit nobody: it reports no token count, takes the cap the ranks holding the cache set, carves
 * nothing for the group and still holds the shared table's block count. Without a cap there is no
 * count to hold, and the carve says so rather than leaving the group at zero blocks. */
TEST(kv_a_rank_whose_only_paged_group_is_zero_takes_the_others_block_count) {
    std::vector<KVGroupInfo> zg = hybrid_groups();
    zg[1].zero = true;                           /* the one paged group */
    Fixture a, z;
    a.cfg.checkpoint_slots = z.cfg.checkpoint_slots = 0;
    CHECK_OK(a.pools.configure(a.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_OK(z.pools.configure(z.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_OK(a.kv.plan_groups(hybrid_groups(), a.cfg));
    CHECK_OK(z.kv.plan_groups(zg, z.cfg));
    const int64_t ta = a.kv.paged_tokens(a.pools.kv().avail());
    CHECK(ta > 0);
    CHECK_EQ(z.kv.paged_tokens(z.pools.kv().avail()), KVManager::kUnboundedTokens);
    a.kv.cap_paged_tokens(ta);
    z.kv.cap_paged_tokens(ta);
    CHECK_OK(a.kv.carve(a.pools.kv()));
    CHECK_OK(z.kv.carve(z.pools.kv()));
    CHECK_EQ(z.kv.plan(G_FULL)->n_blocks, a.kv.plan(G_FULL)->n_blocks);
    CHECK(z.kv.plan(G_FULL)->base == nullptr);
    CHECK_EQ(z.kv.plan(G_FULL)->pool_bytes, (int64_t)0);
    CHECK(!z.kv.lendable(G_FULL));
    /* Its recurrent state is its own and carved as on any rank. */
    CHECK(z.kv.plan(G_GDN)->base != nullptr);

    Fixture u;
    u.cfg.checkpoint_slots = 0;
    CHECK_OK(u.pools.configure(u.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_OK(u.kv.plan_groups(zg, u.cfg));
    CHECK_EQ(u.kv.carve(u.pools.kv()), RAD_E_STATE);
}

/* AN UNEVEN WORLD'S CHEAPER RANK pages the common token count in a smaller pool. The inverse of
 * paged_tokens is exact at the MiB: the size it names reaches the count and one MiB less does not,
 * and a rank whose only paged group is zero still has to hold its fixed costs. */
TEST(kv_pool_mib_for_tokens_is_the_inverse_of_paged_tokens) {
    constexpr int64_t MiB = 1ll << 20;
    Fixture dear, cheap, zero;
    for (Fixture* f : { &dear, &cheap, &zero }) { f->cfg.checkpoint_slots = 0; f->cfg.vram_kv_mib = 8; }
    std::vector<KVGroupInfo> cg = hybrid_groups();
    cg[G_FULL].decl.n_head_kv = 1;                  /* half the page of the dear rank's */
    std::vector<KVGroupInfo> zg = hybrid_groups();
    zg[G_FULL].zero = true;
    CHECK_OK(dear.kv.plan_groups(hybrid_groups(), dear.cfg));
    CHECK_OK(cheap.kv.plan_groups(cg, cheap.cfg));
    CHECK_OK(zero.kv.plan_groups(zg, zero.cfg));
    const int64_t T = dear.kv.paged_tokens(8 * MiB);
    CHECK(T > 0);
    const int64_t need = cheap.kv.pool_mib_for_tokens(T, 8);
    CHECK(need > 0 && need < 8);
    CHECK(cheap.kv.paged_tokens(need * MiB) >= T);
    CHECK(cheap.kv.paged_tokens((need - 1) * MiB) < T);
    /* The dear rank needs its whole pool for its own count. */
    CHECK_EQ(dear.kv.pool_mib_for_tokens(T, 8), (int64_t)8);
    /* Past what the largest size pages: no answer, not a short pool. */
    CHECK_EQ(cheap.kv.pool_mib_for_tokens(T * 1000, 8), (int64_t)-1);
    /* All paged groups zero: the count is never its limit, its fixed state still is. */
    const int64_t zn = zero.kv.pool_mib_for_tokens(T, 8);
    CHECK(zn >= 0 && zn * MiB >= zero.kv.fixed_bytes());
    CHECK(zn == 0 || (zn - 1) * MiB < zero.kv.fixed_bytes());
}

/* THE INVERSE ON A SHAPE, AND THE CARVE IT PROMISES. The dear rank's pool of `common_mib` sets the
 * token count; the cheap rank's inverse names the least whole MiB that pages it (one MiB less does
 * not, or does not hold the fixed costs); and a pool of exactly that size, capped at the count the
 * way configure_kv caps every rank, carves every paged group the dear rank's block count -- the
 * one block table. Returns the MiB the cheap rank needed. */
static int64_t check_inverse_and_carve(const std::vector<KVGroupInfo>& dear_g,
                                       const std::vector<KVGroupInfo>& cheap_g, int64_t common_mib) {
    constexpr int64_t MiB = 1ll << 20;
    Fixture dear, cheap;
    dear.cfg.checkpoint_slots = cheap.cfg.checkpoint_slots = 0;
    dear.cfg.vram_kv_mib = common_mib;
    CHECK_OK(dear.kv.plan_groups(dear_g, dear.cfg));
    CHECK_OK(cheap.kv.plan_groups(cheap_g, cheap.cfg));
    const int64_t T = dear.kv.paged_tokens(common_mib * MiB);
    CHECK(T > 0 && T != KVManager::kUnboundedTokens);
    const int64_t need = cheap.kv.pool_mib_for_tokens(T, common_mib);
    CHECK(need > 0 && need < common_mib);
    if (need <= 0) return need;
    CHECK(cheap.kv.paged_tokens(need * MiB) >= T);
    CHECK((need - 1) * MiB < cheap.kv.fixed_bytes() || cheap.kv.paged_tokens((need - 1) * MiB) < T);

    cheap.cfg.vram_kv_mib = need;
    CHECK_OK(dear.pools.configure(dear.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_OK(cheap.pools.configure(cheap.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_EQ(cheap.pools.kv().avail(), need * MiB);
    CHECK_EQ(dear.kv.paged_tokens(dear.pools.kv().avail()), T);
    dear.kv.cap_paged_tokens(T);
    cheap.kv.cap_paged_tokens(T);
    CHECK_OK(dear.kv.carve(dear.pools.kv()));
    CHECK_OK(cheap.kv.carve(cheap.pools.kv()));
    for (int32_t g = 1; g < (int32_t)dear_g.size(); ++g) {
        if (!dear.kv.plan(g)->paged()) continue;
        CHECK_EQ(cheap.kv.plan(g)->n_blocks, dear.kv.plan(g)->n_blocks);
        CHECK(dear.kv.plan(g)->n_blocks > 0);
    }
    return need;
}

static KVGroupInfo paged_group(const char* name, int kind, int64_t heads, int64_t block,
                               std::vector<int32_t> layers, int64_t window = 0) {
    KVGroupInfo k;
    k.name = name;
    k.decl.kind = kind;
    k.decl.dtype = RAD_BF16;
    k.decl.n_head_kv = heads;
    k.decl.head_dim = 8;
    k.decl.window = window;
    k.block_size = block;
    k.layers = std::move(layers);
    return k;
}

/* A WINDOWED GROUP is capped at its window on both ranks, and the cap moves the bytes it frees to
 * the other groups -- a kink in paged_split the inverse has to cross exactly. Here the dear rank's
 * count is past the window, so its window group holds max_seqs x window_blocks on both ranks. */
TEST(kv_pool_mib_for_tokens_crosses_a_windowed_groups_cap) {
    std::vector<KVGroupInfo> dear = hybrid_groups();
    dear.push_back(paged_group("swa", RAD_KV_WINDOW, 4, 16, { 6, 7, 8, 9 }, 2048));
    std::vector<KVGroupInfo> cheap = dear;
    cheap[G_FULL].decl.n_head_kv = 1;
    check_inverse_and_carve(dear, cheap, 32);
    /* And with the cheap rank's window group the cheaper one: what it saves is all in the window. */
    cheap = dear;
    cheap[3].decl.n_head_kv = 1;
    check_inverse_and_carve(dear, cheap, 32);
}

/* TWO PAGED GROUPS AT DIFFERENT BLOCK SIZES: one token count, two block counts, each the dear
 * rank's. */
TEST(kv_pool_mib_for_tokens_holds_across_block_sizes) {
    std::vector<KVGroupInfo> dear = hybrid_groups();
    dear.push_back(paged_group("full32", RAD_KV_FULL, 2, 32, { 6, 7, 8 }));
    std::vector<KVGroupInfo> cheap = dear;
    cheap[G_FULL].decl.n_head_kv = 1;
    check_inverse_and_carve(dear, cheap, 8);
    cheap = dear;
    cheap[3].decl.n_head_kv = 1;
    check_inverse_and_carve(dear, cheap, 8);
}

/* AN ATTENTION-ZERO GROUP BESIDE ANOTHER PAGED GROUP: the zero group costs this rank nothing and
 * takes no part in the split, so the other group alone sets what its pool pages -- and the zero
 * group still holds the shared table's block count after the carve. */
TEST(kv_pool_mib_for_tokens_with_a_zero_group_beside_a_paged_one) {
    std::vector<KVGroupInfo> dear = hybrid_groups();
    dear.push_back(paged_group("full32", RAD_KV_FULL, 2, 32, { 6, 7, 8 }));
    std::vector<KVGroupInfo> cheap = dear;
    cheap[G_FULL].zero = true;
    const int64_t need = check_inverse_and_carve(dear, cheap, 8);
    /* Cheaper than a rank that keeps the group at half its heads. */
    std::vector<KVGroupInfo> kept = dear;
    kept[G_FULL].decl.n_head_kv = 1;
    CHECK(need < check_inverse_and_carve(dear, kept, 8));
}

/* The re-split moves what the cheaper rank's KV pool saves into its weights pool, never past every
 * weight resident and never below the common pool; what the cap leaves is reported, not allocated.
 * A need that is not below the common pool changes nothing -- the even world's case. */
TEST(an_uneven_worlds_cheaper_rank_gives_its_saved_kv_to_its_weights_up_to_all_resident) {
    constexpr int64_t MiB = 1ll << 20;
    RankSplit s = vram_budget_rank_split(1000, 800, 200, 1300 * MiB);
    CHECK(s.resplit);
    CHECK_EQ(s.kv_mib, (int64_t)200);
    CHECK_EQ(s.weights_mib, (int64_t)1300);
    CHECK_EQ(s.unusable_mib, (int64_t)300);
    CHECK(s.weights_mib + s.kv_mib <= 1000 + 800);
    s = vram_budget_rank_split(1000, 800, 200, 5000 * MiB);     /* room for all of it */
    CHECK_EQ(s.weights_mib, (int64_t)1600);
    CHECK_EQ(s.unusable_mib, (int64_t)0);
    s = vram_budget_rank_split(1000, 800, 200, 500 * MiB);      /* already all resident */
    CHECK_EQ(s.weights_mib, (int64_t)1000);
    CHECK_EQ(s.kv_mib, (int64_t)200);
    CHECK_EQ(s.unusable_mib, (int64_t)600);
    s = vram_budget_rank_split(1000, 800, 800, 5000 * MiB);     /* costs the same: untouched */
    CHECK(!s.resplit);
    CHECK_EQ(s.weights_mib, (int64_t)1000);
    CHECK_EQ(s.kv_mib, (int64_t)800);
    s = vram_budget_rank_split(1000, 800, -1, 5000 * MiB);      /* no answer: untouched */
    CHECK(!s.resplit);
    CHECK_EQ(s.kv_mib, (int64_t)800);
}

TEST(kv_refuses_a_group_with_no_block_size) {
    /* Block size comes off the resolved attention kernel. A zero here means the attention op did
     * not resolve, and the manager refuses rather than inventing 16. */
    auto g = hybrid_groups();
    g[G_FULL].block_size = 0;
    Fixture f;
    f.cfg.checkpoint_slots = 0;
    CHECK_OK(f.pools.configure(f.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_EQ(f.kv.configure(g, f.cfg, f.pools.kv()), RAD_E_NOKERNEL);
}

TEST(kv_refuses_when_fixed_costs_do_not_fit) {
    /* --max-seqs 128 at 16 KiB of GDN state each is 2 MiB of fixed cost against a 1 MiB
     * the KV pool. Refused by name, with the arithmetic printed; not silently served at fewer
     * sequences than the operator asked for. */
    Fixture f;
    f.cfg.max_seqs = 128;
    f.cfg.checkpoint_slots = 0;
    CHECK_OK(f.pools.configure(f.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_EQ(f.kv.configure(hybrid_groups(), f.cfg, f.pools.kv()), RAD_E_NOMEM);

    /* AND THE SNAPSHOTS ARE NOT PART OF IT. The same config fits at 8 sequences whatever
     * --checkpoint-slots asks for -- 64 x 16 KiB is the whole pool, and it is host memory -- and
     * the paged groups get exactly what they get with none. */
    Fixture g2;
    g2.cfg.max_seqs = 8;
    g2.cfg.checkpoint_slots = 0;
    CHECK_OK(g2.pools.configure(g2.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_OK(g2.kv.configure(hybrid_groups(), g2.cfg, g2.pools.kv()));
    Fixture g3;
    g3.cfg.max_seqs = 8;
    g3.cfg.checkpoint_slots = 64;
    CHECK_OK(g3.pools.configure(g3.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_OK(g3.kv.configure(hybrid_groups(), g3.cfg, g3.pools.kv()));
    CHECK_EQ(g3.kv.fixed_bytes(), g2.kv.fixed_bytes());
    CHECK_EQ(g3.kv.plan(G_FULL)->n_blocks, g2.kv.plan(G_FULL)->n_blocks);
}


TEST(a_manager_whose_carve_was_refused_refuses_rather_than_reads_past_a_vector) {
    /* THE GUARD AND THE READ ARE DIFFERENT VECTORS. plan_groups() fills the plan table and carve()
     * sizes the block state against it, at two points with a failure between them, so a refused
     * carve can leave a manager whose plans_ satisfy every bounds check and whose free_usable_ is
     * empty. reserve_block is that pair exactly -- a check against plans_ and a
     * read of free_usable_ -- and it is the tiers' restore path, reached from a cache in host
     * memory rather than from the scheduler.
     *
     * The operator sees the refusal and the caller aborts; what this pins is that getting there is
     * a return code and not a read past the end of a vector. */
    Fixture f;
    f.cfg.max_seqs = 128;                 /* the fixed costs do not fit, as the case above shows */
    f.cfg.checkpoint_slots = 0;
    CHECK_OK(f.pools.configure(f.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_EQ(f.kv.configure(hybrid_groups(), f.cfg, f.pools.kv()), RAD_E_NOMEM);

    int32_t b = 7;
    CHECK(f.kv.reserve_block(G_FULL, &b) < 0);
    CHECK_EQ(b, -1);                      /* and it did not leave a block id a caller would use */
    CHECK(f.kv.reserve_specific_block(G_FULL, 0) < 0);
    CHECK(f.kv.retain_block(G_FULL, 0) < 0);
    CHECK_EQ(f.kv.n_groups(), 0);
    CHECK_EQ(f.kv.total_blocks(G_FULL), 0);
    CHECK_EQ(f.kv.live_blocks(G_FULL), 0);
    CHECK_EQ(f.kv.free_blocks(G_FULL), 0);
}

TEST(kv_alloc_extend_free_and_the_free_list) {
    Fixture f;
    CHECK_OK(f.build());
    const int64_t n0 = f.kv.free_blocks(G_FULL);
    const int64_t s0 = f.kv.free_states(G_GDN);

    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 1));
    CHECK_EQ((long long)f.kv.block_table(1, G_FULL).size(), 1ll);
    CHECK(f.kv.state_slot(1, G_GDN) >= 0);                 /* the same manager owns the GDN slot */
    CHECK_EQ(f.kv.free_blocks(G_FULL), n0 - 1);
    CHECK_EQ(f.kv.free_states(G_GDN), s0 - 1);

    CHECK_OK(f.kv.ensure(1, 16));
    CHECK_EQ((long long)f.kv.block_table(1, G_FULL).size(), 1ll);
    CHECK_OK(f.kv.ensure(1, 17));
    CHECK_EQ((long long)f.kv.block_table(1, G_FULL).size(), 2ll);
    CHECK_OK(f.kv.ensure(1, 64));
    CHECK_EQ((long long)f.kv.block_table(1, G_FULL).size(), 4ll);

    f.kv.free_sequence(1);
    CHECK_EQ(f.kv.free_blocks(G_FULL), n0);
    CHECK_EQ(f.kv.free_states(G_GDN), s0);
    CHECK(!f.kv.has_sequence(1));

    /* Churn: a free list that leaks one block per sequence looks fine for an hour. */
    for (uint64_t i = 0; i < 200; ++i) {
        CHECK_OK(f.kv.add_sequence(100 + i));
        CHECK_OK(f.kv.ensure(100 + i, 40));
        f.kv.free_sequence(100 + i);
    }
    CHECK_EQ(f.kv.free_blocks(G_FULL), n0);
    CHECK_EQ(f.kv.free_states(G_GDN), s0);
}

TEST(kv_copy_on_write_on_fork) {
    Fixture f;
    CHECK_OK(f.build());
    const int64_t n0 = f.kv.free_blocks(G_FULL);

    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 24));                       /* 2 blocks; the second is half full */
    auto parent = f.kv.block_table(1, G_FULL);
    CHECK_EQ((long long)parent.size(), 2ll);

    CHECK_OK(f.kv.fork(1, 2));
    CHECK(f.kv.block_table(2, G_FULL) == parent);            /* shared, no copy yet */
    CHECK_EQ(f.kv.block_refcount(G_FULL, parent[0]), 2);
    CHECK_EQ(f.kv.block_refcount(G_FULL, parent[1]), 2);
    CHECK_EQ(f.kv.free_blocks(G_FULL), n0 - 2);              /* a fork of 24 tokens costs no blocks */
    CHECK(f.kv.state_slot(2, G_GDN) >= 0);
    CHECK(f.kv.state_slot(2, G_GDN) != f.kv.state_slot(1, G_GDN));  /* a GDN state cannot be shared */

    /* The child writes. Only the PARTIAL block is copied; the full one stays shared. */
    CHECK_OK(f.kv.ensure(2, 32));
    auto child = f.kv.block_table(2, G_FULL);
    CHECK_EQ(child[0], parent[0]);
    CHECK(child[1] != parent[1]);
    CHECK_EQ(f.kv.block_refcount(G_FULL, parent[1]), 1);
    auto copies = f.kv.take_pending_copies();
    CHECK_EQ((long long)copies.size(), 1ll);
    CHECK_EQ(copies[0].src_block, parent[1]);
    CHECK_EQ(copies[0].dst_block, child[1]);
    CHECK_EQ(copies[0].group, (int32_t)G_FULL);

    /* The parent now holds its partial block alone, so its next write copies nothing. */
    CHECK_OK(f.kv.ensure(1, 32));
    CHECK_EQ(f.kv.block_table(1, G_FULL)[1], parent[1]);
    CHECK_EQ((long long)f.kv.take_pending_copies().size(), 0ll);

    f.kv.free_sequence(1);
    f.kv.free_sequence(2);
    CHECK_EQ(f.kv.free_blocks(G_FULL), n0);
}

TEST(kv_preemption_by_recompute_and_the_undeadlockable_request) {
    Fixture f;
    CHECK_OK(f.build());
    const int64_t total = f.kv.total_blocks(G_FULL);

    /* A request bigger than the whole pool is failed, not preempted for. Preempting the entire
     * server would not make it fit, and pretending otherwise is the deadlock §17 names. */
    CHECK(!f.kv.can_ever_fit((total + 8) * 16));
    CHECK_OK(f.kv.add_sequence(1));
    std::vector<uint64_t> none;
    CHECK_EQ(f.kv.make_room(1, (total + 8) * 16, none, nullptr), RAD_E_FULL);
    f.kv.free_sequence(1);

    /* Fill the pool, then admit one more by preempting. Four sequences, leaving state slots for
     * the hog and the newcomer -- max_seqs is 8 and a state slot is per sequence, exactly. */
    const int64_t per = 16 * 16;             /* 16 blocks each */
    for (uint64_t i = 1; i <= 4; ++i) {
        CHECK_OK(f.kv.add_sequence(i));
        CHECK_OK(f.kv.ensure(i, per));
    }

    /* Drain the rest so the next ensure genuinely cannot be served. */
    uint64_t hog = 900;
    CHECK_OK(f.kv.add_sequence(hog));
    CHECK_OK(f.kv.ensure(hog, f.kv.free_blocks(G_FULL) * 16));
    CHECK_EQ(f.kv.free_blocks(G_FULL), 0ll);

    uint64_t newcomer = 999;
    CHECK_OK(f.kv.add_sequence(newcomer));
    CHECK_EQ(f.kv.ensure(newcomer, per), RAD_E_FULL);

    std::vector<uint64_t> victims = { hog, 1 };
    std::vector<uint64_t> preempted;
    CHECK_OK(f.kv.make_room(newcomer, per, victims, &preempted));
    CHECK(!preempted.empty());
    /* seq_tokens() == 0 IS the "needs re-prefill" reading -- see KVManager::preempt. */
    CHECK_EQ(f.kv.seq_tokens(preempted[0]), 0ll);
    CHECK(f.kv.block_table(preempted[0], G_FULL).empty());
    CHECK_OK(f.kv.ensure(newcomer, per));
}

/* THE WINDOW BEHIND A PREFIX HIT: a checkpoint slot carries window_blocks pages a WINDOW group,
 * window_rows() names the pages a query at P reads from the table that wrote them, seed_window()
 * gives a resumed sequence fresh pages at the same floor -- and a table whose floor moved past it
 * answers false, which is the fallback (no snapshot, a full re-prefill). */
TEST(kv_window_snapshot_rows_and_seed) {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    KVGroupInfo k;
    k.name = "swa";
    k.decl.kind = RAD_KV_WINDOW;
    k.decl.dtype = RAD_BF16;
    k.decl.n_head_kv = 2;
    k.decl.head_dim = 8;
    k.decl.window = 64;
    k.block_size = 16;
    k.layers = { 0 };
    g.push_back(k);
    Config cfg = small_config();
    Pools p;
    CHECK_OK(p.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    KVManager kv;
    CHECK_OK(kv.configure(g, cfg, p.kv()));
    CHECK(kv.has_window_snapshot());
    CHECK_EQ(kv.checkpoint_bytes(), 5 * kv.plan(G_FULL)->bytes_per_block);

    CHECK_OK(kv.add_sequence(1));
    CHECK_OK(kv.ensure(1, 400));                         /* a prompt of 400 in one chunk */
    std::vector<int32_t> rows;
    CHECK(kv.window_rows(1, 400, &rows));
    CHECK_EQ((long long)rows.size(), 5ll);               /* padded to window_blocks */
    /* A query at 400 reads back to 337: blocks 21..24, the floor recycle_window keeps. */
    for (int j = 0; j < 4; ++j) CHECK_EQ(rows[(size_t)j], kv.block_table(1, G_FULL)[(size_t)(21 + j)]);
    CHECK_EQ(rows[4], -1);
    /* A short prefix: the window starts at zero and the rest is padding. */
    CHECK(kv.window_rows(1, 32, &rows));
    CHECK_EQ(rows[0], kv.block_table(1, G_FULL)[0]);
    CHECK_EQ(rows[2], -1);
    /* Not on a block: no rows. */
    CHECK(!kv.window_rows(1, 40, &rows));
    /* The floor moved past it (a decode step slid the window): the fallback. */
    CHECK_OK(kv.ensure(1, 401));
    CHECK(!kv.window_rows(1, 64, &rows));
    CHECK(kv.window_rows(1, 400, &rows));

    /* A resumed sequence: seeded at the floor, extended from there, and its rows are its own. */
    CHECK_OK(kv.add_sequence(2));
    CHECK_OK(kv.seed_window(2, 400));
    CHECK_EQ(kv.first_block_pos(2, G_FULL), 336ll);
    CHECK_EQ((long long)kv.block_table(2, G_FULL).size(), 4ll);
    CHECK_EQ(kv.seq_tokens(2), 400ll);
    CHECK_OK(kv.ensure(2, 432, 400));
    CHECK_EQ(kv.first_block_pos(2, G_FULL), 336ll);
    CHECK_EQ((long long)kv.block_table(2, G_FULL).size(), 6ll);
    std::vector<int32_t> rows2;
    CHECK(kv.window_rows(2, 400, &rows2));
    for (int j = 0; j < 4; ++j) CHECK_EQ(rows2[(size_t)j], kv.block_table(2, G_FULL)[(size_t)j]);
    /* A table already holding window pages is not seeded twice. */
    CHECK_EQ(kv.seed_window(2, 400), RAD_E_STATE);
    kv.free_sequence(1);
    kv.free_sequence(2);
}

/* A SNAPSHOT ONLY WHERE A HIT RESUMES FROM IT: the trunk's window in a model with no linear state.
 * A model with linear state re-prefills a hit on a windowed trunk, and a drafter's window is
 * recomputed by draft refill, so a window section in either slot would be copied at every
 * checkpoint and never read. */
TEST(kv_window_snapshot_only_where_a_hit_resumes_from_it) {
    KVGroupInfo w;
    w.name = "swa";
    w.decl.kind = RAD_KV_WINDOW;
    w.decl.dtype = RAD_BF16;
    w.decl.n_head_kv = 2;
    w.decl.head_dim = 8;
    w.decl.window = 64;
    w.block_size = 16;
    w.layers = { 6 };
    const Config cfg = small_config();
    {   /* beside linear state: the slot holds the state and nothing more */
        std::vector<KVGroupInfo> g = hybrid_groups();
        g.push_back(w);
        Pools p;
        CHECK_OK(p.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
        KVManager kv;
        CHECK_OK(kv.configure(g, cfg, p.kv()));
        CHECK(!kv.has_window_snapshot());
        CHECK_EQ(kv.checkpoint_bytes(), 16384ll);
    }
    {   /* a drafter's own window over a full-attention trunk */
        std::vector<KVGroupInfo> g = hybrid_groups(/*with_linear=*/false);
        w.drafter = true;
        g.push_back(w);
        Pools p;
        CHECK_OK(p.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
        KVManager kv;
        CHECK_OK(kv.configure(g, cfg, p.kv()));
        CHECK(!kv.has_window_snapshot());
        CHECK_EQ(kv.checkpoint_bytes(), 0ll);
    }
}

TEST(kv_window_group_recycles_below_the_window) {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    KVGroupInfo k;
    k.name = "swa";
    k.decl.kind = RAD_KV_WINDOW;
    k.decl.dtype = RAD_BF16;
    k.decl.n_head_kv = 2;
    k.decl.head_dim = 8;
    k.decl.window = 64;
    k.block_size = 16;
    k.layers = { 0 };
    g.push_back(k);

    Config cfg = small_config();
    Pools p;
    CHECK_OK(p.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    KVManager kv;
    cfg.checkpoint_slots = 0;
    CHECK_OK(kv.configure(g, cfg, p.kv()));
    /* window 64 at block 16 is 4 blocks plus one for the straddle. */
    CHECK_EQ(kv.plan(G_FULL)->window_blocks, 5ll);
    /* Sized against the cap, not against the whole budget: a window a sequence plus one step's
     * worth of tokens (max_tok) and a straddle block a sequence -- a chunk's own blocks are held
     * beside the window behind its first query. Past that is memory nobody will ever address. */
    CHECK_EQ(kv.total_blocks(G_FULL), cfg.max_seqs * 5 + (cfg.max_tok + 15) / 16 + cfg.max_seqs);

    const int64_t n0 = kv.free_blocks(G_FULL);
    CHECK_OK(kv.add_sequence(1));
    CHECK_OK(kv.ensure(1, 80));                          /* a chunk from 0: all of it held */
    CHECK_EQ(kv.first_block_pos(1, G_FULL), 0ll);
    CHECK_EQ((long long)kv.block_table(1, G_FULL).size(), 5ll);
    /* A chunk from 80 to 400: its first query (80) reads back to 17, so block 1 stays and only
     * block 0 goes -- NOT the window behind 400, which left rows 80..335 with no keys. */
    CHECK_OK(kv.ensure(1, 400));
    CHECK_EQ(kv.first_block_pos(1, G_FULL), 16ll);
    CHECK_EQ((long long)kv.block_table(1, G_FULL).size(), 24ll);
    /* A decode step from 400: back to the window, recycled rather than grown. */
    CHECK_OK(kv.ensure(1, 401));
    CHECK_EQ(kv.first_block_pos(1, G_FULL), 336ll);
    CHECK_EQ((long long)kv.block_table(1, G_FULL).size(), 5ll);
    CHECK_EQ(kv.free_blocks(G_FULL), n0 - 5);
    kv.free_sequence(1);
    CHECK_EQ(kv.free_blocks(G_FULL), n0);

    /* THE PROMPT LONGER THAN THE WINDOW, in one chunk from 0: every position keeps its slot. The
     * old rule rebased this table at ceil(400/16) - 5 = 20 blocks, and a sliding layer read no
     * keys at all for positions 0..319. */
    CHECK_OK(kv.add_sequence(2));
    CHECK_OK(kv.ensure(2, 400));
    CHECK_EQ(kv.first_block_pos(2, G_FULL), 0ll);
    CHECK_EQ((long long)kv.block_table(2, G_FULL).size(), 25ll);
    /* And the scheduler's form names the step's first query itself: a chunk grown in the same
     * step (ensure called twice) keeps what the first call's first query reads. */
    CHECK_OK(kv.ensure(2, 432, 400));
    CHECK_EQ(kv.first_block_pos(2, G_FULL), 336ll);
    CHECK_OK(kv.ensure(2, 464, 400));
    CHECK_EQ(kv.first_block_pos(2, G_FULL), 336ll);
    CHECK_EQ((long long)kv.block_table(2, G_FULL).size(), 8ll);
    kv.free_sequence(2);
    CHECK_EQ(kv.free_blocks(G_FULL), n0);

    /* A COMMITTED CHUNK GIVES ITS BLOCKS BACK at once: trim_window to the new end keeps the window
     * behind it and nothing else, so a prefill that waits a step for its next chunk holds a window
     * and not a chunk. */
    CHECK_OK(kv.add_sequence(3));
    CHECK_OK(kv.ensure(3, 400));
    CHECK_EQ((long long)kv.block_table(3, G_FULL).size(), 25ll);
    kv.trim_window(3, 400);
    CHECK_EQ(kv.first_block_pos(3, G_FULL), 336ll);
    CHECK_EQ((long long)kv.block_table(3, G_FULL).size(), 4ll);
    CHECK_EQ(kv.free_blocks(G_FULL), n0 - 4);
    CHECK_OK(kv.ensure(3, 401));
    CHECK_EQ((long long)kv.block_table(3, G_FULL).size(), 5ll);
    kv.free_sequence(3);
    CHECK_EQ(kv.free_blocks(G_FULL), n0);
}

TEST(kv_conv_slot_is_a_rolling_window) {
    /* libr4d's conv update treats the state cache as a rolling window of width-1 + num_spec
     * entries and reads at the slot the last accepted token left, so a speculative rejection is a
     * change of read offset rather than a recompute (spec §10). The manager's job is to make the
     * slot big enough for that, and it is the SAME manager -- two allocators would disagree about
     * what a sequence owns. */
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    KVGroupInfo k;
    k.name = "conv";
    k.decl.kind = RAD_KV_CONV;
    k.decl.dtype = RAD_BF16;
    k.decl.n_head_kv = 2;
    k.decl.head_dim = 8;
    k.decl.conv_width = 4;
    k.layers = { 0, 1 };
    g.push_back(k);                           /* handle 1 */
    g.push_back(hybrid_groups()[G_FULL]);     /* a paged group beside it, as in a real hybrid */

    Config cfg = small_config();
    cfg.n_spec = 3;
    Pools p;
    CHECK_OK(p.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    KVManager kv;
    cfg.checkpoint_slots = 0;
    CHECK_OK(kv.configure(g, cfg, p.kv()));

    /* The conv group is handle 1 here -- the sentinel is 0, as it is in every group table. */
    const int32_t G_CONV = 1;
    CHECK_EQ(kv.plan(G_CONV)->conv_slots, 6ll);                    /* (4-1) + 3 */
    CHECK_EQ(kv.plan(G_CONV)->bytes_per_state, 6ll * 2 * 8 * 2 * 2);
    CHECK_EQ(kv.plan(G_CONV)->n_states, cfg.max_seqs);

    const int64_t s0 = kv.free_states(G_CONV);
    CHECK_OK(kv.add_sequence(1));
    CHECK_OK(kv.ensure(1, 8));
    CHECK(kv.state_slot(1, G_CONV) >= 0);
    CHECK_EQ(kv.free_states(G_CONV), s0 - 1);
    kv.free_sequence(1);
    CHECK_EQ(kv.free_states(G_CONV), s0);
}

/* THE SLOT THE MANAGER HANDS OUT IS THE SLOT THE KERNELS INDEX. The kernels take a conv slot's
 * layout from the state tensor's last extent, conv_width-1 + n_spec; the manager clears a slot
 * when it hands it out and saves and restores it at a checkpoint in the layout bytes_per_state
 * describes. A builder figure one entry wider a slot -- what the allocation once carried --
 * made handing slot s to a new sequence zero the start of slot s+1's window, the window of a
 * sequence still decoding. The manager refuses the disagreement rather than serve with it. */
TEST(kv_refuses_a_state_slot_the_kernels_would_not_index) {
    auto group = [](int kind, int64_t bytes_per_state) {
        KVGroupInfo k;
        k.name = "state";
        k.decl.kind = kind;
        k.decl.dtype = RAD_BF16;
        k.decl.n_head_kv = 2;
        k.decl.head_dim = 8;
        k.decl.state_dim[0] = 4;
        k.decl.state_dim[1] = 8;
        k.decl.conv_width = 4;
        k.layers = { 0, 1 };
        k.bytes_per_state = bytes_per_state;
        return k;
    };
    auto configure = [](const KVGroupInfo& k) -> int {
        std::vector<KVGroupInfo> g;
        g.push_back(null_group());
        g.push_back(k);
        Config cfg = small_config();
        cfg.n_spec = 3;
        cfg.checkpoint_slots = 0;
        Pools p;
        if (p.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()) < 0)
            return RAD_E_NOMEM;
        KVManager kv;
        return kv.configure(g, cfg, p.kv());
    };
    /* Conv: (4-1) + 3 = 6 entries of 2 x 8 bf16 channels, over two layers. */
    CHECK_OK(configure(group(RAD_KV_CONV, 6LL * 2 * 8 * 2 * 2)));
    CHECK_EQ(configure(group(RAD_KV_CONV, 7LL * 2 * 8 * 2 * 2)), RAD_E_SHAPE);
    /* Linear: 2 heads of 4 x 8 bf16, over two layers. */
    CHECK_OK(configure(group(RAD_KV_LINEAR, 2LL * 4 * 8 * 2 * 2)));
    CHECK_EQ(configure(group(RAD_KV_LINEAR, 2LL * 4 * 8 * 2 * 2 + 64)), RAD_E_SHAPE);
}

TEST(kv_adopts_the_blocks_a_cache_hit_found) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    auto prompt = tok_run(555, 96);
    publish(f, 1, prompt, 64);                 /* 4 blocks indexed */

    std::vector<std::vector<int32_t>> found;
    CacheHit h = f.pc.lookup(prompt, {}, &found);
    CHECK_EQ(h.n_attn_tokens, 64);
    CHECK_EQ((long long)found.size(), 1ll);
    CHECK_EQ((long long)found[0].size(), 4ll);

    const int64_t before = f.kv.free_blocks(G_FULL);
    CHECK_OK(f.kv.add_sequence(2));
    CHECK_OK(f.kv.adopt(2, f.pc.cached_groups()[0], found[0], h.n_attn_tokens));
    CHECK_EQ(f.kv.free_blocks(G_FULL), before);                 /* a hit costs no blocks */
    CHECK_EQ(f.kv.seq_tokens(2), 64ll);
    CHECK_EQ(f.kv.block_refcount(G_FULL, found[0][0]), 3);      /* seq 1 + cache + seq 2 */

    /* A partial block is never adopted: the next token written into it would corrupt whoever else
     * holds it, and the cache cannot serve it to anyone anyway. */
    CHECK_OK(f.kv.add_sequence(3));
    CHECK_EQ(f.kv.adopt(3, 0, found[0], 60), RAD_E_INVAL);

    /* Extending past the hit allocates only the new blocks, and the shared ones stay shared. */
    CHECK_OK(f.kv.ensure(2, 96));
    CHECK_EQ((long long)f.kv.block_table(2, G_FULL).size(), 6ll);
    CHECK_EQ(f.kv.block_table(2, G_FULL)[0], found[0][0]);
    CHECK_EQ((long long)f.kv.take_pending_copies().size(), 0ll);
}

/* ================================================================== chained hash */

TEST(chained_hash_identity) {
    auto a = tok_run(1000, 16);
    auto b = tok_run(2000, 16);
    auto shared = tok_run(3000, 16);
    std::vector<MMContentKey> no_mm;

    BlockHash root;
    BlockHash a0 = chain_block_hash(root, a.data(), 16, nullptr, 0, 0);
    BlockHash b0 = chain_block_hash(root, b.data(), 16, nullptr, 0, 0);
    CHECK(a0 != b0);
    CHECK(a0.valid());

    /* The same prefix hits. */
    CHECK(chain_block_hash(root, a.data(), 16, nullptr, 0, 0) == a0);

    /* The same SPAN under a different prefix does not. This is the whole point of chaining: a
     * block sitting in the middle of another conversation must not be served here. */
    BlockHash a1 = chain_block_hash(a0, shared.data(), 16, nullptr, 0, 16);
    BlockHash b1 = chain_block_hash(b0, shared.data(), 16, nullptr, 0, 16);
    CHECK(a1 != b1);

    /* The same span, same parent, different position is also a different block. */
    CHECK(chain_block_hash(a0, shared.data(), 16, nullptr, 0, 32) != a1);

    /* Multimodal hashes BY CONTENT. Identical placeholder token ids, different embeddings. */
    float img1[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    float img2[8] = { 1, 2, 3, 4, 5, 6, 7, 9 };
    MMContentKey m1{ 0, 16, mm_content_hash(img1, sizeof img1) };
    MMContentKey m2{ 0, 16, mm_content_hash(img2, sizeof img2) };
    CHECK(m1.hash != m2.hash);
    BlockHash h1 = chain_block_hash(root, a.data(), 16, &m1, 1, 0);
    BlockHash h2 = chain_block_hash(root, a.data(), 16, &m2, 1, 0);
    CHECK(h1 != h2);
    CHECK(h1 != a0);                                     /* and neither aliases the text-only block */
    CHECK(chain_block_hash(root, a.data(), 16, &m1, 1, 0) == h1);   /* same image, same hash */
}

/* ================================================================== prefix cache */

TEST(prefix_cache_hits_the_shared_prefix_only) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    CHECK(f.pc.enabled());
    CHECK_EQ(f.pc.block_size(), 16ll);

    auto p1 = tok_run(10, 64);                  /* 4 blocks */
    publish(f, 1, p1, 64);

    /* Same prefix, longer prompt: three blocks reused (the fourth is the tail we keep). */
    auto p2 = p1;
    for (int i = 0; i < 32; ++i) p2.push_back(500 + i);
    CacheHit h = f.pc.lookup(p2, {});
    CHECK_EQ(h.n_attn_tokens, 64);

    /* Diverging at block 2: only blocks 0 and 1 are reusable. */
    auto p3 = p1;
    p3[35] = 77777;
    CacheHit h3 = f.pc.lookup(p3, {});
    CHECK_EQ(h3.n_attn_tokens, 32);

    /* A block in the middle of a DIFFERENT prefix does not collide: p4's blocks 1..3 are
     * byte-identical to p1's, but its block 0 is not, so nothing matches. */
    auto p4 = p1;
    for (int i = 0; i < 16; ++i) p4[i] = 9000 + i;
    CHECK_EQ(f.pc.lookup(p4, {}).n_attn_tokens, 0);

    /* A full hit is backed off by one block: the step needs a token to compute a logit from. */
    CHECK_EQ(f.pc.lookup(p1, {}).n_attn_tokens, 48);
}

TEST(prefix_cache_lru_eviction_order) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));

    auto a = tok_run(100, 32), b = tok_run(200, 32), c = tok_run(300, 32);
    publish(f, 1, a, 16);           /* one indexed block each */
    publish(f, 2, b, 16);
    publish(f, 3, c, 16);
    CHECK_EQ(f.pc.size(), 3ll);
    /* WHILE THE SEQUENCES HOLD THEM, EVICTION TAKES NOTHING: releasing the cache's reference
     * would free no block and only lose the entry. */
    CHECK_EQ(f.pc.evict_lru(3), 0ll);
    CHECK_EQ(f.pc.size(), 3ll);
    for (uint64_t id : { 1, 2, 3 }) f.kv.free_sequence(id);

    /* Touch A, so the order becomes B, C, A. */
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 16);
    auto order = f.pc.lru_order();
    CHECK_EQ((long long)order.size(), 3ll);

    CHECK_EQ(f.pc.evict_lru(1), 1ll);
    CHECK_EQ(f.pc.size(), 2ll);
    CHECK_EQ(f.pc.lookup(b, {}).n_attn_tokens, 0);      /* B was the oldest and went first */
    CHECK_EQ(f.pc.lookup(c, {}).n_attn_tokens, 16);
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 16);
}

TEST(prefix_cache_holds_a_reference_on_every_indexed_block) {
    /* The invariant that makes a hit safe: a block named by the cache is never in the free list,
     * so it can never be handed to another sequence underneath a cache entry. */
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    auto a = tok_run(100, 32);
    publish(f, 1, a, 32);
    int32_t blk = f.kv.block_table(1, G_FULL)[0];
    CHECK_EQ(f.kv.block_refcount(G_FULL, blk), 2);           /* sequence + cache */
    const int64_t before = f.kv.free_blocks(G_FULL);
    f.kv.free_sequence(1);
    CHECK_EQ(f.kv.block_refcount(G_FULL, blk), 1);           /* still held by the cache */
    f.pc.clear();
    CHECK_EQ(f.kv.free_blocks(G_FULL), before + 2);
}

/* WHAT THE CACHE IS HOLDING, which is the figure that tells a full KV pool apart from a leaking
 * one. The pool counts a held block whoever holds it, and on a server carrying conversations most
 * of a full pool is finished turns kept for their next one -- so with a single number an operator
 * watching 300K tokens held behind one live request has no way to see which it is. */
TEST(the_cache_reports_the_context_it_is_holding) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    CHECK_EQ(f.pc.size() * f.pc.block_size(), 0);

    auto a = tok_run(100, 32);
    publish(f, 1, a, 32);
    CHECK_EQ(f.pc.size() * f.pc.block_size(), 32);

    /* A SECOND CONVERSATION SHARING THE PREFIX ADDS ONLY WHAT IT ADDED. Entries are keyed by the
     * chain, so the blocks the two have in common are one entry and are held once -- which is
     * why this figure is a share of the pool and not a sum over sessions. */
    auto b = tok_run(100, 32);
    for (size_t i = 16; i < b.size(); ++i) b[i] += 1000;
    publish(f, 2, b, 32);
    CHECK_EQ(f.pc.size() * f.pc.block_size(), 48);

    /* And it falls when the cache gives blocks back, which is the half that makes it an answer
     * to "is the pool releasing anything". */
    f.kv.free_sequence(1);
    f.kv.free_sequence(2);
    f.pc.evict_lru(1);
    CHECK_EQ(f.pc.size() * f.pc.block_size(), 32);
    f.pc.clear();
    CHECK_EQ(f.pc.size() * f.pc.block_size(), 0);
}

/* ================================================================== the split lookup */

TEST(split_lookup_reuses_attention_between_checkpoints) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    CHECK_EQ(f.pc.checkpoint_interval(), 64ll);
    CHECK_EQ(f.pc.block_size(), 16ll);

    /* The scheduler splits a chunk here, so a checkpoint is always written rather than being
     * whatever a step boundary happened to land on (spec §7.3b). */
    CHECK_EQ(f.pc.next_checkpoint_after(0), 64ll);
    CHECK_EQ(f.pc.next_checkpoint_after(1), 64ll);
    CHECK_EQ(f.pc.next_checkpoint_after(64), 128ll);
    CHECK_EQ(f.pc.next_checkpoint_after(65), 128ll);
    CHECK(f.pc.is_checkpoint(128));
    CHECK(!f.pc.is_checkpoint(80));

    auto prompt = tok_run(7000, 336);         /* 21 blocks */
    publish(f, 1, prompt, 320);               /* 20 blocks indexed */

    /* With no checkpoint the attention hit stands alone and the linear layers replay from zero. */
    CacheHit h0 = f.pc.lookup(prompt, {});
    CHECK_EQ(h0.n_attn_tokens, 320);
    CHECK_EQ(h0.n_linear_tokens, 0);
    CHECK_EQ(h0.checkpoint_slot, -1);

    /* Write a checkpoint at 128. Its key is the chained block hash at that position, which is
     * exactly what the attention lookup already computed. */
    BlockHash parent;
    std::vector<BlockHash> hs;
    for (int b = 0; b < 21; ++b) {
        parent = chain_block_hash(parent, prompt.data() + b * 16, 16, nullptr, 0, b * 16);
        hs.push_back(parent);
    }
    int32_t slot = -1;
    CHECK_OK(f.pc.reserve_checkpoint(/*session=*/1, 128, hs[128 / 16 - 1], &slot));
    CHECK(slot >= 0);

    /* RESERVED IS NOT WRITTEN. The reservation happens when the chunk is PLANNED; the state only
     * exists after the step has run. Offering the slot in between hands out a buffer still
     * holding the previous tenant's history, which is a wrong answer that reads perfectly -- so
     * the lookup must skip it until commit_checkpoint says the copy was issued. */
    CacheHit pending = f.pc.lookup(prompt, {});
    CHECK_EQ(pending.n_attn_tokens, 320);
    CHECK_EQ(pending.n_linear_tokens, 0);
    CHECK_EQ(pending.checkpoint_slot, -1);
    f.pc.commit_checkpoint(hs[128 / 16 - 1]);

    /* The hit lands BETWEEN checkpoints: attention is reused to 320, the linear state is restored
     * at 128, and only [128, 320) replays through the linear layers. That split is the whole
     * reason the interval is decoupled from the block size. */
    CacheHit h1 = f.pc.lookup(prompt, {});
    CHECK_EQ(h1.n_attn_tokens, 320);
    CHECK_EQ(h1.n_linear_tokens, 128);
    CHECK_EQ(h1.checkpoint_slot, slot);

    /* A later checkpoint shortens the replay. */
    int32_t slot2 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(1, 256, hs[256 / 16 - 1], &slot2));
    f.pc.commit_checkpoint(hs[256 / 16 - 1]);
    CacheHit h2 = f.pc.lookup(prompt, {});
    CHECK_EQ(h2.n_attn_tokens, 320);
    CHECK_EQ(h2.n_linear_tokens, 256);
    CHECK_EQ(h2.checkpoint_slot, slot2);

    /* A checkpoint under a DIFFERENT prefix is not reachable from here. */
    auto other = tok_run(9000, 336);
    CHECK_EQ(f.pc.lookup(other, {}).n_linear_tokens, 0);
}

/* A CAPPED LOOKUP (HitCap) ends the hit at a block and the linear state at a checkpoint at or
 * below the cap, never past it -- unless that gives back more than the cap allows, in which case
 * the whole hit stands. The statistics count one lookup either way. */
TEST(a_capped_lookup_lands_on_a_checkpoint_at_or_below_the_cap) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    auto prompt = tok_run(7000, 336);
    publish(f, 1, prompt, 320);
    BlockHash parent;
    std::vector<BlockHash> hs;
    for (int b = 0; b < 21; ++b) {
        parent = chain_block_hash(parent, prompt.data() + b * 16, 16, nullptr, 0, b * 16);
        hs.push_back(parent);
    }
    int32_t s128 = -1, s256 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(1, 128, hs[128 / 16 - 1], &s128));
    f.pc.commit_checkpoint(hs[128 / 16 - 1]);
    CHECK_OK(f.pc.reserve_checkpoint(1, 256, hs[256 / 16 - 1], &s256));
    f.pc.commit_checkpoint(hs[256 / 16 - 1]);

    const CacheHit u = f.pc.lookup(prompt, {});
    CHECK_EQ(u.n_attn_tokens, 320);
    CHECK_EQ(u.n_linear_tokens, 256);
    CHECK_EQ(u.given_back, 0);

    HitCap cap;
    cap.tokens = 250;
    cap.max_give_back = 1000;
    const int64_t lookups = f.pc.stats().lookups;
    const CacheHit c = f.pc.lookup(prompt, {}, nullptr, &cap);
    CHECK_EQ(f.pc.stats().lookups, lookups + 1);
    CHECK_EQ(c.n_attn_tokens, 240);
    CHECK_EQ(c.n_linear_tokens, 128);
    CHECK_EQ(c.checkpoint_slot, s128);
    CHECK_EQ(c.given_back, 128);              /* 256 reusable before, 128 after */

    cap.tokens = 100;                         /* below every checkpoint: all 256 go back */
    const CacheHit z = f.pc.lookup(prompt, {}, nullptr, &cap);
    CHECK_EQ(z.n_attn_tokens, 96);
    CHECK_EQ(z.n_linear_tokens, 0);
    CHECK_EQ(z.checkpoint_slot, -1);
    CHECK_EQ(z.given_back, 256);

    cap.max_give_back = 255;                  /* ...which is more than this cap allows */
    const CacheHit w = f.pc.lookup(prompt, {}, nullptr, &cap);
    CHECK_EQ(w.n_attn_tokens, 320);
    CHECK_EQ(w.n_linear_tokens, 256);
    CHECK_EQ(w.given_back, 0);

    cap.tokens = 100000;                      /* a cap above the hit changes nothing */
    CHECK_EQ(f.pc.lookup(prompt, {}, nullptr, &cap).n_attn_tokens, 320);
}

/* The chained block hashes of `tokens`, one a block, as the cache computes them. */
static std::vector<BlockHash> chain_of(const std::vector<int32_t>& tokens, int bs) {
    std::vector<BlockHash> hs;
    BlockHash parent;
    for (int b = 0; b + bs <= (int)tokens.size(); b += bs) {
        parent = chain_block_hash(parent, tokens.data() + b, bs, nullptr, 0, b);
        hs.push_back(parent);
    }
    return hs;
}
static void ckpt_at(Fixture& f, uint64_t session, const std::vector<BlockHash>& hs, int64_t pos) {
    int32_t slot = -1;
    CHECK_OK(f.pc.reserve_checkpoint(session, pos, hs[(size_t)(pos / 16 - 1)], &slot));
    CHECK(slot >= 0);
    f.pc.commit_checkpoint(hs[(size_t)(pos / 16 - 1)]);
}

/* A CHECKPOINT AT THE TAIL OF A PROMPT, off the interval, is one the next turn resumes from: the
 * lookup takes the deepest snapshot under the hit wherever it sits on a block boundary. */
TEST(a_checkpoint_at_the_tail_of_a_prompt_is_found_and_the_deepest_wins) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    const auto prompt = tok_run(7000, 336);
    publish(f, 1, prompt, 320);
    const std::vector<BlockHash> hs = chain_of(prompt, 16);
    ckpt_at(f, 1, hs, 128);                   /* on the interval */
    ckpt_at(f, 1, hs, 304);                   /* the tail: a block boundary, not an interval */
    CacheHit h = f.pc.lookup(prompt, {});
    CHECK_EQ(h.n_attn_tokens, 320);
    CHECK_EQ(h.n_linear_tokens, 304);

    /* A prompt that shares only the first 200 tokens reaches 128 and not 304. */
    auto other = prompt;
    for (size_t k = 200; k < other.size(); ++k) other[k] += 50000;
    CHECK_EQ(f.pc.lookup(other, {}).n_linear_tokens, 128);

    /* Off a block boundary there is no key to hold it by. */
    int32_t slot = -1;
    CHECK_EQ(f.pc.reserve_checkpoint(1, 300, hs[300 / 16 - 1], &slot), RAD_E_INVAL);
}

/* A CONVERSATION'S SNAPSHOTS BELONG TO ITS LATEST TURN, so the retention policy sees one session
 * with one tip rather than one tip a turn. What a diverging conversation shares moves with the
 * prefix it shares, and only that. */
TEST(a_turn_that_extends_a_conversation_takes_over_its_snapshots) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/8));
    const auto p1 = tok_run(7000, 336);
    publish(f, 1, p1, 320);
    const std::vector<BlockHash> hs = chain_of(p1, 16);
    ckpt_at(f, 1, hs, 128);
    ckpt_at(f, 1, hs, 256);
    CHECK_EQ(f.pc.checkpoints(1).size(), 2u);

    auto p2 = p1;
    for (int k = 0; k < 64; ++k) p2.push_back(90000 + k);
    publish(f, 2, p2, 400);
    CHECK(f.pc.checkpoints(1).empty());
    CHECK(f.pc.checkpoints(2) == (std::vector<int64_t>{ 128, 256 }));
    CHECK_EQ(f.pc.sessions_tracked(), 1ll);

    auto p3 = p1;
    for (size_t k = 200; k < p3.size(); ++k) p3[k] += 50000;
    publish(f, 3, p3, 320);
    CHECK(f.pc.checkpoints(3) == (std::vector<int64_t>{ 128 }));
    CHECK(f.pc.checkpoints(2) == (std::vector<int64_t>{ 256 }));
}

TEST(checkpoint_interval_must_be_a_block_multiple) {
    Fixture f;
    f.cfg.checkpoint_interval = 100;          /* not a multiple of 16 */
    CHECK_OK(f.pools.configure(f.cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    f.cfg.checkpoint_slots = 2;
    CHECK_OK(f.kv.configure(hybrid_groups(true), f.cfg, f.pools.kv()));
    /* It would never produce a single hit, so it is refused rather than served at 0%. */
    CHECK_EQ(f.pc.configure(f.cfg, &f.kv, f.cfg.checkpoint_interval), RAD_E_INVAL);
}

/* ================================================================== retention */

TEST(geometric_backoff_keeps_the_tip_and_its_ancestors) {
    GeometricBackoff g;
    std::vector<int64_t> pos;
    for (int i = 1; i <= 16; ++i) pos.push_back(i * 64);     /* 64 .. 1024 */
    std::vector<int64_t> keep;
    g.select(pos, &keep);
    /* tip, then 1x, 2x, 4x, 8x back: 1024, 960, 896, 768, 512. */
    std::vector<int64_t> want = { 512, 768, 896, 960, 1024 };
    CHECK_EQ((long long)keep.size(), (long long)want.size());
    for (size_t i = 0; i < want.size() && i < keep.size(); ++i) CHECK_EQ(keep[i], want[i]);

    /* log(n): a 128K session at a 2048-token interval is 64 checkpoints and keeps 7 of them. */
    std::vector<int64_t> big;
    for (int i = 1; i <= 64; ++i) big.push_back(i * 2048);
    g.select(big, &keep);
    CHECK_EQ((long long)keep.size(), 7ll);
    CHECK_EQ(keep.back(), 64ll * 2048);                       /* THE TIP IS ALWAYS KEPT */

    /* The alternatives are classes, not a rewrite (spec §19.1). */
    KeepTipOnly t; t.select(big, &keep);
    CHECK_EQ((long long)keep.size(), 1ll);
    CHECK_EQ(keep[0], 64ll * 2048);
    KeepAll a; a.select(big, &keep);
    CHECK_EQ((long long)keep.size(), 64ll);
}

TEST(checkpoint_retention_is_budgeted_and_keeps_the_tip) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/8));

    auto prompt = tok_run(4242, 16 * 64);
    BlockHash parent;
    std::vector<BlockHash> hs;
    for (int b = 0; b < 64; ++b) {
        parent = chain_block_hash(parent, prompt.data() + b * 16, 16, nullptr, 0, b * 16);
        hs.push_back(parent);
    }
    /* Eight checkpoints at 64, 128, ... 512 fill the pool exactly. */
    for (int i = 1; i <= 8; ++i) {
        int32_t s = -1;
        CHECK_OK(f.pc.reserve_checkpoint(1, i * 64, hs[i * 4 - 1], &s));
    }
    CHECK_EQ((long long)f.pc.checkpoints(1).size(), 8ll);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 0ll);

    /* The ninth forces retention. Geometric backoff over {64..512} keeps 512, 448, 384, 256 and
     * releases the rest -- and the tip survives, because that is what an appending agent
     * transcript actually needs. */
    int32_t s9 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(1, 9 * 64, hs[9 * 4 - 1], &s9));
    auto held = f.pc.checkpoints(1);
    CHECK(held.size() <= 6);
    CHECK_EQ(held.back(), 9ll * 64);
    bool has_512 = false;
    for (int64_t p : held) if (p == 512) has_512 = true;
    CHECK(has_512);
}

/* A SNAPSHOT A LOOKUP HANDED OUT THIS STEP KEEPS ITS SLOT until the next step begins. The
 * admission it was handed to restores from the slot ahead of the step, and a later admission in
 * the same step reserves its own checkpoint -- which, on a full store, runs retention and then
 * evicts. Either taking the handed snapshot leaves the restore naming a slot the store gave back,
 * and the engine stopped on it ("could not restore ... from checkpoint slot 1: invalid
 * argument"). Geometric backoff over {64..512} releases 192, so the hit is made to land there. */
TEST(a_snapshot_handed_out_this_step_survives_retention) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/8));

    const auto a = tok_run(4242, 16 * 33);
    publish(f, 1, a, 16 * 32);
    BlockHash parent;
    std::vector<BlockHash> hs;
    for (int b = 0; b < 33; ++b) {
        parent = chain_block_hash(parent, a.data() + b * 16, 16, nullptr, 0, b * 16);
        hs.push_back(parent);
    }
    std::vector<int32_t> slot_at(9, -1);
    for (int i = 1; i <= 8; ++i) {
        CHECK_OK(f.pc.reserve_checkpoint(1, i * 64, hs[i * 4 - 1], &slot_at[(size_t)i]));
        f.pc.commit_checkpoint(hs[i * 4 - 1]);
    }
    CHECK_EQ(f.kv.free_checkpoint_slots(), 0ll);

    /* A prompt that leaves the chain after 200 tokens: the deepest snapshot under it is 192. */
    f.pc.begin_step();
    auto b = a;
    b.resize(16 * 20);
    for (int i = 200; i < (int)b.size(); ++i) b[(size_t)i] = 900000 + i;
    std::vector<std::vector<int32_t>> hit;
    const CacheHit h = f.pc.lookup(b, {}, &hit);
    REQUIRE_EQ(h.n_linear_tokens, 192);
    REQUIRE_EQ(h.checkpoint_slot, slot_at[3]);

    /* The same step's next reservation finds the store full. */
    BlockHash other = hs[3];
    other.lo ^= 0x5eedull;
    int32_t s9 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(2, 64, other, &s9));
    CHECK(s9 >= 0);
    CHECK(s9 != h.checkpoint_slot);
    CHECK(f.kv.checkpoint_held(h.checkpoint_slot));
    bool kept_192 = false;
    for (int64_t p : f.pc.checkpoints(1)) if (p == 192) kept_192 = true;
    CHECK(kept_192);
}

/* AND THE EVICTION AFTER RETENTION passes it over too. With one slot, the handed snapshot is the
 * only thing a reservation could take, so the reservation is refused instead: that chunk goes
 * without a checkpoint, and the restore still has its state. */
TEST(a_snapshot_handed_out_this_step_is_not_evicted) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/1));
    const auto a = tok_run(5151, 16 * 9);
    publish(f, 1, a, 16 * 8);
    BlockHash parent;
    std::vector<BlockHash> hs;
    for (int b = 0; b < 9; ++b) {
        parent = chain_block_hash(parent, a.data() + b * 16, 16, nullptr, 0, b * 16);
        hs.push_back(parent);
    }
    int32_t s1 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(1, 64, hs[3], &s1));
    f.pc.commit_checkpoint(hs[3]);

    f.pc.begin_step();
    std::vector<std::vector<int32_t>> hit;
    const CacheHit h = f.pc.lookup(a, {}, &hit);
    REQUIRE_EQ(h.checkpoint_slot, s1);

    int32_t s2 = -1;
    CHECK_EQ(f.pc.reserve_checkpoint(1, 128, hs[7], &s2), RAD_E_FULL);
    CHECK(f.kv.checkpoint_held(s1));
    CHECK_EQ(f.pc.stats().ckpt_evictions, 0ll);

    f.pc.begin_step();
    CHECK_OK(f.pc.reserve_checkpoint(1, 128, hs[7], &s2));
}

TEST(a_dead_sessions_tip_is_evictable_so_the_slot_pool_cannot_wedge) {
    /* One checkpoint each from more DISTINCT sessions than there are slots. Every one of them is
     * its session's tip, so a rule that made a tip unevictable would wedge the pool here: the
     * ninth request gets RAD_E_FULL and so does every request after it, for the life of the
     * process. A session is a request id, requests end, and nothing retires their checkpoints
     * (they are keyed by prefix hash so the next turn can still find them), so "tip" is not a
     * proxy for "live". */
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/8));

    auto prompt = tok_run(909, 16 * 8);
    BlockHash parent;
    std::vector<BlockHash> hs;
    for (int b = 0; b < 8; ++b) {
        parent = chain_block_hash(parent, prompt.data() + b * 16, 16, nullptr, 0, b * 16);
        hs.push_back(parent);
    }

    /* One checkpoint each, under a DISTINCT prefix, so each takes its own slot rather than
     * sharing the one an equal hash would find. Eight sessions fill eight slots exactly. */
    for (uint64_t sess = 1; sess <= 8; ++sess) {
        BlockHash h = hs[3];
        h.lo ^= sess;
        int32_t s = -1;
        CHECK_OK(f.pc.reserve_checkpoint(sess, 64, h, &s));
        CHECK(s >= 0);
    }
    CHECK_EQ(f.kv.free_checkpoint_slots(), 0ll);

    /* The ninth session must still be served, by retiring the least recently used tip. */
    BlockHash fresh = hs[3];
    fresh.lo ^= 0x5eedull;
    int32_t s9 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(99, 64, fresh, &s9));
    CHECK(s9 >= 0);
    /* AND THE SESSION IT EVICTED IS FORGOTTEN. Eight in, one evicted, one added: eight tracked.
     * An index with no eraser -- retain() and the eviction above emptying a session's vector and
     * leaving the key -- grows by one entry per request for the life of the process, and
     * reserve_checkpoint walks every key on an allocation failure. */
    CHECK_EQ(f.pc.sessions_tracked(), 8ll);
}

TEST(a_sessions_index_entry_does_not_outlive_its_last_snapshot) {
    /* The other way a session empties: RETENTION rather than eviction. Drive one session past the
     * slot budget under keep-tip-only, which releases everything but the tip on the first squeeze,
     * then evict that tip from under it by way of other sessions. Nothing may be left behind. */
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    f.pc.set_policy(std::make_unique<KeepTipOnly>());

    auto prompt = tok_run(313, 16 * 32);
    BlockHash parent;
    std::vector<BlockHash> hs;
    for (int b = 0; b < 32; ++b) {
        parent = chain_block_hash(parent, prompt.data() + b * 16, 16, nullptr, 0, b * 16);
        hs.push_back(parent);
    }
    for (int i = 1; i <= 5; ++i) {
        int32_t s = -1;
        CHECK_OK(f.pc.reserve_checkpoint(7, i * 64, hs[i * 4 - 1], &s));
    }
    /* The squeeze happens on the FIFTH reservation: retention runs over {64,128,192,256} and
     * tip-only keeps 256, then 320 is allocated into the freed space. Two, not one. */
    CHECK_EQ((long long)f.pc.checkpoints(7).size(), 2ll);
    CHECK_EQ(f.pc.sessions_tracked(), 1ll);

    /* Four more sessions, one snapshot each: session 7's lone tip is the LRU and goes. */
    for (uint64_t sess = 20; sess < 24; ++sess) {
        BlockHash h = hs[9];
        h.lo ^= sess;
        int32_t s = -1;
        CHECK_OK(f.pc.reserve_checkpoint(sess, 64, h, &s));
    }
    CHECK(f.pc.checkpoints(7).empty());
    CHECK_EQ(f.pc.sessions_tracked(), 4ll);
}

/* ================================================================== the KV loan
 *
 * WHAT THESE ARE HOLDING DOWN. The cache and the expert plane share one card, and the split
 * between them is made once at startup against a worst case nobody runs. So the cache lends the
 * expert slab the blocks it is not holding, in place -- which is only safe if a block is never lent
 * while anything holds it, and only USEFUL if the held blocks stay packed at the bottom of the pool
 * so that there is a top to lend.
 *
 * Both of those fail quietly. A block lent out from under a sequence is expert weights written
 * over its KV, which reads as fluent text with a different hash; a pool whose held blocks are
 * scattered simply never lends, and the only symptom is that the feature does nothing. So the
 * tests here are about the mark and the ordering. None of them needs virtual memory management:
 * a loan maps and unmaps nothing.
 */

struct Lender {
    Pools     pools;
    KVManager kv;
    Config    cfg = small_config();

    /* THE BUFFER IS SET BY THE CHUNK, because that is what it is derived from: the floor is one
     * step's worth of the group, and a step batches at most --max-num-batched-tokens tokens. A
     * test that wants a bigger buffer asks for a bigger chunk, which is the same lever the
     * running engine has and not a second one built for the test. */
    int build(int64_t chunk_tokens = 2048) {
        cfg.vram_kv_mib     = 256;
        cfg.max_tok         = chunk_tokens;
        cfg.checkpoint_slots = 0;
        int s = pools.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator());
        if (s < 0) return s;
        return kv.configure(hybrid_groups(/*with_linear=*/false), cfg, pools.kv());
    }
    int64_t bs() const { return kv.plan(G_FULL)->block_size; }
};

TEST(a_new_cache_lends_nothing_and_hands_out_its_whole_carve) {
    Lender e;
    CHECK_OK(e.build());
    /* THE FIRST REQUEST OF A RUN DOES NOT WAIT FOR A RECALL. Lending is the arbiter's decision,
     * made once the cache has said what it can spare, and not a starting state. */
    CHECK_EQ(e.kv.loan_tokens(), 0);
    CHECK(e.kv.lendable(G_FULL));
    CHECK_EQ(e.kv.live_blocks(G_FULL), e.kv.total_blocks(G_FULL));
    CHECK_EQ(e.kv.free_blocks(G_FULL), e.kv.total_blocks(G_FULL));
}

TEST(an_idle_cache_lends_all_but_its_buffer) {
    Lender e;
    CHECK_OK(e.build());
    const int64_t all = e.kv.total_blocks(G_FULL);
    const int64_t spare = e.kv.spare_tokens();
    CHECK(spare > 0);
    CHECK_OK(e.kv.set_loan(spare));
    CHECK_EQ(e.kv.loan_tokens(), spare);

    /* What it keeps is the buffer, not nothing: a cache that lent everything would recall on the
     * first request. */
    const int64_t live = e.kv.live_blocks(G_FULL);
    CHECK(live > 0);
    CHECK(live < all);
    CHECK_EQ(all - live, spare / e.bs());
    CHECK_EQ(e.kv.free_blocks(G_FULL), live);
    CHECK_EQ(e.kv.held_blocks(G_FULL), 0);
    /* The cache's own figure shrinks by what it lent; the carve, which is the budget, does not. */
    CHECK_EQ(e.kv.pool_bytes(G_FULL), live * e.kv.plan(G_FULL)->bytes_per_block);
    CHECK_EQ(e.kv.pool_bytes_carved(G_FULL), all * e.kv.plan(G_FULL)->bytes_per_block);

    /* AND THE MARK IS WHAT THE FREE LIST OBEYS. A block above it is the slab's, and a restore
     * that names one is refused rather than written over a unit's weights. */
    CHECK_EQ(e.kv.reserve_specific_block(G_FULL, (int32_t)live), RAD_E_STATE);
    CHECK_OK(e.kv.reserve_specific_block(G_FULL, (int32_t)live - 1));

    /* Giving the loan back is moving the mark: every id was in the list all along. */
    CHECK_OK(e.kv.set_loan(0));
    CHECK_EQ(e.kv.live_blocks(G_FULL), all);
    CHECK_EQ(e.kv.free_blocks(G_FULL), all - 1);
}

TEST(a_block_something_holds_is_never_lent) {
    Lender e;
    CHECK_OK(e.build());
    const int64_t all = e.kv.total_blocks(G_FULL);

    /* Take every block, then give back all but the highest. The pool is now almost empty and the
     * one thing in it is at the top, which is the case the loan has to refuse. */
    std::vector<int32_t> got;
    int32_t b = -1;
    while (e.kv.reserve_block(G_FULL, &b) == RAD_OK) got.push_back(b);
    CHECK_EQ((int64_t)got.size(), all);
    int32_t top = 0;
    for (int32_t g : got) top = std::max(top, g);
    CHECK_EQ((int64_t)top, all - 1);
    for (int32_t g : got) if (g != top) e.kv.release_block(G_FULL, g);

    /* One block, at the top, and nothing can be lent. The alternative -- lending around it -- is
     * not available: the pool is layer-major and the boundary is a mark, not a set. */
    CHECK_EQ(e.kv.spare_tokens(), 0);
    CHECK_EQ(e.kv.set_loan(e.bs()), RAD_E_STATE);
    CHECK_EQ(e.kv.live_blocks(G_FULL), all);            /* and a refusal moved nothing */
    CHECK_EQ(e.kv.loan_tokens(), 0);

    /* Let it go and the loan can follow. Everything that makes a block free -- a sequence ending,
     * an eviction, an idle-tier demotion -- reaches this the same way. */
    e.kv.release_block(G_FULL, top);
    CHECK(e.kv.spare_tokens() > 0);
    CHECK_OK(e.kv.set_loan(e.kv.spare_tokens()));
    CHECK(e.kv.live_blocks(G_FULL) < all);
}

TEST(the_lowest_free_block_is_the_one_handed_out) {
    Lender e;
    CHECK_OK(e.build());
    int32_t a = -1, b = -1, c = -1;
    CHECK_OK(e.kv.reserve_block(G_FULL, &a));
    CHECK_OK(e.kv.reserve_block(G_FULL, &b));
    CHECK_OK(e.kv.reserve_block(G_FULL, &c));
    CHECK(a < b && b < c);

    /* Freed in the WORST order for a stack: the highest last. A stack would hand back c, which
     * after a busy period is exactly the high block a loan wants -- so the highest held block
     * would never fall and nothing could ever be lent. */
    e.kv.release_block(G_FULL, a);
    e.kv.release_block(G_FULL, b);
    e.kv.release_block(G_FULL, c);
    int32_t again = -1;
    CHECK_OK(e.kv.reserve_block(G_FULL, &again));
    CHECK_EQ(again, a);
}

/* THE LIVELOCK THIS DESIGN HAS TO NOT HAVE, and it is worth stating plainly because it is the one
 * way a lending cache can be worse than a fixed one rather than merely no better.
 *
 * A long prefill asks for its blocks in ONE call. A cache that could only take its loan back on a
 * timer would refuse that call; the scheduler would wait, or preempt something, and neither
 * produces the demand that would have brought the loan back -- because the demand it just refused
 * was the demand. So admission is decided against the whole carve, and the recall happens on the
 * path that needed it, through the owner of the slabs. */
TEST(a_request_past_the_mark_takes_the_loan_back) {
    Lender e;
    CHECK_OK(e.build());
    const int64_t all = e.kv.total_blocks(G_FULL);
    const int64_t spare = e.kv.spare_tokens();
    CHECK_OK(e.kv.set_loan(spare));
    const int64_t idle = e.kv.live_blocks(G_FULL);

    int calls = 0;
    int64_t asked = -1;
    e.kv.set_reclaim([&](int64_t t) { ++calls; asked = t; return (int)RAD_OK; });

    /* Longer than the cache is keeping, and well inside what it was carved for. */
    const int64_t tokens = (idle + all) / 2 * e.bs();
    CHECK(e.kv.can_ever_fit(tokens));
    CHECK_OK(e.kv.add_sequence(1));
    CHECK_OK(e.kv.ensure(1, tokens));
    CHECK_EQ(calls, 1);                                 /* one recall for the whole request */
    CHECK(asked >= 0 && asked < spare);
    CHECK_EQ(e.kv.loan_tokens(), asked);
    CHECK(e.kv.live_blocks(G_FULL) > idle);
    CHECK_EQ((int64_t)e.kv.block_table(1, G_FULL).size(), (tokens + e.bs() - 1) / e.bs());
    /* AND THE BUFFER CAME BACK WITH IT, so the next block does not recall again. */
    CHECK(e.kv.free_blocks(G_FULL) > 0);

    /* A REQUEST PAST THE CARVE IS STILL REFUSED. The budget is still the budget: lending decides
     * who uses the carve, never how much the operator said the cache may address. */
    CHECK_OK(e.kv.add_sequence(2));
    CHECK_EQ(e.kv.ensure(2, all * e.bs()), RAD_E_FULL);
}

/* A RECALL THE SLABS CANNOT MAKE IS A REQUEST THAT CANNOT BE SERVED, AND NOTHING MOVES. The
 * alternative -- taking the blocks back regardless -- hands a sequence memory an expert unit is
 * still being read out of. */
TEST(a_recall_the_slab_cannot_make_refuses_the_request) {
    Lender e;
    CHECK_OK(e.build());
    const int64_t all = e.kv.total_blocks(G_FULL);
    CHECK_OK(e.kv.set_loan(e.kv.spare_tokens()));
    const int64_t idle = e.kv.live_blocks(G_FULL);
    const int64_t loan = e.kv.loan_tokens();
    e.kv.set_reclaim([](int64_t) { return (int)RAD_E_DEVICE; });

    CHECK_OK(e.kv.add_sequence(1));
    CHECK(e.kv.ensure(1, (idle + all) / 2 * e.bs()) < 0);
    CHECK_EQ(e.kv.loan_tokens(), loan);
    CHECK_EQ(e.kv.live_blocks(G_FULL), idle);
    CHECK_EQ((int64_t)e.kv.block_table(1, G_FULL).size(), 0);
}

/* AND THE BUFFER IT KEEPS IS ONE STEP'S WORTH, which is the floor a chunked prefill needs so that
 * it recalls once a chunk rather than once a block. A bigger chunk is a bigger buffer, and the
 * relation is the one thing about the derivation an operator can still influence. */
TEST(the_derived_buffer_is_one_chunk_of_the_group) {
    Lender small, big;
    CHECK_OK(small.build(/*chunk_tokens=*/2048));
    CHECK_OK(big.build(/*chunk_tokens=*/65536));
    CHECK(big.kv.buffer_bytes() > small.kv.buffer_bytes());
    CHECK(big.kv.spare_tokens() < small.kv.spare_tokens());
}

/* A RESTORE CANNOT LAND ABOVE THE MARK. Under tensor parallelism rank 0 picks the block id and
 * every other rank takes the same one, so a rank whose loan is deeper than its peers' must refuse
 * rather than write over the weights its slab keeps there. */
TEST(a_named_block_above_the_mark_is_refused) {
    Lender e;
    CHECK_OK(e.build());
    CHECK_OK(e.kv.set_loan(e.kv.spare_tokens()));
    const int64_t live = e.kv.live_blocks(G_FULL);
    CHECK(live < e.kv.total_blocks(G_FULL));

    CHECK_EQ(e.kv.reserve_specific_block(G_FULL, (int32_t)live), RAD_E_STATE);
    CHECK_OK(e.kv.reserve_specific_block(G_FULL, (int32_t)live - 1));
    /* Taken out of the middle, and the free count says so even though the heap still names it. */
    CHECK_EQ(e.kv.free_blocks(G_FULL), live - 1);
    CHECK_EQ(e.kv.reserve_specific_block(G_FULL, (int32_t)live - 1), RAD_E_STATE);
}

/* A FOLLOWER LENDS WHAT THE LEADER LENDS. Block ids are shared across ranks and only rank 0 knows
 * what is held, so a follower's mark is the leader's -- in both directions. */
TEST(a_follower_takes_the_leaders_mark) {
    Lender lead, fol;
    CHECK_OK(lead.build());
    CHECK_OK(fol.build());
    CHECK_OK(lead.kv.set_loan(lead.kv.spare_tokens()));
    CHECK_OK(fol.kv.follow(lead.kv));
    CHECK_EQ(fol.kv.loan_tokens(), lead.kv.loan_tokens());
    CHECK_EQ(fol.kv.live_blocks(G_FULL), lead.kv.live_blocks(G_FULL));
    CHECK_OK(lead.kv.set_loan(0));
    CHECK_OK(fol.kv.follow(lead.kv));
    CHECK_EQ(fol.kv.live_blocks(G_FULL), fol.kv.total_blocks(G_FULL));
}

/* A CHECKPOINT SLOT COSTS THE CARD WHEN IT IS TAKEN, NOT WHEN IT IS CARVED. --checkpoint-slots is
 * how many snapshots the server MAY hold, not how much VRAM it holds regardless: a region backed
 * whole at startup would never be released. On a linear-attention model a slot is a sizeable
 * fraction of a gigabyte, so a server that had never written a snapshot would still pay the expert
 * plane's price for every one of them. */
/* A hybrid model whose recurrent state is DELIBERATELY LARGE -- at least two commit granules of
 * snapshot, whatever this driver's granularity is. That is the shape a real linear-attention
 * model has, and it is the only shape for which a checkpoint slot is managed elastically. The
 * second state dimension is solved for rather than written down, because a constant that happened
 * to clear a 2 MiB granule would quietly stop testing anything on a device with a larger one. */
/* ================================================================== the snapshot store */

/* A SLOT IS HOST MEMORY AND THE POOL NEVER SEES IT. Taking one, writing it and giving it back
 * moves nothing the KV pool reports, and the bytes are the store's: what is written through
 * checkpoint_ptr is what the store holds at that slot. */
TEST(a_checkpoint_slot_is_host_memory_and_costs_the_pool_nothing) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    CHECK_EQ(f.kv.checkpoint_slots(), 4);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 4);
    const int64_t used = f.pools.kv().used();
    int32_t a = -1, b = -1;
    CHECK_OK(f.kv.alloc_checkpoint(&a));
    CHECK_OK(f.kv.alloc_checkpoint(&b));
    CHECK(a != b);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 2);
    CHECK_EQ(f.pools.kv().used(), used);
    char* p = (char*)f.kv.checkpoint_ptr(a);
    REQUIRE(p != nullptr);
    CHECK(p == (char*)f.ck.at(a));
    std::memset(p, 0x5a, (size_t)f.kv.checkpoint_bytes());
    CHECK_EQ(((unsigned char*)f.ck.at(a))[f.kv.checkpoint_bytes() - 1], 0x5a);
    f.kv.free_checkpoint(a);
    f.kv.free_checkpoint(a);                         /* a double free is refused, not counted */
    f.kv.free_checkpoint(b);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 4);
}

/* NO STORE, NO SLOTS: a hybrid model served without one hands nothing out, rather than a pointer
 * to memory nobody owns. */
TEST(without_a_store_no_checkpoint_slot_is_handed_out) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/0));
    CHECK_EQ(f.kv.checkpoint_slots(), 0);
    int32_t s = -1;
    CHECK_EQ(f.kv.alloc_checkpoint(&s), RAD_E_FULL);
    CHECK(f.kv.checkpoint_ptr(0) == nullptr);
}

/* ONE SLOT, EVERY RANK'S SHARD. Rank 0 hands the ids out and each rank addresses its own window
 * of the same slot; two ranks writing at offset zero would leave the last one's shard and restore
 * it on both. */
TEST(each_rank_addresses_its_own_window_of_a_shared_snapshot_slot) {
    Fixture lead, follow;
    CHECK_OK(lead.build(/*with_linear=*/true, /*ckpt_slots=*/0));
    CHECK_OK(follow.build(/*with_linear=*/true, /*ckpt_slots=*/0));
    const int64_t shard = lead.kv.checkpoint_bytes();
    HostArena store;
    CHECK_OK(store.open(2 * shard, 2 * shard * 3));
    lead.kv.set_checkpoint_store(&store, 0);
    follow.kv.set_checkpoint_store(&store, shard);
    int32_t s = -1;
    CHECK_OK(lead.kv.alloc_checkpoint(&s));
    CHECK((char*)lead.kv.checkpoint_ptr(s) == (char*)store.at(s));
    CHECK((char*)follow.kv.checkpoint_ptr(s) == (char*)store.at(s) + shard);
    CHECK_EQ(follow.kv.free_checkpoint_slots(), 2);  /* the follower reads the same free list */
    lead.kv.free_checkpoint(s);
    CHECK_EQ(store.used_slots(), 0);
}

RAD_TEST_MAIN()

/* ================================================================== idle session tiers
 *
 * The state machine only, with no bytes moving: every test here drives IdleTiers directly and
 * reports jobs back as the engine would. That is the whole point of the object being separable --
 * a tier's use-after-free hazards live in this bookkeeping rather than in the transfers, and none
 * of them needs a GPU to exercise.
 */

static IdleTiers::Config tier_config(int64_t host_bytes, int64_t entry) {
    IdleTiers::Config c;
    c.host_cap_bytes = host_bytes;
    c.entry_bytes = entry;
    return c;
}
static BlockHash bh(uint64_t n) { return BlockHash{n, n * 0x9e3779b97f4a7c15ull + 1}; }
static const uint64_t SEC = 1000000000ull;

/* Every copy that is waiting, planned and reported the way the engine does once it has landed. */
static std::vector<TierJob> copy_now(IdleTiers& t, bool ok = true) {
    std::vector<TierJob> jobs;
    t.plan_writeback(1 << 20, &jobs);
    for (TierJob& j : jobs) t.report(j, ok);
    return jobs;
}

/* An entry's VRAM given up the way the cache gives it up, once nothing else is reading it. */
static bool drop_now(IdleTiers& t, const BlockHash& k) {
    if (!t.droppable(k)) return false;
    t.drop_commit(k);
    return true;
}

TEST(a_published_entry_is_copied_at_once_and_keeps_its_vram) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 1024)));
    CHECK(t.host_on());

    t.note_used(bh(1), {10}, 1 * SEC);
    CHECK(t.writeback_waiting());

    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    CHECK(jobs[0].kind == TierJob::Kind::WriteBack);
    CHECK(jobs[0].key == bh(1));
    REQUIRE_EQ(jobs[0].blocks.size(), 1u);
    CHECK_EQ(jobs[0].blocks[0], 10);
    CHECK(jobs[0].host_slot >= 0);

    /* PENDING EXCLUDES IT FROM A SECOND PLAN, and from a drop: a copy still out has not landed,
     * so the VRAM copy is the only complete one there is. */
    std::vector<TierJob> again;
    t.plan_writeback(16, &again);
    CHECK_EQ(again.size(), 0u);
    CHECK(!t.droppable(bh(1)));

    t.report(jobs[0], true);
    /* A COPY IS NOT A MOVE. The entry is still in VRAM, where a return finds it for free; what has
     * changed is that giving the VRAM up now costs nothing. */
    CHECK(t.tier_of(bh(1)) == KVTier::Device);
    CHECK(t.droppable(bh(1)));
    CHECK_EQ(t.stats().to_host, 1);
    CHECK_EQ(t.stats().dropped, 0);
    CHECK_EQ(t.arena().used_slots(), 1);
    CHECK(!t.writeback_waiting());
    std::vector<BlockHash> clean;
    t.take_clean(&clean);
    REQUIRE_EQ(clean.size(), 1u);
    CHECK(clean[0] == bh(1));
}

TEST(giving_up_a_clean_entry_s_vram_moves_no_bytes) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 1024)));
    t.note_used(bh(1), {10}, 1 * SEC);
    copy_now(t);

    CHECK(drop_now(t, bh(1)));
    CHECK(t.tier_of(bh(1)) == KVTier::Host);
    CHECK(t.resident_off_device(bh(1)));
    CHECK_EQ(t.stats().dropped, 1);
    CHECK_EQ(t.stats().to_host, 1);               /* the one copy, made before */
    CHECK_EQ(t.arena().used_slots(), 1);          /* the copy IS the entry now */
    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    CHECK_EQ(jobs.size(), 0u);                    /* and nothing is left to copy */
}

TEST(a_failed_copy_leaves_the_entry_in_vram_and_tries_again) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(1024, 1024)));   /* exactly one slot */
    t.note_used(bh(1), {10}, 0);

    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    CHECK(t.arena().full());

    t.report(jobs[0], false);
    CHECK(t.tier_of(bh(1)) == KVTier::Device);    /* the pool still holds the blocks */
    CHECK(!t.droppable(bh(1)));            /* and nothing may let them go */
    CHECK(!t.arena().full());                     /* the slot went back */
    CHECK_EQ(t.stats().failures, 1);
    CHECK_EQ(t.stats().to_host, 0);

    /* A copy that failed is not a decision, it is an attempt. */
    std::vector<TierJob> retry;
    t.plan_writeback(16, &retry);
    CHECK_EQ(retry.size(), 1u);
}

/* ONE COPY A RUN is what makes a long conversation affordable at a 4-token block, so entries
 * planned together have to sit together. */
TEST(a_batch_of_copies_takes_consecutive_arena_slots) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(64 * 1024, 1024)));
    for (uint64_t i = 1; i <= 8; ++i) t.note_used(bh(i), {(int32_t)(10 + i)}, i * SEC);
    std::vector<TierJob> jobs;
    t.plan_writeback(64, &jobs);
    REQUIRE_EQ(jobs.size(), 8u);
    for (size_t i = 0; i < jobs.size(); ++i)
        CHECK_EQ(jobs[i].host_slot, jobs[0].host_slot + (int32_t)i);
    /* And oldest first: the conversation that finished first is the first copied. */
    CHECK(jobs[0].key == bh(1));
    CHECK(jobs[7].key == bh(8));
}

TEST(host_arena_runs_stop_at_a_used_slot_and_wrap) {
    HostArena a;
    CHECK_OK(a.open(1024, 8 * 1024));
    REQUIRE_EQ(a.capacity(), 8);
    int32_t got = 0;
    CHECK_EQ(a.alloc_run(3, &got), 0);
    CHECK_EQ(got, 3);
    CHECK_EQ(a.alloc_run(3, &got), 3);
    CHECK_EQ(got, 3);
    a.free(1);
    /* The run ends where the arena ends; the rest is asked for again. */
    CHECK_EQ(a.alloc_run(4, &got), 6);
    CHECK_EQ(got, 2);
    /* And the cursor wraps to the hole it left behind. */
    CHECK_EQ(a.alloc_run(4, &got), 1);
    CHECK_EQ(got, 1);
    CHECK(a.full());
    CHECK_EQ(a.alloc_run(1, &got), -1);
    CHECK_EQ(got, 0);
    /* A double free is refused rather than counted: the used count is what the view reports. */
    a.free(1);
    a.free(1);
    CHECK_EQ(a.used_slots(), 7);
}

TEST(a_full_arena_holds_the_copy_back_and_makes_room_from_the_oldest_host_entry) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(2048, 1024)));   /* two slots */
    std::vector<BlockHash> lost;
    t.set_drop_sink([&lost](const BlockHash& h) { lost.push_back(h); });

    t.note_used(bh(1), {11}, 1 * SEC);
    t.note_used(bh(2), {12}, 2 * SEC);
    CHECK_EQ(copy_now(t).size(), 2u);
    drop_now(t, bh(1));
    drop_now(t, bh(2));

    t.note_used(bh(3), {13}, 3 * SEC);
    CHECK_EQ(copy_now(t).size(), 0u);             /* nowhere to put it */
    CHECK(t.starved());
    /* Still queued, not forgotten -- but not planned again until a slot is freed: asking again
     * before then pops the queue, finds no slot and puts it back. */
    CHECK(!t.writeback_waiting());
    CHECK(t.tier_of(bh(3)) == KVTier::Device);

    /* The maintenance plan frees exactly what was refused, least recently used first, and the
     * cache is told so it stops offering the entry. */
    std::vector<TierJob> room;
    t.plan(16, &room);
    REQUIRE_EQ(room.size(), 1u);
    CHECK(room[0].kind == TierJob::Kind::DropHost);
    CHECK(room[0].key == bh(1));
    t.report(room[0], true);
    CHECK_EQ(t.stats().host_drops, 1);
    REQUIRE_EQ(lost.size(), 1u);
    CHECK(lost[0] == bh(1));
    CHECK(!t.starved());
    CHECK(t.writeback_waiting());

    CHECK_EQ(copy_now(t).size(), 1u);
    CHECK(t.droppable(bh(3)));
}

TEST(forgetting_an_entry_mid_copy_does_not_erase_it_under_the_engine) {
    /* THE LIFETIME BUG THIS CLASS IS FOR. An entry evicted from the prefix cache while the engine
     * is still copying it must not have its record erased: the job the engine holds names it, and
     * report() dereferences it afterwards. */
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 1024)));
    t.note_used(bh(7), {70}, 0);

    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    CHECK(!t.doomed(bh(7)));

    t.forget(bh(7));                              /* evicted while pending */
    CHECK(t.doomed(bh(7)));

    t.report(jobs[0], true);                      /* the engine finishes anyway */
    CHECK(t.doomed(bh(7)));                       /* gone altogether: unknown reads as doomed */
    CHECK(t.tier_of(bh(7)) == KVTier::Device);    /* and as the device, which is "not ours" */
    CHECK_EQ(t.arena().used_slots(), 0);          /* its slot came back */
    CHECK(!t.droppable(bh(7)));
}

TEST(a_restore_keeps_the_host_copy_so_the_entry_lands_clean) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 1024)));
    t.note_used(bh(5), {50}, 0);
    copy_now(t);
    drop_now(t, bh(5));

    TierJob up;
    CHECK(t.plan_promote(bh(5), &up));
    CHECK(up.kind == TierJob::Kind::ToDevice);
    CHECK(up.host_slot >= 0);
    /* A promotion already outstanding is not handed out twice. */
    TierJob dup;
    CHECK(!t.plan_promote(bh(5), &dup));

    up.blocks = {99};                             /* the engine allocated a different block */
    up.started = true;
    t.report(up, true, 100 * SEC);
    CHECK(t.tier_of(bh(5)) == KVTier::Device);
    CHECK_EQ(t.stats().promoted, 1);
    CHECK_EQ(t.stats().host_hits, 1);
    /* THE COPY STAYED, so the VRAM can go again for nothing and there is nothing to copy. */
    CHECK_EQ(t.arena().used_slots(), 1);
    CHECK(t.droppable(bh(5)));
    CHECK(!t.writeback_waiting());
    std::vector<BlockHash> clean;
    t.take_clean(&clean);
    CHECK(std::find(clean.begin(), clean.end(), bh(5)) != clean.end());
}

TEST(only_what_a_turn_added_is_copied_after_a_restore) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(64 * 1024, 1024)));
    for (uint64_t i = 1; i <= 3; ++i) t.note_used(bh(i), {(int32_t)i}, 1 * SEC);
    copy_now(t);
    for (uint64_t i = 1; i <= 3; ++i) {
        drop_now(t, bh(i));
        TierJob up;
        REQUIRE(t.plan_promote(bh(i), &up));
        up.blocks = {(int32_t)(20 + i)};
        up.started = true;
        t.report(up, true, 10 * SEC);
    }
    /* The next turn publishes the whole chain again, two blocks longer. */
    for (uint64_t i = 1; i <= 5; ++i) t.note_used(bh(i), {(int32_t)(20 + i)}, 20 * SEC);
    std::vector<TierJob> jobs;
    t.plan_writeback(64, &jobs);
    REQUIRE_EQ(jobs.size(), 2u);
    CHECK(jobs[0].key == bh(4));
    CHECK(jobs[1].key == bh(5));
}

TEST(a_failed_restore_that_started_drops_the_entry) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 1024)));
    std::vector<BlockHash> lost;
    t.set_drop_sink([&lost](const BlockHash& h) { lost.push_back(h); });
    t.note_used(bh(3), {30}, 0);
    copy_now(t);
    drop_now(t, bh(3));

    TierJob up;
    CHECK(t.plan_promote(bh(3), &up));
    up.blocks = {31};
    up.started = true;
    t.report(up, false);
    /* Gone, not left on the host tier: the scatter may have written some layers and not others,
     * so the blocks no longer hold a function of the tokens and a miss is the only safe answer --
     * and the cache has to hear it, or it would serve the ids it last had. */
    CHECK(t.tier_of(bh(3)) == KVTier::Device);
    CHECK(!t.resident_off_device(bh(3)));
    CHECK_EQ(t.arena().used_slots(), 0);
    REQUIRE_EQ(lost.size(), 1u);
    CHECK(lost[0] == bh(3));
}

TEST(a_restore_that_never_started_leaves_the_entry_where_it_was) {
    /* THE POOL RAN OUT BEFORE THIS ENTRY. Nothing about it was touched, so its host copy is exactly
     * what it was and dropping it would throw away a conversation for nothing. */
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 1024)));
    t.note_used(bh(3), {30}, 0);
    copy_now(t);
    drop_now(t, bh(3));

    TierJob up;
    CHECK(t.plan_promote(bh(3), &up));
    t.report(up, false);
    CHECK(t.tier_of(bh(3)) == KVTier::Host);
    CHECK(t.resident_off_device(bh(3)));
    CHECK_EQ(t.arena().used_slots(), 1);
    TierJob again;
    CHECK(t.plan_promote(bh(3), &again));
}

TEST(a_recomputed_entry_gives_up_its_tier_copy_and_is_copied_afresh) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 1024)));
    t.note_used(bh(2), {20}, 1 * SEC);
    copy_now(t);
    drop_now(t, bh(2));
    CHECK_EQ(t.arena().used_slots(), 1);

    t.rebound(bh(2), {77}, 10 * SEC);
    CHECK(t.tier_of(bh(2)) == KVTier::Device);
    CHECK_EQ(t.arena().used_slots(), 0);
    CHECK(!t.droppable(bh(2)));            /* the new blocks have no copy yet */
    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    REQUIRE_EQ(jobs[0].blocks.size(), 1u);
    CHECK_EQ(jobs[0].blocks[0], 77);
}

TEST(the_session_view_reports_where_a_conversation_actually_is) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(2048, 1024)));
    t.set_model("q38-flashnext");

    std::vector<BlockHash> chain = {bh(1), bh(2), bh(3)};
    for (uint64_t i = 1; i <= 3; ++i) t.note_used(bh(i), {(int32_t)i}, 0);
    t.note_session(42, chain, 48, 0);

    std::vector<IdleTiers::SessionInfo> s = t.sessions();
    REQUIRE_EQ(s.size(), 1u);
    CHECK_EQ(s[0].id, 42u);
    CHECK(s[0].model == "q38-flashnext");
    CHECK_EQ(s[0].n_tokens, 48);
    CHECK_EQ(s[0].bytes, 3 * 1024);
    CHECK_EQ(s[0].on_device, 3);

    /* Two of the three fit in the arena, so the session straddles two tiers and the view says so
     * rather than picking one. */
    copy_now(t);
    drop_now(t, bh(1));
    drop_now(t, bh(2));
    s = t.sessions();
    REQUIRE_EQ(s.size(), 1u);
    CHECK_EQ(s[0].on_host, 2);
    CHECK_EQ(s[0].on_device, 1);
}

/* WHAT A RETURN REUSES IS THE DEEPEST SNAPSHOT, not the blocks: on a hybrid model the scheduler
 * clamps a hit to the last linear checkpoint it can reach. And what is off the card is counted
 * once, however many conversations share it. */
TEST(the_session_view_says_what_a_return_reuses_and_what_is_off_the_card) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(2048, 1024)));

    for (uint64_t i = 1; i <= 3; ++i) t.note_used(bh(i), {(int32_t)i}, 0);
    t.note_session(1, {bh(1), bh(2), bh(3)}, 48, 0);
    /* A second conversation past the same first two blocks. */
    t.note_used(bh(9), {9}, 100 * SEC);
    t.note_session(2, {bh(1), bh(2), bh(9)}, 48, 100 * SEC);

    IdleTiers::Occupancy occ;
    std::vector<IdleTiers::SessionInfo> s = t.sessions(&occ);
    CHECK_EQ(s.size(), 2u);
    CHECK_EQ(s[0].resume_tokens, 0);             /* no snapshot: a return reuses nothing */
    CHECK_EQ(occ.host_bytes, 0);
    CHECK_EQ(occ.host_cap_bytes, 2048);

    /* The shared blocks are copied first -- they were published first -- and the arena holds
     * two; each is one slot however many rows name it. */
    copy_now(t);
    drop_now(t, bh(1));
    drop_now(t, bh(2));
    s = t.sessions(&occ);
    CHECK_EQ(occ.host_bytes, t.arena().used_slots() * 1024);
    CHECK_EQ(t.arena().used_slots(), 2);
    int64_t named = 0;
    for (const auto& r : s) named += r.on_host;
    CHECK(named > t.arena().used_slots());

    /* Snapshots on the chains: the deepest one a chain reaches is where its return resumes, and a
     * snapshot on the shared prefix is reachable from both. */
    IdleTiers u;
    CHECK_OK(u.configure(tier_config(0, 1024)));
    for (uint64_t i : {1, 2, 3, 9}) u.note_used(bh(i), {(int32_t)i}, 0);
    u.note_snapshot(bh(1), 0, 16);
    u.note_snapshot(bh(9), 1, 48);
    u.note_session(1, {bh(1), bh(2), bh(3)}, 48, 0);
    u.note_session(2, {bh(1), bh(2), bh(9)}, 48, 1 * SEC);
    s = u.sessions();
    CHECK_EQ(s.size(), 2u);
    for (const auto& r : s) {
        if (r.id == 2) CHECK_EQ(r.resume_tokens, 48);
        if (r.id == 1) CHECK_EQ(r.resume_tokens, 16);
    }
}

TEST(tiers_stay_off_until_a_size_is_given) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(0, 1024)));
    CHECK(!t.host_on());
    CHECK(!t.any_on());
    t.note_used(bh(1), {10}, 0);
    CHECK(!t.writeback_waiting());
    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    t.plan(16, &jobs);
    CHECK_EQ(jobs.size(), 0u);
    /* AND NO PER-BLOCK STATE, which is the cost that would otherwise be paid by every operator
     * who never asked for a tier: a record per cached entry is a map proportional to the cache.
     * The SESSION record is a different size of thing -- one row a conversation -- and is kept
     * either way; the test below is the one that pins that. */
    CHECK_EQ(t.sessions().size(), 0u);
}

/* ================================================================== the linear half
 *
 * On a hybrid model an entry's attention blocks are only part of a session. The rest is a
 * recurrent-state SNAPSHOT in a slot of the engine's checkpoint store -- pinned host memory,
 * --checkpoint-slots of them for the whole server. A snapshot never follows its blocks through
 * the tiers: it is restored straight out of its slot, and the tiers only record it for the
 * session view.
 */

TEST(a_record_that_only_ever_held_a_snapshot_is_forgotten_with_it) {
    /* note_snapshot runs at COMMIT, which is mid-request; note_used runs when the chain is
     * published, which is when the request closes. A request that never publishes -- a rebased
     * block table, a cancel before the first whole block -- would otherwise leave a record with
     * no blocks, no storage and no way to be reached again, one per checkpoint, forever. */
    IdleTiers t;
    IdleTiers::Config c = tier_config(1 << 20, 1024);
    c.ckpt_bytes = 2048;
    c.ckpt_slots = 8;
    CHECK_OK(t.configure(c));
    t.note_snapshot(bh(1), 7, 2048);
    std::vector<TierJob> jobs;
    t.plan(16, &jobs);
    CHECK_EQ(jobs.size(), 0u);                         /* a request that never publishes costs nothing */
    t.release_snapshot(bh(1));
    CHECK(t.tier_of(bh(1)) == KVTier::Device);         /* which is what "no record" answers */
    CHECK(!t.writeback_waiting());
    CHECK_EQ(t.sessions().size(), 0u);
}


/* ------------------------------------------------------------------ the cache's half of it */

TEST(the_tiers_are_told_the_cache_s_block_ids_and_not_the_requests) {
    /* TWO REQUESTS COMPUTE THE SAME PREFIX AND ONLY ONE SET OF BLOCKS IS REFERENCED. The second
     * request missed -- its lookup stopped early, which is exactly what happens while the tiers
     * hold the entry -- so it filled its own blocks for tokens the cache already has. insert()
     * keeps the entry it already holds and lets the second request's copy go.
     *
     * Told the second request's ids, the tiers would later gather out of blocks the pool had
     * since handed to somebody else -- putting a stranger's KV into the tier under this prefix's
     * key -- and release a reference nobody took. What that looks like from outside is a log full
     * of "double free of block N" and, on the path that promotes the entry back, fluent text with
     * one word changed: intermittent, plausible, and invisible to a single-session test. */
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false, /*ckpt_slots=*/0));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(1 << 20, 1024)));
    f.pc.set_tiers(&t);

    auto prompt = tok_run(7400, 336);
    publish(f, 1, prompt, 320);
    std::vector<int32_t> first = f.kv.block_table(1, f.pc.cached_groups()[0]);
    CHECK(first.size() >= 20u);

    /* The second sequence gets its own blocks for the same tokens; the cache keeps the first's. */
    publish(f, 2, prompt, 320);
    std::vector<int32_t> second = f.kv.block_table(2, f.pc.cached_groups()[0]);
    CHECK(second[0] != first[0]);

    std::vector<TierJob> jobs;
    t.plan_writeback(512, &jobs);
    CHECK(jobs.size() >= 20u);
    /* Every job names a block the CACHE holds. */
    for (const TierJob& j : jobs) {
        bool from_cache = false, from_second = false;
        for (size_t i = 0; i < first.size(); ++i)  if (j.blocks[0] == first[i])  from_cache = true;
        for (size_t i = 0; i < second.size(); ++i) if (j.blocks[0] == second[i]) from_second = true;
        CHECK(from_cache);
        CHECK(!from_second);
    }
}

/* ================================================================== the pool in bytes
 *
 * WHY THE BYTE FIGURE EXISTS AT ALL. Blocks and tokens are both un-summable across KV groups: a
 * recurrent group has no token axis and hands out states rather than blocks, and two PAGED groups
 * share one token axis, so a token of context takes a slot in each and adding their capacities
 * reports a pool several times the size of the one the card holds. Bytes are disjoint extents of
 * VRAM and are the one measure that survives being added up -- which is what makes them the one
 * measure that can answer "how much of the card is the KV cache holding".
 */

TEST(kv_pool_bytes_are_the_plan_and_used_is_what_is_held) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true));

    const KVGroupPlan* pg = f.kv.plan(G_FULL);
    const KVGroupPlan* ps = f.kv.plan(G_GDN);
    CHECK(pg && ps);

    /* The capacity is the plan's own arithmetic and not a second derivation of it. */
    CHECK_EQ(f.kv.pool_bytes(G_FULL), pg->n_blocks * pg->bytes_per_block);
    CHECK_EQ(f.kv.pool_bytes(G_GDN),  ps->n_states * ps->bytes_per_state);
    /* The sentinel at handle 0 is neither paged nor stateful and must contribute nothing: it is
     * summed over with every real group, and a non-zero answer there would inflate every total. */
    CHECK_EQ(f.kv.pool_bytes(G_NULL), 0ll);
    CHECK_EQ(f.kv.pool_bytes_used(G_NULL), 0ll);
    /* And a handle no group was ever carved for. */
    CHECK_EQ(f.kv.pool_bytes(99), 0ll);
    CHECK_EQ(f.kv.pool_bytes_used(99), 0ll);

    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 0ll);

    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 32));                       /* two 16-token blocks */
    CHECK_EQ(f.kv.block_table(1, G_FULL).size(), 2u);
    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 2 * pg->bytes_per_block);
    /* The recurrent half is one state and it is charged the moment the sequence exists, because
     * that is when the slot stops being available to anyone else. */
    CHECK_EQ(f.kv.pool_bytes_used(G_GDN), ps->bytes_per_state);

    f.kv.free_sequence(1);
    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 0ll);
    CHECK_EQ(f.kv.pool_bytes_used(G_GDN), 0ll);
}

/* USED TRACKS THE FREE LIST AND NOT THE SEQUENCE TABLE, which is the whole reason it can be read
 * as "VRAM this server cannot hand to the next request". A finished turn's blocks are held by the
 * prefix cache with no sequence attached, and they are occupying the card for exactly as long. */
TEST(kv_bytes_in_use_counts_a_block_only_the_cache_still_holds) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    const int64_t per = f.kv.plan(G_FULL)->bytes_per_block;

    publish(f, 1, tok_run(100, 32), 32);
    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 2 * per);

    f.kv.free_sequence(1);                    /* the turn ends; the cache keeps its reference */
    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 2 * per);

    f.pc.clear();                             /* and now nothing holds them */
    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 0ll);
}

/* ================================================================== a dropped copy gives VRAM back
 *
 * THE LINK NEITHER HALF OF THE SUITE COVERS ON ITS OWN. IdleTiers decides that an entry's VRAM may
 * go and KVManager owns the free list, and the release that connects them lives in the cache --
 * PrefixCache::drop_clean, which is also where a block some running sequence still reads is held
 * back. A drop that moved the record and never released the blocks would pass every tier test in
 * this file and every KV test, and the symptom would be a cache that grows to the size of the pool
 * with the tier counters reporting that it was working. So these drive the whole path and ask the
 * pool rather than the policy.
 */
struct TierFixture {
    Fixture f;
    IdleTiers t;
    int64_t per = 0;
    int build() {
        int s = f.build(/*with_linear=*/false, /*ckpt_slots=*/0);
        if (s < 0) return s;
        s = t.configure(tier_config(1 << 20, 1024));
        if (s < 0) return s;
        f.pc.set_tiers(&t);
        per = f.kv.plan(G_FULL)->bytes_per_block;
        return RAD_OK;
    }
};

TEST(dropping_a_clean_entry_gives_its_vram_back_and_keeps_it_restorable) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto prompt = tok_run(7600, 336);
    publish(x.f, 1, prompt, 320);                   /* twenty 16-token blocks */
    x.f.kv.free_sequence(1);                        /* finished: only the cache holds them */
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 20 * x.per);

    CHECK_EQ(copy_now(x.t).size(), 20u);
    CHECK_EQ(x.f.pc.drop_clean(), 20);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 0ll);
    CHECK_EQ(x.f.pc.resident(), 0ll);
    CHECK_EQ(x.f.pc.size(), 20ll);                  /* every entry still indexed */
    CHECK_EQ(x.f.pc.stats().drops, 20ll);
    CHECK_EQ(x.f.pc.stats().evictions, 0ll);        /* nothing was lost */
    /* The lookup stops where VRAM stops, and the restore path sees the whole chain. */
    CHECK_EQ(x.f.pc.lookup(prompt, {}).n_attn_tokens, 0);
    CHECK_EQ(x.f.pc.restorable_from(prompt, {}).size(), 20u);
    CHECK_EQ(x.f.pc.drop_clean(), 0);               /* and there is nothing left to give */
}

TEST(an_entry_a_running_sequence_reads_is_dropped_once_the_sequence_lets_go) {
    /* A BLOCK A SEQUENCE ALSO HOLDS WOULD FREE NOTHING, and its entry would then read off-device
     * while its bytes sat in VRAM under the sequence's table -- the next lookup would restore a
     * second copy beside them. */
    TierFixture x;
    CHECK_OK(x.build());
    publish(x.f, 1, tok_run(7600, 336), 320);       /* sequence 1 is still running */
    copy_now(x.t);
    CHECK_EQ(x.f.pc.drop_clean(), 0);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 21 * x.per);

    x.f.kv.free_sequence(1);
    CHECK_EQ(x.f.pc.drop_clean(), 20);              /* asked again because a sequence let go */
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 0ll);
}

TEST(a_failed_copy_leaves_the_vram_exactly_where_it_was) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto prompt = tok_run(7600, 336);
    publish(x.f, 1, prompt, 320);
    x.f.kv.free_sequence(1);
    copy_now(x.t, /*ok=*/false);
    CHECK_EQ(x.f.pc.drop_clean(), 0);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 20 * x.per);
    CHECK_EQ(x.f.pc.lookup(prompt, {}).n_attn_tokens, 320);
}

TEST(pressure_on_the_pool_moves_a_clean_entry_and_drops_one_with_no_copy) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto a = tok_run(7600, 336);
    const auto b = tok_run(9600, 336);
    publish(x.f, 1, a, 320);
    x.f.kv.free_sequence(1);
    copy_now(x.t);                                  /* a is clean */
    publish(x.f, 2, b, 320);
    x.f.kv.free_sequence(2);                        /* b has no copy yet */

    CHECK_EQ(x.f.pc.evict_lru(1000), 40);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 0ll);
    CHECK_EQ(x.f.pc.restorable_from(a, {}).size(), 20u);   /* kept, off the card */
    CHECK_EQ(x.f.pc.restorable_from(b, {}).size(), 0u);    /* gone */
    CHECK_EQ(x.f.pc.size(), 20ll);
    CHECK_EQ(x.f.pc.stats().drops, 20ll);
    CHECK_EQ(x.f.pc.stats().evictions, 20ll);
    /* And the entries it kept are not on the list pressure walks: evicting again frees nothing
     * and loses nothing. */
    CHECK_EQ(x.f.pc.evict_lru(1000), 0);
    CHECK_EQ(x.f.pc.size(), 20ll);
}

/* THE RESERVE'S TRIM LOSES NO CONVERSATION. It runs right after a publish, while the sequence that
 * published still holds its blocks and before any host copy of them can have landed -- so an entry
 * a sequence holds, and one with no copy yet, are exactly what it must pass over. */
TEST(the_trim_keeps_what_a_sequence_holds_and_what_has_no_copy_yet) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto a = tok_run(7600, 336);
    const auto b = tok_run(9600, 336);
    const auto c = tok_run(11600, 336);
    publish(x.f, 1, a, 320);
    x.f.kv.free_sequence(1);
    copy_now(x.t);                                  /* a is clean */
    publish(x.f, 2, b, 320);
    x.f.kv.free_sequence(2);                        /* b has no copy yet */
    publish(x.f, 3, c, 320);                        /* c's sequence is still running */

    CHECK_EQ(x.f.pc.trim(1000), 20);                /* a, and only a */
    CHECK_EQ(x.f.pc.restorable_from(a, {}).size(), 20u);
    CHECK_EQ(x.f.pc.lookup(b, {}).n_attn_tokens, 320);
    CHECK_EQ(x.f.pc.lookup(c, {}).n_attn_tokens, 320);
    CHECK_EQ(x.f.pc.stats().drops, 20ll);
    CHECK_EQ(x.f.pc.stats().evictions, 0ll);

    copy_now(x.t);                                  /* b and c land */
    CHECK_EQ(x.f.pc.trim(1000), 20);                /* b; c is still read by its sequence */
    CHECK_EQ(x.f.pc.lookup(c, {}).n_attn_tokens, 320);
    x.f.kv.free_sequence(3);
    CHECK_EQ(x.f.pc.trim(5), 5);                    /* and no more than it was asked for */
    CHECK_EQ(x.f.pc.stats().evictions, 0ll);
    CHECK_EQ(x.f.pc.size(), 60ll);                  /* every entry still indexed */
}

TEST(without_tiers_the_trim_frees_only_what_no_sequence_holds) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    const auto a = tok_run(7600, 336);
    const auto b = tok_run(9600, 336);
    publish(f, 1, a, 320);
    f.kv.free_sequence(1);
    publish(f, 2, b, 320);                          /* still running */
    CHECK_EQ(f.pc.trim(1000), 20);
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 0);
    CHECK_EQ(f.pc.lookup(b, {}).n_attn_tokens, 320);
    CHECK_EQ(f.pc.stats().evictions, 20ll);
    CHECK_EQ(f.pc.trim(1000), 0);
}

/* PRESSURE LANDS A WAITING COPY INSTEAD OF ERASING. The binding drops what already has a host
 * copy, then has the engine land the copies still on their way and drops those too; only what the
 * tiers then still cannot take is erased. */
TEST(pressure_lands_a_waiting_copy_instead_of_erasing_the_entry) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto a = tok_run(7600, 336);
    publish(x.f, 1, a, 320);
    x.f.kv.free_sequence(1);                        /* finished; its copy has not landed */
    PCBinding<PrefixCache> b(&x.f.pc);
    int flushes = 0;
    b.flush_fn = [&]() { ++flushes; return !copy_now(x.t).empty(); };
    CHECK_EQ(b.evict(20), 20);
    CHECK_EQ(flushes, 1);
    CHECK_EQ(x.f.pc.stats().evictions, 0ll);
    CHECK_EQ(x.f.pc.stats().drops, 20ll);
    CHECK_EQ(x.f.pc.restorable_from(a, {}).size(), 20u);

    /* With nothing to land the copies, the binding has no lossless way left and erases. */
    const auto c = tok_run(9600, 336);
    publish(x.f, 2, c, 320);
    x.f.kv.free_sequence(2);
    PCBinding<PrefixCache> bare(&x.f.pc);
    CHECK_EQ(bare.evict(20), 20);
    CHECK_EQ(x.f.pc.stats().evictions, 20ll);
}

/* A WAITING REQUEST'S CHAIN STAYS ON THE CARD. An admission refused after its restore gives the hit
 * back, and the entries it brought back are clean; held, the eager drop passes them over until the
 * request is gone, so the next attempt finds them where they are instead of copying them again. */
TEST(a_held_chain_is_not_dropped_until_its_request_lets_go) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto prompt = tok_run(7600, 336);
    publish(x.f, 1, prompt, 320);
    x.f.kv.free_sequence(1);
    copy_now(x.t);
    x.f.pc.hold(7, prompt, {});
    x.f.pc.hold(7, prompt, {});                     /* once per request */
    CHECK_EQ(x.f.pc.drop_clean(), 0);
    CHECK_EQ(x.f.pc.lookup(prompt, {}).n_attn_tokens, 320);
    /* Pressure still takes it: a held entry has its copy, and trim loses nothing. */
    CHECK_EQ(x.f.pc.trim(5), 5);
    x.f.pc.release(7);
    CHECK_EQ(x.f.pc.drop_clean(), 15);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 0ll);
    CHECK_EQ(x.f.pc.stats().evictions, 0ll);
}

/* TIERS ATTACHED AND OFF ARE NO TIERS: the engine attaches them whatever the flags say, and a trim
 * that waited for a copy nobody will make would never keep the reserve. */
TEST(with_the_tiers_attached_and_off_the_trim_still_frees) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(0, 1024)));
    f.pc.set_tiers(&t);
    const auto a = tok_run(7600, 336);
    publish(f, 1, a, 320);
    f.kv.free_sequence(1);
    CHECK_EQ(f.pc.trim(1000), 20);
    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 0ll);
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 0);
    CHECK_EQ(f.pc.size(), 0ll);
}

TEST(a_prefix_computed_again_takes_the_new_blocks_as_its_device_copy) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto prompt = tok_run(7600, 336);
    publish(x.f, 1, prompt, 320);
    x.f.kv.free_sequence(1);
    copy_now(x.t);
    CHECK_EQ(x.f.pc.drop_clean(), 20);
    CHECK_EQ(x.t.arena().used_slots(), 20ll);

    /* A request that could not reach the tier copy computed the same tokens. */
    publish(x.f, 2, prompt, 320);
    CHECK_EQ(x.f.pc.lookup(prompt, {}).n_attn_tokens, 320);
    CHECK_EQ(x.f.pc.resident(), 20ll);
    CHECK_EQ(x.t.arena().used_slots(), 0ll);        /* the older copy went */
    CHECK(x.t.writeback_waiting());                 /* and the new blocks want one of their own */
    const std::vector<int32_t> bt = x.f.kv.block_table(2, G_FULL);
    CHECK_EQ(x.f.kv.block_refcount(G_FULL, bt[0]), 2);
    x.f.kv.free_sequence(2);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 20 * x.per);
}

TEST(evicting_or_clearing_an_entry_on_its_way_back_releases_nothing) {
    /* AN ENTRY WITH A PROMOTION OUT ON IT STILL NAMES THE IDS IT LEFT WITH, and those belong to
     * whoever the pool gave them to since. */
    TierFixture x;
    CHECK_OK(x.build());
    const auto prompt = tok_run(7700, 336);
    publish(x.f, 1, prompt, 320);
    x.f.kv.free_sequence(1);
    copy_now(x.t);
    CHECK_EQ(x.f.pc.drop_clean(), 20);

    const std::vector<BlockHash> keys = x.f.pc.restorable_from(prompt, {});
    REQUIRE(!keys.empty());
    CHECK_OK(x.f.kv.add_sequence(2));
    CHECK_OK(x.f.kv.ensure(2, 16));
    const int32_t taken = x.f.kv.block_table(2, G_FULL)[0];

    TierJob up;
    CHECK(x.t.plan_promote(keys[0], &up));          /* out, on its way back */
    x.f.pc.evict_lru(1000);
    CHECK_EQ(x.f.kv.block_refcount(G_FULL, taken), 1);
    x.f.pc.clear();
    CHECK_EQ(x.f.kv.block_refcount(G_FULL, taken), 1);
    CHECK(x.t.doomed(keys[0]));                     /* forgotten, and finished off on report */
    CHECK_EQ(x.t.arena().used_slots(), 1ll);        /* the one the promotion is reading */
    x.t.report(up, false);
    CHECK_EQ(x.t.arena().used_slots(), 0ll);
}

/* ================================================================== expiry
 *
 * --prefix-cache-ttl: an entry no request has used for that long leaves the cache from whichever
 * tier holds it -- its VRAM blocks back to the pool, its host slot back to the arena -- and a
 * snapshot goes with its prefix. Nothing is moved first. The cache stamps its uses with the
 * monotonic clock, so each test reads that clock between two uses and expires against it.
 */

/* THE HOST LIST IS ORDERED BY USE, not by when an entry arrived on it. An entry arrives when its
 * VRAM is given up, which says nothing about when a request last read it, and a full arena gives
 * up the one used longest ago. */
TEST(a_full_arena_drops_the_least_recently_used_host_entry) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(2048, 1024)));
    std::vector<BlockHash> lost;
    t.set_drop_sink([&lost](const BlockHash& h) { lost.push_back(h); });
    t.note_used(bh(1), {11}, 5 * SEC);                 /* used last */
    t.note_used(bh(2), {12}, 1 * SEC);
    CHECK_EQ(copy_now(t).size(), 2u);
    drop_now(t, bh(1));                                /* but on the host list first */
    drop_now(t, bh(2));
    t.note_used(bh(3), {13}, 6 * SEC);
    CHECK_EQ(copy_now(t).size(), 0u);
    std::vector<TierJob> room;
    t.plan(16, &room);
    REQUIRE_EQ(room.size(), 1u);
    CHECK(room[0].key == bh(2));
    t.report(room[0], true);
    REQUIRE_EQ(lost.size(), 1u);
    CHECK(lost[0] == bh(2));
}

/* A CONVERSATION WHOSE FIRST BLOCK THE ARENA DROPPED IS GONE, and so is its row: no hit reaches
 * past a missing first block, and a row left behind would count the blocks it names as in VRAM
 * for as long as the server runs. */
TEST(a_full_arena_dropping_a_conversation_s_first_block_takes_its_row) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(2048, 1024)));
    t.note_used(bh(1), {11}, 1 * SEC);
    t.note_used(bh(2), {12}, 2 * SEC);
    t.note_session(7, {bh(1), bh(2)}, 32, 2 * SEC);
    CHECK_EQ(copy_now(t).size(), 2u);
    drop_now(t, bh(1));
    drop_now(t, bh(2));
    CHECK_EQ(t.sessions().size(), 1u);
    t.note_used(bh(3), {13}, 3 * SEC);
    CHECK_EQ(copy_now(t).size(), 0u);
    std::vector<TierJob> room;
    t.plan(16, &room);
    REQUIRE_EQ(room.size(), 1u);
    CHECK(room[0].key == bh(1));
    t.report(room[0], true);
    CHECK_EQ(t.sessions().size(), 0u);
}

TEST(the_host_tier_names_what_is_older_than_the_cutoff_oldest_first) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 1024)));
    for (uint64_t i = 1; i <= 3; ++i) t.note_used(bh(i), {(int32_t)i}, (4 - i) * SEC);
    copy_now(t);
    for (uint64_t i = 1; i <= 3; ++i) drop_now(t, bh(i));
    t.note_used(bh(4), {4}, 0);                        /* in VRAM: the cache's own list has it */
    std::vector<BlockHash> out;
    t.expired(2 * SEC + 1, 16, &out);
    REQUIRE_EQ(out.size(), 2u);
    CHECK(out[0] == bh(3));                            /* used at 1 s */
    CHECK(out[1] == bh(2));                            /* used at 2 s */
    out.clear();
    t.expired(10 * SEC, 1, &out);                      /* bounded */
    CHECK_EQ(out.size(), 1u);
    /* A record with a job out is not named: it is asked about again once the job reports. */
    TierJob up;
    REQUIRE(t.plan_promote(bh(3), &up));
    out.clear();
    t.expired(10 * SEC, 16, &out);
    CHECK_EQ(out.size(), 2u);
    CHECK(std::find(out.begin(), out.end(), bh(3)) == out.end());
}

TEST(an_entry_no_request_used_expires_from_vram_and_a_hit_is_a_use) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false, /*ckpt_slots=*/0));
    const auto a = tok_run(7600, 336), b = tok_run(9600, 336);
    const uint64_t t0 = rad_mono_ns();
    publish(f, 1, a, 320);
    f.kv.free_sequence(1);
    publish(f, 2, b, 320);
    f.kv.free_sequence(2);
    const uint64_t mid = rad_mono_ns();
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 320);  /* a hit is a use */
    const int64_t free0 = f.kv.free_blocks(G_FULL);
    const uint64_t ttl = 3600 * SEC;
    CHECK_EQ(f.pc.expire(t0 + ttl, ttl), 0ll);        /* nothing is that old */
    CHECK_EQ(f.pc.expire(mid + ttl, 0), 0ll);         /* 0 keeps everything */
    CHECK_EQ(f.pc.expire(mid + ttl, ttl), 20ll);      /* b, and only b */
    CHECK_EQ(f.kv.free_blocks(G_FULL), free0 + 20);
    CHECK_EQ(f.pc.lookup(b, {}).n_attn_tokens, 0);
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 320);
    CHECK_EQ(f.pc.stats().expired, 20ll);
    CHECK_EQ(f.pc.stats().evictions, 0ll);            /* nothing wanted the memory */
}

/* A SHARED PREFIX OUTLIVES THE CONVERSATION THAT EXPIRES PAST IT: the other conversation's turn
 * stamped the shared blocks along with its own. */
TEST(a_conversation_expires_from_its_tip_and_a_shared_prefix_stays) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false, /*ckpt_slots=*/0));
    const auto a = tok_run(7600, 336);
    auto b = a;
    for (int i = 160; i < 336; ++i) b[(size_t)i] = 20000 + i;   /* ten blocks shared */
    publish(f, 1, a, 320);
    f.kv.free_sequence(1);
    const uint64_t mid = rad_mono_ns();
    publish(f, 2, b, 320);
    f.kv.free_sequence(2);
    const uint64_t ttl = 60 * SEC;
    CHECK_EQ(f.pc.expire(mid + ttl, ttl), 10ll);      /* a's own ten */
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 160);
    CHECK_EQ(f.pc.lookup(b, {}).n_attn_tokens, 320);
}

TEST(an_entry_only_host_memory_holds_expires_and_its_slot_goes_back) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto a = tok_run(7600, 336);
    publish(x.f, 1, a, 320);
    x.f.kv.free_sequence(1);
    CHECK_EQ(copy_now(x.t).size(), 20u);
    CHECK_EQ(x.f.pc.drop_clean(), 20);
    CHECK_EQ(x.f.pc.restorable_from(a, {}).size(), 20u);
    CHECK_EQ(x.t.arena().used_slots(), 20);
    CHECK_EQ(x.t.sessions().size(), 1u);
    const uint64_t later = rad_mono_ns();
    const uint64_t ttl = 60 * SEC;
    CHECK_EQ(x.f.pc.expire(later + ttl, ttl), 20ll);
    CHECK_EQ(x.t.arena().used_slots(), 0);
    CHECK_EQ(x.f.pc.size(), 0ll);
    CHECK_EQ(x.f.pc.restorable_from(a, {}).size(), 0u);
    CHECK_EQ(x.t.sessions().size(), 0u);              /* its row went with its first block */
    CHECK_EQ(x.f.pc.stats().evictions, 0ll);
}

/* A SNAPSHOT EXPIRES WITH ITS PREFIX, and one a lookup handed out this step is kept until the
 * step's admission has restored from it. */
TEST(a_snapshot_expires_with_its_prefix_and_its_slot_goes_back) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    const auto prompt = tok_run(7000, 336);
    publish(f, 1, prompt, 320);
    f.kv.free_sequence(1);
    const std::vector<BlockHash> hs = chain_of(prompt, 16);
    const int64_t free0 = f.kv.free_checkpoint_slots();
    ckpt_at(f, 1, hs, 128);
    ckpt_at(f, 1, hs, 256);
    CHECK_EQ(f.pc.lookup(prompt, {}).n_linear_tokens, 256);
    const uint64_t later = rad_mono_ns();
    const uint64_t ttl = 60 * SEC;
    CHECK_EQ(f.pc.expire(later + ttl, ttl), 20ll);
    CHECK(f.pc.checkpoints(1) == (std::vector<int64_t>{ 256 }));
    CHECK_EQ(f.pc.stats().ckpt_expired, 1ll);
    f.pc.begin_step();
    f.pc.expire(later + ttl, ttl);
    CHECK(f.pc.checkpoints(1).empty());
    CHECK_EQ(f.kv.free_checkpoint_slots(), free0);
    CHECK_EQ(f.pc.sessions_tracked(), 0ll);
    CHECK_EQ(f.pc.stats().ckpt_evictions, 0ll);
}

/* ================================================================== the cache-only count
 *
 * KVManager keeps the number of blocks only the cache holds current at every reference change, so
 * that reading it is free. A count maintained at six call sites is a count that drifts the first
 * time one of them is missed, so these check it after every operation against the figure taken
 * from scratch: a block at a count of one that no live sequence holds can only be the cache's.
 */
static int64_t cache_only_recount(Fixture& f, const std::vector<uint64_t>& live) {
    const int64_t nb = f.kv.plan(G_FULL)->n_blocks;
    std::vector<uint8_t> in_seq((size_t)nb, 0);
    for (uint64_t id : live)
        for (int32_t b : f.kv.block_table(id, G_FULL)) in_seq[(size_t)b] = 1;
    int64_t n = 0;
    for (int64_t b = 0; b < nb; ++b)
        if (f.kv.block_refcount(G_FULL, (int32_t)b) == 1 && !in_seq[(size_t)b]) ++n;
    return n;
}

TEST(the_cache_only_count_follows_every_reference_change) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    uint64_t x = 0x2545f4914f6cdd1dull;
    auto rnd = [&x](uint64_t n) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x % n; };
    std::vector<uint64_t> live;
    uint64_t next_id = 1;
    int64_t hits = 0, forks = 0, evicted = 0, bad = 0;
    for (int it = 0; it < 4000; ++it) {
        const uint64_t op = rnd(6);
        if (op <= 1 && live.size() < 8) {
            /* A prompt from one of six families: a prefix of a family's run is a hit on whatever
             * that family has published before. */
            const auto prompt = tok_run(1000 * (int32_t)rnd(6), 16 + (int)rnd(160));
            const uint64_t id = next_id++;
            CHECK_OK(f.kv.add_sequence(id));
            std::vector<std::vector<int32_t>> found;
            const CacheHit h = f.pc.lookup(prompt, {}, &found);
            if (h.n_attn_tokens > 0) {
                CHECK_OK(f.kv.adopt(id, G_FULL, found[0], h.n_attn_tokens));
                ++hits;
            }
            int st = f.kv.ensure(id, (int64_t)prompt.size());
            if (st == RAD_E_FULL) { evicted += f.pc.evict_lru(64); st = f.kv.ensure(id, (int64_t)prompt.size()); }
            if (st != RAD_OK) { f.kv.free_sequence(id); continue; }
            std::vector<std::vector<int32_t>> bt{ f.kv.block_table(id, G_FULL) };
            CHECK_OK(f.pc.insert(id, prompt, {}, bt, (int64_t)prompt.size()));
            live.push_back(id);
        } else if (op == 2 && !live.empty() && live.size() < 8) {
            const uint64_t id = next_id++;
            if (f.kv.fork(live[(size_t)rnd(live.size())], id) == RAD_OK) { live.push_back(id); ++forks; }
        } else if (op <= 4 && !live.empty()) {
            const size_t k = (size_t)rnd(live.size());
            f.kv.free_sequence(live[k]);
            live.erase(live.begin() + (ptrdiff_t)k);
        } else if (op == 5) {
            evicted += f.pc.evict_lru((int64_t)rnd(9));
        }
        if (f.pc.unshared_tokens() != 16 * cache_only_recount(f, live)) ++bad;
    }
    CHECK_EQ(bad, 0ll);
    /* Every kind of change happened, or the loop proved less than it says. */
    CHECK(hits > 100);
    CHECK(forks > 100);
    CHECK(evicted > 100);
    for (uint64_t id : live) f.kv.free_sequence(id);
    live.clear();
    CHECK_EQ(f.pc.unshared_tokens(), 16 * f.pc.resident());
    f.pc.clear();
    CHECK_EQ(f.pc.unshared_tokens(), 0ll);
    CHECK_EQ(cache_only_recount(f, live), 0ll);
}

TEST(the_cache_only_count_follows_a_drop_and_a_restore) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto prompt = tok_run(7600, 336);
    publish(x.f, 1, prompt, 320);                   /* twenty blocks, sequence 1 reading them */
    CHECK_EQ(x.f.pc.unshared_tokens(), 0ll);
    x.f.kv.free_sequence(1);
    CHECK_EQ(x.f.pc.unshared_tokens(), 320ll);

    copy_now(x.t);
    CHECK_EQ(x.f.pc.drop_clean(), 20);              /* off the card: none of it counts */
    CHECK_EQ(x.f.pc.unshared_tokens(), 0ll);

    /* A restore reserves blocks and hands each reservation to the cache as its reference. */
    const std::vector<BlockHash> keys = x.f.pc.restorable_from(prompt, {});
    REQUIRE(keys.size() == 20u);
    int64_t handed = 0;
    for (const BlockHash& k : keys) {
        int32_t b = -1;
        CHECK_OK(x.f.kv.reserve_block(G_FULL, &b));
        CHECK_EQ(x.f.pc.unshared_tokens(), 16 * handed);   /* a reservation is not the cache's */
        CHECK_OK(x.f.pc.rebind(k, { b }));
        ++handed;
    }
    CHECK_EQ(x.f.pc.unshared_tokens(), 320ll);

    std::vector<std::vector<int32_t>> found;
    const CacheHit h = x.f.pc.lookup(prompt, {}, &found);
    REQUIRE(h.n_attn_tokens == 320);
    CHECK_OK(x.f.kv.add_sequence(2));
    CHECK_OK(x.f.kv.adopt(2, G_FULL, found[0], h.n_attn_tokens));
    CHECK_EQ(x.f.pc.unshared_tokens(), 0ll);
    x.f.kv.free_sequence(2);
    CHECK_EQ(x.f.pc.unshared_tokens(), 320ll);
    x.f.pc.clear();
    CHECK_EQ(x.f.pc.unshared_tokens(), 0ll);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 0ll);
}


/* ================================================================== the tier names
 *
 * THESE ARE READ BY A PERSON AND BY A SCRAPER, on the sessions view, and the value of a name is
 * that it is stable and distinct. A "?" where a tier belongs means the
 * enum grew a value this switch does not know -- which is exactly when a reader most needs to be
 * told, rather than shown the name of a neighbouring tier.
 */
TEST(every_tier_has_its_own_stable_name) {
    CHECK_EQ(std::string(kv_tier_name(KVTier::Device)), std::string("device"));
    CHECK_EQ(std::string(kv_tier_name(KVTier::Host)), std::string("host"));
    /* A value outside the enum says so instead of picking one. */
    CHECK_EQ(std::string(kv_tier_name((KVTier)99)), std::string("?"));
}

/* ================================================================== the loan */

/* A RESTORE IS SIZED BEFORE IT STARTS, so it recalls the loan for the whole chain at once. One
 * recall a block would drain the slab a buffer at a time and stop at whatever the cache kept in
 * hand. */
TEST(a_restore_reserves_its_whole_chain_against_one_recall) {
    Lender e;
    CHECK_OK(e.build());
    const int64_t all = e.kv.total_blocks(G_FULL);
    CHECK_OK(e.kv.set_loan(e.kv.spare_tokens()));
    const int64_t idle = e.kv.live_blocks(G_FULL);
    int calls = 0;
    e.kv.set_reclaim([&](int64_t) { ++calls; return (int)RAD_OK; });

    const int64_t want = (idle + all) / 2;
    std::vector<int32_t> got;
    CHECK_EQ(e.kv.reserve_blocks(G_FULL, want, &got), want);
    CHECK_EQ((int64_t)got.size(), want);
    CHECK_EQ(calls, 1);
    for (int32_t b : got) CHECK(b < e.kv.live_blocks(G_FULL));

    /* Past the carve it takes what there is and says how much. */
    std::vector<int32_t> more;
    const int64_t n = e.kv.reserve_blocks(G_FULL, all, &more);
    CHECK_EQ(n, all - want);
    CHECK_EQ((int64_t)more.size(), n);
}


/* ================================================================== the transfer, end to end
 *
 * EVERYTHING ABOVE IS BOOKKEEPING, and this is the one piece that can put another conversation's
 * context into a restored entry: the layer-major addressing, the staging pitch, a batch cut into
 * staging-sized pieces, arena slots that are not consecutive, and under tensor parallelism a rank's
 * window of a slot every rank shares. So a chain is copied out through the real TierExec, its VRAM
 * given up and written over, restored into DIFFERENT blocks, and every byte of every layer compared
 * with what was there before -- on the host backend, whose streams are real queues on threads, so
 * a missing order between the copies and the row kernels would show up here as wrong bytes.
 */
static std::string rows_plugin() {
    const char* d = std::getenv("RAD_TESTPLUGIN_HOME");
    if (!d) d = std::getenv("RADIANCE_HOME");
    return std::string(d ? d : "radiance_home") + "/testrows/radtest_rows.so";
}

/* The byte a pool holds at (rank, layer, block, i) before anything moves. */
/* A full avalanche, so that neighbouring blocks, layers and bytes all differ: a multiply-only hash
 * moves a small input difference into the high bits and leaves the byte taken here unchanged,
 * which is a pattern every wrong offset reproduces. */
static unsigned char pat(int rank, int64_t layer, int32_t block, int64_t i) {
    uint64_t z = ((uint64_t)rank << 60) ^ ((uint64_t)layer << 48) ^ ((uint64_t)block << 20) ^
                 (uint64_t)i;
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return (unsigned char)(z ^ (z >> 31));
}

static void fill(Fixture& f, int rank) {
    for (int32_t g : f.pc.cached_groups()) {
        const KVGroupPlan* p = f.kv.plan(g);
        if (p->zero) continue;
        const int64_t frag = p->layer_bytes_per_block();
        for (int64_t l = 0; l < p->n_layers; ++l)
            for (int32_t b = 0; b < (int32_t)p->n_blocks; ++b) {
                unsigned char* row = (unsigned char*)p->base + l * p->layer_stride() + b * frag;
                for (int64_t i = 0; i < frag; ++i) row[i] = pat(rank, g * 16 + l, b, i);
            }
    }
}

static void scribble(Fixture& f) {
    for (int32_t g : f.pc.cached_groups()) {
        const KVGroupPlan* p = f.kv.plan(g);
        if (p->zero) continue;
        for (int64_t l = 0; l < p->n_layers; ++l)
            std::memset((char*)p->base + l * p->layer_stride(), 0xEE, (size_t)p->layer_stride());
    }
}

/* How many entries' blocks in `now` hold, in every cached group and layer, exactly what the
 * blocks in `was` held before the move. */
static int64_t same_bytes(Fixture& f, int rank, const std::vector<std::vector<int32_t>>& was,
                          const std::vector<std::vector<int32_t>>& now) {
    const std::vector<int32_t>& cg = f.pc.cached_groups();
    int64_t good = 0;
    for (size_t k = 0; k < was[0].size(); ++k) {
        bool ok = true;
        for (size_t gi = 0; gi < cg.size() && ok; ++gi) {
            const KVGroupPlan* p = f.kv.plan(cg[gi]);
            if (p->zero) continue;
            const int64_t frag = p->layer_bytes_per_block();
            for (int64_t l = 0; l < p->n_layers && ok; ++l) {
                const unsigned char* row =
                    (const unsigned char*)p->base + l * p->layer_stride() + now[gi][k] * frag;
                for (int64_t i = 0; i < frag && ok; ++i)
                    ok = row[i] == pat(rank, cg[gi] * 16 + l, was[gi][k], i);
            }
        }
        good += ok;
    }
    return good;
}

/* Two paged groups of different widths, the shape a production model has: the attention cache
 * beside a narrower per-block key cache. One entry is a block in each, and a restore that confused
 * their offsets inside a staging entry would put one group's bytes in the other. */
static std::vector<KVGroupInfo> two_paged_groups() {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    KVGroupInfo a;
    a.name = "attn";
    a.decl.kind = RAD_KV_FULL;
    a.decl.dtype = RAD_BF16;
    a.decl.n_head_kv = 2;
    a.decl.head_dim = 8;
    a.block_size = 16;
    a.layers = { 0, 1 };
    g.push_back(a);
    KVGroupInfo k;
    k.name = "bkey";
    k.decl.kind = RAD_KV_FULL;
    k.decl.dtype = RAD_BF16;
    k.decl.n_head_kv = 1;
    k.decl.head_dim = 4;
    k.block_size = 16;
    k.layers = { 2, 3, 4 };
    g.push_back(k);
    return g;
}

/* `zero_last`: the last rank keeps no cache for the attention group (rad_kv_group_zero), as an
 * attention-zero rank does. Its window of a slot is the narrower group alone, the ranks' windows
 * are unequal, and every rank still takes part in every job. */
static void run_transfer(int world, bool fragment, bool zero_last = false) {
    /* THE HOST BACKEND ONLY. The stand-in kernels copy rows with the CPU, which on the host backend
     * is where "device" memory is; on a card the staging buffer is VRAM and the production kernels
     * are the ones to run. */
    if (!device_is_host()) {
        std::printf("  SKIP  the tier transfer needs the host backend (%s here)\n",
                    device_backend_name());
        return;
    }
    Registry reg;
    REQUIRE_EQ(reg.load_plugin(rows_plugin(), 0), RAD_OK);
    CHECK_OK(reg.check_complete());

    std::vector<std::unique_ptr<Fixture>> fx;
    for (int r = 0; r < world; ++r) {
        fx.push_back(std::make_unique<Fixture>());
        fx.back()->cfg.checkpoint_slots = 0;
        std::vector<KVGroupInfo> groups = two_paged_groups();
        if (zero_last && r == world - 1) groups[1].zero = true;
        REQUIRE_EQ(fx.back()->build_groups(groups), RAD_OK);
    }
    Fixture& f = *fx[0];
    const std::vector<int32_t>& cg = f.pc.cached_groups();
    REQUIRE_EQ(cg.size(), 2u);
    /* A slot is the sum of the ranks' entries, each rank's window where the ones before it end
     * (Engine::configure_kv). */
    std::vector<int64_t> win_off((size_t)world, 0);
    int64_t slot = 0;
    for (int r = 0; r < world; ++r) {
        win_off[(size_t)r] = slot;
        for (int32_t g : cg) slot += fx[(size_t)r]->kv.plan(g)->bytes_per_block;
    }

    /* FORTY SLOTS. Fragmented, every other one is taken and the copy goes out as twenty runs of
     * one; whole, it goes out as one run of twenty, which is the case where a rank's pitch through
     * a slot every rank shares decides where its rows land. */
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(40 * slot, slot)));
    REQUIRE_EQ(t.arena().capacity(), 40);
    if (fragment) {
        for (int i = 0; i < 40; ++i) CHECK_EQ(t.arena().alloc(), i);
        for (int i = 1; i < 40; i += 2) t.arena().free(i);
    }
    const int64_t held_before = t.arena().used_slots();
    f.pc.set_tiers(&t);

    std::vector<RadStream> compute((size_t)world, nullptr);
    std::vector<std::unique_ptr<TierExec>> x;
    for (int r = 0; r < world; ++r) {
        REQUIRE_EQ(rad_stream_create(&compute[(size_t)r], 0), RAD_OK);
        x.push_back(std::make_unique<TierExec>());
        /* A three-entry staging buffer, so twenty entries cross it in seven pieces. */
        REQUIRE_EQ(x.back()->configure(&reg, &fx[(size_t)r]->kv, &t, cg, /*max_batch=*/3,
                                       /*max_copy=*/64, /*max_chain=*/64, /*leader=*/r == 0,
                                       /*device=*/0, /*slot_off=*/win_off[(size_t)r],
                                       /*shared=*/nullptr), RAD_OK);
    }

    /* THE GROUPS' IDS ARE MADE TO DIFFER. They allocate in step, so an entry is normally the same
     * id in each -- and an index row read for the wrong group would then be right by accident. */
    int32_t skew = -1;
    CHECK_OK(f.kv.reserve_block(cg[1], &skew));

    const auto prompt = tok_run(4100, 336);
    publish(f, 1, prompt, 320);
    f.kv.free_sequence(1);
    std::vector<std::vector<int32_t>> hit;
    REQUIRE_EQ(f.pc.lookup(prompt, {}, &hit).n_attn_tokens, 320);
    const std::vector<std::vector<int32_t>> was = hit;
    CHECK(was[0][0] != was[1][0]);
    for (int r = 0; r < world; ++r) fill(*fx[(size_t)r], r);

    /* ---- out ---- */
    std::vector<TierJob> jobs;
    t.plan_writeback(64, &jobs);
    REQUIRE_EQ(jobs.size(), 20u);
    for (int r = 0; r < world; ++r)
        CHECK_OK(x[(size_t)r]->issue_copy(jobs, compute[(size_t)r]));
    for (int r = 0; r < world; ++r) {
        int st = 0;
        for (int spin = 0; spin < 20000 && (st = x[(size_t)r]->copy_state()) == 0; ++spin)
            usleep(100);
        CHECK_EQ(st, 1);
    }
    x[0]->report_pass(jobs, std::vector<char>(jobs.size(), 1), 0);
    CHECK_EQ(t.stats().to_host, 20);

    /* ---- the VRAM goes, and somebody else writes over it ---- */
    CHECK_EQ(f.pc.drop_clean(), 20);
    for (auto& fp : fx) scribble(*fp);
    /* A new sequence takes the lowest blocks, so the chain cannot come back where it was. */
    CHECK_OK(f.kv.add_sequence(2));
    CHECK_OK(f.kv.ensure(2, 5 * 16));

    /* ---- back ---- */
    const std::vector<BlockHash> keys = f.pc.restorable_from(prompt, {});
    REQUIRE_EQ(keys.size(), 20u);
    std::vector<TierJob> up;
    for (const BlockHash& k : keys) {
        TierJob j;
        REQUIRE(x[0]->plan_promote(k, &j));
        up.push_back(j);
    }
    std::vector<char> ok(up.size(), 1);
    for (int r = 0; r < world; ++r) {
        std::vector<char> mine;
        CHECK_OK(x[(size_t)r]->issue_restore(up, compute[(size_t)r], &mine));
        for (size_t i = 0; i < ok.size(); ++i) ok[i] = ok[i] && i < mine.size() && mine[i];
    }
    for (char c : ok) CHECK(c);
    x[0]->report_pass(up, ok, 0);
    for (const TierJob& j : up) CHECK_OK(f.pc.rebind(j.key, j.blocks));
    /* THE STEP THAT READS THEM WAITS ON THE DEVICE, not the host: draining the compute stream is
     * what reading them would do. */
    for (int r = 0; r < world; ++r) CHECK_OK(rad_stream_sync(compute[(size_t)r]));

    REQUIRE_EQ(f.pc.lookup(prompt, {}, &hit).n_attn_tokens, 320);
    const std::vector<std::vector<int32_t>> now = hit;
    int64_t moved = 0;
    for (size_t k = 0; k < was[0].size(); ++k) moved += was[0][k] != now[0][k];
    CHECK(moved > 0);
    for (int r = 0; r < world; ++r)
        CHECK_EQ(same_bytes(*fx[(size_t)r], r, was, now), 20);
    /* AND THE COPY STAYED: the entries are clean the moment they land. */
    CHECK_EQ(t.arena().used_slots(), held_before + 20);
    for (const BlockHash& k : keys) CHECK(t.droppable(k));

    for (auto& xp : x) xp->close();
    for (RadStream s : compute) rad_stream_destroy(s);
    f.pc.set_tiers(nullptr);
    (void)cg;
}

TEST(a_conversation_restored_from_host_is_byte_identical_in_new_blocks) {
    run_transfer(1, /*fragment=*/false);
    run_transfer(1, /*fragment=*/true);
}
TEST(each_rank_restores_its_own_window_of_a_shared_slot) {
    run_transfer(2, /*fragment=*/false);
    run_transfer(2, /*fragment=*/true);
}
TEST(a_rank_with_a_zero_group_moves_only_its_own_groups) {
    run_transfer(3, /*fragment=*/false, /*zero_last=*/true);
    run_transfer(3, /*fragment=*/true, /*zero_last=*/true);
}

/* A SPECULATING DEPLOYMENT KEEPS TWO PHYSICAL SLOTS A SEQUENCE, and the free list counts logical
 * ones. Subtracting one from the other reads an idle pool as half full. */
TEST(kv_bytes_in_use_count_a_speculating_state_once_per_sequence) {
    Fixture f;
    f.cfg.n_spec = 2;
    CHECK_OK(f.build(/*with_linear=*/true));
    const KVGroupPlan* ps = f.kv.plan(G_GDN);
    REQUIRE(ps != nullptr);
    CHECK_EQ(ps->state_copies, 2ll);

    CHECK_EQ(f.kv.pool_bytes_used(G_GDN), 0ll);
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16));
    CHECK_EQ(f.kv.pool_bytes_used(G_GDN), 2 * ps->bytes_per_state);
    f.kv.free_sequence(1);
    CHECK_EQ(f.kv.pool_bytes_used(G_GDN), 0ll);
}
