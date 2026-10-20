/* audio_gen_test.cpp -- the audio generator's pass driver (core/gen/audio_driver.h) for a vocoder
 * that plays frames as they are made (RadAudioGenDecl::voc_chunk), against sched_fakes.h's KV
 * manager and device layer. The plugin's side of each pass is played by the test: it writes the
 * FRAME pass's status and the VOCODE pass's samples where the driver reads them back.
 */
#include "sched_fakes.h"

using namespace sched_fake;

#include "gen/audio_driver.h"

#include <string>
#include <vector>

using namespace rad;

namespace {

/* Two frames a chunk, three frames of context, at most five frames, every frame kept. */
RadAudioGenDecl decl_chunked() {
    RadAudioGenDecl d{};
    d.max_prompt = 16;
    d.max_frames = 5;
    d.frame_rate_num = 25;
    d.frame_rate_den = 2;
    d.lat_num = 1;
    d.lat_den = 1;
    d.samples_per_latent = 4;
    d.sample_rate = 50;
    d.channels = 1;
    d.default_cfg = 1.0;
    d.max_lanes = 2;
    d.status = 1;
    d.audio_out = 2;
    d.prompt_whole = 1;
    d.first_frame = 0;
    d.voc_chunk = 2;
    d.voc_context = 3;
    return d;
}

struct AudioFixture {
    Program prog;
    FakeKV  kv;
    gen::AudioDriver drv;
    void build() {
        prog.audio_gen = decl_chunked();
        prog.agen_str.name = "test";
        prog.agen_str.neg_tmpl = "<n>{input}";
        prog.ctx.max_tok = 64;
        prog.ctx.max_seqs = 8;
        prog.kv_groups.emplace_back();
        KVGroupInfo g;
        g.name = "lm";
        g.decl.kind = RAD_KV_FULL;
        g.block_size = 16;
        prog.kv_groups.push_back(g);
        kv.setup(prog.kv_groups, 1 << 20, 256, 4);
    }
    int start() { return drv.init(prog, &kv, { gen::AudioDriver::RankIO{} }); }
    std::unique_ptr<gen::AudioRequest> req(float cfg, int64_t frames = 5) {
        auto r = std::make_unique<gen::AudioRequest>();
        r->tokens = { 1, 2, 3, 4 };
        r->neg_tokens = { 9, 9 };
        r->max_frames = frames;
        r->cfg = cfg;
        r->seed = 7;
        r->ticket = std::make_shared<gen::AudioTicket>();
        return r;
    }
    /* The plugin's answer to the pass in flight: each lane's FRAME status, or a VOCODE pass's
     * samples numbered from `sample0`. */
    void answer(const RadBatch* b, int32_t status, float sample0) {
        for (const gen::IPassDriver::Readback& rb : drv.readbacks()) {
            if (b->gen == RAD_GEN_PASS_FRAME) {
                int32_t* st = static_cast<int32_t*>(rb.host);
                for (int64_t i = 0; i < rb.bytes / 4; ++i) st[i] = status;
            } else {
                float* x = static_cast<float*>(rb.host);
                for (int64_t i = 0; i < rb.bytes / 4; ++i) x[i] = sample0 + (float)i;
            }
        }
    }
};

}  /* namespace */

/* A request's passes: its whole prompt (both branches, of their own lengths), then frames, a chunk
 * vocoded as soon as two are kept (before the next frame), and the last frame vocoded on its own
 * once the length is reached. Each chunk reads up to three earlier frames again: it names its
 * first frame, the frames it reads and how many lead it, and the frames' positions in the piece
 * are the pass's positions. The samples come back in order and end the ticket. */
TEST(a_chunked_request_vocodes_each_chunk_as_its_frames_are_made) {
    AudioFixture f;
    f.build();
    CHECK_OK(f.start());
    std::string why;
    auto r = f.req(3.0f);
    auto t = r->ticket;
    CHECK_OK(f.drv.submit(std::move(r), &why));
    std::string kinds;
    std::vector<std::vector<int32_t>> voc;   /* frame0, frames, left per VOCODE */
    float next_sample = 0.0f;
    /* Until nothing is left to run: the ticket's samples are read after the loop. */
    for (int i = 0; i < 20; ++i) {
        const RadBatch* b = f.drv.next(i);
        if (!b) break;
        const RadAudioGenBatch* A = b->agen_b;
        REQUIRE(A != nullptr);
        switch (b->gen) {
        case RAD_GEN_PASS_PROMPT:
            kinds += 'P';
            /* Both branches whole: four conditional rows, two unconditional. */
            CHECK_EQ(b->n_tok, (int64_t)6);
            CHECK_EQ(b->n_seq, (int64_t)2);
            break;
        case RAD_GEN_PASS_FRAME:
            kinds += 'F';
            CHECK_EQ(b->n_tok, (int64_t)2);
            break;
        case RAD_GEN_PASS_VOCODE:
            kinds += 'V';
            voc.push_back({ (int32_t)A->voc_frame0, A->voc_latents, A->voc_left });
            CHECK_EQ(b->n_tok, (int64_t)A->voc_latents);
            break;
        default:
            kinds += '?';
        }
        f.answer(b, 0, next_sample);
        if (b->gen == RAD_GEN_PASS_VOCODE)
            next_sample += (float)((A->voc_latents - A->voc_left) * 4);
        CHECK_OK(f.drv.done());
    }
    CHECK_EQ(kinds, std::string("PFFVFFVFV"));
    REQUIRE_EQ(voc.size(), (size_t)3);
    CHECK(voc[0] == std::vector<int32_t>({ 0, 2, 0 }));   /* frames 0, 1 */
    CHECK(voc[1] == std::vector<int32_t>({ 0, 4, 2 }));   /* 0, 1 again, then 2, 3 */
    CHECK(voc[2] == std::vector<int32_t>({ 1, 4, 3 }));   /* 1..3 again, then 4 */
    std::vector<float> all;
    gen::AudioChunk c;
    bool to = false;
    int32_t k = 0;
    while (t->next(&c, 0, &to)) {
        CHECK_EQ(c.window, k++);
        all.insert(all.end(), c.samples.begin(), c.samples.end());
    }
    REQUIRE_EQ(all.size(), (size_t)20);                   /* five frames of four samples */
    for (size_t i = 0; i < all.size(); ++i) CHECK_EQ(all[i], (float)i);
    CHECK_EQ(t->status(), (int)RAD_OK);
    CHECK_EQ(t->frames, (int64_t)5);
    CHECK_EQ(t->windows, (int64_t)3);
}

/* The model ending the piece right after a chunk was played leaves nothing owed: the request ends
 * at that FRAME pass, with no empty vocode. Ending it before the first frame fails the request. */
TEST(a_chunked_request_ends_with_the_model_and_fails_before_its_first_frame) {
    AudioFixture f;
    f.build();
    CHECK_OK(f.start());
    std::string why;
    auto r = f.req(1.0f);
    auto t = r->ticket;
    CHECK_OK(f.drv.submit(std::move(r), &why));
    std::string kinds;
    for (int i = 0; i < 20; ++i) {
        const RadBatch* b = f.drv.next(i);
        if (!b) break;
        kinds += b->gen == RAD_GEN_PASS_PROMPT ? 'P' : b->gen == RAD_GEN_PASS_FRAME ? 'F' : 'V';
        /* The third frame pass reports the model's end. */
        f.answer(b, kinds == "PFFVF" ? -1 : 0, 0.0f);
        CHECK_OK(f.drv.done());
    }
    CHECK_EQ(kinds, std::string("PFFVF"));
    CHECK(t->finished() || t->status() == RAD_OK);
    CHECK_EQ(t->status(), (int)RAD_OK);
    CHECK(t->model_ended);
    CHECK_EQ(t->frames, (int64_t)2);

    auto e = f.req(1.0f);
    auto te = e->ticket;
    CHECK_OK(f.drv.submit(std::move(e), &why));
    for (int i = 0; i < 4 && !te->finished(); ++i) {
        const RadBatch* b = f.drv.next(100 + i);
        REQUIRE(b != nullptr);
        f.answer(b, b->gen == RAD_GEN_PASS_FRAME ? -1 : 0, 0.0f);
        CHECK_OK(f.drv.done());
    }
    CHECK(te->finished());
    CHECK_EQ(te->status(), (int)RAD_E_INVAL);
    CHECK(has(te->why(), "before its first frame"));
}

/* Guidance 1 runs the conditional prompt in both branches; other guidance the request's own
 * unconditional prompt, at its own length and positions. */
TEST(the_unconditional_branch_is_the_conditional_prompt_at_guidance_one) {
    for (float cfg : { 1.0f, 2.0f }) {
        AudioFixture f;
        f.build();
        CHECK_OK(f.start());
        std::string why;
        CHECK_OK(f.drv.submit(f.req(cfg), &why));
        const RadBatch* b = f.drv.next(0);
        REQUIRE(b != nullptr && b->gen == RAD_GEN_PASS_PROMPT);
        CHECK_EQ(b->n_tok, (int64_t)(cfg == 1.0f ? 8 : 6));
        f.answer(b, 0, 0.0f);
        CHECK_OK(f.drv.done());
        b = f.drv.next(1);
        REQUIRE(b != nullptr && b->gen == RAD_GEN_PASS_FRAME);
    }
}

/* A hybrid language model (paged attention beside a linear-attention state and a convolution
 * group, as Qwen3.5's delta-net layers keep) of one branch: every pass stages one sequence a
 * request, the state groups' rows are the sequence's slots, and each entry carries its context
 * and a one-based num_accepted. The slots held at the prompt are the frames' too; guidance other
 * than 1 is refused. */
TEST(a_hybrid_model_of_one_branch_stages_its_state_groups) {
    AudioFixture f;
    f.prog.audio_gen = decl_chunked();
    f.prog.audio_gen.branches = 1;
    f.prog.agen_str.name = "hybrid";
    f.prog.ctx.max_tok = 64;
    f.prog.ctx.max_seqs = 8;
    f.prog.kv_groups.emplace_back();
    for (int kind : { RAD_KV_FULL, RAD_KV_LINEAR, RAD_KV_CONV }) {
        KVGroupInfo g;
        g.name = kind == RAD_KV_FULL ? "attn" : kind == RAD_KV_LINEAR ? "state" : "conv";
        g.decl.kind = kind;
        g.block_size = kind == RAD_KV_FULL ? 16 : 0;
        f.prog.kv_groups.push_back(g);
    }
    f.kv.setup(f.prog.kv_groups, 1 << 20, 256, 4);
    CHECK_OK(f.start());
    std::string why;
    CHECK(f.drv.submit(f.req(2.0f), &why) < 0);
    CHECK(has(why, "without guidance"));
    CHECK_OK(f.drv.submit(f.req(1.0f), &why));

    auto state_rows = [](const RadBatch* b, int32_t* st, int32_t* cv) {
        *st = *cv = -2;
        for (int i = 0; i < b->n_kv_groups; ++i) {
            const RadKVGroupBatch& k = b->kv[i];
            if (k.group == 1) continue;
            REQUIRE(k.state_index != nullptr && k.block_table != nullptr);
            CHECK_EQ(k.block_table[0], k.state_index[0]);
            (k.group == 2 ? *st : *cv) = k.state_index[0];
        }
    };
    const RadBatch* b = f.drv.next(0);
    REQUIRE(b != nullptr && b->gen == RAD_GEN_PASS_PROMPT);
    CHECK_EQ(b->n_seq, (int64_t)1);
    CHECK_EQ(b->n_tok, (int64_t)4);
    REQUIRE_EQ(b->n_kv_groups, 3);
    int32_t st0 = 0, cv0 = 0;
    state_rows(b, &st0, &cv0);
    CHECK(st0 >= 0 && cv0 >= 0);
    REQUIRE(b->ctx_lens != nullptr && b->num_accepted != nullptr);
    CHECK_EQ(b->ctx_lens[0], 0);
    CHECK_EQ(b->num_accepted[0], 1);
    CHECK_EQ(b->agen_b->slot[0], b->agen_b->lane_h[0]);   /* one branch: the slot is the lane */
    f.answer(b, 0, 0.0f);
    CHECK_OK(f.drv.done());

    b = f.drv.next(1);
    REQUIRE(b != nullptr && b->gen == RAD_GEN_PASS_FRAME);
    CHECK_EQ(b->n_seq, (int64_t)1);
    CHECK_EQ(b->n_tok, (int64_t)1);
    CHECK_EQ(b->positions[0], 4);                          /* after the four prompt rows */
    CHECK_EQ(b->ctx_lens[0], 4);
    /* The row the frame is sampled from: the prompt's last, at its slot. */
    CHECK_EQ(b->agen_b->prev_pos[0], 3);
    CHECK_EQ(b->agen_b->prev_used[0], 4);
    CHECK_EQ(b->agen_b->prev_slot[0], b->kv[0].slot_mapping[0] - 1);
    CHECK_EQ(b->num_accepted[0], 1);
    int32_t st1 = 0, cv1 = 0;
    state_rows(b, &st1, &cv1);
    CHECK_EQ(st1, st0);
    CHECK_EQ(cv1, cv0);
}

/* The driver stages one paged group beside state groups: a second paged group is refused at init,
 * by name. */
TEST(a_second_paged_group_is_refused) {
    AudioFixture f;
    f.prog.audio_gen = decl_chunked();
    f.prog.agen_str.name = "two";
    f.prog.ctx.max_tok = 64;
    f.prog.ctx.max_seqs = 8;
    f.prog.kv_groups.emplace_back();
    for (int i = 0; i < 2; ++i) {
        KVGroupInfo g;
        g.name = i ? "b" : "a";
        g.decl.kind = RAD_KV_FULL;
        g.block_size = 16;
        f.prog.kv_groups.push_back(g);
    }
    f.kv.setup(f.prog.kv_groups, 1 << 20, 256, 4);
    CHECK(f.start() < 0);
}

RAD_TEST_MAIN()
