/* minimaxh3 -- MiniMax-H3: a video and its stereo soundtrack from text.
 *
 * ============================== THE MODEL ==============================
 *
 * Four networks behind one prompt (diffusers' MiniMaxH3ModularPipeline, workflow t2va):
 *
 *   Text encoder: Qwen3-VL-32B's language model over the prompt's tokens (no template, no special
 *   tokens). The conditioning is the hidden state after its 50th layer, before any norm, so only
 *   the first 50 layers exist here.
 *
 *   Denoiser: a 33B single-stream diffusion transformer over ONE packed sequence
 *   [text | soundtrack | video] -- the conditioning through `context_embedder` and a two-block
 *   token refiner, the soundtrack's latents (40 a second, two channels one after the other, 32
 *   values each) through `audio_proj_in`, the video's latents in 1 x 2 x 2 patches through
 *   `proj_in` -- with full attention and a 3-axis rotary over (time, row, column). 50 blocks of
 *   `h += gate * attn(rms(h) w1 (1 + scale) + shift); h += gate' * mlp(...)`, each block's six
 *   modulation vectors chosen per row by its modality (video, text, audio) at its timestep. It is
 *   guidance-distilled: one forward a step. The video and the soundtrack step down their own
 *   schedules together.
 *
 *   Video decoder: a 36-layer ViT over the latents, decoded in 16 x 16-latent tiles of 7 latent
 *   frames (a 256-pixel tile, 28 pixel frames), blended where tiles and chunks overlap; frames
 *   come out at 24 a second.
 *
 *   Audio decoder: a BigVGAN with anti-aliased SnakeBeta activations, 800 samples a latent at
 *   32 kHz, each stereo channel a sequence of its own.
 *
 * The passes, their batches and the lanes a request's state lives in are the core's video
 * generator (abi/rad_builder.h, rad_declare_video_generator; core/gen/video_driver.h).
 *
 * ============================== A DISTILLED RELEASE ==============================
 *
 * FastVideo's FastH3 is the same denoiser distilled to eight evaluations on fixed timesteps with
 * video sparse attention, and says so in its inference settings (fastvideo_inference.json): the
 * timesteps on a 1000-step clock, the shifts (10 for the video), the attention and its sparsity.
 * Its attention cuts the packed sequence into tiles -- the prompt's and the soundtrack's rows in
 * 64-row chunks, the video's patches in 4 x 4 x 4 boxes of (time, row, column) -- and a video
 * query tile attends to every chunk and to the video tiles of best mean-query . mean-key score,
 * (1 - sparsity) of them; a chunk's queries attend to everything. A fourth projection beside q, k
 * and v (`to_gate_compress`) gates in an attention over the tiles' means. Everything else is the
 * base model's.
 *
 * ============================== ONCE A REQUEST ==============================
 *
 * The conditioning pass runs the text encoder and the refiner, and also every step's modulation:
 * a block's six vectors depend only on the step's timesteps, which the request's schedule fixes,
 * so the 13B parameters of the modulation projections are read once a request rather than once a
 * step. They are a table in the lane, [step][block][modality][this rank's share], gathered across
 * ranks one step at a time. Both the text encoder and the modulation projections are therefore
 * read once a request (RAD_ACCESS_PER_REQUEST), which lets placement keep them in host memory
 * (--vision-tower-placement ram).
 *
 * ============================== WHAT IS QUANTISED ==============================
 *
 * Every linear of the text encoder, of the refiner and of the DiT's blocks, and the modulation
 * projections, are block fp8 (LinearFP8 / DenseFP8 read what the container holds). The small
 * linears around them keep the checkpoint's precision: the timestep MLP and the four patch
 * projections in f32, `context_embedder` and `norm_out` in bf16. The DiT's dense attention takes
 * its scores in int8 (keys per row and head, the query inside the op; P.V in bf16), which runs the
 * score half at twice the rate. The video decoder's linears are bf16 with f32 norms and layer
 * scales; the audio decoder is f32 throughout.
 *
 * ============================== TENSOR PARALLEL ==============================
 *
 * The text encoder, the refiner and the DiT's blocks split by heads and feed-forward width, an
 * all-reduce after each output projection. A modulation projection splits by its output rows
 * (6 / world of the six vectors a rank) and the table is gathered a step at a time. The video
 * decoder splits its tiles between ranks with whole weights on each, and gathers the decoded
 * tiles; the audio decoder runs whole on every rank.
 */
#include <arch/rad_arch.h>
#include <arch/rad_fp8.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace minimaxh3 {

using namespace rad::arch;

/* ================================================================== the model's constants */
/* The text encoder's layer the conditioning is read after. */
enum { kTextLayers = 50 };
/* The longest prompt, in tokens; a structured prompt in the model's own style is about 1500. */
enum { kMaxText = 2048 };
/* Requests holding state at once: one denoising, one conditioned and waiting. */
enum { kMaxLanes = 2 };
/* Model evaluations a request may take (kMaxSteps), and the default sigma grid points (the
 * terminal 0 included, so one more than the evaluations). */
enum { kMaxSteps = 64, kDefSteps = 50 };
/* Time: 24 frames a second; a video of n chunks is 17 n + 5 frames over 5 n + 2 latent frames,
 * and a decode pass reads 7 latent frames from 5 c. 5 to 15 seconds: 7 to 20 chunks. */
enum { kFps = 24, kFpc = 17, kFtail = 5, kLpc = 5, kLtail = 2, kDecLat = 7 };
enum { kMinChunks = 7, kMaxChunks = 20, kDefChunks = 7 };
/* Space: 16 pixels a latent, 2 x 2 latents a patch; sides multiples of 32 from 256, at most the
 * 1344 x 768 canvas's area, 4:1 at most. */
enum { kScale = 16, kPatch = 2, kMult = 32, kMinSide = 256, kMaxAspect = 4 };
enum { kDefW = 1344, kDefH = 768, kMaxArea = kDefW * kDefH };
/* The soundtrack. */
enum { kRate = 32000, kAudioRate = 40, kAudioCh = 2, kAudioC = 32 };
/* The video decoder's tiles, in latents: 16 square, 4 apart at least, 4 register tokens and one
 * zero token after a tile's 7 x 16 x 16. */
enum { kTile = 16, kOverlap = 4, kRegs = 4, kTileTok = kDecLat * kTile * kTile,
       kPeriod = kTileTok + kRegs + 1 };
/* Rows the DiT's heads run a part at a time: the f32 copy of the normed rows they read. */
enum { kHeadRows = 16384 };
/* Rows a DiT all-reduce carries. The library's peer scratch is sized by the largest message an
 * all-reduce declares, two slots of it and outside the VRAM budget, so a step's whole 106K x 5376
 * residual would hold over 4 GiB a card; 16K rows is 176 MB a message, past where the link
 * saturates. */
enum { kArRows = 16384 };

/* The latent channels and the patch's values. */
enum { kC = 24, kPatchVals = kC * kPatch * kPatch };

/* Video sparse attention's tiles, in patches (time, row, column), for a checkpoint distilled with
 * it: 64 rows a tile, the dense segments cut into 64-row chunks of their own. */
enum { kVsaT = 4, kVsaH = 4, kVsaW = 4, kVsaRows = kVsaT * kVsaH * kVsaW };

/* ImageNet's normalisation, which the video decoder's pixels are in. */
static const char* kPixMean = "0.485 0.456 0.406";
static const char* kPixStd = "0.229 0.224 0.225";

/* The packed sequence's segments, in order, and each one's modulation table row: text rows take
 * the video's timestep with modality 1, the soundtrack its own with 2, the video its own with 0. */
enum { kSegText = 0, kSegAudio = 1, kSegVideo = 2, kSegs = 3 };
static const int kSegMod[kSegs] = { 1, 2, 0 };

/* ================================================================== geometry, on the host */
/* diffusers' AutoencoderKLMiniMaxH3._split_tiles in latents: the first latent of each tile across
 * a side of `L` latents. The overlaps start at the minimum and the slack goes to them round-robin,
 * a latent at a time. A side no longer than a tile is one tile (min_side keeps it exactly one). */
static std::vector<int64_t> tile_starts(int64_t L) {
    std::vector<int64_t> st{ 0 };
    if (kTile >= L) return st;
    int64_t n = (L + kTile - 1) / kTile;
    while (kTile * n - kOverlap * (n - 1) - L < 0) ++n;
    std::vector<int64_t> ov((size_t)(n - 1), kOverlap);
    const int64_t rem = kTile * n - kOverlap * (n - 1) - L;
    for (int64_t i = 0; i < rem; ++i) ov[(size_t)(i % (n - 1))] += 1;
    for (int64_t i = 0; i + 1 < n; ++i) st.push_back(st.back() + kTile - ov[(size_t)i]);
    return st;
}

static int64_t tiles_of(int64_t lat_h, int64_t lat_w) {
    return (int64_t)tile_starts(lat_h).size() * (int64_t)tile_starts(lat_w).size();
}

/* Whether a frame of w x h pixels is one this model makes. */
static bool size_ok(int64_t w, int64_t h) {
    return w % kMult == 0 && h % kMult == 0 && w >= kMinSide && h >= kMinSide &&
           w * h <= kMaxArea && std::max(w, h) <= kMaxAspect * std::min(w, h);
}

/* The most tiles a decode pass carries, over every size. */
static int64_t max_tiles() {
    int64_t best = 0;
    for (int64_t w = kMinSide; w <= (int64_t)kMaxArea / kMinSide; w += kMult)
        for (int64_t h = kMinSide; h * w <= kMaxArea; h += kMult)
            if (size_ok(w, h)) best = std::max(best, tiles_of(h / kScale, w / kScale));
    return best;
}

static int64_t max_lat_frames() { return (int64_t)kLpc * kMaxChunks + kLtail; }
static int64_t max_patches() { return (int64_t)kMaxArea / (kScale * kPatch * kScale * kPatch); }
static int64_t max_video_rows() { return max_lat_frames() * max_patches(); }
static int64_t max_audio() {
    const int64_t frames = (int64_t)kFpc * kMaxChunks + kFtail;
    return (frames * kAudioRate + kFps / 2) / kFps + 1;
}
static int64_t max_step_rows() { return kMaxText + kAudioCh * max_audio() + max_video_rows(); }

/* Video sparse attention's video tiles over a grid of F x ph x pw patches, and the ones a video
 * query keeps at `sparsity`: ceil((1 - sparsity) V) in f64, at least one. */
static int64_t vsa_video_tiles(int64_t F, int64_t ph, int64_t pw) {
    return ((F + kVsaT - 1) / kVsaT) * ((ph + kVsaH - 1) / kVsaH) * ((pw + kVsaW - 1) / kVsaW);
}
static int64_t vsa_keep(double sparsity, int64_t V) {
    const double keep = std::ceil((1.0 - sparsity) * (double)V);
    return std::max<int64_t>(1, std::min<int64_t>((int64_t)keep, V));
}
/* The most tiles a step has (the prompt's and the soundtrack's chunks, then the video's boxes) and
 * the most a video query keeps, over every size at the longest length. */
static void vsa_bounds(double sparsity, int64_t* tiles, int64_t* keep) {
    int64_t best = 0;
    for (int64_t w = kMinSide; w <= (int64_t)kMaxArea / kMinSide; w += kMult)
        for (int64_t h = kMinSide; h * w <= kMaxArea; h += kMult)
            if (size_ok(w, h))
                best = std::max(best, vsa_video_tiles(max_lat_frames(), h / (kScale * kPatch),
                                                      w / (kScale * kPatch)));
    const int64_t prefix = ((int64_t)kMaxText + kVsaRows - 1) / kVsaRows +
                           (kAudioCh * max_audio() + kVsaRows - 1) / kVsaRows;
    *tiles = prefix + best;
    *keep = vsa_keep(sparsity, best);
}

/* The packed sequence's rotary positions (diffusers' MiniMaxH3PrepareLayoutStep), in f64 as the
 * reference builds them, f32 as its rope reads them. Text rows count up on the time axis; the
 * media's clock starts after them. A latent frame spans 5/3 (1, 4, 4, 4, 4)[i % 5] units of time,
 * a soundtrack latent one; the spatial axes are each side's latents over the square root of the
 * frame's area, centred on the unit interval and scaled by 32 (numpy's linspace without its
 * endpoint), and a soundtrack's two channels sit at the two ends of the width axis. */
static int64_t positions(const RadVideoGeom* g, float* pos, int64_t cap) {
    if (!g || g->n_text < 1 || g->lat_frames < 1 || g->lat_h < kPatch || g->lat_w < kPatch ||
        g->n_audio < 1)
        return RAD_E_INVAL;
    const int64_t nt = g->n_text, na = g->n_audio, F = g->lat_frames;
    const int64_t ph = g->lat_h / kPatch, pw = g->lat_w / kPatch;
    const int64_t rows = nt + kAudioCh * na + F * ph * pw;
    if (!pos) return rows;
    if (cap < rows) return RAD_E_INVAL;
    float* P0 = pos;
    float* P1 = pos + rows;
    float* P2 = pos + 2 * rows;
    const double sq = std::sqrt((double)g->lat_h * (double)g->lat_w);
    auto grid = [&](int64_t dim, std::vector<double>* out) {
        const int64_t n = dim / kPatch;
        const double ratio = (double)dim / sq, left = (1.0 - ratio) / 2.0;
        const double step = ((left + ratio) - left) / (double)n;
        out->resize((size_t)n);
        for (int64_t i = 0; i < n; ++i) (*out)[(size_t)i] = ((double)i * step + left) * 32.0;
    };
    std::vector<double> hg, wg;
    grid(g->lat_h, &hg);
    grid(g->lat_w, &wg);
    int64_t r = 0;
    for (int64_t i = 0; i < nt; ++i, ++r) {
        P0[r] = (float)(double)i;
        P1[r] = 0.0f;
        P2[r] = 0.0f;
    }
    for (int64_t c = 0; c < kAudioCh; ++c)
        for (int64_t k = 0; k < na; ++k, ++r) {
            P0[r] = (float)((double)nt + (double)k);
            P1[r] = 0.0f;
            P2[r] = (float)(c == 0 ? wg.front() : wg.back());
        }
    static const int kPattern[5] = { 1, 4, 4, 4, 4 };
    double t = (double)nt;
    for (int64_t f = 0; f < F; ++f) {
        for (int64_t y = 0; y < ph; ++y)
            for (int64_t x = 0; x < pw; ++x, ++r) {
                P0[r] = (float)t;
                P1[r] = (float)hg[(size_t)y];
                P2[r] = (float)wg[(size_t)x];
            }
        t += 5.0 / 3.0 * (double)kPattern[f % 5];
    }
    return rows;
}

/* A decode pass's rows: every tile of the frame, each kPeriod rows. */
static int64_t decode_rows(const RadVideoGeom* g) {
    if (!g || g->lat_h < kTile || g->lat_w < kTile) return RAD_E_INVAL;
    return tiles_of(g->lat_h, g->lat_w) * kPeriod;
}

/* ================================================================== the model's parts */
struct TeLayer {
    rad_weight w_n1 = 0, w_n2 = 0, w_qn = 0, w_kn = 0;
    LinearFP8  qkv{}, o{}, gate_up{}, down{};
    QuantFP8   q_h{}, q_attn{}, q_h2{};
    GateQuantFP8 gq{};
    rad_op     op_n1 = 0, op_qkr = 0, op_attn = 0, op_ar1 = 0, op_add1 = 0;
    rad_op     op_n2 = 0, op_ar2 = 0, op_add2 = 0;
};

/* A token refiner block: a plain pre-norm transformer block, no rotary, no modulation. */
struct RefBlock {
    rad_weight w_n1 = 0, w_n2 = 0, w_qn = 0, w_kn = 0;
    LinearFP8  qkv{}, o{}, gate_up{}, down{};
    QuantFP8   q_h{}, q_attn{}, q_h2{};
    GateQuantFP8 gq{};
    rad_op     op_n1 = 0, op_qkn = 0, op_attn = 0, op_ar1 = 0, op_add1 = 0;
    rad_op     op_n2 = 0, op_ar2 = 0, op_add2 = 0;
};

/* A DiT block. Its two modulated norms carry the gated residual in front of them: the first the
 * previous block's MLP branch (none for block 0), the second this block's attention branch. */
struct DitBlock {
    rad_weight w_n1 = 0, w_n2 = 0, w_qn = 0, w_kn = 0;
    LinearFP8  qkv{}, o{}, gate_up{}, down{};
    QuantFP8   q_attn{};
    GateQuantFP8 gq{};
    rad_op     op_mq1 = 0, op_mq2 = 0, op_qkr = 0, op_attn = 0, op_ar1 = 0, op_ar2 = 0;
    rad_op     op_sel = 0;         /* video sparse attention's tile selection */
    rad_op     op_kq = 0;          /* the keys to int8 codes for the attention's scores */
    rad_op     op_vq = 0;          /* the values to E4M3 codes for the attention's P.V */
    /* The three modulation projections, one a modality, this rank's share of their rows. */
    DenseFP8   ada[3];
    rad_weight ada_b[3] = {};
    rad_op     op_ada[3] = {};
};

/* A video decoder block. */
struct VitBlock {
    rad_weight w_n1 = 0, w_n2 = 0, w_qkv = 0, b_qkv = 0, w_o = 0, b_o = 0;
    rad_weight w_up = 0, b_up = 0, w_down = 0, b_down = 0, s1 = 0, s2 = 0;
    rad_op     op_n1 = 0, op_qkv = 0, op_qkr = 0, op_attn = 0, op_o = 0, op_s1 = 0, op_r1 = 0;
    rad_op     op_n2 = 0, op_up = 0, op_act = 0, op_down = 0, op_s2 = 0, op_r2 = 0;
};

/* An audio decoder convolution (weight-normalised in the checkpoint, folded at convert). */
struct AConv {
    rad_weight w = 0, b = 0;
    rad_op     op = 0;
};
/* An anti-aliased SnakeBeta. */
struct AAct {
    rad_weight alpha = 0, beta = 0, up = 0, down = 0;
    rad_op     op = 0;
};
/* One dilation of an AMP block: act, dilated conv, act, conv with the residual. */
struct AmpUnit {
    AAct  a1{}, a2{};
    AConv c1{}, c2{};
};
struct AmpBlock {
    AmpUnit u[3];
};
struct AStage {
    AConv    up{};
    rad_op   op_cu = 0;
    AmpBlock amp[3];
    rad_op   op_sum = 0, op_avg = 0;
    int64_t  ch = 0, rate = 0, k = 0;
};

/* Weight groups (movement units) by network: the text encoder's layers 0.., the refiner, the
 * timestep MLP with norm_out's projection, the DiT's model-level weights and its blocks after
 * them, each block's modulation projections, the decoders. A unit takes the strongest access class
 * of its weights, so what is read once a request (the modulation, the timestep MLP) is never
 * grouped with what a step reads: that would keep it off the tower placement. */
enum : int {
    kGrpRefiner = 1000,
    kGrpTemb    = 1010,
    kGrpDit     = 2000,
    kGrpAda     = 2200,
    kGrpVae     = 3000,
    kGrpAudio   = 3100,
};

struct Model {
    Names nm{""};
    Geom  gt{}, gd{}, gr{};      /* text encoder, DiT (step rows), refiner (prompt rows) */
    int64_t D = 0, S = kMaxSteps;
    int64_t W = 1, Ws = 0;       /* ranks, and a rank's share of a modality's six vectors */
    int64_t maxV = 0, maxA = 0;  /* a lane's video and soundtrack latent rows */
    int64_t tiles = 0, tpr = 0;  /* the most tiles a decode carries, and a rank's share of them */

    /* ---- the text encoder */
    rad_weight w_tok = 0;
    rad_op  op_embed = 0;
    rad_buf b_tx = 0, b_tkv = 0, b_th2 = 0, b_tgu = 0;
    ActFP8  a_th{}, a_tattn{}, a_tffn{};
    std::vector<TeLayer> te;

    /* ---- context_embedder and the refiner, over the prompt's rows */
    rad_weight w_ce = 0, b_ce = 0, w_fn = 0;
    rad_op  op_ce = 0, op_fn = 0;
    rad_buf b_rh = 0, b_rqkv = 0, b_rao = 0, b_rgu = 0;
    ActFP8  a_rh{}, a_rattn{}, a_rffn{};
    RefBlock rb[2];

    /* ---- the timestep MLP and the per-step tables */
    rad_weight w_t1 = 0, b_t1 = 0, w_t2 = 0, b_t2 = 0, w_no = 0, b_no = 0;
    rad_op  op_temb = 0, op_t1 = 0, op_ts = 0, op_t2 = 0, op_ts2 = 0, op_no = 0;
    QuantFP8 q_temb{};
    rad_buf b_tproj = 0, b_te1 = 0, b_te1s = 0, b_temb = 0;
    ActFP8  a_temb{};

    /* ---- the lanes (PERSIST) */
    rad_buf b_text = 0;          /* [lanes * kMaxText, D] bf16: the refined prompt */
    rad_buf b_ada = 0;           /* [lanes * S, 150 * Ws] bf16: the modulation table */
    rad_buf b_nout = 0;          /* [lanes * S, 3 * 2D] bf16: norm_out's, by segment */
    rad_buf b_vstate = 0;        /* [lanes * maxV, 96] f32 */
    rad_buf b_astate = 0;        /* [lanes * maxA, 32] f32 */
    rad_buf b_prev = 0;          /* [lanes * 5 * area, 3] bf16: the decoder's frames across chunks */

    /* ---- the DiT */
    rad_op  op_rnd_v = 0, op_rnd_a = 0, op_modg = 0, op_text = 0, op_ain = 0, op_vin = 0;
    rad_weight w_ain = 0, b_ain = 0, w_vin = 0, b_vin = 0;
    rad_buf b_modg = 0;          /* [W * 150, Ws] bf16: this step's table, every rank's share */
    /* Video sparse attention (a checkpoint distilled with it): the dropped share of a video
     * query's video tiles, the most tiles a step has and the most a query keeps, and the per-layer
     * selection and pooled-tile attention, [heads, tiles, k] i32 and [heads, tiles, head_dim]. */
    bool    vsa = false;
    double  vsa_sparsity = 0.0;
    int64_t vsa_tiles = 0, vsa_k = 0;
    rad_buf b_sel = 0, b_coarse = 0;
    rad_buf b_k8 = 0, b_ks = 0;  /* the attention's keys at int8 codes and their scales */
    rad_buf b_v8 = 0, b_vs = 0;  /* its values at E4M3 codes and their column scales */
    /* The trained schedule's base times (a few-step checkpoint's), or none. */
    std::vector<double> rungs;
    rad_buf b_h = 0, b_qkv = 0, b_ao = 0, b_gu = 0, b_hn = 0, b_hf = 0, b_vel = 0, b_avel = 0;
    ActFP8  a_hm{}, a_attn{}, a_ffn{};
    std::vector<DitBlock> dit;
    rad_weight w_no_n = 0, w_vout = 0, b_vout = 0, w_aout = 0, b_aout = 0;
    rad_op  op_nout = 0, op_hcast = 0, op_vout = 0, op_aout = 0, op_ev = 0, op_ea = 0;

    /* ---- the audio decoder */
    std::string a_mean, a_std;
    rad_op  op_adenorm = 0, op_acu0 = 0, op_vmix = 0;
    AConv   a_in{}, a_pre{}, a_post{};
    AAct    a_actpost{};
    AStage  ast[7];
    int64_t a_rows[8] = {};
    rad_buf b_aden = 0, b_ain0 = 0, b_acu[8] = {}, b_ah[8] = {}, b_aavg[8] = {}, b_aamp[3][8] = {};
    rad_buf b_at1[8] = {}, b_at2[8] = {}, b_aout0 = 0, b_audio = 0;

    /* ---- the video decoder */
    std::string v_mean, v_std;
    rad_op  op_unfold = 0, op_pqc = 0, op_vpin = 0, op_tiles = 0, op_dpos = 0, op_dcu = 0;
    rad_op  op_vnorm = 0, op_vproj = 0, op_vgather = 0, op_stitch = 0;
    rad_weight w_pqc = 0, b_pqc = 0, w_vpin2 = 0, b_vpin2 = 0, w_reg = 0;
    rad_weight w_vno = 0, b_vno = 0, w_vpo = 0, b_vpo = 0;
    rad_buf b_vox = 0, b_vox2 = 0, b_vpin = 0, b_vx = 0, b_dpos = 0, b_dcu = 0, b_vn = 0;
    rad_buf b_vqkv = 0, b_vattn = 0, b_vo = 0, b_vt = 0, b_vup = 0, b_vact = 0;
    rad_buf b_vpix = 0, b_vpix_all = 0, b_frames = 0;
    std::vector<VitBlock> vit;
};

static Model g_model[MAX_RANKS];

/* A space-separated list off the container (rad-convert flattens arrays that way). */
static std::vector<double> meta_list(const RadModelMeta* m, const char* key) {
    std::vector<double> v;
    const char* s = rad_meta_gets(m, key, nullptr);
    while (s && *s) {
        char* e = nullptr;
        const double x = std::strtod(s, &e);
        if (e == s) { ++s; continue; }
        v.push_back(x);
        s = e;
    }
    return v;
}

static std::string join_list(const std::vector<double>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        char t[40];
        std::snprintf(t, sizeof t, "%s%.9g", i ? " " : "", v[i]);
        s += t;
    }
    return s;
}

/* A weight copied as the checkpoint holds it, under `group`, read whole on every rank. */
static rad_weight wcopy(RadBuilder* b, const char* dn, const char* src, uint32_t dtype,
                        std::initializer_list<int64_t> shape, int access, int group) {
    if (map_copy(b, dn, src) < 0) return 0;
    return decl_w(b, dn, dtype, shape, access, RAD_SHARD_NONE, grp_layer(group));
}

/* A linear with a bias, `gemm_nt_bias` at `dtype` (bf16 or f32), read whole on every rank. */
static rad_op dense_bias(RadBuilder* b, Names& nm, const char* dn, const char* src, uint32_t wdt,
                         const char* dtype, int64_t N, int64_t K, int64_t rows, rad_buf in,
                         rad_buf out, rad_weight* w, rad_weight* bias, int access, int group) {
    *w = wcopy(b, nm.f("%s.weight", dn), nm.ckpt("%s.weight", src), wdt, { N, K }, access, group);
    *bias = wcopy(b, nm.f("%s.bias", dn), nm.ckpt("%s.bias", src), wdt, { N }, access, group);
    if (!*w || !*bias) return 0;
    return rw(b, RAD_OP(b, "gemm_nt_bias",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                   RAD_STR("dtype", dtype)),
                        RAD_WEIGHTS(*w, *bias)),
              { in }, { out });
}

static rad_op unary(RadBuilder* b, const char* op, int64_t rows, int64_t n, const char* dtype,
                    rad_buf in, rad_buf out) {
    return rw(b, RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                          RAD_STR("dtype", dtype)),
                        RAD_NOWEIGHTS),
              { in }, { out });
}

static rad_op add_op(RadBuilder* b, int64_t rows, int64_t n, const char* dtype, rad_buf a, rad_buf c,
                     rad_buf y) {
    return rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", dtype)),
                        RAD_NOWEIGHTS),
              { a, c }, { y });
}

/* (a + b) / div, each step rounded to y's dtype. */
static rad_op add_div_op(RadBuilder* b, int64_t rows, int64_t n, const char* dtype, rad_buf a,
                         rad_buf c, rad_buf y, double div) {
    return rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", dtype),
                                   RAD_F64("div", div)),
                        RAD_NOWEIGHTS),
              { a, c }, { y });
}

static rad_op rmsnorm_op(RadBuilder* b, int64_t rows, int64_t n, double eps, rad_weight w, rad_buf x,
                         rad_buf y) {
    return rw(b, RAD_OP(b, "rmsnorm",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                   RAD_STR("dtype", "bf16")),
                        RAD_WEIGHTS(w)),
              { x }, { y });
}

/* `overlap`: issued on the second lane beside compute, which picks a narrow launch. */
static rad_op all_reduce(RadBuilder* b, const Geom& g, rad_buf x, int64_t rows, int64_t n,
                         const char* dtype = "bf16", bool overlap = false) {
    if (g.world <= 1) return 0;
    return rw(b, RAD_OP(b, "all_reduce",
                        RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", n, rows * n),
                                   RAD_STR("dtype", dtype), RAD_INT("exact", g.wire_exact),
                                   RAD_INT("min_bytes", g.wire_min_bytes),
                                   RAD_INT("overlap", overlap ? 1 : 0)),
                        RAD_NOWEIGHTS),
              { x }, { x });
}

/* ================================================================== check_format */
static int check_format(const RadModelMeta* meta) {
    const char* cls = rad_meta_gets(meta, "pipeline._class_name", "");
    if (std::strcmp(cls, "MiniMaxH3ModularPipeline") != 0 &&
        std::strcmp(cls, "MiniMaxH3Pipeline") != 0) {
        fprintf(stderr, "radiance: minimaxh3: the container's pipeline is '%s', not "
                        "MiniMaxH3ModularPipeline\n", cls);
        return RAD_E_UNSUPPORTED;
    }
    const int64_t ps[3] = { rad_meta_geti(meta, "transformer.patch_size.0", 1),
                            rad_meta_geti(meta, "transformer.patch_size.1", 2),
                            rad_meta_geti(meta, "transformer.patch_size.2", 2) };
    const std::vector<double> pl = meta_list(meta, "transformer.patch_size");
    const double pp = (double)kPatch;
    const bool patch_ok = pl.size() == 3 ? pl[0] == 1.0 && pl[1] == pp && pl[2] == pp
                                         : ps[0] == 1 && ps[1] == kPatch && ps[2] == kPatch;
    if (!patch_ok || rad_meta_geti(meta, "transformer.in_channels", 0) != kC ||
        rad_meta_geti(meta, "transformer.audio_in_channels", 0) != kAudioC ||
        rad_meta_geti(meta, "transformer.rope_freq_dim", 0) != 16 ||
        rad_meta_geti(meta, "vae.latent_channels", 0) != kC ||
        rad_meta_geti(meta, "vae.clip_length", 0) != kFpc ||
        rad_meta_geti(meta, "vae.token_drop", 0) != 3 ||
        rad_meta_geti(meta, "audio_vae.latent_channels", 0) != kAudioC ||
        rad_meta_geti(meta, "audio_vae.sampling_rate", 0) != kRate) {
        fprintf(stderr, "radiance: minimaxh3: the transformer, video VAE or audio VAE is not the "
                        "released geometry (patch 1x2x2, 24 + 32 latent channels, 17-frame "
                        "chunks, 32 kHz)\n");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* ================================================================== the text encoder */
/* Qwen3-VL's language model, its first kTextLayers layers: plain RMSNorm gains, q/k head norms,
 * SwiGLU. Its interleaved M-RoPE turns each frequency by one of three position components; a text
 * token's three are one position, which makes it NeoX's rotary over the whole head, declared as
 * such. Read once a request. */
static int declare_te(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.gt;
    Names& nm = m.nm;
    const int64_t T = g.max_tok;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();
    const int acc = RAD_ACCESS_PER_REQUEST;

    RAD_ARCH_TRY(map_copy(b, nm.f("te.token_embd.weight"),
                          nm.ckpt("model.language_model.embed_tokens.weight")));
    m.w_tok = decl_w(b, nm.f("te.token_embd.weight"), RAD_BF16, { g.n_vocab_all, g.n_embd },
                     RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    m.b_tx   = decl_b(b, nm.f("te.x"), RAD_BF16, { T, g.n_embd });
    m.a_th.x = decl_b(b, nm.f("te.h"), RAD_BF16, { T, g.n_embd });
    RAD_ARCH_TRY(m.a_th.declare_qs(b, nm, g, "te.h", g.n_embd));
    m.b_tkv  = decl_b(b, nm.f("te.qkv"), RAD_BF16, { T, q_dim + 2 * kv_dim });
    m.a_tattn.x = decl_b(b, nm.f("te.attn"), RAD_BF16, { T, q_dim });
    RAD_ARCH_TRY(m.a_tattn.declare_qs(b, nm, g, "te.attn", q_dim));
    m.b_th2  = decl_b(b, nm.f("te.h2"), RAD_BF16, { T, g.n_embd });
    m.b_tgu  = decl_b(b, nm.f("te.gate_up"), RAD_BF16, { T, 2 * g.n_ff });
    m.a_tffn.x = decl_b(b, nm.f("te.ffn"), RAD_BF16, { T, g.n_ff });
    RAD_ARCH_TRY(m.a_tffn.declare_qs(b, nm, g, "te.ffn", g.n_ff));

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", g.n_embd),
                                         RAD_INT("n_vocab", g.n_vocab_all), RAD_STR("dtype", "bf16"),
                                         RAD_INT("vocab_offset", 0)),
                              RAD_WEIGHTS(m.w_tok)),
                    {}, { m.b_tx });

    m.te.resize((size_t)g.n_layer);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        TeLayer& L = m.te[(size_t)l];
        const int li = (int)l;
        const char* pre = "model.language_model.layers";
        L.w_n1 = wcopy(b, nm.f("te.blk.%d.attn_norm.weight", li),
                       nm.ckpt("%s.%d.input_layernorm.weight", pre, li), RAD_F32, { g.n_embd }, acc, li);
        L.w_n2 = wcopy(b, nm.f("te.blk.%d.ffn_norm.weight", li),
                       nm.ckpt("%s.%d.post_attention_layernorm.weight", pre, li), RAD_F32,
                       { g.n_embd }, acc, li);
        L.w_qn = wcopy(b, nm.f("te.blk.%d.q_norm.weight", li),
                       nm.ckpt("%s.%d.self_attn.q_norm.weight", pre, li), RAD_BF16, { g.head_dim },
                       acc, li);
        L.w_kn = wcopy(b, nm.f("te.blk.%d.k_norm.weight", li),
                       nm.ckpt("%s.%d.self_attn.k_norm.weight", pre, li), RAD_BF16, { g.head_dim },
                       acc, li);
        if (!L.w_n1 || !L.w_n2 || !L.w_qn || !L.w_kn) return RAD_E_INVAL;
        L.op_n1 = rmsnorm_op(b, T, g.n_embd, g.eps, L.w_n1, m.b_tx, m.a_th.x);
        RAD_ARCH_TRY(L.q_h.declare(b, g, m.a_th, g.n_embd));
        L.qkv.access = acc;
        RAD_ARCH_TRY(L.qkv.declare(b, nm, g, nm.f("te.blk.%d.attn_qkv", li), q_dim + 2 * kv_dim,
                                   g.n_embd, RAD_SHARD_ROW, grp_layer(li),
                                   { nm.ckpt("%s.%d.self_attn.q_proj.weight", pre, li),
                                     nm.ckpt("%s.%d.self_attn.k_proj.weight", pre, li),
                                     nm.ckpt("%s.%d.self_attn.v_proj.weight", pre, li) },
                                   m.a_th, m.b_tkv, 0, { q_dim, kv_dim, kv_dim }));
        L.op_qkr = rw(b, RAD_OP(b, "qk_norm_rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", g.head_dim),
                                           RAD_INT("n_head", g.n_head),
                                           RAD_INT("n_head_kv", g.n_head_kv),
                                           RAD_F64("theta", g.theta), RAD_F64("eps", g.eps),
                                           RAD_STR("mode", "neox"),
                                           RAD_INT("rotary_dim", g.head_dim)),
                                RAD_WEIGHTS(L.w_qn, L.w_kn)),
                      { m.b_tkv }, { m.b_tkv });
        L.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", g.head_dim),
                                            RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                            RAD_STR("dtype", "bf16"),
                                            RAD_F64("scale", 1.0 / std::sqrt((double)g.head_dim)),
                                            RAD_INT("max_seqlen", T)),
                                 RAD_NOWEIGHTS),
                       { m.b_tkv }, { m.a_tattn.x });
        RAD_ARCH_TRY(L.q_attn.declare(b, g, m.a_tattn, q_dim));
        L.o.access = acc;
        RAD_ARCH_TRY(L.o.declare(b, nm, g, nm.f("te.blk.%d.attn_output", li), g.n_embd, q_dim,
                                 RAD_SHARD_COL, grp_layer(li),
                                 { nm.ckpt("%s.%d.self_attn.o_proj.weight", pre, li) },
                                 m.a_tattn, m.b_th2));
        L.op_ar1 = all_reduce(b, g, m.b_th2, T, g.n_embd);
        L.op_add1 = add_op(b, T, g.n_embd, "bf16", m.b_tx, m.b_th2, m.b_tx);
        L.op_n2 = rmsnorm_op(b, T, g.n_embd, g.eps, L.w_n2, m.b_tx, m.a_th.x);
        RAD_ARCH_TRY(L.q_h2.declare(b, g, m.a_th, g.n_embd));
        L.gate_up.access = acc;
        RAD_ARCH_TRY(L.gate_up.declare(b, nm, g, nm.f("te.blk.%d.ffn_gate_up", li), 2 * g.n_ff,
                                       g.n_embd, RAD_SHARD_ROW, grp_layer(li),
                                       { nm.ckpt("%s.%d.mlp.gate_proj.weight", pre, li),
                                         nm.ckpt("%s.%d.mlp.up_proj.weight", pre, li) },
                                       m.a_th, m.b_tgu, 0, { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(L.gq.declare(b, g, m.b_tgu, m.a_tffn, g.n_ff, "silu", 0, true));
        L.down.access = acc;
        RAD_ARCH_TRY(L.down.declare(b, nm, g, nm.f("te.blk.%d.ffn_down", li), g.n_embd, g.n_ff,
                                    RAD_SHARD_COL, grp_layer(li),
                                    { nm.ckpt("%s.%d.mlp.down_proj.weight", pre, li) },
                                    m.a_tffn, m.b_th2));
        L.op_ar2 = all_reduce(b, g, m.b_th2, T, g.n_embd);
        L.op_add2 = add_op(b, T, g.n_embd, "bf16", m.b_tx, m.b_th2, m.b_tx);
        if (!L.op_n1 || !L.op_qkr || !L.op_attn || !L.op_add1 || !L.op_n2 || !L.op_add2)
            return RAD_E_INVAL;
    }
    return m.op_embed ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== the token refiner */
/* `context_embedder` (5120 -> 5376, bf16, replicated), two plain blocks with full attention over
 * the prompt and no rotary, and the final RMSNorm into the lane's prompt rows. The feed-forward is
 * diffusers' SwiGLU, `value * silu(gate)` with the value first in the checkpoint; the gate half is
 * moved first at convert, which is the order `silu_mul` reads. */
static int declare_refiner(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.gr;
    Names& nm = m.nm;
    const int64_t T = g.max_tok, D = g.n_embd, qd = g.q_dim(), hd = g.head_dim;
    const int acc = RAD_ACCESS_PER_REQUEST;
    const int gx = kGrpRefiner;
    const double eps = rad_meta_getf(meta, "transformer.norm_eps", 1e-5);
    const double qk_eps = rad_meta_getf(meta, "transformer.qk_norm_eps", 1e-5);
    const int64_t tdim = rad_meta_geti(meta, "transformer.text_dim", 5120);
    if (tdim != m.gt.n_embd) {
        fprintf(stderr, "radiance: minimaxh3: the DiT reads %lld-wide conditioning and the text "
                        "encoder writes %lld\n", (long long)tdim, (long long)m.gt.n_embd);
        return RAD_E_INVAL;
    }

    m.b_rh = decl_b(b, nm.f("ref.h"), RAD_BF16, { T, D });
    m.op_ce = dense_bias(b, nm, "ref.ctx", "transformer.context_embedder", RAD_BF16, "bf16", D,
                         tdim, T, m.b_tx, m.b_rh, &m.w_ce, &m.b_ce, acc, gx);
    m.a_rh.x = decl_b(b, nm.f("ref.hn"), RAD_BF16, { T, D });
    RAD_ARCH_TRY(m.a_rh.declare_qs(b, nm, g, "ref.hn", D));
    m.b_rqkv = decl_b(b, nm.f("ref.qkv"), RAD_BF16, { T, 3 * qd });
    m.a_rattn.x = decl_b(b, nm.f("ref.attn"), RAD_BF16, { T, qd });
    RAD_ARCH_TRY(m.a_rattn.declare_qs(b, nm, g, "ref.attn", qd));
    m.b_rao = decl_b(b, nm.f("ref.ao"), RAD_BF16, { T, D });
    m.b_rgu = decl_b(b, nm.f("ref.gate_up"), RAD_BF16, { T, 2 * g.n_ff });
    m.a_rffn.x = decl_b(b, nm.f("ref.ffn"), RAD_BF16, { T, g.n_ff });
    RAD_ARCH_TRY(m.a_rffn.declare_qs(b, nm, g, "ref.ffn", g.n_ff));
    const int64_t ff_all = g.n_ff_all;

    for (int l = 0; l < 2; ++l) {
        RefBlock& R = m.rb[l];
        const char* pre = nm.ckpt("transformer.token_refiner.refiner_blocks.%d", l);
        const char* dn = nm.f("ref.blk.%d", l);
        R.w_n1 = wcopy(b, nm.f("%s.norm1.weight", dn), nm.ckpt("%s.norm1.weight", pre), RAD_BF16,
                       { D }, acc, gx);
        R.w_n2 = wcopy(b, nm.f("%s.norm2.weight", dn), nm.ckpt("%s.norm2.weight", pre), RAD_BF16,
                       { D }, acc, gx);
        R.w_qn = wcopy(b, nm.f("%s.q_norm.weight", dn), nm.ckpt("%s.attn.norm_q.weight", pre),
                       RAD_BF16, { hd }, acc, gx);
        R.w_kn = wcopy(b, nm.f("%s.k_norm.weight", dn), nm.ckpt("%s.attn.norm_k.weight", pre),
                       RAD_BF16, { hd }, acc, gx);
        if (!R.w_n1 || !R.w_n2 || !R.w_qn || !R.w_kn) return RAD_E_INVAL;
        R.op_n1 = rmsnorm_op(b, T, D, eps, R.w_n1, m.b_rh, m.a_rh.x);
        RAD_ARCH_TRY(R.q_h.declare(b, g, m.a_rh, D));
        R.qkv.access = acc;
        RAD_ARCH_TRY(R.qkv.declare(b, nm, g, nm.f("%s.attn_qkv", dn), 3 * qd, D, RAD_SHARD_ROW,
                                   grp_layer(gx),
                                   { nm.ckpt("%s.attn.to_q.weight", pre),
                                     nm.ckpt("%s.attn.to_k.weight", pre),
                                     nm.ckpt("%s.attn.to_v.weight", pre) },
                                   m.a_rh, m.b_rqkv, 0, { qd, qd, qd }));
        R.op_qkn = rw(b, RAD_OP(b, "qk_norm_rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                           RAD_INT("n_head", g.n_head),
                                           RAD_INT("n_head_kv", g.n_head),
                                           RAD_F64("theta", 10000.0), RAD_F64("eps", qk_eps),
                                           RAD_INT("rotary_dim", 0)),
                                RAD_WEIGHTS(R.w_qn, R.w_kn)),
                      { m.b_rqkv }, { m.b_rqkv });
        R.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                            RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                            RAD_STR("dtype", "bf16"),
                                            RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                            RAD_INT("max_seqlen", T)),
                                 RAD_NOWEIGHTS),
                       { m.b_rqkv }, { m.a_rattn.x });
        RAD_ARCH_TRY(R.q_attn.declare(b, g, m.a_rattn, qd));
        R.o.access = acc;
        RAD_ARCH_TRY(R.o.declare(b, nm, g, nm.f("%s.attn_output", dn), D, qd, RAD_SHARD_COL,
                                 grp_layer(gx), { nm.ckpt("%s.attn.to_out.0.weight", pre) },
                                 m.a_rattn, m.b_rao));
        R.op_ar1 = all_reduce(b, g, m.b_rao, T, D);
        R.op_add1 = add_op(b, T, D, "bf16", m.b_rh, m.b_rao, m.b_rh);
        R.op_n2 = rmsnorm_op(b, T, D, eps, R.w_n2, m.b_rh, m.a_rh.x);
        RAD_ARCH_TRY(R.q_h2.declare(b, g, m.a_rh, D));
        const char* fw = nm.f("%s.ffn_gate_up.weight", dn);
        const char* fsrc = nm.ckpt("%s.ff.net.0.proj.weight", pre);
        RAD_ARCH_TRY(map_rows(b, fw, { { fsrc, (int)ff_all, ff_all }, { fsrc, 0, ff_all } }));
        R.gate_up.mapped = true;
        R.gate_up.access = acc;
        RAD_ARCH_TRY(R.gate_up.declare(b, nm, g, nm.f("%s.ffn_gate_up", dn), 2 * g.n_ff, D,
                                       RAD_SHARD_ROW, grp_layer(gx), { fsrc }, m.a_rh, m.b_rgu, 0,
                                       { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(R.gq.declare(b, g, m.b_rgu, m.a_rffn, g.n_ff, "silu", 0, true));
        R.down.access = acc;
        RAD_ARCH_TRY(R.down.declare(b, nm, g, nm.f("%s.ffn_down", dn), D, g.n_ff, RAD_SHARD_COL,
                                    grp_layer(gx), { nm.ckpt("%s.ff.net.2.weight", pre) },
                                    m.a_rffn, m.b_rao));
        R.op_ar2 = all_reduce(b, g, m.b_rao, T, D);
        R.op_add2 = add_op(b, T, D, "bf16", m.b_rh, m.b_rao, m.b_rh);
        if (!R.op_n1 || !R.op_qkn || !R.op_attn || !R.op_add1 || !R.op_n2 || !R.op_add2)
            return RAD_E_INVAL;
    }
    m.w_fn = wcopy(b, nm.f("ref.final_norm.weight"),
                   nm.ckpt("transformer.token_refiner.final_norm.weight"), RAD_BF16, { D }, acc, gx);
    if (!m.w_fn) return RAD_E_INVAL;
    m.op_fn = rmsnorm_op(b, T, D, rad_meta_getf(meta, "transformer.final_norm_eps", 1e-5), m.w_fn,
                         m.b_rh, m.b_text);
    return m.op_ce && m.op_fn ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== the per-step tables */
/* The timestep embedding of every step's two timesteps -- rows [0, S) the video's, [S, 2S) the
 * soundtrack's -- through the f32 timestep MLP, then silu, rounded to bf16 as the bf16 projections
 * read it, and quantised for the modulation projections. Each block's modulation for segment e is
 * its modality's projection of its timestep's row, into the lane's table at column block
 * (l, e); norm_out's shift and scale are the same of norm_out's own projection, by timestep. */
static int declare_tables(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Names& nm = m.nm;
    const Geom& g = m.gd;
    const int64_t S = m.S, D = m.D, L = g.n_layer;
    const int64_t fd = rad_meta_geti(meta, "transformer.freq_dim", 256);
    const int64_t hd1 = rad_meta_geti(meta, "transformer.time_embed_hidden_dim", 5376);
    const int64_t td = rad_meta_geti(meta, "transformer.time_embed_dim", 2688);
    const int acc = RAD_ACCESS_PER_REQUEST;
    const int gx = kGrpTemb;
    if (td % RAD_FP8_BLOCK) {
        fprintf(stderr, "radiance: minimaxh3: the timestep embedding is %lld wide, not whole "
                        "%d-column blocks\n", (long long)td, RAD_FP8_BLOCK);
        return RAD_E_INVAL;
    }
    m.b_tproj = decl_b(b, nm.f("tmb.proj"), RAD_F32, { 2 * S, fd });
    m.b_te1 = decl_b(b, nm.f("tmb.h1"), RAD_F32, { 2 * S, hd1 });
    m.b_te1s = decl_b(b, nm.f("tmb.h1s"), RAD_F32, { 2 * S, hd1 });
    m.b_temb = decl_b(b, nm.f("tmb.emb"), RAD_F32, { 2 * S, td });
    m.op_temb = rw(b, RAD_OP(b, "timestep_embed",
                             RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", fd),
                                        RAD_F64("scale", 1.0), RAD_F64("max_period", 10000.0),
                                        RAD_STR("dtype", "f32")),
                             RAD_NOWEIGHTS),
                   {}, { m.b_tproj });
    m.op_t1 = dense_bias(b, nm, "tmb.l1", "transformer.time_embedder.linear_1", RAD_F32, "f32", hd1,
                         fd, 2 * S, m.b_tproj, m.b_te1, &m.w_t1, &m.b_t1, acc, gx);
    m.op_ts = unary(b, "silu", 2 * S, hd1, "f32", m.b_te1, m.b_te1s);
    m.op_t2 = dense_bias(b, nm, "tmb.l2", "transformer.time_embedder.linear_2", RAD_F32, "f32", td,
                         hd1, 2 * S, m.b_te1s, m.b_temb, &m.w_t2, &m.b_t2, acc, gx);
    /* silu(temb) in f32, rounded once to the bf16 the projections take. */
    Geom gs = g;
    gs.max_tok = 2 * S;
    m.a_temb.x = decl_b(b, nm.f("tmb.act"), RAD_BF16, { 2 * S, td });
    m.op_ts2 = unary(b, "silu", 2 * S, td, "f32", m.b_temb, m.a_temb.x);
    RAD_ARCH_TRY(m.a_temb.declare_qs(b, nm, gs, "tmb.act", td, 2 * S));
    RAD_ARCH_TRY(m.q_temb.declare(b, gs, m.a_temb, td));
    if (!m.op_temb || !m.op_t1 || !m.op_ts || !m.op_t2 || !m.op_ts2) return RAD_E_INVAL;

    /* The lane's tables. */
    m.b_ada = decl_b(b, nm.f("lane.ada"), RAD_BF16, { (int64_t)kMaxLanes * S, L * kSegs * m.Ws },
                     RAD_BUF_PERSIST);
    m.b_nout = decl_b(b, nm.f("lane.nout"), RAD_BF16, { (int64_t)kMaxLanes * S, kSegs * 2 * D },
                      RAD_BUF_PERSIST);
    const int64_t ada_rows = 6 * D;   /* one modality's six vectors */
    m.dit.resize((size_t)L);
    for (int64_t l = 0; l < L; ++l) {
        DitBlock& B = m.dit[(size_t)l];
        const int li = (int)l;
        const int grp = kGrpAda + li;
        const char* src_w = nm.ckpt("transformer.transformer_blocks.%d.adaln_proj.linear.weight", li);
        const char* src_b = nm.ckpt("transformer.transformer_blocks.%d.adaln_proj.linear.bias", li);
        for (int md = 0; md < 3; ++md) {
            const char* base = nm.f("dit.blk.%d.ada%d", li, md);
            const char* wn = nm.f("%s.weight", base);
            const char* bn = nm.f("%s.bias", base);
            RAD_ARCH_TRY(map_rows(b, wn, { { src_w, (int)(md * ada_rows), ada_rows } }));
            RAD_ARCH_TRY(map_rows(b, bn, { { src_b, (int)(md * ada_rows), ada_rows } }));
            B.ada[md].mapped = true;
            RAD_ARCH_TRY(B.ada[md].declare(b, nm, base, m.Ws, td, acc, RAD_SHARD_ROW,
                                           grp_layer(grp), { src_w }));
            B.ada_b[md] = decl_w(b, bn, RAD_BF16, { m.Ws }, acc, RAD_SHARD_ROW, grp_layer(grp));
            if (!B.ada_b[md]) return RAD_E_INVAL;
            B.op_ada[md] = rw(b, RAD_OP(b, "gemm_nt_q_bias",
                                        RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("N", m.Ws),
                                                   RAD_INT("K", td), RAD_INT("group", RAD_FP8_BLOCK),
                                                   RAD_STR("dtype", "fp8a8")),
                                        RAD_WEIGHTS(B.ada[md].w, B.ada[md].s, B.ada_b[md])),
                              { m.a_temb.q, m.a_temb.s }, { m.b_ada });
            if (!B.op_ada[md]) return RAD_E_INVAL;
        }
    }
    m.op_no = dense_bias(b, nm, "dit.nout.lin", "transformer.norm_out.linear", RAD_BF16, "bf16",
                         2 * D, td, S, m.a_temb.x, m.b_nout, &m.w_no, &m.b_no, acc, gx);
    return m.op_no ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== the DiT */
static int declare_dit(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.gd;
    Names& nm = m.nm;
    const int64_t T = g.max_tok, D = g.n_embd, qd = g.q_dim(), hd = g.head_dim, L = g.n_layer;
    const double eps = rad_meta_getf(meta, "transformer.norm_eps", 1e-5);
    const double qk_eps = rad_meta_getf(meta, "transformer.qk_norm_eps", 1e-5);
    const double theta = rad_meta_getf(meta, "transformer.rope_theta", 10000.0);
    const int64_t rf = rad_meta_geti(meta, "transformer.rope_freq_dim", 16);
    const int gx = kGrpDit;
    const int64_t La = (int64_t)kMaxLanes;

    /* ---- the lanes' latents, and the first step's noise */
    m.b_vstate = decl_b(b, nm.f("lane.video"), RAD_F32, { La * m.maxV, kPatchVals }, RAD_BUF_PERSIST);
    m.b_astate = decl_b(b, nm.f("lane.audio"), RAD_F32, { La * m.maxA, kAudioC }, RAD_BUF_PERSIST);
    m.op_rnd_v = rw(b, RAD_OP(b, "randn_lanes",
                              RAD_PARAMS(RAD_RANGE("M", 1, m.maxV), RAD_INT("n", kPatchVals),
                                         RAD_INT("stride", m.maxV), RAD_STR("dtype", "f32")),
                              RAD_NOWEIGHTS),
                    { m.b_vstate }, { m.b_vstate });
    m.op_rnd_a = rw(b, RAD_OP(b, "randn_lanes",
                              RAD_PARAMS(RAD_RANGE("M", 1, m.maxA), RAD_INT("n", kAudioC),
                                         RAD_INT("stride", m.maxA), RAD_STR("dtype", "f32")),
                              RAD_NOWEIGHTS),
                    { m.b_astate }, { m.b_astate });

    /* ---- this step's modulation, every rank's share of it */
    const int64_t row = L * kSegs * m.Ws;
    m.b_modg = decl_b(b, nm.f("dit.mod"), RAD_BF16, { (int64_t)g.world * L * kSegs, m.Ws });
    if (g.world > 1)
        m.op_modg = rw(b, RAD_OP(b, "all_gather",
                                 RAD_PARAMS(RAD_INT("world_size", g.world),
                                            RAD_RANGE("numel", 1, row), RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                       { m.b_ada }, { m.b_modg });
    else
        m.op_modg = rw(b, RAD_OP(b, "cast",
                                 RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", row),
                                            RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                                 RAD_NOWEIGHTS),
                       { m.b_ada }, { m.b_modg });

    /* ---- the packed sequence: the lane's prompt rows, the soundtrack's and the video's latents
     * through their f32 projections, rounded to bf16 as the reference packs them */
    m.b_h = decl_b(b, nm.f("dit.h"), RAD_BF16, { T, D });
    m.op_text = rw(b, RAD_OP(b, "lane_gather",
                             RAD_PARAMS(RAD_RANGE("M", 1, (int64_t)kMaxText), RAD_INT("n", D),
                                        RAD_INT("stride", (int64_t)kMaxText),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { m.b_text }, { m.b_h });
    m.op_ain = dense_bias(b, nm, "dit.audio_in", "transformer.audio_proj_in", RAD_F32, "f32", D,
                          kAudioC, m.maxA, m.b_astate, m.b_h, &m.w_ain, &m.b_ain,
                          RAD_ACCESS_PER_TOKEN, gx);
    m.op_vin = dense_bias(b, nm, "dit.video_in", "transformer.proj_in", RAD_F32, "f32", D,
                          kPatchVals, m.maxV, m.b_vstate, m.b_h, &m.w_vin, &m.b_vin,
                          RAD_ACCESS_PER_TOKEN, gx);
    if (!m.op_rnd_v || !m.op_rnd_a || !m.op_modg || !m.op_text || !m.op_ain || !m.op_vin)
        return RAD_E_INVAL;

    /* ---- the blocks */
    RAD_ARCH_TRY(m.a_hm.declare_qs(b, nm, g, "dit.hm", D));
    /* With video sparse attention a fourth projection beside q, k and v gates the pooled-tile
     * attention into each row's output. */
    const int64_t nparts = m.vsa ? 4 : 3;
    m.b_qkv = decl_b(b, nm.f("dit.qkv"), RAD_BF16, { T, nparts * qd });
    if (m.vsa) {
        m.b_sel = decl_b(b, nm.f("dit.vsa_sel"), RAD_I32, { g.n_head, m.vsa_tiles, m.vsa_k });
        m.b_coarse = decl_b(b, nm.f("dit.vsa_coarse"), RAD_BF16, { g.n_head, m.vsa_tiles, hd });
    }
    m.b_k8 = decl_b(b, nm.f("dit.k8"), RAD_I8, { T, qd });
    m.b_ks = decl_b(b, nm.f("dit.ks"), RAD_F32, { T, qd / hd });
    m.b_v8 = decl_b(b, nm.f("dit.v8"), RAD_F8E4M3, { T, qd });
    m.b_vs = decl_b(b, nm.f("dit.vs"), RAD_F32, { qd });
    m.a_attn.x = decl_b(b, nm.f("dit.attn"), RAD_BF16, { T, qd });
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "dit.attn", qd));
    m.b_ao = decl_b(b, nm.f("dit.ao"), RAD_BF16, { T, D });
    /* More than one rank: the all-reduces of b_ao's halves run on the second lane (step_dit). */
    if (g.world > 1) RAD_ARCH_TRY(rad_buf_concurrent(b, m.b_ao));
    m.b_gu = decl_b(b, nm.f("dit.gate_up"), RAD_BF16, { T, 2 * g.n_ff });
    m.a_ffn.x = decl_b(b, nm.f("dit.ffn"), RAD_BF16, { T, g.n_ff });
    RAD_ARCH_TRY(m.a_ffn.declare_qs(b, nm, g, "dit.ffn", g.n_ff));
    const char* sections = nm.f("%lld %lld %lld", (long long)rf, (long long)rf, (long long)rf);
    const int64_t ff_all = g.n_ff_all;

    auto mod_q = [&](rad_weight w, bool gated) {
        rad_op op = RAD_OP(b, "rms_mod_quant_fp8",
                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_F64("eps", eps),
                                      RAD_INT("group", RAD_FP8_BLOCK), RAD_INT("gated", gated ? 1 : 0),
                                      RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(w));
        if (!op) return op;
        rad_buf rd[3] = { m.b_modg, m.b_h, m.b_ao };
        rad_buf wr[3] = { m.a_hm.q, m.a_hm.s, m.b_h };
        rad_op_reads(b, op, rd, gated ? 3 : 2);
        rad_op_writes(b, op, wr, 3);
        return op;
    };

    for (int64_t l = 0; l < L; ++l) {
        DitBlock& B = m.dit[(size_t)l];
        const int li = (int)l;
        const int grp = kGrpDit + 1 + li;
        const char* pre = nm.ckpt("transformer.transformer_blocks.%d", li);
        const char* dn = nm.f("dit.blk.%d", li);
        const int acc = RAD_ACCESS_PER_TOKEN;
        B.w_n1 = wcopy(b, nm.f("%s.norm1.weight", dn), nm.ckpt("%s.norm1.weight", pre), RAD_BF16,
                       { D }, acc, grp);
        B.w_n2 = wcopy(b, nm.f("%s.norm2.weight", dn), nm.ckpt("%s.norm2.weight", pre), RAD_BF16,
                       { D }, acc, grp);
        B.w_qn = wcopy(b, nm.f("%s.q_norm.weight", dn), nm.ckpt("%s.attn.norm_q.weight", pre),
                       RAD_BF16, { hd }, acc, grp);
        B.w_kn = wcopy(b, nm.f("%s.k_norm.weight", dn), nm.ckpt("%s.attn.norm_k.weight", pre),
                       RAD_BF16, { hd }, acc, grp);
        if (!B.w_n1 || !B.w_n2 || !B.w_qn || !B.w_kn) return RAD_E_INVAL;
        B.op_mq1 = mod_q(B.w_n1, l > 0);
        if (m.vsa)
            RAD_ARCH_TRY(B.qkv.declare(b, nm, g, nm.f("%s.attn_qkv", dn), 4 * qd, D, RAD_SHARD_ROW,
                                       grp_layer(grp),
                                       { nm.ckpt("%s.attn.to_q.weight", pre),
                                         nm.ckpt("%s.attn.to_k.weight", pre),
                                         nm.ckpt("%s.attn.to_v.weight", pre),
                                         nm.ckpt("%s.attn.to_gate_compress.weight", pre) },
                                       m.a_hm, m.b_qkv, 0, { qd, qd, qd, qd }));
        else
            RAD_ARCH_TRY(B.qkv.declare(b, nm, g, nm.f("%s.attn_qkv", dn), 3 * qd, D, RAD_SHARD_ROW,
                                       grp_layer(grp),
                                       { nm.ckpt("%s.attn.to_q.weight", pre),
                                         nm.ckpt("%s.attn.to_k.weight", pre),
                                         nm.ckpt("%s.attn.to_v.weight", pre) },
                                       m.a_hm, m.b_qkv, 0, { qd, qd, qd }));
        B.op_qkr = rw(b, RAD_OP(b, "qk_norm_rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                           RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", g.n_head),
                                           RAD_F64("theta", theta), RAD_F64("eps", qk_eps),
                                           RAD_STR("mode", "axial"), RAD_INT("rotary_dim", 6 * rf),
                                           RAD_STR("sections", sections)),
                                RAD_WEIGHTS(B.w_qn, B.w_kn)),
                      { m.b_qkv }, { m.b_qkv });
        /* The keys to int8 codes and scales per (row, head) for the attention's scores. */
        B.op_kq = rw(b, RAD_OP(b, "quant_act_i8g",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", qd),
                                          RAD_INT("group", hd), RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.b_qkv }, { m.b_k8, m.b_ks });
        /* The values to E4M3 with a scale per column (one dimension of one head), for P.V on
         * the fp8 matrix instruction. */
        B.op_vq = rw(b, RAD_OP(b, "quant_cols_fp8",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", qd),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.b_qkv }, { m.b_v8, m.b_vs });
        if (m.vsa) {
            /* Tiles of the packed sequence: each query tile's best video tiles by pooled score
             * and the softmax over every tile's pooled keys, then attention over the kept tiles
             * with that pooled term gated in. */
            B.op_sel = rw(b, RAD_OP(b, "vsa_select",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                               RAD_INT("tile_t", kVsaT), RAD_INT("tile_h", kVsaH),
                                               RAD_INT("tile_w", kVsaW), RAD_INT("patch", kPatch),
                                               RAD_F64("sparsity", m.vsa_sparsity),
                                               RAD_F64("scale", 1.0 / std::sqrt((double)hd))),
                                    RAD_NOWEIGHTS),
                          { m.b_qkv }, { m.b_sel, m.b_coarse });
            /* The kept tiles' scores in int8, as the dense form below takes them. */
            B.op_attn = rw(b, RAD_OP(b, "attn_vsa",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                                RAD_INT("tile_t", kVsaT), RAD_INT("tile_h", kVsaH),
                                                RAD_INT("tile_w", kVsaW), RAD_INT("patch", kPatch),
                                                RAD_F64("sparsity", m.vsa_sparsity),
                                                RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                                RAD_STR("qk", "i8"), RAD_STR("pv", "fp8")),
                                     RAD_NOWEIGHTS),
                           { m.b_qkv, m.b_sel, m.b_coarse, m.b_k8, m.b_ks, m.b_v8, m.b_vs },
                           { m.a_attn.x });
            if (!B.op_sel) return RAD_E_INVAL;
        } else {
            /* Dense attention with int8 scores: the keys quantised per (row, head) by
             * quant_act_i8g, the query inside the op, the score an exact integer product times
             * both scales, P.V in bf16. The prefix-free form of attn_prefix_self, the op that
             * carries the int8 form. */
            B.op_attn = rw(b, RAD_OP(b, "attn_prefix_self",
                                     RAD_PARAMS(RAD_INT("q_len", T), RAD_INT("head_dim", hd),
                                                RAD_INT("gqa", 1), RAD_INT("block_size", 16),
                                                RAD_INT("causal", 0), RAD_STR("q_dtype", "bf16"),
                                                RAD_STR("kv_dtype", "bf16"),
                                                RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                                RAD_INT("max_ctx", 0), RAD_INT("n_head", g.n_head),
                                                RAD_INT("max_seqs", 1), RAD_STR("qk", "i8"),
                                                RAD_STR("pv", "fp8")),
                                     RAD_NOWEIGHTS),
                           { m.b_qkv, m.b_k8, m.b_ks, m.b_v8, m.b_vs }, { m.a_attn.x });
        }
        if (!B.op_kq || !B.op_vq) return RAD_E_INVAL;
        RAD_ARCH_TRY(B.q_attn.declare(b, g, m.a_attn, qd));
        RAD_ARCH_TRY(B.o.declare(b, nm, g, nm.f("%s.attn_output", dn), D, qd, RAD_SHARD_COL,
                                 grp_layer(grp), { nm.ckpt("%s.attn.to_out.0.weight", pre) },
                                 m.a_attn, m.b_ao));
        B.op_ar1 = all_reduce(b, g, m.b_ao, std::min<int64_t>(T, kArRows), D, "bf16", true);
        B.op_mq2 = mod_q(B.w_n2, true);
        const char* fw = nm.f("%s.ffn_gate_up.weight", dn);
        const char* fsrc = nm.ckpt("%s.ff.net.0.proj.weight", pre);
        RAD_ARCH_TRY(map_rows(b, fw, { { fsrc, (int)ff_all, ff_all }, { fsrc, 0, ff_all } }));
        B.gate_up.mapped = true;
        RAD_ARCH_TRY(B.gate_up.declare(b, nm, g, nm.f("%s.ffn_gate_up", dn), 2 * g.n_ff, D,
                                       RAD_SHARD_ROW, grp_layer(grp), { fsrc }, m.a_hm, m.b_gu, 0,
                                       { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(B.gq.declare(b, g, m.b_gu, m.a_ffn, g.n_ff, "silu", 0, false));
        RAD_ARCH_TRY(B.down.declare(b, nm, g, nm.f("%s.ffn_down", dn), D, g.n_ff, RAD_SHARD_COL,
                                    grp_layer(grp), { nm.ckpt("%s.ff.net.2.weight", pre) },
                                    m.a_ffn, m.b_ao));
        B.op_ar2 = all_reduce(b, g, m.b_ao, std::min<int64_t>(T, kArRows), D, "bf16", true);
        if (!B.op_mq1 || !B.op_mq2 || !B.op_qkr || !B.op_attn) return RAD_E_INVAL;
    }

    /* ---- the output: norm_out's modulated norm (the last block's branch gated in), the heads in
     * f32 over parts of the rows, and the two Euler steps into the lane */
    m.b_hn = decl_b(b, nm.f("dit.hn"), RAD_BF16, { T, D });
    m.w_no_n = wcopy(b, nm.f("dit.nout.norm.weight"), nm.ckpt("transformer.norm_out.norm.weight"),
                     RAD_BF16, { D }, RAD_ACCESS_PER_TOKEN, gx);
    if (!m.w_no_n) return RAD_E_INVAL;
    {
        rad_op op = RAD_OP(b, "rms_mod",
                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                      RAD_F64("eps", rad_meta_getf(meta, "transformer.final_norm_eps",
                                                                   1e-5)),
                                      RAD_INT("gated", 1), RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(m.w_no_n));
        if (!op) return RAD_E_INVAL;
        rad_buf rd[4] = { m.b_ao, m.b_modg, m.b_nout, m.b_h };
        rad_buf wr[2] = { m.b_hn, m.b_h };
        rad_op_reads(b, op, rd, 4);
        rad_op_writes(b, op, wr, 2);
        m.op_nout = op;
    }
    m.b_hf = decl_b(b, nm.f("dit.hf"), RAD_F32, { (int64_t)kHeadRows, D });
    m.op_hcast = rw(b, RAD_OP(b, "cast",
                              RAD_PARAMS(RAD_RANGE("M", 1, (int64_t)kHeadRows), RAD_INT("n", D),
                                         RAD_STR("from", "bf16"), RAD_STR("to", "f32")),
                              RAD_NOWEIGHTS),
                    { m.b_hn }, { m.b_hf });
    m.b_vel = decl_b(b, nm.f("dit.vel"), RAD_F32, { m.maxV, kPatchVals });
    m.b_avel = decl_b(b, nm.f("dit.avel"), RAD_F32, { m.maxA, kAudioC });
    m.op_vout = dense_bias(b, nm, "dit.video_out", "transformer.proj_out", RAD_F32, "f32",
                           kPatchVals, D, kHeadRows, m.b_hf, m.b_vel, &m.w_vout, &m.b_vout,
                           RAD_ACCESS_PER_TOKEN, gx);
    m.op_aout = dense_bias(b, nm, "dit.audio_out", "transformer.audio_proj_out", RAD_F32, "f32",
                           kAudioC, D, kHeadRows, m.b_hf, m.b_avel, &m.w_aout, &m.b_aout,
                           RAD_ACCESS_PER_TOKEN, gx);
    /* The cast and the two heads are issued in a loop over parts of the rows, the cast again after
     * a head has run, so the normed rows outlive both heads: declared as read by them, their
     * storage is never the velocities'. */
    if (m.op_vout && m.op_aout) {
        RAD_ARCH_TRY(rad_op_reads(b, m.op_vout, &m.b_hn, 1));
        RAD_ARCH_TRY(rad_op_reads(b, m.op_aout, &m.b_hn, 1));
    }
    m.op_ev = rw(b, RAD_OP(b, "flow_x0_step",
                           RAD_PARAMS(RAD_RANGE("M", 1, m.maxV), RAD_INT("n", kPatchVals),
                                      RAD_STR("dtype", "f32")),
                           RAD_NOWEIGHTS),
                 { m.b_vel, m.b_vstate }, { m.b_vstate });
    m.op_ea = rw(b, RAD_OP(b, "flow_x0_step",
                           RAD_PARAMS(RAD_RANGE("M", 1, m.maxA), RAD_INT("n", kAudioC),
                                      RAD_STR("dtype", "f32")),
                           RAD_NOWEIGHTS),
                 { m.b_avel, m.b_astate }, { m.b_astate });
    return m.op_hcast && m.op_vout && m.op_aout && m.op_ev && m.op_ea ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== the audio decoder */
/* A weight-normalised convolution's folded weight and its bias (none for conv_post). */
static int aconv(RadBuilder* b, Names& nm, AConv& c, const char* dn, const char* src, int64_t cin,
                 int64_t cout, int64_t k, int64_t dil, bool wnorm, bool bias, int64_t rows, rad_buf x,
                 rad_buf cu, rad_buf res, rad_buf y, const char* act) {
    const char* wn = nm.f("%s.weight", dn);
    if (wnorm) RAD_ARCH_TRY(map_wnorm(b, wn, nm.ckpt("%s.weight_g", src), nm.ckpt("%s.weight_v", src)));
    else RAD_ARCH_TRY(map_copy(b, wn, nm.ckpt("%s.weight", src)));
    c.w = decl_w(b, wn, RAD_F32, { cout, cin, k }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                 grp_layer(kGrpAudio));
    if (bias)
        c.b = wcopy(b, nm.f("%s.bias", dn), nm.ckpt("%s.bias", src), RAD_F32, { cout },
                    RAD_ACCESS_PER_TOKEN, kGrpAudio);
    if (!c.w || (bias && !c.b)) return RAD_E_INVAL;
    rad_op op = bias ? RAD_OP(b, "conv1d",
                              RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("cin", cin),
                                         RAD_INT("cout", cout), RAD_INT("k", k), RAD_INT("dil", dil),
                                         RAD_STR("act", act), RAD_STR("dtype", "f32")),
                              RAD_WEIGHTS(c.w, c.b))
                     : RAD_OP(b, "conv1d",
                              RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("cin", cin),
                                         RAD_INT("cout", cout), RAD_INT("k", k), RAD_INT("dil", dil),
                                         RAD_STR("act", act), RAD_STR("dtype", "f32")),
                              RAD_WEIGHTS(c.w));
    c.op = res ? rw(b, op, { x, cu, res }, { y }) : rw(b, op, { x, cu }, { y });
    return c.op ? RAD_OK : RAD_E_INVAL;
}

/* An anti-aliased SnakeBeta: its log-space alpha and beta, and the two Kaiser filters. */
static int aact(RadBuilder* b, Names& nm, AAct& a, const char* dn, const char* src, int64_t ch,
                int64_t rows, rad_buf x, rad_buf cu, rad_buf y) {
    a.alpha = wcopy(b, nm.f("%s.alpha", dn), nm.ckpt("%s.act.alpha", src), RAD_F32, { ch },
                    RAD_ACCESS_PER_TOKEN, kGrpAudio);
    a.beta = wcopy(b, nm.f("%s.beta", dn), nm.ckpt("%s.act.beta", src), RAD_F32, { ch },
                   RAD_ACCESS_PER_TOKEN, kGrpAudio);
    a.up = wcopy(b, nm.f("%s.up", dn), nm.ckpt("%s.upsample.filter", src), RAD_F32, { 1, 1, 12 },
                 RAD_ACCESS_PER_TOKEN, kGrpAudio);
    a.down = wcopy(b, nm.f("%s.down", dn), nm.ckpt("%s.downsample.lowpass.filter", src), RAD_F32,
                   { 1, 1, 12 }, RAD_ACCESS_PER_TOKEN, kGrpAudio);
    if (!a.alpha || !a.beta || !a.up || !a.down) return RAD_E_INVAL;
    a.op = rw(b, RAD_OP(b, "snake_aa",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", ch), RAD_INT("ratio", 2),
                                   RAD_INT("k", 12), RAD_INT("log", 1), RAD_STR("dtype", "f32")),
                        RAD_WEIGHTS(a.alpha, a.beta, a.up, a.down)),
              { x, cu }, { y });
    return a.op ? RAD_OK : RAD_E_INVAL;
}

static rad_op cu_op(RadBuilder* b, int64_t mult, rad_buf y) {
    return rw(b, RAD_OP(b, "cu_repeat",
                        RAD_PARAMS(RAD_RANGE("M", 1, 3), RAD_INT("mult", mult)),
                        RAD_NOWEIGHTS),
              {}, { y });
}

/* dec_in_proj and BigVGAN over the soundtrack's two channels, each a sequence (`cu`); every stage
 * an upsampling transposed convolution and the mean of three AMP blocks (kernel 3, 7 and 11), each
 * three dilations of act, dilated conv, act, conv and the residual. */
static int declare_audio(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Names& nm = m.nm;
    const std::vector<double> rates = meta_list(meta, "audio_vae.decoder_rates");
    const std::vector<double> ks = meta_list(meta, "audio_vae.decoder_kernel_sizes");
    const std::vector<double> rk = meta_list(meta, "audio_vae.resblock_kernel_sizes");
    const int64_t dd = rad_meta_geti(meta, "audio_vae.decoder_dim", 1024);
    const int64_t ld = rad_meta_geti(meta, "audio_vae.latent_dim", 2048);
    const std::vector<double> mean = meta_list(meta, "audio_vae.latents_mean");
    const std::vector<double> sd = meta_list(meta, "audio_vae.latents_std");
    int64_t hop = 1;
    for (double r : rates) hop *= (int64_t)r;
    if (rates.size() != 7 || ks.size() != 7 || rk.size() != 3 || hop != kRate / kAudioRate ||
        (int64_t)mean.size() != kAudioC || (int64_t)sd.size() != kAudioC) {
        fprintf(stderr, "radiance: minimaxh3: the audio VAE is not the released BigVGAN (7 stages "
                        "to 800 samples a latent, three AMP kernels)\n");
        return RAD_E_UNSUPPORTED;
    }
    static const int64_t kDil[3] = { 1, 3, 5 };
    m.a_mean = join_list(mean);
    m.a_std = join_list(sd);
    m.a_rows[0] = m.maxA;
    for (int s = 0; s < 7; ++s) m.a_rows[s + 1] = m.a_rows[s] * (int64_t)rates[(size_t)s];

    m.b_aden = decl_b(b, nm.f("aud.den"), RAD_F32, { m.maxA, kAudioC });
    m.op_adenorm = rw(b, RAD_OP(b, "chan_affine",
                                RAD_PARAMS(RAD_RANGE("M", 1, m.maxA), RAD_INT("n", kAudioC),
                                           RAD_STR("mul", m.a_std.c_str()),
                                           RAD_STR("add", m.a_mean.c_str()), RAD_STR("mode", "fma"),
                                           RAD_INT("round", 0), RAD_STR("dtype", "f32")),
                                RAD_NOWEIGHTS),
                      { m.b_astate }, { m.b_aden });
    for (int s = 0; s < 8; ++s) m.b_acu[s] = decl_b(b, nm.f("aud.cu%d", s), RAD_I32, { 3 });
    m.op_acu0 = cu_op(b, 1, m.b_acu[0]);
    m.b_ain0 = decl_b(b, nm.f("aud.in"), RAD_F32, { m.maxA, ld });
    RAD_ARCH_TRY(aconv(b, nm, m.a_in, "aud.in_proj", "audio_vae.dec_in_proj", kAudioC, ld, 1, 1,
                       false, true, m.maxA, m.b_aden, m.b_acu[0], 0, m.b_ain0, "none"));
    m.b_aavg[0] = decl_b(b, nm.f("aud.h0"), RAD_F32, { m.maxA, dd });
    RAD_ARCH_TRY(aconv(b, nm, m.a_pre, "aud.conv_pre", "audio_vae.decoder.conv_pre", ld, dd, 7, 1,
                       true, true, m.maxA, m.b_ain0, m.b_acu[0], 0, m.b_aavg[0], "none"));
    int64_t mult = 1;
    for (int s = 0; s < 7; ++s) {
        AStage& St = m.ast[s];
        const int64_t ci = dd >> s, co = dd >> (s + 1), R = m.a_rows[s + 1];
        St.ch = co;
        St.rate = (int64_t)rates[(size_t)s];
        St.k = (int64_t)ks[(size_t)s];
        mult *= St.rate;
        St.op_cu = cu_op(b, mult, m.b_acu[s + 1]);
        m.b_ah[s + 1] = decl_b(b, nm.f("aud.s%d.in", s), RAD_F32, { R, co });
        const char* wn = nm.f("aud.s%d.up.weight", s);
        const char* src = nm.ckpt("audio_vae.decoder.ups.%d.0", s);
        RAD_ARCH_TRY(map_wnorm(b, wn, nm.ckpt("%s.weight_g", src), nm.ckpt("%s.weight_v", src)));
        St.up.w = decl_w(b, wn, RAD_F32, { ci, co, St.k }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                         grp_layer(kGrpAudio));
        St.up.b = wcopy(b, nm.f("aud.s%d.up.bias", s), nm.ckpt("%s.bias", src), RAD_F32, { co },
                        RAD_ACCESS_PER_TOKEN, kGrpAudio);
        if (!St.up.w || !St.up.b) return RAD_E_INVAL;
        St.up.op = rw(b, RAD_OP(b, "conv_t1d",
                                RAD_PARAMS(RAD_RANGE("M", 1, m.a_rows[s]), RAD_INT("cin", ci),
                                           RAD_INT("cout", co), RAD_INT("stride", St.rate),
                                           RAD_INT("k", St.k), RAD_INT("pad", (St.k - St.rate) / 2),
                                           RAD_STR("dtype", "f32")),
                                RAD_WEIGHTS(St.up.w, St.up.b)),
                      { m.b_aavg[s], m.b_acu[s] }, { m.b_ah[s + 1] });
        if (!St.up.op || !St.op_cu) return RAD_E_INVAL;
        m.b_at1[s + 1] = decl_b(b, nm.f("aud.s%d.t1", s), RAD_F32, { R, co });
        m.b_at2[s + 1] = decl_b(b, nm.f("aud.s%d.t2", s), RAD_F32, { R, co });
        for (int j = 0; j < 3; ++j) m.b_aamp[j][s + 1] = decl_b(b, nm.f("aud.s%d.amp%d", s, j), RAD_F32,
                                                               { R, co });
        for (int j = 0; j < 3; ++j) {
            AmpBlock& A = St.amp[j];
            const int64_t k = (int64_t)rk[(size_t)j];
            const int ri = s * 3 + j;
            for (int d = 0; d < 3; ++d) {
                AmpUnit& U = A.u[d];
                const char* bn = nm.f("aud.s%d.amp%d.d%d", s, j, d);
                const char* bs = nm.ckpt("audio_vae.decoder.resblocks.%d", ri);
                /* The unit reads the stage's input on its first dilation, its own running sum
                 * after. */
                const rad_buf x = d == 0 ? m.b_ah[s + 1] : m.b_aamp[j][s + 1];
                RAD_ARCH_TRY(aact(b, nm, U.a1, nm.f("%s.act1", bn),
                                  nm.ckpt("%s.activations.%d", bs, 2 * d), co, R, x, m.b_acu[s + 1],
                                  m.b_at1[s + 1]));
                RAD_ARCH_TRY(aconv(b, nm, U.c1, nm.f("%s.conv1", bn),
                                   nm.ckpt("%s.convs1.%d", bs, d), co, co, k, kDil[d], true, true, R,
                                   m.b_at1[s + 1], m.b_acu[s + 1], 0, m.b_at2[s + 1], "none"));
                RAD_ARCH_TRY(aact(b, nm, U.a2, nm.f("%s.act2", bn),
                                  nm.ckpt("%s.activations.%d", bs, 2 * d + 1), co, R, m.b_at2[s + 1],
                                  m.b_acu[s + 1], m.b_at1[s + 1]));
                RAD_ARCH_TRY(aconv(b, nm, U.c2, nm.f("%s.conv2", bn),
                                   nm.ckpt("%s.convs2.%d", bs, d), co, co, k, 1, true, true, R,
                                   m.b_at1[s + 1], m.b_acu[s + 1], x, m.b_aamp[j][s + 1], "none"));
            }
        }
        m.b_aavg[s + 1] = decl_b(b, nm.f("aud.s%d.out", s), RAD_F32, { R, co });
        St.op_sum = add_op(b, R, co, "f32", m.b_aamp[0][s + 1], m.b_aamp[1][s + 1], m.b_aavg[s + 1]);
        St.op_avg = add_div_op(b, R, co, "f32", m.b_aavg[s + 1], m.b_aamp[2][s + 1], m.b_aavg[s + 1],
                               3.0);
        if (!St.op_sum || !St.op_avg) return RAD_E_INVAL;
    }
    const int64_t R7 = m.a_rows[7], c7 = dd >> 7;
    RAD_ARCH_TRY(aact(b, nm, m.a_actpost, "aud.act_post", nm.ckpt("audio_vae.decoder.activation_post"),
                      c7, R7, m.b_aavg[7], m.b_acu[7], m.b_at1[7]));
    m.b_aout0 = decl_b(b, nm.f("aud.mono"), RAD_F32, { R7, 1 });
    RAD_ARCH_TRY(aconv(b, nm, m.a_post, "aud.conv_post", "audio_vae.decoder.conv_post", c7, 1, 7, 1,
                       true, false, R7, m.b_at1[7], m.b_acu[7], 0, m.b_aout0, "clamp"));
    m.b_audio = decl_b(b, nm.f("audio_out"), RAD_F32, { R7 / kAudioCh + 1, kAudioCh },
                       RAD_BUF_PERSIST);
    m.op_vmix = rw(b, RAD_OP(b, "interleave2",
                             RAD_PARAMS(RAD_RANGE("M", 1, R7 / kAudioCh), RAD_STR("dtype", "f32")),
                             RAD_NOWEIGHTS),
                   { m.b_aout0 }, { m.b_audio });
    return m.op_adenorm && m.op_acu0 && m.op_vmix ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== the video decoder */
/* A decode pass is one chunk: latent frames [5 c, 5 c + 7) unpatchified out of the lane and
 * denormalised, post_quant_conv and proj_in (f32) over every latent, then this rank's share of the
 * tiles (7 x 16 x 16 latents, its registers and a zero token) through the ViT, its pixels gathered
 * across ranks, and the tiles stitched into the chunk's frames. */
static int declare_video(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Names& nm = m.nm;
    const int64_t dim = rad_meta_geti(meta, "vae.decoder_num_attention_heads", 32) *
                        rad_meta_geti(meta, "vae.decoder_attention_head_dim", 64);
    const int64_t hd = rad_meta_geti(meta, "vae.decoder_attention_head_dim", 64);
    const int64_t nl = rad_meta_geti(meta, "vae.decoder_num_layers", 36);
    const int64_t ff = dim * rad_meta_geti(meta, "vae.decoder_ffn_mult", 4);
    const double eps = rad_meta_getf(meta, "vae.decoder_norm_eps", 1e-5);
    const double theta = rad_meta_getf(meta, "vae.decoder_rope_theta", 100.0);
    const double ratio = rad_meta_getf(meta, "vae.decoder_rope_dim_ratio", 0.75);
    const std::vector<double> mean = meta_list(meta, "vae.latents_mean");
    const std::vector<double> sd = meta_list(meta, "vae.latents_std");
    const int64_t rot = (int64_t)(hd * ratio);
    const int64_t px = 3 * 4 * kScale * kScale;    /* a token's pixels: 3 colours x 4 frames x 16^2 */
    if ((int64_t)mean.size() != kC || (int64_t)sd.size() != kC || rot % 6 ||
        rad_meta_geti(meta, "vae.decoder_num_register_tokens", 0) != kRegs) {
        fprintf(stderr, "radiance: minimaxh3: the video VAE decoder is not the released ViT\n");
        return RAD_E_UNSUPPORTED;
    }
    m.v_mean = join_list(mean);
    m.v_std = join_list(sd);
    const int64_t maxhw = (int64_t)kMaxArea / (kScale * kScale);
    const int64_t vox = (int64_t)kDecLat * maxhw;
    const int64_t R = m.tpr * kPeriod;               /* this rank's tile rows */
    const int64_t RA = (int64_t)m.W * R;             /* every rank's, gathered */
    const int gv = kGrpVae;
    const int acc = RAD_ACCESS_PER_TOKEN;

    m.b_vox = decl_b(b, nm.f("vae.vox"), RAD_F32, { vox, kC });
    m.op_unfold = rw(b, RAD_OP(b, "patch_unfold",
                               RAD_PARAMS(RAD_RANGE("M", 1, vox), RAD_INT("C", kC), RAD_INT("pt", 1),
                                          RAD_INT("ph", kPatch), RAD_INT("pw", kPatch),
                                          RAD_STR("mul", m.v_std.c_str()),
                                          RAD_STR("add", m.v_mean.c_str()), RAD_STR("dtype", "f32")),
                               RAD_NOWEIGHTS),
                     { m.b_vstate }, { m.b_vox });
    m.b_vox2 = decl_b(b, nm.f("vae.vox2"), RAD_F32, { vox, kC });
    {
        RAD_ARCH_TRY(map_copy(b, nm.f("vae.pqc.weight"), nm.ckpt("vae.post_quant_conv.weight")));
        m.w_pqc = decl_w(b, nm.f("vae.pqc.weight"), RAD_F32, { kC, kC, 1, 1, 1 }, acc,
                         RAD_SHARD_NONE, grp_layer(gv));
        m.b_pqc = wcopy(b, nm.f("vae.pqc.bias"), nm.ckpt("vae.post_quant_conv.bias"), RAD_F32,
                        { kC }, acc, gv);
        if (!m.w_pqc || !m.b_pqc) return RAD_E_INVAL;
        m.op_pqc = rw(b, RAD_OP(b, "gemm_nt_bias",
                                RAD_PARAMS(RAD_RANGE("M", 1, vox), RAD_INT("N", kC), RAD_INT("K", kC),
                                           RAD_STR("dtype", "f32")),
                                RAD_WEIGHTS(m.w_pqc, m.b_pqc)),
                      { m.b_vox }, { m.b_vox2 });
    }
    m.b_vpin = decl_b(b, nm.f("vae.pin"), RAD_F32, { vox, dim });
    m.op_vpin = dense_bias(b, nm, "vae.proj_in", "vae.decoder.proj_in", RAD_F32, "f32", dim, kC, vox,
                           m.b_vox2, m.b_vpin, &m.w_vpin2, &m.b_vpin2, acc, gv);
    m.b_vx = decl_b(b, nm.f("vae.x"), RAD_F32, { (int64_t)m.tiles * kPeriod, dim });
    m.w_reg = wcopy(b, nm.f("vae.registers"), nm.ckpt("vae.decoder.register_tokens"), RAD_F32,
                    { 1, kRegs, dim }, acc, gv);
    if (!m.w_reg || !m.op_vpin || !m.op_pqc) return RAD_E_INVAL;
    m.op_tiles = rw(b, RAD_OP(b, "vtile_gather",
                              RAD_PARAMS(RAD_RANGE("M", 1, (int64_t)m.tiles * kPeriod),
                                         RAD_INT("n", dim), RAD_INT("tile", kTile),
                                         RAD_INT("overlap", kOverlap), RAD_INT("frames", kDecLat),
                                         RAD_INT("period", kPeriod), RAD_INT("n_reg", kRegs),
                                         RAD_STR("dtype", "f32")),
                              RAD_WEIGHTS(m.w_reg)),
                    { m.b_vpin }, { m.b_vx });
    m.b_dpos = decl_b(b, nm.f("vae.pos"), RAD_F32, { 3, R });
    m.op_dpos = rw(b, RAD_OP(b, "grid_pos3",
                             RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("T", kDecLat),
                                        RAD_INT("H", kTile), RAD_INT("W", kTile),
                                        RAD_INT("period", kPeriod), RAD_F64("scale", 2.0 * M_PI)),
                             RAD_NOWEIGHTS),
                   {}, { m.b_dpos });
    m.b_dcu = decl_b(b, nm.f("vae.cu"), RAD_I32, { m.tpr + 1 });
    m.op_dcu = rw(b, RAD_OP(b, "iota",
                            RAD_PARAMS(RAD_RANGE("M", 1, m.tpr + 1), RAD_INT("start", 0),
                                       RAD_INT("step", kPeriod)),
                            RAD_NOWEIGHTS),
                  {}, { m.b_dcu });
    if (!m.op_tiles || !m.op_dpos || !m.op_dcu) return RAD_E_INVAL;

    m.b_vn = decl_b(b, nm.f("vae.hn"), RAD_BF16, { R, dim });
    m.b_vqkv = decl_b(b, nm.f("vae.qkv"), RAD_BF16, { R, 3 * dim });
    m.b_vattn = decl_b(b, nm.f("vae.attn"), RAD_BF16, { R, dim });
    m.b_vo = decl_b(b, nm.f("vae.o"), RAD_BF16, { R, dim });
    m.b_vt = decl_b(b, nm.f("vae.t"), RAD_F32, { R, dim });
    m.b_vup = decl_b(b, nm.f("vae.up"), RAD_BF16, { R, 2 * ff });
    m.b_vact = decl_b(b, nm.f("vae.act"), RAD_BF16, { R, ff });
    const char* sections = nm.f("%lld %lld %lld", (long long)(rot / 6), (long long)(rot / 6),
                                (long long)(rot / 6));
    m.vit.resize((size_t)nl);
    for (int64_t l = 0; l < nl; ++l) {
        VitBlock& V = m.vit[(size_t)l];
        const int li = (int)l;
        const char* pre = nm.ckpt("vae.decoder.transformer_blocks.%d", li);
        const char* dn = nm.f("vae.blk.%d", li);
        V.w_n1 = wcopy(b, nm.f("%s.norm1.weight", dn), nm.ckpt("%s.norm1.weight", pre), RAD_F32,
                       { dim }, acc, gv);
        V.w_n2 = wcopy(b, nm.f("%s.norm2.weight", dn), nm.ckpt("%s.norm2.weight", pre), RAD_F32,
                       { dim }, acc, gv);
        V.s1 = wcopy(b, nm.f("%s.scale1", dn), nm.ckpt("%s.scale1", pre), RAD_F32, { dim }, acc, gv);
        V.s2 = wcopy(b, nm.f("%s.scale2", dn), nm.ckpt("%s.scale2", pre), RAD_F32, { dim }, acc, gv);
        const char* qn = nm.f("%s.attn_qkv.weight", dn);
        const char* qb = nm.f("%s.attn_qkv.bias", dn);
        RAD_ARCH_TRY(map_concat(b, qn, { nm.ckpt("%s.attn.to_q.weight", pre),
                                         nm.ckpt("%s.attn.to_k.weight", pre),
                                         nm.ckpt("%s.attn.to_v.weight", pre) }));
        RAD_ARCH_TRY(map_concat(b, qb, { nm.ckpt("%s.attn.to_q.bias", pre),
                                         nm.ckpt("%s.attn.to_k.bias", pre),
                                         nm.ckpt("%s.attn.to_v.bias", pre) }));
        V.w_qkv = decl_w(b, qn, RAD_BF16, { 3 * dim, dim }, acc, RAD_SHARD_NONE, grp_layer(gv));
        V.b_qkv = decl_w(b, qb, RAD_F32, { 3 * dim }, acc, RAD_SHARD_NONE, grp_layer(gv));
        V.w_o = wcopy(b, nm.f("%s.attn_out.weight", dn), nm.ckpt("%s.attn.to_out.0.weight", pre),
                      RAD_BF16, { dim, dim }, acc, gv);
        V.b_o = wcopy(b, nm.f("%s.attn_out.bias", dn), nm.ckpt("%s.attn.to_out.0.bias", pre),
                      RAD_F32, { dim }, acc, gv);
        const char* un = nm.f("%s.ffn_up.weight", dn);
        const char* ub = nm.f("%s.ffn_up.bias", dn);
        const char* usrc = nm.ckpt("%s.ff.net.0.proj.weight", pre);
        const char* ubsrc = nm.ckpt("%s.ff.net.0.proj.bias", pre);
        RAD_ARCH_TRY(map_rows(b, un, { { usrc, (int)ff, ff }, { usrc, 0, ff } }));
        RAD_ARCH_TRY(map_rows(b, ub, { { ubsrc, (int)ff, ff }, { ubsrc, 0, ff } }));
        V.w_up = decl_w(b, un, RAD_BF16, { 2 * ff, dim }, acc, RAD_SHARD_NONE, grp_layer(gv));
        V.b_up = decl_w(b, ub, RAD_F32, { 2 * ff }, acc, RAD_SHARD_NONE, grp_layer(gv));
        V.w_down = wcopy(b, nm.f("%s.ffn_down.weight", dn), nm.ckpt("%s.ff.net.2.weight", pre),
                         RAD_BF16, { dim, ff }, acc, gv);
        V.b_down = wcopy(b, nm.f("%s.ffn_down.bias", dn), nm.ckpt("%s.ff.net.2.bias", pre), RAD_F32,
                         { dim }, acc, gv);
        if (!V.w_n1 || !V.w_n2 || !V.s1 || !V.s2 || !V.w_qkv || !V.b_qkv || !V.w_o || !V.b_o ||
            !V.w_up || !V.b_up || !V.w_down || !V.b_down)
            return RAD_E_INVAL;
        auto gemm = [&](rad_weight w, rad_weight bias, int64_t N, int64_t K, rad_buf x, rad_buf y) {
            return rw(b, RAD_OP(b, "gemm_nt_bias",
                                RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("N", N), RAD_INT("K", K),
                                           RAD_STR("dtype", "bf16")),
                                RAD_WEIGHTS(w, bias)),
                      { x }, { y });
        };
        V.op_n1 = rmsnorm_op(b, R, dim, eps, V.w_n1, m.b_vx, m.b_vn);
        V.op_qkv = gemm(V.w_qkv, V.b_qkv, 3 * dim, dim, m.b_vn, m.b_vqkv);
        V.op_qkr = rw(b, RAD_OP(b, "qk_norm_rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("head_dim", hd),
                                           RAD_INT("n_head", dim / hd), RAD_INT("n_head_kv", dim / hd),
                                           RAD_F64("theta", theta), RAD_F64("eps", eps),
                                           RAD_STR("mode", "axial"), RAD_INT("rotary_dim", rot),
                                           RAD_STR("sections", sections)),
                                RAD_NOWEIGHTS),
                      { m.b_vqkv, m.b_dpos }, { m.b_vqkv });
        V.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("head_dim", hd),
                                            RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                            RAD_STR("dtype", "bf16"),
                                            RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                            RAD_INT("max_seqlen", kPeriod)),
                                 RAD_NOWEIGHTS),
                       { m.b_vqkv, m.b_dcu }, { m.b_vattn });
        V.op_o = gemm(V.w_o, V.b_o, dim, dim, m.b_vattn, m.b_vo);
        V.op_s1 = rw(b, RAD_OP(b, "mul_weight",
                               RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", dim),
                                          RAD_STR("dtype", "f32")),
                               RAD_WEIGHTS(V.s1)),
                     { m.b_vo }, { m.b_vt });
        V.op_r1 = add_op(b, R, dim, "f32", m.b_vx, m.b_vt, m.b_vx);
        V.op_n2 = rmsnorm_op(b, R, dim, eps, V.w_n2, m.b_vx, m.b_vn);
        V.op_up = gemm(V.w_up, V.b_up, 2 * ff, dim, m.b_vn, m.b_vup);
        V.op_act = rw(b, RAD_OP(b, "silu_mul",
                                RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", ff),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_vup }, { m.b_vact });
        V.op_down = gemm(V.w_down, V.b_down, dim, ff, m.b_vact, m.b_vo);
        V.op_s2 = rw(b, RAD_OP(b, "mul_weight",
                               RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", dim),
                                          RAD_STR("dtype", "f32")),
                               RAD_WEIGHTS(V.s2)),
                     { m.b_vo }, { m.b_vt });
        V.op_r2 = add_op(b, R, dim, "f32", m.b_vx, m.b_vt, m.b_vx);
        if (!V.op_n1 || !V.op_qkv || !V.op_qkr || !V.op_attn || !V.op_o || !V.op_s1 || !V.op_r1 ||
            !V.op_n2 || !V.op_up || !V.op_act || !V.op_down || !V.op_s2 || !V.op_r2)
            return RAD_E_INVAL;
    }
    m.w_vno = wcopy(b, nm.f("vae.norm_out.weight"), nm.ckpt("vae.decoder.norm_out.weight"), RAD_F32,
                    { dim }, acc, gv);
    m.b_vno = wcopy(b, nm.f("vae.norm_out.bias"), nm.ckpt("vae.decoder.norm_out.bias"), RAD_F32,
                    { dim }, acc, gv);
    if (!m.w_vno || !m.b_vno) return RAD_E_INVAL;
    m.op_vnorm = rw(b, RAD_OP(b, "layernorm",
                              RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", dim), RAD_F64("eps", eps),
                                         RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(m.w_vno, m.b_vno)),
                    { m.b_vx }, { m.b_vn });
    m.b_vpix = decl_b(b, nm.f("vae.pix"), RAD_BF16, { R, px });
    m.op_vproj = dense_bias(b, nm, "vae.proj_out", "vae.decoder.proj_out", RAD_BF16, "bf16", px, dim,
                            R, m.b_vn, m.b_vpix, &m.w_vpo, &m.b_vpo, acc, gv);
    if (!m.op_vnorm || !m.op_vproj) return RAD_E_INVAL;
    if (m.W > 1) {
        m.b_vpix_all = decl_b(b, nm.f("vae.pix_all"), RAD_BF16, { RA, px });
        m.op_vgather = rw(b, RAD_OP(b, "all_gather",
                                    RAD_PARAMS(RAD_INT("world_size", m.W),
                                               RAD_RANGE("numel", 1, R * px), RAD_STR("dtype", "bf16")),
                                    RAD_NOWEIGHTS),
                          { m.b_vpix }, { m.b_vpix_all });
        if (!m.op_vgather) return RAD_E_INVAL;
    }
    const int64_t fmax = (int64_t)kFpc + kFtail;
    m.b_prev = decl_b(b, nm.f("lane.prev"), RAD_BF16, { (int64_t)kMaxLanes * kFtail * kMaxArea, 3 },
                      RAD_BUF_PERSIST);
    m.b_frames = decl_b(b, nm.f("frames_out"), RAD_U8, { fmax * kMaxArea, 3 }, RAD_BUF_PERSIST);
    m.op_stitch = rw(b, RAD_OP(b, "vtile_stitch",
                               RAD_PARAMS(RAD_RANGE("M", 1, fmax * kMaxArea), RAD_INT("tile", kTile),
                                          RAD_INT("overlap", kOverlap), RAD_INT("scale", kScale),
                                          RAD_INT("pt", 4), RAD_INT("skip", 3), RAD_INT("keep", kFpc),
                                          RAD_INT("blend", kFtail), RAD_INT("period", kPeriod),
                                          RAD_STR("mean", kPixMean), RAD_STR("std", kPixStd)),
                               RAD_NOWEIGHTS),
                     { m.W > 1 ? m.b_vpix_all : m.b_vpix, m.b_prev }, { m.b_prev, m.b_frames });
    return m.op_stitch ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== declare */
static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    Model& m = g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    Names& nm = m.nm;
    RAD_ARCH_TRY(check_format(meta));
    RAD_ARCH_TRY(geom_from(m.gt, meta, ctx, "bf16", RAD_BF16));
    if (m.gt.n_layer <= kTextLayers) {
        fprintf(stderr, "radiance: minimaxh3: the conditioning is the text encoder's state after "
                        "layer %d, and it has %lld layers\n", kTextLayers, (long long)m.gt.n_layer);
        return RAD_E_INVAL;
    }
    m.gt.n_layer = kTextLayers;
    m.gt.max_tok = kMaxText;
    m.W = m.gt.world;

    /* ---- the DiT's geometry, off the transformer's own config */
    Geom& gd = m.gd;
    gd = m.gt;
    gd.max_tok = ctx->max_tok;
    gd.n_layer    = rad_meta_geti(meta, "transformer.num_layers", 50);
    gd.head_dim   = rad_meta_geti(meta, "transformer.attention_head_dim", 128);
    gd.n_head_all = rad_meta_geti(meta, "transformer.num_attention_heads", 56);
    gd.n_head_kv_all = gd.n_head_all;
    gd.n_embd     = rad_meta_geti(meta, "transformer.hidden_size", 5376);
    gd.n_ff_all   = rad_meta_geti(meta, "transformer.ffn_dim", 14336);
    gd.eps        = (float)rad_meta_getf(meta, "transformer.norm_eps", 1e-5);
    gd.rope_mc    = false;
    RAD_ARCH_TRY(divide_or_fail(gd.n_head_all, gd.world, "transformer.num_attention_heads",
                                &gd.n_head));
    gd.n_head_kv = gd.n_head;
    gd.kv_off = 0;
    RAD_ARCH_TRY(divide_or_fail(gd.n_ff_all, gd.world, "the DiT's feed-forward width", &gd.n_ff));
    m.D = gd.n_embd;
    /* A rank holds 6 / world of a modality's six vectors, which keeps each [shift | scale] pair and
     * its gate on one rank. */
    if (6 % m.W || (6 * m.D / m.W) % RAD_FP8_BLOCK) {
        fprintf(stderr, "radiance: minimaxh3: %d ranks do not split the six modulation vectors "
                        "into whole fp8 blocks; use 1 or 2\n", (int)m.W);
        return RAD_E_UNSUPPORTED;
    }
    m.Ws = 6 * m.D / m.W;

    /* A checkpoint distilled for few steps with video sparse attention says so in its inference
     * settings (fastvideo_inference.json): the attention, its sparsity and tile, the timesteps it
     * was trained on (a 1000-step clock) and the shifts it was trained with. */
    const char* backend = rad_meta_gets(meta, "fastvideo_inference.attention_backend", "");
    if (!std::strcmp(backend, "VIDEO_SPARSE_ATTN_H3")) {
        m.vsa = true;
        m.vsa_sparsity = rad_meta_getf(meta, "fastvideo_inference.vsa_sparsity", -1.0);
        const long long tile = rad_meta_geti(meta, "fastvideo_inference.vsa_tile_size", 0);
        if (tile != kVsaRows || !(m.vsa_sparsity >= 0.0 && m.vsa_sparsity < 1.0)) {
            fprintf(stderr, "radiance: minimaxh3: video sparse attention at %lld-row tiles and "
                            "sparsity %g; this plugin runs %d-row (4 x 4 x 4) tiles at a sparsity "
                            "in [0, 1)\n", tile, m.vsa_sparsity, (int)kVsaRows);
            return RAD_E_UNSUPPORTED;
        }
        vsa_bounds(m.vsa_sparsity, &m.vsa_tiles, &m.vsa_k);
    } else if (*backend && std::strcmp(backend, "FLASH_ATTN")) {
        fprintf(stderr, "radiance: minimaxh3: the checkpoint was made for the attention '%s'; this "
                        "plugin runs dense attention and VIDEO_SPARSE_ATTN_H3\n", backend);
        return RAD_E_UNSUPPORTED;
    }
    for (double t : meta_list(meta, "fastvideo_inference.dmd_denoising_steps"))
        m.rungs.push_back(t / 1000.0);
    if (!m.rungs.empty()) {
        const long long fw = rad_meta_geti(meta, "fastvideo_inference.transformer_forwards",
                                           (long long)m.rungs.size());
        if (fw != (long long)m.rungs.size() || (int64_t)m.rungs.size() > kMaxSteps) {
            fprintf(stderr, "radiance: minimaxh3: a trained schedule of %zu timesteps for %lld "
                            "evaluations; at most %d, one a timestep\n", m.rungs.size(), fw,
                    (int)kMaxSteps);
            return RAD_E_UNSUPPORTED;
        }
        m.S = (int64_t)m.rungs.size();
    }
    for (const char* k : { "video", "audio" }) {
        const double want = rad_meta_getf(meta, m.nm.f("fastvideo_inference.%s_scheduler_shift", k),
                                          0.0);
        const double have = rad_meta_getf(meta, !std::strcmp(k, "video") ? "scheduler.shift"
                                                                          : "audio_scheduler.shift",
                                          !std::strcmp(k, "video") ? 12.0 : 3.0);
        if (want > 0.0 && want != have) {
            fprintf(stderr, "radiance: minimaxh3: the checkpoint was trained with a %s shift of %g "
                            "and its scheduler says %g\n", k, want, have);
            return RAD_E_INVAL;
        }
    }
    m.gr = gd;
    m.gr.max_tok = kMaxText;
    m.maxV = max_video_rows();
    m.maxA = kAudioCh * max_audio();
    m.tiles = max_tiles();
    m.tpr = (m.tiles + m.W - 1) / m.W;
    if (gd.n_layer >= kGrpAda - kGrpDit - 1 || gd.n_layer > kGrpVae - kGrpAda ||
        m.gt.n_layer >= kGrpRefiner) {
        fprintf(stderr, "radiance: minimaxh3: the networks are too deep for the weight groups\n");
        return RAD_E_UNSUPPORTED;
    }
    if (ctx->max_tok < max_step_rows() || ctx->max_tok < m.tpr * kPeriod) {
        fprintf(stderr, "radiance: minimaxh3: a pass carries %lld rows; the longest video's step "
                        "needs %lld and its decode %lld\n", (long long)ctx->max_tok,
                (long long)max_step_rows(), (long long)(m.tpr * kPeriod));
        return RAD_E_INVAL;
    }

    /* ---- in issue order: the conditioning pass, the step, the soundtrack, the frames */
    m.b_text = decl_b(b, nm.f("lane.text"), RAD_BF16, { (int64_t)kMaxLanes * kMaxText, m.D },
                      RAD_BUF_PERSIST);
    RAD_ARCH_TRY(declare_te(b, meta, m));
    RAD_ARCH_TRY(declare_refiner(b, meta, m));
    RAD_ARCH_TRY(declare_tables(b, meta, m));
    RAD_ARCH_TRY(declare_dit(b, meta, m));
    if (m.dit[0].qkv.bf16) {
        fprintf(stderr, "radiance: minimaxh3: the DiT's linears are bf16; this plugin serves the "
                        "block-fp8 container (data/recipes/minimax-h3-fp8.recipe)\n");
        return RAD_E_UNSUPPORTED;
    }
    RAD_ARCH_TRY(declare_audio(b, meta, m));
    RAD_ARCH_TRY(declare_video(b, meta, m));

    if (!ctx->shape_probe) {
        RadVideoGenDecl d{};
        /* A distilled release is served under its own id's last part, lower case. */
        std::string name = "minimax-h3";
        if (const char* id = rad_meta_gets(meta, "fastvideo_inference.model_id", nullptr)) {
            const char* last = std::strrchr(id, '/');
            name = last ? last + 1 : id;
            for (char& ch : name) ch = (char)std::tolower((unsigned char)ch);
        }
        d.name = name.c_str();
        d.tmpl = "{prompt}";
        d.max_text = kMaxText;
        d.fps_num = kFps;
        d.fps_den = 1;
        d.frames_per_chunk = kFpc;
        d.frames_tail = kFtail;
        d.latents_per_chunk = kLpc;
        d.latents_tail = kLtail;
        d.dec_latents = kDecLat;
        d.min_chunks = kMinChunks;
        d.max_chunks = kMaxChunks;
        d.latent_scale = kScale;
        d.latent_channels = kC;
        d.patch = kPatch;
        d.size_multiple = kMult;
        d.min_side = kMinSide;
        d.max_area = kMaxArea;
        d.max_aspect = kMaxAspect;
        d.default_width = kDefW;
        d.default_height = kDefH;
        d.default_chunks = kDefChunks;
        d.sample_rate = kRate;
        d.audio_rate = kAudioRate;
        d.audio_channels = kAudioCh;
        d.audio_latent_channels = kAudioC;
        d.default_steps = kDefSteps;
        d.max_steps = kMaxSteps + 1;
        if (!m.rungs.empty()) {
            d.n_rungs = (int32_t)m.rungs.size();
            d.rungs = m.rungs.data();
            d.default_steps = d.max_steps = d.n_rungs + 1;
        }
        d.shift_video = rad_meta_getf(meta, "scheduler.shift", 12.0);
        d.shift_audio = rad_meta_getf(meta, "audio_scheduler.shift", 3.0);
        d.max_lanes = kMaxLanes;
        d.frames_out = m.b_frames;
        d.audio_out = m.b_audio;
        RAD_ARCH_TRY(rad_declare_video_generator(b, &d));
    }

    rad_note(b, "rank %d/%d: MiniMax-H3 t2va -- text encoder %lld layers (%lld heads a rank), "
                "refiner 2 and DiT %lld blocks (%lld heads, ff %lld a rank), modulation %lld of "
                "6 x %lld a rank, video decoder %lld tiles at most (%lld a rank)",
             m.gt.rank, m.gt.world, (long long)m.gt.n_layer, (long long)m.gt.n_head,
             (long long)gd.n_layer, (long long)gd.n_head, (long long)gd.n_ff, (long long)m.Ws,
             (long long)m.D, (long long)m.tiles, (long long)m.tpr);
    return RAD_OK;
}

/* ================================================================== step */
/* The conditioning pass: the text encoder, context_embedder and the refiner into the lane's prompt
 * rows, then every step's modulation into the lane's tables. */
static void step_cond(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadVideoGenBatch* V = batch->vgen_b;
    const Geom& g = m.gt;
    const int64_t T = batch->n_tok, lane = V->lane, S = V->n_steps;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();
    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.b_tx, T));
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, 2);
    for (const TeLayer& L : m.te) {
        RAD_ISSUE(c, L.op_n1, brows(m.b_tx, T), RAD_W(L.w_n1), brows(m.a_th.x, T));
        L.q_h.step(c, T);
        L.qkv.step(c, m.a_th, m.b_tkv, T);
        RAD_ISSUE(c, L.op_qkr, brows(m.b_tkv, T), pos, RAD_W(L.w_qn), RAD_W(L.w_kn));
        RAD_ISSUE_N(c, L.op_attn, T, bcol(m.b_tkv, 0, q_dim, T), bcol(m.b_tkv, q_dim, kv_dim, T),
                    bcol(m.b_tkv, q_dim + kv_dim, kv_dim, T), cu, brows(m.a_tattn.x, T));
        L.q_attn.step(c, T);
        L.o.step(c, m.a_tattn, m.b_th2, T);
        if (L.op_ar1) RAD_ISSUE_N(c, L.op_ar1, T * g.n_embd, brows(m.b_th2, T), RAD_NONE);
        RAD_ISSUE(c, L.op_add1, brows(m.b_tx, T), brows(m.b_th2, T), brows(m.b_tx, T));
        RAD_ISSUE(c, L.op_n2, brows(m.b_tx, T), RAD_W(L.w_n2), brows(m.a_th.x, T));
        L.q_h2.step(c, T);
        L.gate_up.step(c, m.a_th, m.b_tgu, T);
        L.gq.step(c, m.b_tgu, T);
        L.down.step(c, m.a_tffn, m.b_th2, T);
        if (L.op_ar2) RAD_ISSUE_N(c, L.op_ar2, T * g.n_embd, brows(m.b_th2, T), RAD_NONE);
        RAD_ISSUE(c, L.op_add2, brows(m.b_tx, T), brows(m.b_th2, T), brows(m.b_tx, T));
    }

    const Geom& r = m.gr;
    const int64_t D = m.D, qd = r.q_dim();
    RAD_ISSUE_N(c, m.op_ce, T, brows(m.b_tx, T), RAD_W(m.w_ce), RAD_W(m.b_ce), RAD_NONE,
                brows(m.b_rh, T));
    for (const RefBlock& R : m.rb) {
        RAD_ISSUE_N(c, R.op_n1, T, brows(m.b_rh, T), RAD_W(R.w_n1), brows(m.a_rh.x, T));
        R.q_h.step(c, T);
        R.qkv.step(c, m.a_rh, m.b_rqkv, T);
        RAD_ISSUE_N(c, R.op_qkn, T, brows(m.b_rqkv, T), RAD_NONE, RAD_W(R.w_qn), RAD_W(R.w_kn));
        RAD_ISSUE_N(c, R.op_attn, T, bcol(m.b_rqkv, 0, qd, T), bcol(m.b_rqkv, qd, qd, T),
                    bcol(m.b_rqkv, 2 * qd, qd, T), cu, brows(m.a_rattn.x, T));
        R.q_attn.step(c, T);
        R.o.step(c, m.a_rattn, m.b_rao, T);
        if (R.op_ar1) RAD_ISSUE_N(c, R.op_ar1, T * D, brows(m.b_rao, T), RAD_NONE);
        RAD_ISSUE_N(c, R.op_add1, T, brows(m.b_rh, T), brows(m.b_rao, T), brows(m.b_rh, T));
        RAD_ISSUE_N(c, R.op_n2, T, brows(m.b_rh, T), RAD_W(R.w_n2), brows(m.a_rh.x, T));
        R.q_h2.step(c, T);
        R.gate_up.step(c, m.a_rh, m.b_rgu, T);
        R.gq.step(c, m.b_rgu, T);
        R.down.step(c, m.a_rffn, m.b_rao, T);
        if (R.op_ar2) RAD_ISSUE_N(c, R.op_ar2, T * D, brows(m.b_rao, T), RAD_NONE);
        RAD_ISSUE_N(c, R.op_add2, T, brows(m.b_rh, T), brows(m.b_rao, T), brows(m.b_rh, T));
    }
    RAD_ISSUE_N(c, m.op_fn, T, brows(m.b_rh, T), RAD_W(m.w_fn),
                brow_slice(m.b_text, lane * kMaxText, T, D));

    /* ---- every step's timestep embedding, video rows then soundtrack rows */
    const int64_t fd = 256, td = 2688, S2 = 2 * S;
    RAD_ISSUE_N(c, m.op_temb, S, praw(V->t_video, RAD_F32, S), brow_slice(m.b_tproj, 0, S, fd));
    RAD_ISSUE_N(c, m.op_temb, S, praw(V->t_audio, RAD_F32, S), brow_slice(m.b_tproj, S, S, fd));
    RAD_ISSUE_N(c, m.op_t1, S2, brows(m.b_tproj, S2), RAD_W(m.w_t1), RAD_W(m.b_t1), RAD_NONE,
                brows(m.b_te1, S2));
    RAD_ISSUE_N(c, m.op_ts, S2, brows(m.b_te1, S2), brows(m.b_te1s, S2));
    RAD_ISSUE_N(c, m.op_t2, S2, brows(m.b_te1s, S2), RAD_W(m.w_t2), RAD_W(m.b_t2), RAD_NONE,
                brows(m.b_temb, S2));
    RAD_ISSUE_N(c, m.op_ts2, S2, brows(m.b_temb, S2), brows(m.a_temb.x, S2));
    m.q_temb.step(c, S2);

    /* ---- the modulation table: block l, segment e from its modality's projection of its
     * timestep's rows, at the table's column block (l, e) */
    const int64_t L = m.gd.n_layer, row = L * kSegs * m.Ws, base = lane * m.S;
    for (int64_t l = 0; l < L; ++l) {
        const DitBlock& B = m.dit[(size_t)l];
        for (int e = 0; e < kSegs; ++e) {
            const int md = kSegMod[e];
            const int64_t r0 = e == kSegAudio ? S : 0;
            RAD_ISSUE_N(c, B.op_ada[md], S, brow_slice(m.a_temb.q, r0, S, td),
                        brow_slice(m.a_temb.s, r0, S, td / RAD_FP8_BLOCK), RAD_W(B.ada[md].w),
                        RAD_W(B.ada[md].s), RAD_W(B.ada_b[md]),
                        bcol_at(m.b_ada, base, row, (l * kSegs + e) * m.Ws, m.Ws, S));
        }
    }
    for (int e = 0; e < kSegs; ++e) {
        const int64_t r0 = e == kSegAudio ? S : 0;
        RAD_ISSUE_N(c, m.op_no, S, brow_slice(m.a_temb.x, r0, S, td), RAD_W(m.w_no), RAD_W(m.b_no),
                    RAD_NONE, bcol_at(m.b_nout, base, kSegs * 2 * D, e * 2 * D, 2 * D, S));
    }
}

/* One denoising step of the lane: the packed sequence, the blocks, the heads and both Euler
 * steps. */
static void step_dit(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadVideoGenBatch* V = batch->vgen_b;
    const Geom& g = m.gd;
    const int64_t R = batch->n_tok, lane = V->lane, nt = V->geom.n_text;
    const int64_t arows = (int64_t)kAudioCh * V->geom.n_audio, vrows = R - nt - arows;
    const int64_t D = m.D, qd = g.q_dim(), L = g.n_layer, La = kMaxLanes;
    const RadOperand lane_d = praw(V->lane_d, RAD_I32, 1);

    if (V->step == 0) {
        RAD_ISSUE_N(c, m.op_rnd_v, vrows, praw(V->first_d, RAD_I32, 1), praw(V->seed, RAD_U32, 2),
                    lane_d, praw(V->lat_cu, RAD_I32, 2), brows(m.b_vstate, La * m.maxV));
        RAD_ISSUE_N(c, m.op_rnd_a, arows, praw(V->first_d, RAD_I32, 1),
                    praw(V->seed + 2, RAD_U32, 2), lane_d, praw(V->lat_cu + 2, RAD_I32, 2),
                    brows(m.b_astate, La * m.maxA));
    }
    const int64_t row = L * kSegs * m.Ws, ti = lane * m.S + V->step;
    if (g.world > 1)
        RAD_ISSUE_N(c, m.op_modg, row, brow_slice(m.b_ada, ti, 1, row),
                    brows(m.b_modg, (int64_t)g.world * L * kSegs));
    else
        RAD_ISSUE_N(c, m.op_modg, 1, brow_slice(m.b_ada, ti, 1, row),
                    RAD_P_T2(rad_buf_ptr(c, m.b_modg), RAD_BF16, 1, row));

    RAD_ISSUE_N(c, m.op_text, nt, brows(m.b_text, La * kMaxText), lane_d,
                praw(V->seg_cu, RAD_I32, 2), brows(m.b_h, nt));
    RAD_ISSUE_N(c, m.op_ain, arows, brow_slice(m.b_astate, lane * m.maxA, arows, kAudioC),
                RAD_W(m.w_ain), RAD_W(m.b_ain), RAD_NONE, brow_slice(m.b_h, nt, arows, D));
    RAD_ISSUE_N(c, m.op_vin, vrows, brow_slice(m.b_vstate, lane * m.maxV, vrows, kPatchVals),
                RAD_W(m.w_vin), RAD_W(m.b_vin), RAD_NONE, brow_slice(m.b_h, nt + arows, vrows, D));

    const RadOperand segs = praw(V->seg_cu, RAD_I32, kSegs + 1);
    const RadOperand geom = praw(V->geom_d, RAD_I32, RAD_VGEOM_N);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, 2);
    const RadOperand pos = praw2(V->pos, RAD_F32, 3, R);
    /* Chunk k of block l's six, for the three segments: on rank k / (6 / world), at its column. */
    const int64_t per = 6 / g.world, nD = D / RAD_FP8_BLOCK;
    auto mod = [&](int64_t l, int k, int chunks) {
        const int64_t r = k / per;
        return bcol_at(m.b_modg, r * L * kSegs + l * kSegs, m.Ws, (k % per) * D, chunks * D, kSegs);
    };
    /* Two halves of the rows with more than one rank. Everything between two attentions is
     * row-local, so a half computes exactly the bits the whole would, and each half's all-reduces
     * run on lane 1 while lane 0 computes the other half; the attention stays whole. The second
     * half starts at the video's middle row, so it is video alone: the batch's one-sequence
     * cu_seqlens and the video's modulation row stand for the segment table. Lane 0 waits for a
     * half's all-reduce before its first reader, issued before the next all-reduce goes to lane 1
     * (a join orders a lane behind everything issued on the other). */
    const bool split = g.world > 1;
    const int64_t cut = split ? nt + arows + vrows / 2 : R;
    const int64_t hr0[2] = { 0, cut }, hn[2] = { cut, R - cut };
    const int nh = split ? 2 : 1;
    const RadOperand segh[2] = { segs, cu };
    auto modh = [&](int h, int64_t l, int k, int chunks) {
        if (h == 0) return mod(l, k, chunks);
        const int64_t r = k / per;
        return bcol_at(m.b_modg, r * L * kSegs + l * kSegs + kSegVideo, m.Ws, (k % per) * D,
                       chunks * D, 1);
    };
    /* A half's partial sums across the ranks, kArRows rows a message, on lane 1 when split. */
    auto all_reduce_half = [&](rad_op op, int64_t r0, int64_t n) {
        if (!op) return;
        if (split) {
            rad_lane_join(c, 0, 1);
            rad_lane(c, 1);
        }
        for (int64_t q0 = 0; q0 < n; q0 += kArRows) {
            const int64_t k = std::min<int64_t>(kArRows, n - q0);
            RAD_ISSUE_N(c, op, k * D, brow_slice(m.b_ao, r0 + q0, k, D), RAD_NONE);
        }
        if (split) rad_lane(c, 0);
    };
    auto wait_ar = [&]() { if (split) rad_lane_join(c, 1, 0); };
    for (int64_t l = 0; l < L; ++l) {
        const DitBlock& B = m.dit[(size_t)l];
        /* The first half's norm reads the all-reduce the previous block waited for last; the
         * second half's waits here, so its reduction ran under the first half's norm and qkv. */
        for (int h = 0; h < nh; ++h) {
            const int64_t r0 = hr0[h], n = hn[h];
            if (h > 0 && l > 0) wait_ar();
            RAD_ISSUE_N(c, B.op_mq1, n, l > 0 ? brow_slice(m.b_ao, r0, n, D) : RAD_NONE,
                        l > 0 ? modh(h, l - 1, 5, 1) : RAD_NONE, modh(h, l, 0, 2), RAD_W(B.w_n1),
                        segh[h], brow_slice(m.b_h, r0, n, D), brow_slice(m.a_hm.q, r0, n, D),
                        brow_slice(m.a_hm.s, r0, n, nD), RAD_NONE);
            B.qkv.step(c, m.a_hm, m.b_qkv, R, r0, n);
        }
        RAD_ISSUE_N(c, B.op_qkr, R, bcol(m.b_qkv, 0, 3 * qd, R), pos, RAD_W(B.w_qn),
                    RAD_W(B.w_kn));
        RAD_ISSUE_N(c, B.op_kq, R, bcol(m.b_qkv, qd, qd, R), brows(m.b_k8, R), brows(m.b_ks, R));
        RAD_ISSUE_N(c, B.op_vq, R, bcol(m.b_qkv, 2 * qd, qd, R), brows(m.b_v8, R), RAD_B(m.b_vs));
        if (m.vsa) {
            RAD_ISSUE_N(c, B.op_sel, R, bcol(m.b_qkv, 0, qd, R), bcol(m.b_qkv, qd, qd, R),
                        bcol(m.b_qkv, 2 * qd, qd, R), segs, geom, RAD_B(m.b_sel),
                        RAD_B(m.b_coarse));
            RAD_ISSUE_N(c, B.op_attn, R, bcol(m.b_qkv, 0, qd, R), bcol(m.b_qkv, qd, qd, R),
                        brows(m.b_v8, R), segs, geom, RAD_B(m.b_sel), RAD_B(m.b_coarse),
                        bcol(m.b_qkv, 3 * qd, qd, R), brows(m.a_attn.x, R), brows(m.b_k8, R),
                        brows(m.b_ks, R), RAD_B(m.b_vs));
        } else {
            RAD_ISSUE_N(c, B.op_attn, R, bcol(m.b_qkv, 0, qd, R), bcol(m.b_qkv, qd, qd, R),
                        brows(m.b_v8, R), RAD_NONE, RAD_NONE, RAD_NONE, cu, RAD_NONE, RAD_NONE,
                        brows(m.b_k8, R), brows(m.b_ks, R), brows(m.a_attn.x, R), RAD_B(m.b_vs));
        }
        /* Each wait below is for the all-reduce lane 1 holds last, which is the one lane 0 reads
         * next: the second half's projection runs under the first half's reduction, the first
         * half's feed-forward under the second's, and so on. */
        for (int h = 0; h < nh; ++h) {
            B.q_attn.step(c, R, hr0[h], hn[h]);
            B.o.step(c, m.a_attn, m.b_ao, R, hr0[h], hn[h]);
            if (h > 0) wait_ar();
            all_reduce_half(B.op_ar1, hr0[h], hn[h]);
        }
        for (int h = 0; h < nh; ++h) {
            const int64_t r0 = hr0[h], n = hn[h];
            RAD_ISSUE_N(c, B.op_mq2, n, brow_slice(m.b_ao, r0, n, D), modh(h, l, 2, 1),
                        modh(h, l, 3, 2), RAD_W(B.w_n2), segh[h], brow_slice(m.b_h, r0, n, D),
                        brow_slice(m.a_hm.q, r0, n, D), brow_slice(m.a_hm.s, r0, n, nD), RAD_NONE);
            B.gate_up.step(c, m.a_hm, m.b_gu, R, r0, n);
            B.gq.step(c, m.b_gu, R, r0, n);
            B.down.step(c, m.a_ffn, m.b_ao, R, r0, n);
            wait_ar();
            all_reduce_half(B.op_ar2, r0, n);
        }
    }
    wait_ar();
    const char* nout = (const char*)rad_buf_ptr(c, m.b_nout) +
                       (lane * m.S + V->step) * kSegs * 2 * D * (int64_t)sizeof(uint16_t);
    RAD_ISSUE_N(c, m.op_nout, R, brows(m.b_ao, R), mod(L - 1, 5, 1),
                RAD_P_T2(nout, RAD_BF16, kSegs, 2 * D), RAD_W(m.w_no_n), segs, brows(m.b_h, R),
                brows(m.b_hn, R));

    /* The heads, in f32 over parts of the rows. */
    auto heads = [&](rad_op op, rad_weight w, rad_weight bias, int64_t first, int64_t n, int64_t nout_c,
                     rad_buf y) {
        for (int64_t r0 = 0; r0 < n; r0 += kHeadRows) {
            const int64_t k = std::min<int64_t>(kHeadRows, n - r0);
            RAD_ISSUE_N(c, m.op_hcast, k, brow_slice(m.b_hn, first + r0, k, D), brows(m.b_hf, k));
            RAD_ISSUE_N(c, op, k, brows(m.b_hf, k), RAD_W(w), RAD_W(bias), RAD_NONE,
                        brow_slice(y, r0, k, nout_c));
        }
    };
    heads(m.op_aout, m.w_aout, m.b_aout, nt, arows, kAudioC, m.b_avel);
    heads(m.op_vout, m.w_vout, m.b_vout, nt + arows, vrows, kPatchVals, m.b_vel);
    RAD_ISSUE_N(c, m.op_ev, vrows, brows(m.b_vel, vrows), praw(V->sched, RAD_F32, 1),
                praw(V->sched + 1, RAD_F32, 1), praw(V->sched + 2, RAD_F32, 1),
                brow_slice(m.b_vstate, lane * m.maxV, vrows, kPatchVals));
    RAD_ISSUE_N(c, m.op_ea, arows, brows(m.b_avel, arows), praw(V->sched + 3, RAD_F32, 1),
                praw(V->sched + 4, RAD_F32, 1), praw(V->sched + 5, RAD_F32, 1),
                brow_slice(m.b_astate, lane * m.maxA, arows, kAudioC));
}

/* The soundtrack: both channels, each a sequence, through the audio decoder, then interleaved. */
static void step_audio(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadVideoGenBatch* V = batch->vgen_b;
    const int64_t lane = V->lane, na = V->geom.n_audio, r0 = kAudioCh * na;
    const RadOperand nlen = praw(V->geom_d + RAD_VGEOM_N_AUDIO, RAD_I32, 1);
    auto conv = [&](const AConv& k, rad_buf x, rad_buf res, rad_buf y, int64_t R, rad_buf cu) {
        RAD_ISSUE_N(c, k.op, R, brows(x, R), RAD_W(k.w), brows(cu, 3), res ? brows(res, R) : RAD_NONE,
                    brows(y, R), k.b ? RAD_W(k.b) : RAD_NONE);
    };
    auto act = [&](const AAct& a, rad_buf x, rad_buf y, int64_t R, rad_buf cu) {
        RAD_ISSUE_N(c, a.op, R, brows(x, R), RAD_W(a.alpha), RAD_W(a.beta), RAD_W(a.up),
                    RAD_W(a.down), brows(cu, 3), brows(y, R));
    };
    RAD_ISSUE_N(c, m.op_adenorm, r0, brow_slice(m.b_astate, lane * m.maxA, r0, kAudioC),
                brows(m.b_aden, r0));
    RAD_ISSUE_N(c, m.op_acu0, 3, nlen, brows(m.b_acu[0], 3));
    conv(m.a_in, m.b_aden, 0, m.b_ain0, r0, m.b_acu[0]);
    conv(m.a_pre, m.b_ain0, 0, m.b_aavg[0], r0, m.b_acu[0]);
    int64_t rows = r0;
    for (int s = 0; s < 7; ++s) {
        const AStage& St = m.ast[s];
        const int64_t R = rows * St.rate;
        RAD_ISSUE_N(c, St.op_cu, 3, nlen, brows(m.b_acu[s + 1], 3));
        RAD_ISSUE_N(c, St.up.op, rows, brows(m.b_aavg[s], rows), RAD_W(St.up.w), RAD_W(St.up.b),
                    brows(m.b_acu[s], 3), brows(m.b_ah[s + 1], R));
        for (int j = 0; j < 3; ++j)
            for (int d = 0; d < 3; ++d) {
                const AmpUnit& U = St.amp[j].u[d];
                const rad_buf x = d == 0 ? m.b_ah[s + 1] : m.b_aamp[j][s + 1];
                act(U.a1, x, m.b_at1[s + 1], R, m.b_acu[s + 1]);
                conv(U.c1, m.b_at1[s + 1], 0, m.b_at2[s + 1], R, m.b_acu[s + 1]);
                act(U.a2, m.b_at2[s + 1], m.b_at1[s + 1], R, m.b_acu[s + 1]);
                conv(U.c2, m.b_at1[s + 1], x, m.b_aamp[j][s + 1], R, m.b_acu[s + 1]);
            }
        RAD_ISSUE_N(c, St.op_sum, R, brows(m.b_aamp[0][s + 1], R), brows(m.b_aamp[1][s + 1], R),
                    brows(m.b_aavg[s + 1], R));
        RAD_ISSUE_N(c, St.op_avg, R, brows(m.b_aavg[s + 1], R), brows(m.b_aamp[2][s + 1], R),
                    brows(m.b_aavg[s + 1], R));
        rows = R;
    }
    act(m.a_actpost, m.b_aavg[7], m.b_at1[7], rows, m.b_acu[7]);
    conv(m.a_post, m.b_at1[7], 0, m.b_aout0, rows, m.b_acu[7]);
    const int64_t n = rows / kAudioCh;
    RAD_ISSUE_N(c, m.op_vmix, n, brow_slice(m.b_aout0, 0, n, 1), brow_slice(m.b_aout0, n, n, 1),
                brows(m.b_audio, n));
}

/* One chunk of frames: the latents, this rank's tiles through the ViT, every rank's gathered, and
 * the stitch into the chunk's frames. */
static void step_video(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadVideoGenBatch* V = batch->vgen_b;
    const int64_t lane = V->lane, lh = V->geom.lat_h, lw = V->geom.lat_w;
    const int64_t ppf = (lh / kPatch) * (lw / kPatch), vox = (int64_t)kDecLat * lh * lw;
    const int64_t dim = 2048, px = 3 * 4 * kScale * kScale;
    const int64_t nt = tiles_of(lh, lw), tpr = (nt + m.W - 1) / m.W;
    const int64_t t0 = (int64_t)m.gt.rank * tpr;
    const int64_t mine = std::max<int64_t>(0, std::min<int64_t>(tpr, nt - t0));
    const int64_t R = mine * kPeriod, a = t0 * kPeriod;

    RAD_ISSUE_N(c, m.op_unfold, vox,
                brow_slice(m.b_vstate, lane * m.maxV + (int64_t)V->chunk * kLpc * ppf,
                           (int64_t)kDecLat * ppf, kPatchVals),
                praw(V->geom_d + RAD_VGEOM_PATCH_FRAMES, RAD_I32, 3), brows(m.b_vox, vox));
    RAD_ISSUE_N(c, m.op_pqc, vox, brows(m.b_vox, vox), RAD_W(m.w_pqc), RAD_W(m.b_pqc), RAD_NONE,
                brows(m.b_vox2, vox));
    RAD_ISSUE_N(c, m.op_vpin, vox, brows(m.b_vox2, vox), RAD_W(m.w_vpin2), RAD_W(m.b_vpin2),
                RAD_NONE, brows(m.b_vpin, vox));
    RAD_ISSUE_N(c, m.op_tiles, nt * kPeriod, brows(m.b_vpin, vox), RAD_W(m.w_reg),
                praw(V->geom_d + RAD_VGEOM_CHUNK_FRAMES, RAD_I32, 3),
                brows(m.b_vx, nt * kPeriod));
    if (mine > 0) {
        const RadOperand x = brow_slice(m.b_vx, a, R, dim);
        const RadOperand pos = bcol(m.b_dpos, 0, R, 3);
        RAD_ISSUE_N(c, m.op_dpos, R, pos);
        RAD_ISSUE_N(c, m.op_dcu, mine + 1, brows(m.b_dcu, mine + 1));
        for (const VitBlock& B : m.vit) {
            RAD_ISSUE_N(c, B.op_n1, R, x, RAD_W(B.w_n1), brows(m.b_vn, R));
            RAD_ISSUE_N(c, B.op_qkv, R, brows(m.b_vn, R), RAD_W(B.w_qkv), RAD_W(B.b_qkv), RAD_NONE,
                        brows(m.b_vqkv, R));
            RAD_ISSUE_N(c, B.op_qkr, R, brows(m.b_vqkv, R), pos, RAD_NONE, RAD_NONE);
            RAD_ISSUE_N(c, B.op_attn, R, bcol(m.b_vqkv, 0, dim, R), bcol(m.b_vqkv, dim, dim, R),
                        bcol(m.b_vqkv, 2 * dim, dim, R), brows(m.b_dcu, mine + 1),
                        brows(m.b_vattn, R));
            RAD_ISSUE_N(c, B.op_o, R, brows(m.b_vattn, R), RAD_W(B.w_o), RAD_W(B.b_o), RAD_NONE,
                        brows(m.b_vo, R));
            RAD_ISSUE_N(c, B.op_s1, R, brows(m.b_vo, R), RAD_W(B.s1), brows(m.b_vt, R));
            RAD_ISSUE_N(c, B.op_r1, R, x, brows(m.b_vt, R), x);
            RAD_ISSUE_N(c, B.op_n2, R, x, RAD_W(B.w_n2), brows(m.b_vn, R));
            RAD_ISSUE_N(c, B.op_up, R, brows(m.b_vn, R), RAD_W(B.w_up), RAD_W(B.b_up), RAD_NONE,
                        brows(m.b_vup, R));
            RAD_ISSUE_N(c, B.op_act, R, brows(m.b_vup, R), brows(m.b_vact, R));
            RAD_ISSUE_N(c, B.op_down, R, brows(m.b_vact, R), RAD_W(B.w_down), RAD_W(B.b_down),
                        RAD_NONE, brows(m.b_vo, R));
            RAD_ISSUE_N(c, B.op_s2, R, brows(m.b_vo, R), RAD_W(B.s2), brows(m.b_vt, R));
            RAD_ISSUE_N(c, B.op_r2, R, x, brows(m.b_vt, R), x);
        }
        RAD_ISSUE_N(c, m.op_vnorm, R, x, RAD_W(m.w_vno), RAD_W(m.b_vno), brows(m.b_vn, R));
        RAD_ISSUE_N(c, m.op_vproj, R, brows(m.b_vn, R), RAD_W(m.w_vpo), RAD_W(m.b_vpo), RAD_NONE,
                    brows(m.b_vpix, R));
    }
    rad_buf pix = m.b_vpix;
    if (m.W > 1) {
        RAD_ISSUE_N(c, m.op_vgather, tpr * kPeriod * px, brows(m.b_vpix, tpr * kPeriod),
                    brows(m.b_vpix_all, m.W * tpr * kPeriod));
        pix = m.b_vpix_all;
    }
    const int64_t hp = lh * kScale, wp = lw * kScale, n = (int64_t)V->n_frames_out * hp * wp;
    RAD_ISSUE_N(c, m.op_stitch, n, brows(pix, nt * kPeriod),
                praw(V->geom_d + RAD_VGEOM_CHUNK_FRAMES, RAD_I32, 5),
                brow_slice(m.b_prev, lane * kFtail * (int64_t)kMaxArea, (int64_t)kFtail * hp * wp, 3),
                brows(m.b_frames, n));
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_rank(c)];
    if (!batch->gen || !batch->vgen_b) {
        rad_step_fail(c, "minimaxh3 is a video generator and runs its generator passes only");
        return;
    }
    switch (batch->gen) {
    case RAD_GEN_PASS_VCOND:   step_cond(c, batch, m); return;
    case RAD_GEN_PASS_VSTEP:   step_dit(c, batch, m); return;
    case RAD_GEN_PASS_ADECODE: step_audio(c, batch, m); return;
    case RAD_GEN_PASS_VDECODE: step_video(c, batch, m); return;
    default:
        rad_step_fail(c, "minimaxh3: a generator pass of a kind this plugin does not know");
    }
}

/* ================================================================== the probe */
/* The rows of the largest pass: a step of the longest video at the largest size, or a decode of
 * the most tiles. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->gen_rows = std::max<int64_t>(max_step_rows(), max_tiles() * kPeriod);
    out->gen_video = 1;
    return RAD_OK;
}

}  /* namespace minimaxh3 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(minimaxh3)
RAD_ARCH_PLUGIN(minimaxh3, "minimaxh3", "", "0.1.0",
                "MiniMax-H3 (t2va): Qwen3-VL-32B conditioning, a 33B single-stream diffusion "
                "transformer over text, soundtrack and video, a tiled ViT video decoder and a "
                "BigVGAN audio decoder; block-fp8 linears")

extern "C" int64_t rad_arch_video_positions(int which, const RadVideoGeom* g, float* pos,
                                            int64_t cap) {
    if (!g || g->n_text < 1 || g->n_text > minimaxh3::kMaxText) return RAD_E_INVAL;
    if (which == RAD_VPOS_STEP) return minimaxh3::positions(g, pos, cap);
    if (which == RAD_VPOS_DECODE) return minimaxh3::decode_rows(g);
    return RAD_E_INVAL;
}
#endif
