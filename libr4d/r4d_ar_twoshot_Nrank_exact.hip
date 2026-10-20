// r4d_ar_twoshot_Nrank_exact.hip: r4d_ar_twoshot_{3,4,8}rank_exact -- reduce-scatter +
// all-gather at 3, 4 or 8 ranks, exact bf16/fp16/fp32 wire. Per-block pipelined handshake
// instead of a grid arrival election. Sends 2*(ws-1)/ws*N bytes per rank in two hops, so it
// covers the sizes where the one-shot's (ws-1)*N send is too heavy.
//
// Shard s = a contiguous run of the message's 16-byte words, owned by rank s. Phase 1: push my
// contribution to each peer-owned shard into the owner's slot[me] region A, flag 2s-1; reduce my
// shard fp32 in ascending rank order (own contribution from the input). Phase 2: reduced words go
// to my out and every peer's slot[me] region B, flag 2s; assemble out from region B. Region A at
// 0, B at sh_max; the slot stride must hold 2*sh_max.
//
// The shard table (r4d_ar_shards.h) comes in as a kernel argument. By default it is the even
// split (the first n16 % ws shards one word longer); at 3 ranks engine residuals rarely divide (a
// 2048-wide bf16 row is 256 words). At 8 ranks it may be weighted per link
// (RADIANCE_AR8_SHARD_WEIGHTS); a rank with weight 0 owns nothing and only pushes, gathers and
// handshakes. Over both phases a rank moves N + (ws-2)*len_r each way, but phase 1 only sends
// N - len_r (receiving (ws-1)*len_r) and phase 2 the reverse, so a short shard on a slow link pays
// only as far as the phases overlap on that link (docs/GUIDE.md, 4.7). Each word is the same fp32
// fold in ascending rank order whoever owns it, so the bits match the one-shot's.
//
// At 8 ranks the message may also run in pieces (RADIANCE_AR8_PIECES, by default a count from
// the message size, r4d_ar_shards.h): each block's words split into `pieces` runs pipelined so
// piece j+1's phase 1 shares the link with piece j's phase 2. A rank with a short shard then
// sends and receives at once instead of one direction per phase.
//
// Shares the wide family's flag/seq/scratch set (r4d_ar_wide.h): it publishes in the 2*seq flag
// space plus the 2s-1 half-step (with pieces: pieces+1 values from 2s-1 on, after reserving one
// step a piece), so one buffer set can carry any mix of one-shot and two-shot launches. The slot
// stride must hold the largest requirement in the mix.
#include "r4d.h"
#include "r4d_ar_wide.h"
#include "r4d_ar_shards.h"
#include "r4d_chain.h"
#include <stdexcept>
#include <string>

template <int WS>
struct R4DArShardOff {
  int off[WS + 1];
  int sh_max;
};

// PIECED false fixes pieces at 1, so the unpipelined launch keeps the straight-line kernel.
// CHAIN: the chained form's wait (before the sequence take, so a block of the next launch on the
// instance never starts ahead of this one's) and add (r4d_chain.h); no block returns early.
template <typename T, int LANES, int WS, bool PIECED, bool CHAIN>
__device__ __forceinline__ void r4d_ar_twoshot_wide_exact_body(
    const int4* __restrict__ in4, R4DArWidePeers<int4, WS - 1> peers,
    const int4* __restrict__ my_base, unsigned int* __restrict__ my_flags,
    int4* __restrict__ out4, unsigned int* __restrict__ seq_ctrs, R4DArShardOff<WS> sh,
    int slot_stride16, int rank, int drain, int acq, int pub,
    long long* gate, unsigned int* gate_ctr, int pieces_arg) {
  if (CHAIN) r4d_chain_wait();
  constexpr int NPEERS = WS - 1;
  const int pieces = PIECED ? pieces_arg : 1;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  // device-resident, replay-safe; `pieces` steps reserve the 2*pieces flag values this launch uses
  r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot, (unsigned int)pieces);
  const unsigned int v0 = 2u * s_seq - 1u;
  const int parity = (int)s_slot;

  const int sh_max = sh.sh_max;
#define R4D_TS_OFF(s) (sh.off[s])
#define R4D_TS_LEN(s) (sh.off[(s) + 1] - sh.off[s])

  const int w = (sh_max + nb - 1) / nb;    // block b owns words [w0,w1) of the SHARD space
  const int w0 = b * w;
  int w1 = w0 + w; if (w1 > sh_max) w1 = sh_max;
  const int pw = (w + pieces - 1) / pieces;  // piece j of the block: [w0 + j*pw, +pw) within w1

  // region A of my slot in every peer's scratch (B sits at +sh_max)
  int4* pa[NPEERS];
#pragma unroll
  for (int p = 0; p < NPEERS; ++p)
    pa[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16;

  const int4* src[WS];
#pragma unroll
  for (int r = 0; r < WS; ++r)
    src[r] = (r == rank) ? (in4 + (size_t)R4D_TS_OFF(rank))
                         : (my_base + ((size_t)parity * WS + r) * (size_t)slot_stride16);
  int4* my_out = out4 + (size_t)R4D_TS_OFF(rank);

  // phase 1: my contribution to each peer-owned shard -> owner's region A
  auto push1 = [&](int j) {
    const int p0 = w0 + j * pw, p1 = p0 + pw < w1 ? p0 + pw : w1;
    for (int p = 0; p < NPEERS; ++p) {
      const int owner = peers.rank[p];
      const int4* s1 = in4 + (size_t)R4D_TS_OFF(owner);
      int4* dst = pa[p];
      const int lim = p1 < R4D_TS_LEN(owner) ? p1 : R4D_TS_LEN(owner);
      for (int k = p0 + tid; k < lim; k += nt) dst[k] = s1[k];
    }
  };
  // reduce my shard (fp32, ascending rank, own contribution from the input) fused with the phase-2
  // push: the reduced word goes to my out and every peer's region B in the same pass
  auto reduce2 = [&](int j) {
    const int p0 = w0 + j * pw, p1 = p0 + pw < w1 ? p0 + pw : w1;
    const int my_lim = p1 < R4D_TS_LEN(rank) ? p1 : R4D_TS_LEN(rank);
    for (int k = p0 + tid; k < my_lim; k += nt) {
      float acc[LANES];
#pragma unroll
      for (int l = 0; l < LANES; ++l) acc[l] = 0.0f;
#pragma unroll
      for (int r = 0; r < WS; ++r) {
        const int4 v = src[r][k];
        const T* pv = reinterpret_cast<const T*>(&v);
#pragma unroll
        for (int l = 0; l < LANES; ++l) acc[l] += r4d_ar_to_f(pv[l]);
      }
      int4 o;
      T* po = reinterpret_cast<T*>(&o);
#pragma unroll
      for (int l = 0; l < LANES; ++l) r4d_ar_from_f(acc[l], po[l]);
      my_out[k] = o;
#pragma unroll
      for (int p = 0; p < NPEERS; ++p) pa[p][sh_max + k] = o;  // region B
    }
  };
  // every peer-owned shard from my region B copies
  auto assemble = [&](int j) {
    const int p0 = w0 + j * pw, p1 = p0 + pw < w1 ? p0 + pw : w1;
    for (int p = 0; p < NPEERS; ++p) {
      const int owner = peers.rank[p];
      const int4* srcB = my_base + ((size_t)parity * WS + owner) * (size_t)slot_stride16 + sh_max;
      int4* dst = out4 + (size_t)R4D_TS_OFF(owner);
      const int lim = p1 < R4D_TS_LEN(owner) ? p1 : R4D_TS_LEN(owner);
      for (int k = p0 + tid; k < lim; k += nt) dst[k] = srcB[k];
    }
  };

  /* Pieces run as a pipeline so both directions of a link carry bytes at once: step j pushes
   * piece j+1's phase 1 and piece j's phase 2. One flag word per (block, source) is a monotone
   * counter, so a step's single handshake (v0 + 1 + j) also vouches for every earlier one. A rank
   * owning little sends nearly all its bytes in phase 1 and receives them in phase 2; without the
   * overlap one direction idles in each phase. With one piece the sequence is push, flag 2s-1,
   * reduce, flag 2s, assemble. A word is touched by the same thread in all three steps, which keeps
   * the in-place form safe. */
  push1(0);
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, v0, drain, acq, pub);
  for (int j = 0; j < pieces; ++j) {
    if (j + 1 < pieces) push1(j + 1);
    reduce2(j);
    if (j > 0) assemble(j - 1);
    r4d_ar_handshake<WS>(peers, my_flags, b, rank, v0 + 1u + (unsigned int)j, drain, acq, pub);
  }
  /* The hold ends after the last handshake, not the first: the reduce is fused with the phase-2
   * push (every reduced word goes to each peer's region B), so until then the link still carries
   * this launch's phase-2 bytes. Here every phase-1 and phase-2 remote store has retired behind
   * its flag store and every peer's pushes to this rank have been waited on; only the local
   * assemble of the last piece is left (r4d_ar_seq.h). */
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);
  assemble(pieces - 1);
#undef R4D_TS_OFF
#undef R4D_TS_LEN
  if (CHAIN) r4d_chain_done();
}

#define R4D_ARTS_PARAMS const int4* __restrict__ in4, R4DArWidePeers<int4, WS - 1> peers,       \
    const int4* __restrict__ my_base, unsigned int* __restrict__ my_flags,                      \
    int4* __restrict__ out4, unsigned int* __restrict__ seq_ctrs, R4DArShardOff<WS> sh,         \
    int slot_stride16, int rank, int drain, int acq, int pub, long long* gate,                  \
    unsigned int* gate_ctr, int pieces_arg
#define R4D_ARTS_ARGS in4, peers, my_base, my_flags, out4, seq_ctrs, sh, slot_stride16, rank,    \
    drain, acq, pub, gate, gate_ctr, pieces_arg
template <typename T, int LANES, int WS, bool PIECED>
__global__ void r4d_ar_twoshot_wide_exact_kernel(R4D_ARTS_PARAMS) {
  r4d_ar_twoshot_wide_exact_body<T, LANES, WS, PIECED, false>(R4D_ARTS_ARGS);
}
template <typename T, int LANES, int WS, bool PIECED>
__global__ void r4dchain_ar_twoshot_wide_exact(R4D_ARTS_PARAMS) {
  r4d_ar_twoshot_wide_exact_body<T, LANES, WS, PIECED, true>(R4D_ARTS_ARGS);
}
#undef R4D_ARTS_ARGS
#undef R4D_ARTS_PARAMS

// --- host side ---
template <int WS>
static void ar_twoshot_launch(const long* peer_scratch, const long* peer_flags, long my_scratch,
                              long my_flags, long seq_ctrs, long slot_stride16, long inp,
                              long out, long n_elem, long dtype, long rank, long stream_i,
                              long nblocks, long nthreads, long drain, long acq, long pub,
                              long gate, long gate_ctr, const int* shard_weights,
                              long pieces, bool chain = false) {
  constexpr int NPEERS = WS - 1;
  if (rank < 0 || rank >= WS)
    throw std::runtime_error("all_reduce: rank " + std::to_string(rank) +
                             " out of range for the " + std::to_string(WS) +
                             " ranks this entry point is compiled for");
  const int esize = (dtype == 2) ? 4 : 2;
  const long nbytes = (long)n_elem * esize;
  if (nbytes % 16 != 0) throw std::runtime_error("all_reduce: n_elem*esize not 16B-aligned");
  if (n_elem > R4D_AR_TWOSHOT_WIDE_MAX_ELEMS)
    throw std::runtime_error("all_reduce: n_elem " + std::to_string(n_elem) +
                             " exceeds the twoshot band (" +
                             std::to_string(R4D_AR_TWOSHOT_WIDE_MAX_ELEMS) +
                             " elems); use the fallback path");
  const long n16 = nbytes / 16;
  if (!shard_weights && n16 < WS)
    throw std::runtime_error("all_reduce: message (" + std::to_string(n16) +
                             " 16B words) is shorter than one word a rank");
  if (n16 < 1) throw std::runtime_error("all_reduce: empty message");
  if (shard_weights) {
    long wsum = 0;
    for (int s = 0; s < WS; ++s) {
      if (shard_weights[s] < 0 || shard_weights[s] > R4D_AR_SHARD_WEIGHT_MAX)
        throw std::runtime_error("all_reduce: shard weight out of range");
      wsum += shard_weights[s];
    }
    if (wsum <= 0) throw std::runtime_error("all_reduce: shard weights sum to zero");
  }
  long off[WS + 1];
  const long sh_max = r4d_ar_shard_table(n16, WS, shard_weights, off);
  R4DArShardOff<WS> sh{};
  for (int s = 0; s <= WS; ++s) sh.off[s] = (int)off[s];
  sh.sh_max = (int)sh_max;
  // regions A and B each hold the longest shard
  if (2 * sh_max > (long)slot_stride16)
    throw std::runtime_error("all_reduce: shard (" + std::to_string(sh_max) +
                             " 16B words) needs 2x inside the per-source slot stride (" +
                             std::to_string(slot_stride16) + ")");

  int nb = (int)nblocks;
  if (nb < 1) nb = R4D_AR_NB_DESIGN;
  if (nb > sh_max) nb = (int)sh_max;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
  if (pieces < 1 || pieces > R4D_AR_TS_MAX_PIECES)
    throw std::runtime_error("all_reduce: pieces " + std::to_string(pieces) + " outside 1.." +
                             std::to_string((int)R4D_AR_TS_MAX_PIECES));
  // a block's share of the longest shard bounds the pieces worth a handshake each
  const long words_a_block = (sh_max + nb - 1) / nb;
  const int np = (int)(pieces < words_a_block ? pieces : words_a_block);

  R4DArWidePeers<int4, NPEERS> peers{};
  for (int i = 0; i < NPEERS; ++i) {
    if (!peer_scratch[i] || !peer_flags[i])
      throw std::runtime_error("all_reduce: null peer scratch/flag pointer at index " +
                               std::to_string(i));
    peers.scratch[i] = (int4*)peer_scratch[i];
    peers.flags[i] = (unsigned int*)peer_flags[i];
    peers.rank[i] = (i < (int)rank) ? i : i + 1;
  }

  // Checked against the allocations: scratch backs 2*WS slots of the stride, flags one word per
  // (block, source rank), seq a handshake and a parity word per block of capacity (r4d_ar_seq.h).
  const long slot_bytes = 2L * WS * (long)slot_stride16 * 16;
  const long flag_bytes = (long)nb * WS * (long)sizeof(unsigned int);
  r4d_ar_check_extent(my_scratch, slot_bytes, "all_reduce", "my_scratch");
  r4d_ar_check_extent(my_flags, flag_bytes, "all_reduce", "my_flags");
  r4d_ar_check_extent(seq_ctrs, 2L * R4D_AR_WIDE_MAX_BLOCKS * (long)sizeof(unsigned int),
                      "all_reduce", "seq_ctrs");
  for (int i = 0; i < NPEERS; ++i) {
    r4d_ar_check_extent(peer_scratch[i], slot_bytes, "all_reduce", "peer scratch");
    r4d_ar_check_extent(peer_flags[i], flag_bytes, "all_reduce", "peer flags");
  }

  hipStream_t st = (hipStream_t)stream_i;
#define R4D_AR_TS_LAUNCH_P(T, LN, P)                                                  \
  (chain ? r4dchain_ar_twoshot_wide_exact<T, LN, WS, P>                               \
         : r4d_ar_twoshot_wide_exact_kernel<T, LN, WS, P>)<<<nb, nt, 0, st>>>(        \
      (const int4*)inp, peers, (const int4*)my_scratch, (unsigned int*)my_flags,      \
      (int4*)out, (unsigned int*)seq_ctrs, sh, (int)slot_stride16,                    \
      (int)rank, (int)drain, (int)acq, (int)pub, (long long*)gate,                    \
      (unsigned int*)gate_ctr, np)
#define R4D_AR_TS_LAUNCH(T, LN)                                                       \
  do {                                                                                \
    if (np > 1) R4D_AR_TS_LAUNCH_P(T, LN, true);                                      \
    else R4D_AR_TS_LAUNCH_P(T, LN, false);                                            \
  } while (0)
  if (dtype == 0) R4D_AR_TS_LAUNCH(__hip_bfloat16, 8);
  else if (dtype == 1) R4D_AR_TS_LAUNCH(__half, 8);
  else if (dtype == 2) R4D_AR_TS_LAUNCH(float, 4);
  else throw std::runtime_error("all_reduce: bad dtype");
#undef R4D_AR_TS_LAUNCH
#undef R4D_AR_TS_LAUNCH_P
}

void r4d_ar_twoshot_3rank_exact(long peer_scratch0, long peer_scratch1, long peer_flags0,
                                long peer_flags1, long my_scratch, long my_flags, long seq_ctrs,
                                long slot_stride16, long inp, long out, long n_elem, long dtype,
                                long rank, long stream, long nblocks, long nthreads, long drain,
                                long acq, long pub) {
  const long ps[2] = {peer_scratch0, peer_scratch1};
  const long pf[2] = {peer_flags0, peer_flags1};
  /* No link gate at three ranks: the row answers 1, the engine releases by barrier. */
  ar_twoshot_launch<3>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                       dtype, rank, stream, nblocks, nthreads, drain, acq, pub, 0L, 0L, nullptr,
                       1L);
}

void r4d_ar_twoshot_4rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_flags0, long peer_flags1, long peer_flags2,
                                long my_scratch, long my_flags, long seq_ctrs,
                                long slot_stride16, long inp, long out, long n_elem, long dtype,
                                long rank, long stream, long nblocks, long nthreads, long drain,
                                long acq, long pub, long gate, long gate_ctr, long chain) {
  const long ps[3] = {peer_scratch0, peer_scratch1, peer_scratch2};
  const long pf[3] = {peer_flags0, peer_flags1, peer_flags2};
  ar_twoshot_launch<4>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                       dtype, rank, stream, nblocks, nthreads, drain, acq, pub, gate, gate_ctr,
                       nullptr, 1L, chain != 0);
}

void r4d_ar_twoshot_8rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_scratch3, long peer_scratch4, long peer_scratch5,
                                long peer_scratch6, long peer_flags0, long peer_flags1,
                                long peer_flags2, long peer_flags3, long peer_flags4,
                                long peer_flags5, long peer_flags6, long my_scratch,
                                long my_flags, long seq_ctrs, long slot_stride16, long inp,
                                long out, long n_elem, long dtype, long rank, long stream,
                                long nblocks, long nthreads, long drain, long acq, long pub,
                                const int* shard_weights, long pieces) {
  const long ps[7] = {peer_scratch0, peer_scratch1, peer_scratch2, peer_scratch3,
                      peer_scratch4, peer_scratch5, peer_scratch6};
  const long pf[7] = {peer_flags0, peer_flags1, peer_flags2, peer_flags3,
                      peer_flags4, peer_flags5, peer_flags6};
  /* No link gate at eight ranks: the row answers 1, the engine releases by barrier. */
  ar_twoshot_launch<8>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                       dtype, rank, stream, nblocks, nthreads, drain, acq, pub, 0L, 0L,
                       shard_weights, pieces);
}
