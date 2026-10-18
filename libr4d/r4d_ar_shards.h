#pragma once
// r4d_ar_shards.h: the exact two-shot's shard table. Shard s is the 16-byte words
// [off[s], off[s+1]) and is reduced by rank s. Host only and HIP-free: the launcher, the plugin
// shim (which sizes scratch and picks the block count from it) and the selftest (which prints it)
// must derive the same table from the same inputs.
//
// The even split is n16 / ws words a shard with the first n16 % ws shards one word longer. Weights
// split proportionally (floor(n16 * w_s / sum)) and hand the few remaining words, one each, to the
// weighted ranks in ascending order; equal weights therefore reproduce the even split exactly, and
// a zero weight always means an empty shard. Which rank owns a word changes only the link it
// crosses, not its fold (fp32, ascending rank), so every table gives the same bits.
#include <cstdlib>

#define R4D_AR8_SHARD_WEIGHTS_ENV "RADIANCE_AR8_SHARD_WEIGHTS"
enum { R4D_AR_SHARD_WEIGHT_MAX = 65535 };

// `ws` non-negative integers separated by commas, each at most R4D_AR_SHARD_WEIGHT_MAX, summing
// to more than zero; nothing else (no blanks, signs or trailing text).
inline bool r4d_ar_shard_weights_parse(const char* s, int ws, int* w) {
  if (!s) return false;
  long sum = 0;
  for (int i = 0; i < ws; ++i) {
    if (*s < '0' || *s > '9') return false;
    long v = 0;
    while (*s >= '0' && *s <= '9') {
      v = v * 10 + (*s++ - '0');
      if (v > R4D_AR_SHARD_WEIGHT_MAX) return false;
    }
    w[i] = (int)v;
    sum += v;
    if (i + 1 < ws) {
      if (*s != ',') return false;
      ++s;
    }
  }
  return *s == '\0' && sum > 0;
}

// Fills off[0..ws] and returns the longest shard. `w` null: the even split.
inline long r4d_ar_shard_table(long n16, int ws, const int* w, long* off) {
  long len[16];
  if (!w) {
    const long base = n16 / ws, rem = n16 % ws;
    for (int s = 0; s < ws; ++s) len[s] = base + (s < rem ? 1 : 0);
  } else {
    long sum = 0;
    for (int s = 0; s < ws; ++s) sum += w[s];
    long left = n16;
    for (int s = 0; s < ws; ++s) {
      len[s] = n16 * w[s] / sum;
      left -= len[s];
    }
    for (int s = 0; s < ws && left > 0; ++s)
      if (w[s] > 0) { ++len[s]; --left; }
  }
  long sh_max = 0;
  off[0] = 0;
  for (int s = 0; s < ws; ++s) {
    off[s + 1] = off[s] + len[s];
    if (len[s] > sh_max) sh_max = len[s];
  }
  return sh_max;
}

// The longest shard any message of at most n16 words can get under `w` (null: even): what the
// scratch slot must hold twice.
inline long r4d_ar_shard_max_bound(long n16, int ws, const int* w) {
  if (!w) return (n16 + ws - 1) / ws;
  long sum = 0, wmax = 0;
  for (int s = 0; s < ws; ++s) {
    sum += w[s];
    if (w[s] > wmax) wmax = w[s];
  }
  const long b = n16 * wmax / sum + 1;
  return b < n16 ? b : n16;
}
