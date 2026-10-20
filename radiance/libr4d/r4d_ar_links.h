#pragma once
// r4d_ar_links.h: the eight-rank two-shot's shard weights from the PCIe links (host only).
//
// A card's link to the CPU is as fast as its slowest hop. The R9700 sits behind an on-card switch
// whose endpoint reports full speed in any slot, so a degraded slot shows only at the root port:
// every hop from the root port down is read, as core/diag/sysinfo.cpp (read_card) reads it, from
// the kernel's sysfs under `root` ("" on a real machine, a directory tree in tests). Nothing that
// a file does not say is guessed; a card with no readable hop has no rate.
//
// Weight from the rate relative to the fastest card: w = round(4 * (rate / max)^2), so x16 4,
// half the rate 1, a quarter 0. A rank's bytes per direction grow with its shard, so the weights
// fall faster than the rate; the measured best on four Gen5 x16, two Gen5 x8 and two Gen4 x8 was
// 4,1,0,4,0,1,4,4 (docs/GUIDE.md, 4.7).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

struct R4DArLink {
  std::string bdf;     // the slowest hop, "" when none was readable
  double gts = 0.0;    // its current speed in GT/s
  int width = 0;       // its current width in lanes
};

// Payload GB/s of one direction: 8b/10b below 8 GT/s, 128b/130b from Gen3 on.
inline double r4d_ar_link_gbs(const R4DArLink& l) {
  if (l.gts <= 0.0 || l.width <= 0) return 0.0;
  const double enc = l.gts < 8.0 ? 8.0 / 10.0 : 128.0 / 130.0;
  return l.gts * l.width * enc / 8.0;
}

inline std::string r4d_ar_sysfs_line(const std::string& path) {
  std::ifstream f(path);
  std::string s;
  if (!f || !std::getline(f, s)) return std::string();
  return s;
}

inline bool r4d_ar_is_bdf(const std::string& s) {
  unsigned d, b, dv, f;
  char tail;
  return s.size() == 12 && std::sscanf(s.c_str(), "%4x:%2x:%2x.%1x%c", &d, &b, &dv, &f, &tail) == 4;
}

// The slowest hop between the root port and the card `bdf` ("0000:03:00.0", lower case). False
// when no hop on the path says its current speed and width.
inline bool r4d_ar_link_bottleneck(const std::string& root, const std::string& bdf,
                                   R4DArLink* out) {
  namespace fs = std::filesystem;
  const std::string devs = root + "/sys/bus/pci/devices/";
  std::error_code ec;
  const fs::path real = fs::canonical(devs + bdf, ec);
  if (ec) return false;
  R4DArLink best;
  for (const fs::path& part : real) {
    const std::string p = part.string();
    if (!r4d_ar_is_bdf(p)) continue;
    R4DArLink h;
    h.bdf = p;
    h.gts = std::strtod(r4d_ar_sysfs_line(devs + p + "/current_link_speed").c_str(), nullptr);
    h.width = std::atoi(r4d_ar_sysfs_line(devs + p + "/current_link_width").c_str());
    const double r = r4d_ar_link_gbs(h);
    if (r <= 0.0) continue;
    if (best.bdf.empty() || r < r4d_ar_link_gbs(best)) best = h;
  }
  if (best.bdf.empty()) return false;
  *out = best;
  return true;
}

// Weights from every rank's rate (GB/s). False when a rank's rate is unknown.
inline bool r4d_ar_weights_from_rates(const double* gbs, int ws, int* w) {
  double top = 0.0;
  for (int r = 0; r < ws; ++r) {
    if (!(gbs[r] > 0.0)) return false;
    if (gbs[r] > top) top = gbs[r];
  }
  for (int r = 0; r < ws; ++r) {
    const double q = gbs[r] / top;
    w[r] = (int)std::lround(4.0 * q * q);
  }
  return true;
}
