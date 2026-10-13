// Hybrid filter for the arm64 substring search (simdutf::search).
//
// On NEON every searcher works 16 bytes at a time, so the cost of the filter
// itself decides the speed: two needle bytes per position beat three when the
// two are rare. This kernel therefore starts with the two rarest bytes of the
// needle (by a static byte-frequency table) and switches, for the rest of the
// haystack, to three needle bytes chosen by position (first, middle, last,
// moved to distinct values) once the survivors of the pair cost more than the
// third compare would save. Each survivor is compared against the first (up
// to) 64 bytes of the needle; the verification work beyond those 64 bytes is
// counted, and once it exceeds a budget proportional to the bytes scanned the
// search resumes with the linear-time two-way algorithm. Every search is
// therefore linear in the haystack.
//
// Credits: the byte-frequency table and the rarest-pair rule are those of the
// Rust memchr crate (src/arch/all/packedpair/default_rank.rs, Andrew Gallant,
// MIT or Unlicense). The positional choice of three bytes is StringZilla's
// sz_locate_needle_anomalies_ (Ash Vardanian, Apache-2.0).
//
// Measured against the Rust memchr crate (memmem::find and a prebuilt
// memmem::Finder, whichever is faster per cell) on ten 1 MiB corpora, needles
// of 8 to 2048 bytes, present and absent: faster in 69 of 79 cells on an Apple
// M4 Max and in 67 of 79 on AWS Graviton 4 (Neoverse V3).

namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {
namespace {
namespace search_hybrid {

using simdutf::scalar::search::npos;

// Lower is rarer.
static constexpr uint8_t byte_rank[256] = {
     55,  52,  51,  50,  49,  48,  47,  46,  45, 103, 242,  66,  67, 229,  44,  43,
     42,  41,  40,  39,  38,  37,  36,  35,  34,  33,  56,  32,  31,  30,  29,  28,
    255, 148, 164, 149, 136, 160, 155, 173, 221, 222, 134, 122, 232, 202, 215, 224,
    208, 220, 204, 187, 183, 179, 177, 168, 178, 200, 226, 195, 154, 184, 174, 126,
    120, 191, 157, 194, 170, 189, 162, 161, 150, 193, 142, 137, 171, 176, 185, 167,
    186, 112, 175, 192, 188, 156, 140, 143, 123, 133, 128, 147, 138, 146, 114, 223,
    151, 249, 216, 238, 236, 253, 227, 218, 230, 247, 135, 180, 241, 233, 246, 244,
    231, 139, 245, 243, 251, 235, 201, 196, 240, 214, 152, 182, 205, 181, 127,  27,
    212, 211, 210, 213, 228, 197, 169, 159, 131, 172, 105,  80,  98,  96,  97,  81,
    207, 145, 116, 115, 144, 130, 153, 121, 107, 132, 109, 110, 124, 111,  82, 108,
    118, 141, 113, 129, 119, 125, 165, 117,  92, 106,  83,  72,  99,  93,  65,  79,
    166, 237, 163, 199, 190, 225, 209, 203, 198, 217, 219, 206, 234, 248, 158, 239,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};

// The two rarest bytes of the needle within its first 255 bytes, at distinct
// offsets (memchr's Pair::with_ranker).
simdutf_really_inline void rare_pair(const uint8_t *s, size_t m, size_t *r1,
                                     size_t *r2) {
  size_t i1 = 0, i2 = 1;
  uint8_t k1 = byte_rank[s[0]], k2 = byte_rank[s[1]];
  if (k2 < k1) {
    std::swap(i1, i2);
    std::swap(k1, k2);
  }
  const size_t lim = m < 255 ? m : 255;
  for (size_t i = 2; i < lim; ++i) {
    const uint8_t k = byte_rank[s[i]];
    if (k < k1) {
      i2 = i1;
      k2 = k1;
      i1 = i;
      k1 = k;
    } else if (s[i] != s[i1] && k < k2) {
      i2 = i;
      k2 = k;
    }
  }
  *r1 = i1;
  *r2 = i2;
}

// Three offsets by position: first, middle, last, moved to distinct byte
// values and, for needles over 8 bytes, away from UTF-8 lead bytes (which carry
// little information). StringZilla's sz_locate_needle_anomalies_.
simdutf_really_inline void positional_three(const uint8_t *s, size_t m,
                                            size_t *a, size_t *b, size_t *c) {
  size_t first = 0, second = m / 2, third = m - 1;
  const bool dup =
      s[first] == s[second] || s[first] == s[third] || s[second] == s[third];
  if (m > 3 && dup) {
    while (s[second] == s[first] && second + 1 < third) {
      ++second;
    }
    while ((s[third] == s[second] || s[third] == s[first]) &&
           third > second + 1) {
      --third;
    }
  }
  if (m > 8) {
    size_t vf = first, vs = second;
    const size_t vt = third;
    while ((s[vs] > 191 || s[vs] == s[vt]) && vs + 1 < vt) {
      ++vs;
    }
    if (s[vs] < 191) {
      second = vs;
    } else {
      vs = second;
    }
    while ((s[vf] > 191 || s[vf] == s[vs] || s[vf] == s[vt]) && vf + 1 < vs) {
      ++vf;
    }
    if (s[vf] < 191) {
      first = vf;
    }
  }
  *a = first;
  *b = second;
  *c = third;
}

// One bit (bit 3 of a nibble) per lane.
simdutf_really_inline uint64_t lane_bits(uint8x16_t v) {
  return vget_lane_u64(
             vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(v), 4)), 0) &
         0x8888888888888888ull;
}

// First index below len where a and b differ, or len.
simdutf_really_inline size_t mismatch(const uint8_t *a, const uint8_t *b,
                                      size_t len) {
  size_t i = 0;
  for (; i + 16 <= len; i += 16) {
    const uint8x16_t e = vceqq_u8(vld1q_u8(a + i), vld1q_u8(b + i));
    if (vminvq_u8(e) != 0xFF) {
      return i + size_t(__builtin_ctzll(lane_bits(vmvnq_u8(e)))) / 4;
    }
  }
  if (i < len) {
    if (len >= 16) { // overlapping last window
      const size_t j = len - 16;
      const uint8x16_t e = vceqq_u8(vld1q_u8(a + j), vld1q_u8(b + j));
      if (vminvq_u8(e) != 0xFF) {
        return j + size_t(__builtin_ctzll(lane_bits(vmvnq_u8(e)))) / 4;
      }
    } else {
      for (; i < len; ++i) {
        if (a[i] != b[i]) {
          return i;
        }
      }
    }
  }
  return len;
}

// Verification work budget: 64-byte compares beyond each candidate's first 64
// bytes, at most budget_per_64 per 64 positions scanned plus a slack.
static constexpr size_t budget_per_64 = 4;

struct verifier {
  const uint8_t *t;
  const uint8_t *p;
  size_t m;
  size_t charge;
  size_t slack;
  size_t found;
  size_t over;

  // Past the first 64 bytes, out of line: counted and budgeted.
  simdutf_never_inline int deep(size_t q) {
    size_t d = 64;
    while (d < m) {
      const size_t len = m - d < 64 ? m - d : 64;
      ++charge;
      if (mismatch(t + q + d, p + d, len) != len) {
        if (charge > budget_per_64 * (q >> 6) + slack) {
          over = q;
          return 2;
        }
        return 0;
      }
      d += len;
    }
    found = q;
    return 1;
  }
  // 0: no match at q; 1: match (found = q); 2: budget exhausted (over = q).
  simdutf_really_inline int candidate(size_t q) {
    const size_t head = m < 64 ? m : 64;
    if (mismatch(t + q, p, head) != head) {
      return 0;
    }
    if (m <= 64) {
      found = q;
      return 1;
    }
    return deep(q);
  }
};

simdutf_really_inline uint8x16_t eq2(const uint8_t *at, size_t a, uint8x16_t va,
                                     size_t b, uint8x16_t vb) {
  return vandq_u8(vceqq_u8(vld1q_u8(at + a), va),
                  vceqq_u8(vld1q_u8(at + b), vb));
}
simdutf_really_inline uint8x16_t eq3(const uint8_t *at, size_t a, uint8x16_t va,
                                     size_t b, uint8x16_t vb, size_t c,
                                     uint8x16_t vc) {
  return vandq_u8(vandq_u8(vceqq_u8(vld1q_u8(at + a), va),
                           vceqq_u8(vld1q_u8(at + b), vb)),
                  vceqq_u8(vld1q_u8(at + c), vc));
}

// Requires m >= 4 and n >= m + 64.
inline size_t find_long(const char *text, size_t n, const char *needle,
                        size_t m) {
  const uint8_t *t = reinterpret_cast<const uint8_t *>(text);
  const uint8_t *P = reinterpret_cast<const uint8_t *>(needle);
  size_t o0, o1, o2, r1, r2;
  positional_three(P, m, &o0, &o1, &o2);
  rare_pair(P, m, &r1, &r2);
  const uint8x16_t v0 = vdupq_n_u8(P[o0]), v1 = vdupq_n_u8(P[o1]),
                   v2 = vdupq_n_u8(P[o2]);
  const uint8x16_t w1 = vdupq_n_u8(P[r1]), w2 = vdupq_n_u8(P[r2]);
  verifier V{t, P, m, 0, (m >> 3) + 64, 0, 0};
  const size_t last = n - m; // last candidate position
  // A survivor of the pair costs about this many compares (Needle-Hammer's
  // escalation rule).
  const size_t cost = 20 + m / 32;
  int r = 0;
  size_t i = 0;
  auto survivors = [&](size_t at, uint64_t w) -> int {
    for (; w; w &= w - 1) {
      if (int c = V.candidate(at + size_t(__builtin_ctzll(w)) / 4)) {
        return c;
      }
    }
    return 0;
  };
  // Phase 1: the rare pair, 64 positions per step, survivors counted.
  size_t surv = 0;
  bool three = false;
  for (; i + 63 <= last; i += 64) {
    const uint8x16_t a = eq2(t + i, r1, w1, r2, w2);
    const uint8x16_t b = eq2(t + i + 16, r1, w1, r2, w2);
    const uint8x16_t c = eq2(t + i + 32, r1, w1, r2, w2);
    const uint8x16_t d = eq2(t + i + 48, r1, w1, r2, w2);
    if (simdutf_unlikely(vmaxvq_u8(vorrq_u8(vorrq_u8(a, b), vorrq_u8(c, d))) !=
                         0)) {
      const uint64_t na = lane_bits(a), nb = lane_bits(b), nc = lane_bits(c),
                     nd = lane_bits(d);
      surv += size_t(__builtin_popcountll(na) + __builtin_popcountll(nb) +
                     __builtin_popcountll(nc) + __builtin_popcountll(nd));
      if (surv * cost > (i >> 7) + 128) {
        three = true; // this block is redone with three bytes
        break;
      }
      if ((r = survivors(i, na)) || (r = survivors(i + 16, nb)) ||
          (r = survivors(i + 32, nc)) || (r = survivors(i + 48, nd))) {
        break;
      }
    }
  }
  // Phase 2: three positional bytes.
  if (!r && three) {
    for (; i + 63 <= last; i += 64) {
      const uint8x16_t a = eq3(t + i, o0, v0, o1, v1, o2, v2);
      const uint8x16_t b = eq3(t + i + 16, o0, v0, o1, v1, o2, v2);
      const uint8x16_t c = eq3(t + i + 32, o0, v0, o1, v1, o2, v2);
      const uint8x16_t d = eq3(t + i + 48, o0, v0, o1, v1, o2, v2);
      if (simdutf_unlikely(
              vmaxvq_u8(vorrq_u8(vorrq_u8(a, b), vorrq_u8(c, d))) != 0)) {
        if ((r = survivors(i, lane_bits(a))) ||
            (r = survivors(i + 16, lane_bits(b))) ||
            (r = survivors(i + 32, lane_bits(c))) ||
            (r = survivors(i + 48, lane_bits(d)))) {
          break;
        }
      }
    }
  }
  if (!r) {
    for (; i + 15 <= last; i += 16) {
      if (const uint64_t w = lane_bits(eq3(t + i, o0, v0, o1, v1, o2, v2))) {
        if ((r = survivors(i, w))) {
          break;
        }
      }
    }
  }
  if (!r) {
    for (; i <= last; ++i) {
      if (t[i + o0] == P[o0] && t[i + o1] == P[o1] && t[i + o2] == P[o2]) {
        if ((r = V.candidate(i))) {
          break;
        }
      }
    }
  }
  if (r == 1) {
    return V.found;
  }
  if (r == 2) {
    return simdutf::scalar::search::twoway(text, n, needle, m, V.over);
  }
  return npos;
}

template <class ops>
simdutf_really_inline const char *
search_range(const char *start, const char *end, const char *needle_start,
             const char *needle_end) {
  const size_t n = start < end ? size_t(end - start) : 0;
  const size_t m =
      needle_start < needle_end ? size_t(needle_end - needle_start) : 0;
  size_t r;
  if (m <= 3 || n < m + 64) {
    // Short needles need no verification and tiny haystacks are bounded:
    // the Needle-Hammer kernel handles both.
    r = search::needle_hammer<ops>::find(start, n, needle_start, m);
  } else {
    r = find_long(start, n, needle_start, m);
  }
  return r == npos ? end : start + r;
}

} // namespace search_hybrid
} // unnamed namespace
} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf
