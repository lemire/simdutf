#ifndef SIMDUTF_SEARCH_H
#define SIMDUTF_SEARCH_H

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace simdutf {
namespace scalar {
namespace {
namespace search {

// Returned by the search routines when the needle does not occur.
constexpr size_t npos = SIZE_MAX;

// Critical factorization of the Crochemore-Perrin two-way algorithm: split the
// needle into needle[0..crit) . needle[crit..m) at a point where the local
// period equals the global period. With an ordered alphabet one of the two
// maximal suffixes (under the byte order and under its reverse) starts at such
// a point, so both are computed and the one that starts later is kept. Returns
// crit and sets *period to the period of the right half.
//
// max_suffix is the index of the last byte of the left half, or SIZE_MAX when
// the left half is empty; needle[max_suffix + k] then wraps to needle[k - 1],
// which is the byte the algorithm wants (unsigned wraparound is deliberate).
inline size_t critical_factorization(const uint8_t *needle, size_t m,
                                     size_t *period) {
  if (m < 3) {
    *period = 1;
    return m - 1;
  }
  size_t max_suffix = SIZE_MAX, j = 0, k = 1, p = 1;
  while (j + k < m) {
    const uint8_t a = needle[j + k], b = needle[max_suffix + k];
    if (a < b) {
      j += k;
      k = 1;
      p = j - max_suffix;
    } else if (a == b) {
      if (k != p) {
        ++k;
      } else {
        j += p;
        k = 1;
      }
    } else {
      max_suffix = j++;
      k = p = 1;
    }
  }
  *period = p;
  size_t max_suffix_rev = SIZE_MAX;
  j = 0;
  k = p = 1;
  while (j + k < m) {
    const uint8_t a = needle[j + k], b = needle[max_suffix_rev + k];
    if (b < a) {
      j += k;
      k = 1;
      p = j - max_suffix_rev;
    } else if (a == b) {
      if (k != p) {
        ++k;
      } else {
        j += p;
        k = 1;
      }
    } else {
      max_suffix_rev = j++;
      k = p = 1;
    }
  }
  // Keep the factorization with the shorter left half (+1 maps "last byte of
  // the left half" to "first byte of the right half", and SIZE_MAX to 0).
  if (max_suffix_rev + 1 < max_suffix + 1) {
    return max_suffix + 1;
  }
  *period = p;
  return max_suffix_rev + 1;
}

// Two-way search for needle[0..m) in text[from..n), m >= 1. Linear time and
// constant space: every byte of the text is compared a bounded number of
// times whatever the input. Returns the index of the first occurrence at or
// after `from`, or npos.
inline size_t twoway(const char *text, size_t n, const char *needle, size_t m,
                     size_t from) noexcept {
  if (m == 0) {
    return from <= n ? from : npos;
  }
  if (n < m || from > n - m) {
    return npos;
  }
  const uint8_t *hay = reinterpret_cast<const uint8_t *>(text);
  const uint8_t *nd = reinterpret_cast<const uint8_t *>(needle);
  size_t period;
  const size_t crit = critical_factorization(nd, m, &period);
  // The needle is periodic iff its left half repeats the period; otherwise no
  // two occurrences overlap that much and a mismatch in the left half allows
  // a shift past the larger half.
  const bool periodic = std::memcmp(nd, nd + period, crit) == 0;
  if (!periodic) {
    period = (crit > m - crit ? crit : m - crit) + 1;
  }
  const size_t last = n - m;
  size_t j = from, memory = 0;
  while (j <= last) {
    size_t i = memory > crit ? memory : crit;
    while (i < m && nd[i] == hay[j + i]) {
      ++i;
    }
    if (i < m) {
      j += i - crit + 1;
      memory = 0;
      continue;
    }
    size_t k = crit;
    while (k > memory && nd[k - 1] == hay[j + k - 1]) {
      --k;
    }
    if (k <= memory) {
      return j;
    }
    j += period;
    memory = periodic ? m - period : 0;
  }
  return npos;
}

// ---------------------------------------------------------------------------
// The parts of the Needle-Hammer search that do not depend on the register
// width, shared by the SIMD kernels (src/generic/search.h and the AVX-512
// kernel).
//
// Needle-Hammer filters every candidate position on two to four needle bytes
// (the anchors), a block of positions per iteration, then narrows the
// survivors one needle byte at a time. It starts with two anchors and adds a
// third and a fourth when survivors turn out to be frequent. Narrowing rounds
// are counted, and once they exceed a budget proportional to the haystack the
// search resumes with the linear-time two-way above, so that every search
// runs in O(n + m) time.
// ---------------------------------------------------------------------------

struct anchors {
  size_t o[4]; // offsets into the needle, ascending; unused slots are spares
  int k;       // anchors in use: 2, 3 or 4
};

// Positional anchors: first, quarter, middle and last byte, each walked
// inward (at most 64 steps) so that the four byte values differ where
// possible. The middle anchor moves only while it equals the first byte;
// the last moves while it equals the first or the middle; the quarter point
// moves while it equals the first byte.
inline void positional(const uint8_t *s, size_t m, size_t o[4]) {
  o[0] = 0;
  o[1] = m / 4;
  o[2] = m / 2;
  o[3] = m - 1;
  for (int w = 64; w-- && s[o[1]] == s[o[0]] && o[1] + 1 < o[2];) {
    ++o[1];
  }
  for (int w = 64; w-- && s[o[2]] == s[o[0]] && o[2] + 1 < o[3];) {
    ++o[2];
  }
  for (int w = 64;
       w-- && (s[o[3]] == s[o[2]] || s[o[3]] == s[o[0]]) && o[3] > o[2] + 1;) {
    --o[3];
  }
}

// The start of a needle with a wide alphabet: the first and the last
// positional anchors; the middle and then the quarter point are the spares,
// in the order the escalation adds them.
inline void first_and_last(anchors &a) {
  const size_t quarter = a.o[1], middle = a.o[2];
  a.o[1] = a.o[3];
  a.o[2] = middle;
  a.o[3] = quarter;
  a.k = 2;
}

// Cost of a surviving position, in units of one filter compare, for the
// rule that adds an anchor: a mispredicted branch plus, for longer needles,
// the restart of load streams far into the window.
inline size_t survivor_cost(size_t m) {
#ifdef SIMDUTF_SEARCH_COST_CAP
  return 20 + (m < 256 ? m : 256) / 32;
#else
  return 20 + m / 32;
#endif
}

// Sorts the first k anchor offsets (k <= 4).
inline void sort_anchors(size_t *o, int k) {
  for (int i = 1; i < k; ++i) {
    const size_t v = o[i];
    int j = i;
    while (j > 0 && o[j - 1] > v) {
      o[j] = o[j - 1];
      --j;
    }
    o[j] = v;
  }
}

// What a pass of a kernel reports back to the driver.
struct result {
  bool found;
  size_t index;
  int state;              // 0 done; 1 budget exhausted; 2 add an anchor
  size_t resume;          // first position not yet ruled out (states 1 and 2)
  size_t rounds;          // narrowing rounds counted so far
  size_t survivors;       // positions that passed the anchors so far (k < 4)
  size_t hint = SIZE_MAX; // state 2: a needle offset where survivors failed
};

inline result done(size_t rounds, size_t survivors) {
  return {false, 0, 0, 0, rounds, survivors};
}
inline result match(size_t index, size_t rounds) {
  return {true, index, 0, 0, rounds, 0};
}
inline result stop(int state, size_t resume, size_t rounds, size_t survivors,
                   size_t hint = SIZE_MAX) {
  return {false, 0, state, resume, rounds, survivors, hint};
}

// The escalation driver; it updates `an` in place (a copy of the freshly
// written anchors as one 32-byte load stalls on the scalar stores). `run(anchors, carried_rounds, from)` is one pass of
// a kernel with the given anchors from position `from`. When survivors are
// too frequent (state 2), the next anchor joins and the search continues from
// where the pass stopped, carrying the rounds already spent. When the budget
// is spent with fewer than four anchors (state 1), the search continues with
// all four and a fresh budget. Out of budget with four anchors, the two-way
// search takes over from the first position not ruled out.
template <class Run>
simdutf_really_inline size_t drive(Run run, anchors &an, const char *text,
                                   size_t n, const char *needle, size_t m) {
  result r = run(an, 0, 0);
  while (r.state != 0 && an.k < 4) {
    if (r.state == 1) {
      an.k = 4;
      sort_anchors(an.o, 4);
      r = run(an, 0, r.resume);
    } else {
#ifdef SIMDUTF_SEARCH_HINT
      // EXPERIMENT (not Needle-Hammer): the next anchor is the needle offset
      // where survivors failed, when the pass reports one.
      if (r.hint < m) {
        int j = 0;
        while (j < 4 && an.o[j] != r.hint) {
          ++j;
        }
        if (j == 4) {
          an.o[an.k] = r.hint;
        } else if (j > an.k) {
          const size_t t = an.o[an.k];
          an.o[an.k] = an.o[j];
          an.o[j] = t;
        }
      }
#endif
      ++an.k;
      sort_anchors(an.o, an.k);
      r = run(an, r.rounds, r.resume);
    }
  }
  if (r.state == 0) {
    return r.found ? r.index : npos;
  }
  return twoway(text, n, needle, m, r.resume);
}

} // namespace search
} // unnamed namespace
} // namespace scalar
} // namespace simdutf

#endif // SIMDUTF_SEARCH_H
