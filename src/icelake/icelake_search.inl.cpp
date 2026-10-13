// Needle-Hammer substring search, AVX-512 (BW). A window is 64 candidate
// positions (one mask register), a block four windows, 256 positions. The
// ends of the haystack are covered by masked loads, so no byte outside the
// haystack or the needle is read. See include/simdutf/scalar/search.h for the
// parts shared with the other kernels (anchors, escalation, two-way).
namespace search {

using scalar::search::anchors;
using scalar::search::npos;
using scalar::search::result;
using scalar::search::survivor_cost;
// Narrowing rounds allowed before the two-way search takes over: n/64. A
// round is a broadcast and four compares over a 256-byte block.
constexpr size_t budget_den = 64;
// Below this needle length the guard is absent. A block admits at most m - 2
// rounds, so the block loop admits n(m - 2)/256 rounds, about n/7.5 at
// m = 36; the masked windows admit at most m rounds per 64 positions. Both
// are bounded by construction and cheaper than the two-way search on the
// short periodic needles that would trip a guard.
constexpr size_t free_below = 36;
// Haystacks shorter than this take masked windows rather than blocks.
constexpr size_t min_wide = 1024;
// How far ahead of the current block the block loop prefetches: the filter
// rejects almost every block without a branch, and from memory the demand
// misses alone do not keep enough lines in flight.
constexpr size_t prefetch_ahead = 4096;

simdutf_really_inline __mmask64 low_lanes(size_t k) {
  return k >= 64 ? ~__mmask64(0) : (__mmask64(1) << k) - 1;
}
simdutf_really_inline size_t lane(uint64_t mask) {
  return size_t(_tzcnt_u64(mask));
}
simdutf_really_inline size_t lanes(uint64_t mask) {
  return size_t(_mm_popcnt_u64(mask));
}
simdutf_really_inline __mmask64 match_at(const char *p, __m512i c) {
  return _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(p), c);
}

// Leading equal bytes of a[0..len) and b[0..len).
simdutf_really_inline size_t lcp(const char *a, const char *b, size_t len) {
  size_t k = 0;
  for (; k + 64 <= len; k += 64) {
    const __mmask64 ne = _mm512_cmpneq_epi8_mask(_mm512_loadu_si512(a + k),
                                                 _mm512_loadu_si512(b + k));
    if (ne) {
      return k + lane(ne);
    }
  }
  if (k < len) {
    const __mmask64 live = low_lanes(len - k);
    const __mmask64 ne =
        _mm512_cmpneq_epi8_mask(_mm512_maskz_loadu_epi8(live, a + k),
                                _mm512_maskz_loadu_epi8(live, b + k));
    if (ne) {
      return k + lane(ne);
    }
  }
  return len;
}

// Candidate positions to skip so that text + head sits on a 64-byte
// boundary, at most the number of candidate positions (n - m + 1).
simdutf_really_inline size_t align_head(const char *text, size_t n, size_t m) {
  const size_t off = reinterpret_cast<uintptr_t>(text) & 63;
  if (off == 0) {
    return 0;
  }
  const size_t head = 64 - off;
  const size_t positions = n - m + 1;
  return head < positions ? head : positions;
}

// The first `count` (< 64) candidate positions, one masked window narrowed
// through the whole needle.
simdutf_really_inline size_t masked_head(const char *text, const char *needle,
                                         size_t m, size_t count) {
  __mmask64 f = (__mmask64(1) << count) - 1;
  for (size_t k = 0; k < m && f != 0; ++k) {
    f = _mm512_mask_cmpeq_epi8_mask(f, _mm512_maskz_loadu_epi8(f, text + k),
                                    _mm512_set1_epi8(needle[k]));
  }
  return f != 0 ? lane(f) : npos;
}

// Anchor selection for m >= 5 (Needle-Hammer, AVX-512 variant). If few
// needle bytes (at most max(1, m/8)) are not one of the four positional
// values, the first of them is rare in the needle and replaces the quarter
// point, with all four anchors in use. Otherwise the search starts with the
// first and the last byte; the middle and the quarter point are the spares,
// which the escalation adds when survivors are frequent.
simdutf_really_inline anchors select(const char *needle, size_t m) {
  const uint8_t *s = reinterpret_cast<const uint8_t *>(needle);
  anchors a;
  scalar::search::positional(s, m, a.o);
  const __m512i v0 = _mm512_set1_epi8(char(s[a.o[0]]));
  const __m512i v1 = _mm512_set1_epi8(char(s[a.o[1]]));
  const __m512i v2 = _mm512_set1_epi8(char(s[a.o[2]]));
  const __m512i v3 = _mm512_set1_epi8(char(s[a.o[3]]));
  const size_t limit = m / 8 > 1 ? m / 8 : 1;
  size_t outside = 0, first_out = m;
  for (size_t k = 0; k < m; k += 64) {
    const __mmask64 live = low_lanes(m - k);
    const __m512i x = _mm512_maskz_loadu_epi8(live, needle + k);
    const __mmask64 in =
        _mm512_cmpeq_epi8_mask(x, v0) | _mm512_cmpeq_epi8_mask(x, v1) |
        _mm512_cmpeq_epi8_mask(x, v2) | _mm512_cmpeq_epi8_mask(x, v3);
    const __mmask64 out = live & ~in;
    if (out) {
      if (first_out == m) {
        first_out = k + lane(out);
      }
      outside += lanes(out);
      if (outside > limit) {
        break;
      }
    }
  }
  if (outside != 0 && outside <= limit) {
    a.o[1] = first_out; // a byte rare in the needle: the best anchor there is
    a.k = 4;
  } else {
    scalar::search::first_and_last(a);
  }
  scalar::search::sort_anchors(a.o, a.k);
  return a;
}

// Masked windows over [first, last_exclusive): 64 candidate positions per
// step, every load masked to the live lanes, the anchors tested first and the
// other needle bytes in order. Used for the alignment head, the remainder
// past the block loop and whole small haystacks. Candidate p is live only
// while p + m <= n, so every load stays inside the haystack. With fewer than
// four anchors the survivors are counted and the escalation rule applies as
// in the block loop, `allowance` standing for the blocks scanned so far.
template <bool guarded, int K>
simdutf_really_inline result masked_windows(
    const char *text, const char *needle, size_t m, size_t first,
    size_t last_exclusive, const size_t *o, const __m512i *pv, size_t budget,
    size_t rounds, size_t survivors, size_t allowance) {
  for (size_t i = first; i < last_exclusive; i += 64) {
    const __mmask64 active = low_lanes(last_exclusive - i);
    // Every load is masked by `active`, not by the live mask f: a load whose
    // mask depends on the previous compare makes each round wait for the one
    // before, and an adversarial input narrows through the whole needle.
    __mmask64 f = active;
    for (int k = 0; k < K && f != 0; ++k) {
      f &= _mm512_mask_cmpeq_epi8_mask(
          active, _mm512_maskz_loadu_epi8(active, text + i + o[k]), pv[k]);
    }
    if (K < 4 && f != 0) {
      survivors += lanes(f);
      if (survivors * survivor_cost(m) > allowance) {
        return scalar::search::stop(2, i, rounds, survivors);
      }
    }
    // the other needle bytes, as the segments between the sorted anchors
    size_t k = 0;
    for (int q = 0; q <= K && f != 0; ++q) {
      const size_t end = (q < K) ? o[q] : m;
      for (; k < end && f != 0; ++k) {
        f &= _mm512_mask_cmpeq_epi8_mask(
            active, _mm512_maskz_loadu_epi8(active, text + i + k),
            _mm512_set1_epi8(needle[k]));
        if (guarded && ++rounds > budget) {
          return scalar::search::stop(1, i, rounds, survivors);
        }
      }
      k = end + 1;
    }
    if (f != 0) {
      return scalar::search::match(i + lane(f), rounds);
    }
  }
  return scalar::search::done(rounds, survivors);
}

// The block loop: 256 positions per iteration as four 64-byte chunks; each
// chunk loads the haystack at the chunk base plus each anchor offset and ANDs
// the compares. A lone survivor in the block is verified in place; several
// are narrowed together, one needle byte per round, the rounds counted when
// guarded. With K < 4 the survivors are also counted and the pass returns
// state 2 when they cost more than another anchor would. Out of line so that
// the broadcasts stay in registers.
template <bool guarded, int K>
simdutf_never_inline result wide(const char *text, size_t n, const char *needle,
                                 size_t m, const anchors &a, size_t budget,
                                 size_t rounds, size_t start) {
  static_assert(K >= 2 && K <= 4, "two to four anchors");
  const bool fits = (m <= 64);
  const __mmask64 nmask = fits ? low_lanes(m) : 0;
  const __m512i nvec = _mm512_maskz_loadu_epi8(nmask, needle);
  const size_t o0 = a.o[0], o1 = a.o[1], o2 = a.o[K > 2 ? 2 : 1],
               o3 = a.o[K - 1];
  const __m512i p0 = _mm512_set1_epi8(needle[o0]);
  const __m512i p1 = _mm512_set1_epi8(needle[o1]);
  const __m512i p2 = _mm512_set1_epi8(needle[o2]);
  const __m512i p3 = _mm512_set1_epi8(needle[o3]);
  const size_t oo[4] = {o0, o1, o2, o3};
  const __m512i pv[4] = {p0, p1, p2, p3};
  // One base pointer per anchor, indexed by the block position: a single
  // induction variable and base+index addressing.
  const char *const t0 = text + o0;
  const char *const t1 = text + o1;
  const char *const t2 = text + o2;
  const char *const t3 = text + o3;
  size_t i = start;
  size_t survivors = 0;
  if (start == 0) {
    // Walk to the position where the loads of the first anchor are aligned:
    // a quarter of the loads (half with two anchors) then never split a
    // cache line. One masked window covers the positions skipped.
    const size_t head = align_head(t0, n - o0, m - o0);
    if (head) {
      const result r = masked_windows<guarded, K>(text, needle, m, 0, head, oo,
                                                  pv, budget, rounds, 0, 128);
      if (r.found || r.state != 0) {
        return r;
      }
      rounds = r.rounds;
      survivors = r.survivors;
      i = head;
    }
  }
  size_t kill = SIZE_MAX; // needle offset where the last survivors failed
  // Blocks scanned so far are (i - first_block) / 256; a counter
  // incremented every block was spilled to the stack by the compiler.
  const size_t first_block = i;
#define SIMDUTF_SEARCH_CHUNK(OFF)                                              \
  (K == 4   ? ((match_at(t0 + i + (OFF), p0) & match_at(t1 + i + (OFF), p1)) & \
             (match_at(t2 + i + (OFF), p2) & match_at(t3 + i + (OFF), p3)))  \
   : K == 3 ? ((match_at(t0 + i + (OFF), p0) & match_at(t1 + i + (OFF), p1)) & \
               match_at(t2 + i + (OFF), p2))                                   \
            : (match_at(t0 + i + (OFF), p0) & match_at(t1 + i + (OFF), p1)))
  if (n >= m + 255) {
    const size_t last_block = n - m - 255;
    for (; i <= last_block; i += 256) {
      _mm_prefetch(text + i + prefetch_ahead, _MM_HINT_T0);
      _mm_prefetch(text + i + prefetch_ahead + 64, _MM_HINT_T0);
      _mm_prefetch(text + i + prefetch_ahead + 128, _MM_HINT_T0);
      _mm_prefetch(text + i + prefetch_ahead + 192, _MM_HINT_T0);
      __mmask64 fA = SIMDUTF_SEARCH_CHUNK(0), fB = SIMDUTF_SEARCH_CHUNK(64);
      __mmask64 fC = SIMDUTF_SEARCH_CHUNK(128), fD = SIMDUTF_SEARCH_CHUNK(192);
      if ((fA | fB | fC | fD) == 0) {
        continue;
      }
      if (K < 4) {
        survivors += lanes(fA) + lanes(fB) + lanes(fC) + lanes(fD);
        // each anchor dropped saves about two compares per block
        if (survivors * survivor_cost(m) > ((i - first_block) >> 7) + 128) {
          return scalar::search::stop(2, i, rounds, survivors, kill);
        }
      }
      const int nz = (fA != 0) + (fB != 0) + (fC != 0) + (fD != 0);
      const __mmask64 one = fA | fB | fC | fD;
      if (nz == 1 && (one & (one - 1)) == 0) {
        const size_t off = (fA != 0)   ? 0
                           : (fB != 0) ? 64
                           : (fC != 0) ? 128
                                       : 192;
        const size_t b = off + lane(one);
        const __mmask64 ne =
            fits
                ? _mm512_mask_cmpneq_epi8_mask(
                      nmask, _mm512_maskz_loadu_epi8(nmask, text + i + b), nvec)
                : _mm512_cmpneq_epi8_mask(_mm512_loadu_si512(text + i + b),
                                          _mm512_loadu_si512(needle));
        if (ne != 0) {
          kill = lane(ne);
        } else if (fits) {
          return scalar::search::match(i + b, rounds);
        } else {
          // The first 64 bytes match: compare the rest and charge the bytes
          // compared to the budget, one round per 256. Without the charge a
          // near-match placed once per block would cost m bytes per block
          // that no counter sees.
          const size_t k = 64 + lcp(text + i + b + 64, needle + 64, m - 64);
          if (k == m) {
            return scalar::search::match(i + b, rounds);
          }
          kill = k;
          if (guarded) {
            rounds += (k + 255) >> 8;
            if (rounds > budget) {
              return scalar::search::stop(1, i + 256, rounds, survivors);
            }
          }
        }
        continue;
      }
      size_t last_k = SIZE_MAX;
      for (size_t k = 0; k < m; ++k) {
        if (k == o0 || k == o1 || k == o2 || k == o3) {
          continue;
        }
        if ((fA | fB | fC | fD) == 0) {
          kill = last_k;
          break;
        }
        last_k = k;
        const __m512i pk = _mm512_set1_epi8(needle[k]);
        fA = _mm512_mask_cmpeq_epi8_mask(fA, _mm512_loadu_si512(text + i + k),
                                         pk);
        fB = _mm512_mask_cmpeq_epi8_mask(
            fB, _mm512_loadu_si512(text + i + k + 64), pk);
        fC = _mm512_mask_cmpeq_epi8_mask(
            fC, _mm512_loadu_si512(text + i + k + 128), pk);
        fD = _mm512_mask_cmpeq_epi8_mask(
            fD, _mm512_loadu_si512(text + i + k + 192), pk);
        // A give-up resumes at the start of the block, whose positions are
        // not all ruled out.
        if (guarded && ++rounds > budget) {
          return scalar::search::stop(1, i, rounds, survivors);
        }
      }
      if (fA) {
        return scalar::search::match(i + lane(fA), rounds);
      }
      if (fB) {
        return scalar::search::match(i + 64 + lane(fB), rounds);
      }
      if (fC) {
        return scalar::search::match(i + 128 + lane(fC), rounds);
      }
      if (fD) {
        return scalar::search::match(i + 192 + lane(fD), rounds);
      }
    }
  }
#undef SIMDUTF_SEARCH_CHUNK
  return masked_windows<guarded, K>(text, needle, m, i, n - m + 1, oo, pv,
                                    budget, rounds, survivors,
                                    ((i - first_block) >> 7) + 128);
}

// Four-byte needles: the anchors are the whole needle, at offsets 0 to 3, so
// a survivor is a match.
simdutf_really_inline size_t four(const char *text, size_t n,
                                  const char *needle) {
  const __m512i p0 = _mm512_set1_epi8(needle[0]);
  const __m512i p1 = _mm512_set1_epi8(needle[1]);
  const __m512i p2 = _mm512_set1_epi8(needle[2]);
  const __m512i p3 = _mm512_set1_epi8(needle[3]);
  size_t i = 0;
  if (n >= 4 + 255) {
    const size_t head = align_head(text, n, 4);
    if (head) {
      const size_t r = masked_head(text, needle, 4, head);
      if (r != npos) {
        return r;
      }
      i = head;
    }
#define SIMDUTF_SEARCH_FOUR(OFF)                                               \
  ((match_at(text + i + (OFF), p0) & match_at(text + i + (OFF) + 1, p1)) &     \
   (match_at(text + i + (OFF) + 2, p2) & match_at(text + i + (OFF) + 3, p3)))
    for (; i + 4 + 255 <= n; i += 256) {
      const __mmask64 fA = SIMDUTF_SEARCH_FOUR(0), fB = SIMDUTF_SEARCH_FOUR(64);
      const __mmask64 fC = SIMDUTF_SEARCH_FOUR(128),
                      fD = SIMDUTF_SEARCH_FOUR(192);
      if ((fA | fB | fC | fD) == 0) {
        continue;
      }
      if (fA) {
        return i + lane(fA);
      }
      if (fB) {
        return i + 64 + lane(fB);
      }
      if (fC) {
        return i + 128 + lane(fC);
      }
      return i + 192 + lane(fD);
    }
#undef SIMDUTF_SEARCH_FOUR
  }
  for (; i + 4 <= n; i += 64) {
    __mmask64 f = low_lanes(n - 4 - i + 1);
    for (size_t k = 0; k < 4 && f != 0; ++k) {
      f = _mm512_mask_cmpeq_epi8_mask(
          f, _mm512_maskz_loadu_epi8(f, text + i + k), _mm512_set1_epi8(needle[k]));
    }
    if (f != 0) {
      return i + lane(f);
    }
  }
  return npos;
}

// Needles of one to three bytes: the filter is the whole needle, so a
// survivor is a match.
template <int M>
simdutf_really_inline size_t short_search(const char *text, size_t n,
                                          const char *needle) {
  static_assert(M >= 1 && M <= 3, "short needles only");
  if (M == 1) {
    const __m512i v = _mm512_set1_epi8(needle[0]);
    size_t i = 0;
    // 256 bytes per iteration with plain loads; masked loads only for the
    // last 255 bytes or fewer
    for (; i + 256 <= n; i += 256) {
      const __mmask64 fA = match_at(text + i, v), fB = match_at(text + i + 64, v);
      const __mmask64 fC = match_at(text + i + 128, v),
                      fD = match_at(text + i + 192, v);
      if ((fA | fB | fC | fD) == 0) {
        continue;
      }
      if (fA) {
        return i + lane(fA);
      }
      if (fB) {
        return i + 64 + lane(fB);
      }
      if (fC) {
        return i + 128 + lane(fC);
      }
      return i + 192 + lane(fD);
    }
    for (; i < n; i += 64) {
      const __mmask64 active = low_lanes(n - i);
      const __mmask64 eq = _mm512_mask_cmpeq_epi8_mask(
          active, _mm512_maskz_loadu_epi8(active, text + i), v);
      if (eq) {
        return i + lane(eq);
      }
    }
    return npos;
  }
  const size_t m = M;
  if (n < m) {
    return npos;
  }
  size_t i = 0;
  const __m512i p0 = _mm512_set1_epi8(needle[0]);
  const __m512i p1 = _mm512_set1_epi8(needle[1]);
  const __m512i p2 = _mm512_set1_epi8(needle[M > 2 ? 2 : 1]);
#define SIMDUTF_SEARCH_SHORT(OFF)                                              \
  (M > 2 ? (match_at(text + i + (OFF), p0) &                                   \
            match_at(text + i + (OFF) + 1, p1) &                               \
            match_at(text + i + (OFF) + 2, p2))                                \
         : (match_at(text + i + (OFF), p0) &                                   \
            match_at(text + i + (OFF) + 1, p1)))
  if (n >= m + 255) {
    // Most short-needle searches on text end within the first 64 bytes:
    // look there first with one unaligned window, then align.
    {
      const __mmask64 f = SIMDUTF_SEARCH_SHORT(0);
      if (f) {
        return lane(f);
      }
      i = 64;
    }
    const size_t head = align_head(text + i, n - i, m);
    if (head) {
      const size_t r = masked_head(text + i, needle, m, head);
      if (r != npos) {
        return i + r;
      }
      i += head;
    }
    for (; i + m + 255 <= n; i += 256) {
      const __mmask64 fA = SIMDUTF_SEARCH_SHORT(0),
                      fB = SIMDUTF_SEARCH_SHORT(64);
      const __mmask64 fC = SIMDUTF_SEARCH_SHORT(128),
                      fD = SIMDUTF_SEARCH_SHORT(192);
      if ((fA | fB | fC | fD) == 0) {
        continue;
      }
      if (fA) {
        return i + lane(fA);
      }
      if (fB) {
        return i + 64 + lane(fB);
      }
      if (fC) {
        return i + 128 + lane(fC);
      }
      return i + 192 + lane(fD);
    }
  }
#undef SIMDUTF_SEARCH_SHORT
  for (; i + m <= n; i += 64) {
    __mmask64 f = low_lanes(n - m - i + 1);
    for (size_t k = 0; k < m && f != 0; ++k) {
      f = _mm512_mask_cmpeq_epi8_mask(f,
                                      _mm512_maskz_loadu_epi8(f, text + i + k),
                                      _mm512_set1_epi8(needle[k]));
    }
    if (f != 0) {
      return i + lane(f);
    }
  }
  return npos;
}

template <int K>
simdutf_really_inline result run_windows(bool guarded, const char *text,
                                         size_t n, const char *needle, size_t m,
                                         const anchors &an, size_t budget,
                                         size_t carried, size_t from) {
  __m512i pv[4];
  for (int k = 0; k < K; ++k) {
    pv[k] = _mm512_set1_epi8(needle[an.o[k]]);
  }
  return guarded ? masked_windows<true, K>(text, needle, m, from, n - m + 1,
                                           an.o, pv, budget, carried, 0, 128)
                 : masked_windows<false, K>(text, needle, m, from, n - m + 1,
                                            an.o, pv, budget, carried, 0, 128);
}

// m >= 4. Out of line so that a short-needle search does not pay its
// prologue.
simdutf_never_inline size_t search_long(const char *text, size_t n,
                                        const char *needle, size_t m) {
  if (n < m) {
    return npos;
  }
#ifndef SIMDUTF_ICELAKE_FOUR_WIDE
  if (m == 4) {
    return four(text, n, needle);
  }
#endif
  const size_t budget = n / budget_den + 1;
  const bool guarded = m > free_below;
  anchors a = select(needle, m);
  if (n < min_wide || n < m + 255) {
    return scalar::search::drive(
        [&](const anchors &an, size_t carried, size_t from) -> result {
          switch (an.k) {
          case 2:
            return run_windows<2>(guarded, text, n, needle, m, an, budget,
                                  carried, from);
          case 3:
            return run_windows<3>(guarded, text, n, needle, m, an, budget,
                                  carried, from);
          default:
            return run_windows<4>(guarded, text, n, needle, m, an, budget,
                                  carried, from);
          }
        },
        a, text, n, needle, m);
  }
  return scalar::search::drive(
      [&](const anchors &an, size_t carried, size_t from) -> result {
        switch (an.k) {
        case 2:
          return guarded ? wide<true, 2>(text, n, needle, m, an, budget,
                                         carried, from)
                         : wide<false, 2>(text, n, needle, m, an, budget,
                                          carried, from);
        case 3:
          return guarded ? wide<true, 3>(text, n, needle, m, an, budget,
                                         carried, from)
                         : wide<false, 3>(text, n, needle, m, an, budget,
                                          carried, from);
        default:
          return guarded ? wide<true, 4>(text, n, needle, m, an, budget,
                                         carried, from)
                         : wide<false, 4>(text, n, needle, m, an, budget,
                                          carried, from);
        }
      },
      a, text, n, needle, m);
}

simdutf_really_inline const char *search_range(const char *start,
                                               const char *end,
                                               const char *needle_start,
                                               const char *needle_end) {
  const size_t n = start < end ? size_t(end - start) : 0;
  const size_t m =
      needle_start < needle_end ? size_t(needle_end - needle_start) : 0;
  size_t r;
  switch (m) {
  case 0:
    return start;
  case 1:
    r = short_search<1>(start, n, needle_start);
    break;
  case 2:
    r = short_search<2>(start, n, needle_start);
    break;
  case 3:
    r = short_search<3>(start, n, needle_start);
    break;
  default:
    r = search_long(start, n, needle_start, m);
  }
  return r == npos ? end : start + r;
}

} // namespace search
