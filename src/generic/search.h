// Needle-Hammer substring search, written once for the kernels whose
// registers hold 16 or 32 bytes. Each kernel provides a `search_ops` struct
// (see e.g. src/arm64/arm_search.cpp) with:
//
//   vec, bvec           a data register and a compare result (may be the same)
//   W                   bytes per register (16 or 32)
//   skip                (experiment) blocks tested at once while none survives
//   lane_shift          log2 of the bits per lane in mask(): 0, or 2 on NEON
//   all_lanes           mask() of a compare result true in every lane
//   splat(c), load(p)   broadcast, unaligned load
//   load_aligned(p)     load from a W-aligned address
//   eq(a, b)            byte compare
//   and_(x, y), or_(x, y), any(x), mask(x)
//
// A block is four registers, 4W candidate positions. See
// include/simdutf/scalar/search.h for the parts that do not depend on the
// width (anchors, escalation, the two-way fallback).
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {
namespace {
namespace search {

using scalar::search::anchors;
using scalar::search::npos;
using scalar::search::result;
using scalar::search::survivor_cost;

template <class ops> struct needle_hammer {
  using vec = typename ops::vec;
  using bvec = typename ops::bvec;
  static constexpr size_t W = ops::W;
  static constexpr size_t B = 4 * W;
  static constexpr int block_shift = (W == 16) ? 6 : 7;
  static_assert(B == (size_t(1) << block_shift), "B must be 64 or 128");
  // Guard budget: narrowing rounds allowed before the two-way search takes
  // over, as a fraction of the haystack. A round is a broadcast and four
  // compares over one block.
  static constexpr size_t budget_den = B / 4;
  // Below this needle length the guard is absent: a block admits at most
  // m - 2 rounds, so the block loop admits n(m - 2)/B rounds, a small
  // multiple of the filter's own work.
  static constexpr size_t free_below = B / 8;
  // Haystacks shorter than this take single windows rather than blocks.
  static constexpr size_t min_wide = 8 * B;

  static simdutf_really_inline size_t lane_index(uint64_t mask) {
    return size_t(trailing_zeroes(mask)) >> ops::lane_shift;
  }
  static simdutf_really_inline size_t lane_count(uint64_t mask) {
    return size_t(count_ones(mask));
  }
  static simdutf_really_inline bvec match_at(const char *p, vec c) {
    return ops::eq(ops::load(p), c);
  }

  // Leading equal bytes of a[0..len) and b[0..len).
  static size_t lcp(const char *a, const char *b, size_t len) {
    size_t k = 0;
    for (; k + W <= len; k += W) {
      const uint64_t ne =
          ~ops::mask(ops::eq(ops::load(a + k), ops::load(b + k))) &
          ops::all_lanes;
      if (ne) {
        return k + lane_index(ne);
      }
    }
    while (k < len && a[k] == b[k]) {
      ++k;
    }
    return k;
  }

  // Anchor selection for m >= 4 (Needle-Hammer, NEON variant). After the
  // positional choice, one pass over the needle counts the bytes that are
  // NOT one of the four positional values. A needle made only of those
  // values (DNA, say) starts with four anchors, or three when the values
  // are exactly two. If the other bytes are few (at most max(1, m/8)), the
  // first of them is rare in the needle, so it replaces the quarter point
  // and all four anchors are used. Otherwise the search starts with the
  // first and the last byte, and the middle and the quarter point are the
  // spares.
  static anchors select(const char *needle, size_t m) {
    const uint8_t *s = reinterpret_cast<const uint8_t *>(needle);
    anchors a;
    scalar::search::positional(s, m, a.o);
    const vec v0 = ops::splat(s[a.o[0]]), v1 = ops::splat(s[a.o[1]]);
    const vec v2 = ops::splat(s[a.o[2]]), v3 = ops::splat(s[a.o[3]]);
    const uint8_t b0 = s[a.o[0]], b1 = s[a.o[1]], b2 = s[a.o[2]],
                  b3 = s[a.o[3]];
    const size_t limit = m / 8 > 1 ? m / 8 : 1;
    size_t outside = 0, first_out = m;
    size_t k = 0;
    for (; k + W <= m; k += W) {
      const vec x = ops::load(needle + k);
      const bvec in = ops::or_(ops::or_(ops::eq(x, v0), ops::eq(x, v1)),
                               ops::or_(ops::eq(x, v2), ops::eq(x, v3)));
      const uint64_t out = ~ops::mask(in) & ops::all_lanes;
      if (out) {
        if (first_out == m) {
          first_out = k + lane_index(out);
        }
        outside += lane_count(out);
        if (outside > limit) {
          break;
        }
      }
    }
    // The last partial register, one byte at a time: copying it to a buffer
    // to load it as a vector stalls the load on the byte stores.
    for (; k < m && outside <= limit; ++k) {
      const uint8_t c = s[k];
      if (c != b0 && c != b1 && c != b2 && c != b3) {
        if (first_out == m) {
          first_out = k;
        }
        ++outside;
      }
    }
    const int distinct = 1 + (b1 != b0) + (b2 != b0 && b2 != b1) +
                         (b3 != b0 && b3 != b1 && b3 != b2);
    if (outside == 0 && distinct != 2 && m > 8) {
      a.k = 4;
    } else if (outside != 0 && outside <= limit) {
      a.o[1] = first_out;
      a.k = 4;
    } else if (outside == 0 && m > 8) {
      // two values: first, middle and last carry both; the quarter is spare
      const size_t quarter = a.o[1];
      a.o[1] = a.o[2];
      a.o[2] = a.o[3];
      a.o[3] = quarter;
      a.k = 3;
    } else {
      scalar::search::first_and_last(a);
    }
    scalar::search::sort_anchors(a.o, a.k);
    return a;
  }

  // Single windows of W positions over [first, n - m], for the remainder
  // past the block loop and for whole small haystacks. The last window
  // overlaps the one before: it starts at the last position whose loads stay
  // in bounds, and its lanes below `first` (already ruled out) are ignored.
  // With fewer than four anchors the survivors of the anchor filter are
  // counted and the escalation rule applies as in the block loop,
  // `allowance` standing for the blocks scanned so far.
  template <bool guarded, int K>
  static result windows(const char *text, size_t n, const char *needle,
                        size_t m, size_t first, const size_t *o, const vec *pv,
                        size_t budget, size_t rounds, size_t survivors,
                        size_t allowance) {
    if (n < m || first > n - m) {
      return scalar::search::done(rounds, survivors);
    }
    const size_t positions = n - m + 1;
    if (positions < W) {
      for (size_t p = first; p < positions; ++p) {
        if (std::memcmp(text + p, needle, m) == 0) {
          return scalar::search::match(p, rounds);
        }
      }
      return scalar::search::done(rounds, survivors);
    }
    const size_t last = positions - W;
    // Returns the lane mask of the window's matches, or 0. Sets `stop_state`
    // to 1 when the narrowing rounds ran out, to 2 when the survivors call
    // for another anchor; the window is then not resolved.
    int stop_state = 0;
    auto window = [&](size_t i) -> uint64_t {
      bvec f = match_at(text + i + o[0], pv[0]);
      for (int k = 1; k < K; ++k) {
        f = ops::and_(f, match_at(text + i + o[k], pv[k]));
      }
      if (!ops::any(f)) {
        return 0;
      }
      if (K < 4) {
        survivors += lane_count(ops::mask(f));
        if (survivors * survivor_cost(m) > allowance) {
          stop_state = 2;
          return 0;
        }
      }
      // The other needle bytes in order, as the segments between the sorted
      // anchor offsets.
      size_t k = 0;
      for (int q = 0; q <= K; ++q) {
        const size_t end = (q < K) ? o[q] : m;
        for (; k < end; ++k) {
          f = ops::and_(f,
                        match_at(text + i + k, ops::splat(uint8_t(needle[k]))));
          if (!ops::any(f)) {
            return 0;
          }
          if (guarded && ++rounds > budget) {
            stop_state = 1;
            return 0;
          }
        }
        k = end + 1;
      }
      return ops::mask(f);
    };
    size_t i = first;
    for (; i <= last; i += W) {
      const uint64_t r = window(i);
      if (r) {
        return scalar::search::match(i + lane_index(r), rounds);
      }
      if (stop_state) {
        return scalar::search::stop(stop_state, i, rounds, survivors);
      }
    }
    if (i < positions) {
      uint64_t r = window(last);
      if (stop_state) {
        return scalar::search::stop(stop_state, i, rounds, survivors);
      }
      while (r) {
        const size_t b = last + lane_index(r);
        if (b >= i) {
          return scalar::search::match(b, rounds);
        }
        r &= r - 1;
      }
    }
    return scalar::search::done(rounds, survivors);
  }

  // The block loop: B = 4W positions per iteration, four registers per
  // anchor. A lone survivor in a block is verified in place; several are
  // narrowed together, one needle byte per round, the rounds counted against
  // the budget when guarded. With K < 4 the survivors are also counted, and
  // the pass returns state 2 when they cost more than another anchor would.
  // Out of line so that the broadcasts stay in registers.
  template <bool guarded, int K>
  static simdutf_never_inline result wide(const char *text, size_t n,
                                          const char *needle, size_t m,
                                          const anchors &a, size_t budget,
                                          size_t rounds, size_t start) {
    static_assert(K >= 2 && K <= 4, "two to four anchors");
    // unused anchor slots repeat an active one
    const size_t o0 = a.o[0], o1 = a.o[1], o2 = a.o[K > 2 ? 2 : 1],
                 o3 = a.o[K - 1];
    const vec p0 = ops::splat(uint8_t(needle[o0]));
    const vec p1 = ops::splat(uint8_t(needle[o1]));
    const vec p2 = ops::splat(uint8_t(needle[o2]));
    const vec p3 = ops::splat(uint8_t(needle[o3]));
    const size_t oo[4] = {o0, o1, o2, o3};
    const vec pv[4] = {p0, p1, p2, p3};
    const char *const t0 = text + o0;
    const char *const t1 = text + o1;
    const char *const t2 = text + o2;
    const char *const t3 = text + o3;
    // A needle of at most W bytes is verified with one compare against a
    // register that holds it. (Building the lane mask arithmetically, or
    // loading the needle copy at each verification, made the scan 11 to 15%
    // slower for m = 4..32 on Emerald Rapids, for reasons of code layout.)
    const bool fits = (m <= W);
    char nbuf[W] = {};
    char nlive[W] = {};
    if (fits) {
      std::memcpy(nbuf, needle, m);
      std::memset(nlive, -1, m);
    }
    const vec nvec = ops::load(nbuf);
    const uint64_t nbits =
        ops::mask(ops::eq(ops::load(nlive), ops::splat(0xFF)));
    // A longer needle is compared W bytes at a time, and the bytes compared
    // are charged to the budget, one round per B: without the charge, a
    // near-match placed once per block would cost m bytes per block that no
    // counter sees.
    size_t kill = SIZE_MAX; // needle offset where the last survivors failed
    auto verify = [&](size_t pos) -> bool {
      if (fits && pos + W <= n) {
        const uint64_t ne =
            ~ops::mask(ops::eq(ops::load(text + pos), nvec)) &
            nbits;
        if (ne != 0) {
          kill = lane_index(ne);
        }
        return ne == 0;
      }
      const size_t k = lcp(text + pos, needle, m);
      if (k < m) {
        kill = k;
      }
      if (guarded) {
        rounds += (k + B - 1) >> block_shift;
      }
      return k == m;
    };
    size_t i = start;
    // Walk to the position where the loads of the first anchor are aligned
    // on W bytes, with one window over the positions skipped. Then a quarter
    // of the filter loads (half with two anchors) never split a cache line,
    // and SSE can fold them into the compares.
    if (i + m + B - 1 <= n) {
      const size_t head =
          (W - (reinterpret_cast<uintptr_t>(t0 + i) & (W - 1))) & (W - 1);
      if (head != 0) {
        bvec f = match_at(text + i + a.o[0], pv[0]);
        for (int k = 1; k < K; ++k) {
          f = ops::and_(f, match_at(text + i + a.o[k], pv[k]));
        }
        uint64_t r =
            ops::mask(f) & ((uint64_t(1) << (head << ops::lane_shift)) - 1);
        if (r != 0) {
          // lanes at and past `head` are narrowed too (their loads are in
          // bounds) and dropped from the result
          size_t k = 0;
          for (int q = 0; q <= K && ops::any(f); ++q) {
            const size_t end = (q < K) ? a.o[q] : m;
            for (; k < end && ops::any(f); ++k) {
              f = ops::and_(
                  f, match_at(text + i + k, ops::splat(uint8_t(needle[k]))));
              if (guarded && ++rounds > budget) {
                return scalar::search::stop(1, i, rounds, 0);
              }
            }
            k = end + 1;
          }
          r = ops::mask(f) & ((uint64_t(1) << (head << ops::lane_shift)) - 1);
          if (r != 0) {
            return scalar::search::match(i + lane_index(r), rounds);
          }
        }
        i += head;
      }
    }
    const size_t first_block = i;
    size_t survivors = 0;
// the loads of the first anchor are aligned (see above)
#define SIMDUTF_SEARCH_A0(OFF) ops::eq(ops::load_aligned(t0 + i + (OFF)), p0)
#define SIMDUTF_SEARCH_CHUNK(OFF)                                              \
  (K == 4   ? ops::and_(ops::and_(SIMDUTF_SEARCH_A0(OFF),                      \
                                  match_at(t1 + i + (OFF), p1)),               \
                        ops::and_(match_at(t2 + i + (OFF), p2),                \
                                  match_at(t3 + i + (OFF), p3)))               \
   : K == 3 ? ops::and_(ops::and_(SIMDUTF_SEARCH_A0(OFF),                      \
                                  match_at(t1 + i + (OFF), p1)),               \
                        match_at(t2 + i + (OFF), p2))                          \
            : ops::and_(SIMDUTF_SEARCH_A0(OFF), match_at(t1 + i + (OFF), p1)))
    for (; i + m + B - 1 <= n; i += B) {
      bvec fA = SIMDUTF_SEARCH_CHUNK(0), fB = SIMDUTF_SEARCH_CHUNK(W);
      bvec fC = SIMDUTF_SEARCH_CHUNK(2 * W), fD = SIMDUTF_SEARCH_CHUNK(3 * W);
#ifdef SIMDUTF_SEARCH_SKIP
      // EXPERIMENT (not Needle-Hammer).
      // With fewer than four anchors most blocks have no survivor: test
      // ops::skip blocks at once, and fall back to block by block only when
      // one of them has a survivor. With four anchors survivors are what the
      // input is made of (DNA, say) and the extra blocks would be wasted.
      if (K < 4 && i + m + ops::skip * B - 1 <= n) {
        bvec acc = ops::or_(ops::or_(fA, fB), ops::or_(fC, fD));
        for (size_t b = 1; b < ops::skip; ++b) {
          acc = ops::or_(
              acc, ops::or_(ops::or_(SIMDUTF_SEARCH_CHUNK(b * B),
                                     SIMDUTF_SEARCH_CHUNK(b * B + W)),
                            ops::or_(SIMDUTF_SEARCH_CHUNK(b * B + 2 * W),
                                     SIMDUTF_SEARCH_CHUNK(b * B + 3 * W))));
        }
        if (!ops::any(acc)) {
          i += (ops::skip - 1) * B;
          continue;
        }
      }
#endif
      if (!ops::any(ops::or_(ops::or_(fA, fB), ops::or_(fC, fD)))) {
        continue;
      }
      const uint64_t mA = ops::mask(fA), mB = ops::mask(fB);
      const uint64_t mC = ops::mask(fC), mD = ops::mask(fD);
      if (K < 4) {
        survivors +=
            lane_count(mA) + lane_count(mB) + lane_count(mC) + lane_count(mD);
        // blocks scanned = (i - first_block) / B; each anchor dropped saves
        // about two compares per block
        if (survivors * survivor_cost(m) >
            ((i - first_block) >> (block_shift - 1)) + 128) {
          return scalar::search::stop(2, i, rounds, survivors, kill);
        }
      }
      const int nz = (mA != 0) + (mB != 0) + (mC != 0) + (mD != 0);
      const uint64_t one = mA | mB | mC | mD;
      if (nz == 1 && (one & (one - 1)) == 0) {
        const size_t off = (mA != 0)   ? 0
                           : (mB != 0) ? W
                           : (mC != 0) ? 2 * W
                                       : 3 * W;
        const size_t b = i + off + lane_index(one);
        if (verify(b)) {
          return scalar::search::match(b, rounds);
        }
        if (guarded && rounds > budget) {
          return scalar::search::stop(1, i + B, rounds, survivors);
        }
        continue;
      }
      size_t last_k = SIZE_MAX;
      for (size_t k = 0; k < m; ++k) {
        if (k == o0 || k == o1 || k == o2 || k == o3) {
          continue;
        }
        if (!ops::any(ops::or_(ops::or_(fA, fB), ops::or_(fC, fD)))) {
          kill = last_k;
          break;
        }
        last_k = k;
        const vec pk = ops::splat(uint8_t(needle[k]));
        fA = ops::and_(fA, match_at(text + i + k, pk));
        fB = ops::and_(fB, match_at(text + i + k + W, pk));
        fC = ops::and_(fC, match_at(text + i + k + 2 * W, pk));
        fD = ops::and_(fD, match_at(text + i + k + 3 * W, pk));
        // A give-up resumes at the start of the block, whose positions are
        // not all ruled out.
        if (guarded && ++rounds > budget) {
          return scalar::search::stop(1, i, rounds, survivors);
        }
      }
      uint64_t r;
      if ((r = ops::mask(fA))) {
        return scalar::search::match(i + lane_index(r), rounds);
      }
      if ((r = ops::mask(fB))) {
        return scalar::search::match(i + W + lane_index(r), rounds);
      }
      if ((r = ops::mask(fC))) {
        return scalar::search::match(i + 2 * W + lane_index(r), rounds);
      }
      if ((r = ops::mask(fD))) {
        return scalar::search::match(i + 3 * W + lane_index(r), rounds);
      }
    }
#undef SIMDUTF_SEARCH_CHUNK
#undef SIMDUTF_SEARCH_A0
    return windows<guarded, K>(text, n, needle, m, i, oo, pv, budget, rounds,
                               survivors,
                               ((i - first_block) >> (block_shift - 1)) + 128);
  }

  // Needles of one to three bytes: the filter is the whole needle, so a
  // survivor is a match.
  template <int M>
  static size_t short_search(const char *text, size_t n, const char *needle) {
    static_assert(M >= 1 && M <= 3, "short needles only");
    const size_t m = M;
    if (n < m) {
      return npos;
    }
    const size_t positions = n - m + 1;
    const vec q0 = ops::splat(uint8_t(needle[0]));
    const vec q1 = ops::splat(uint8_t(needle[M > 1 ? 1 : 0]));
    const vec q2 = ops::splat(uint8_t(needle[M > 2 ? 2 : 0]));
    if (positions < W) {
      for (size_t p = 0; p < positions; ++p) {
        if (std::memcmp(text + p, needle, m) == 0) {
          return p;
        }
      }
      return npos;
    }
#define SIMDUTF_SEARCH_SHORT(BASE)                                             \
  (M == 1 ? match_at(BASE, q0)                                                 \
   : M == 2                                                                    \
       ? ops::and_(match_at(BASE, q0), match_at((BASE) + 1, q1))               \
       : ops::and_(ops::and_(match_at(BASE, q0), match_at((BASE) + 1, q1)),    \
                   match_at((BASE) + 2, q2)))
    const size_t last = positions - W;
    size_t i = 0;
    for (; i + B - W <= last; i += B) {
#ifdef SIMDUTF_SEARCH_SHORTSKIP
      // EXPERIMENT (not Needle-Hammer).
      // As in the long-needle loop, several blocks at once while none has a
      // match; only once the search has run for a while, so that the
      // searches that end in their first blocks do not pay for it.
      if (M > 1 && i >= 16 * B &&
          i + ops::skip * B - W <= last) {
        bvec acc = SIMDUTF_SEARCH_SHORT(text + i);
        for (size_t c = 1; c < 4 * ops::skip; ++c) {
          acc = ops::or_(acc, SIMDUTF_SEARCH_SHORT(text + i + c * W));
        }
        if (!ops::any(acc)) {
          i += (ops::skip - 1) * B;
          continue;
        }
      }
#endif
      const bvec fA = SIMDUTF_SEARCH_SHORT(text + i);
      const bvec fB = SIMDUTF_SEARCH_SHORT(text + i + W);
      const bvec fC = SIMDUTF_SEARCH_SHORT(text + i + 2 * W);
      const bvec fD = SIMDUTF_SEARCH_SHORT(text + i + 3 * W);
      if (!ops::any(ops::or_(ops::or_(fA, fB), ops::or_(fC, fD)))) {
        continue;
      }
      uint64_t r;
      if ((r = ops::mask(fA))) {
        return i + lane_index(r);
      }
      if ((r = ops::mask(fB))) {
        return i + W + lane_index(r);
      }
      if ((r = ops::mask(fC))) {
        return i + 2 * W + lane_index(r);
      }
      return i + 3 * W + lane_index(ops::mask(fD));
    }
    for (; i <= last; i += W) {
      const bvec f = SIMDUTF_SEARCH_SHORT(text + i);
      if (ops::any(f)) {
        return i + lane_index(ops::mask(f));
      }
    }
    if (i < positions) {
      uint64_t r = ops::mask(SIMDUTF_SEARCH_SHORT(text + last));
      while (r) {
        const size_t b = last + lane_index(r);
        if (b >= i) {
          return b;
        }
        r &= r - 1;
      }
    }
#undef SIMDUTF_SEARCH_SHORT
    return npos;
  }

  template <int K>
  static result run_windows(bool guarded, const char *text, size_t n,
                            const char *needle, size_t m, const anchors &an,
                            size_t budget, size_t carried, size_t from) {
    vec pv[4];
    for (int k = 0; k < K; ++k) {
      pv[k] = ops::splat(uint8_t(needle[an.o[k]]));
    }
    return guarded ? windows<true, K>(text, n, needle, m, from, an.o, pv,
                                      budget, carried, 0, 128)
                   : windows<false, K>(text, n, needle, m, from, an.o, pv,
                                       budget, carried, 0, 128);
  }

  // m >= 4. Out of line so that a short-needle search does not pay its
  // prologue.
  static simdutf_never_inline size_t search_long(const char *text, size_t n,
                                                 const char *needle, size_t m) {
    if (n < m) {
      return npos;
    }
      const size_t budget = n / budget_den + 1;
    const bool guarded = m > free_below;
    anchors a = select(needle, m);
  if (n < min_wide || n < m + B - 1) {
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

  // First occurrence of needle[0..m) in text[0..n), or npos.
  static simdutf_really_inline size_t find(const char *text, size_t n,
                                           const char *needle, size_t m) {
    switch (m) {
    case 0:
      return 0;
    case 1:
      return short_search<1>(text, n, needle);
    case 2:
      return short_search<2>(text, n, needle);
    case 3:
      return short_search<3>(text, n, needle);
    default:
      return search_long(text, n, needle, m);
    }
  }
};

// The search in the form of the public API: pointers in, a pointer to the
// first occurrence or `end` out.
template <class ops>
simdutf_really_inline const char *
search_range(const char *start, const char *end, const char *needle_start,
             const char *needle_end) {
  const size_t n = start < end ? size_t(end - start) : 0;
  const size_t m =
      needle_start < needle_end ? size_t(needle_end - needle_start) : 0;
  const size_t r = needle_hammer<ops>::find(start, n, needle_start, m);
  return r == npos ? end : start + r;
}

} // namespace search
} // unnamed namespace
} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf
