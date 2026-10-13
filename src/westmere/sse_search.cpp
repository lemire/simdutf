// Primitives for the Needle-Hammer search (src/generic/search.h): 16 lanes,
// one mask bit per lane.
struct search_ops {
  using vec = __m128i;
  using bvec = __m128i;
  static constexpr size_t W = 16;
  // blocks tested at once while no survivor turns up (measured on Emerald
  // Rapids)
  static constexpr size_t skip = 3;
  static constexpr int lane_shift = 0;
  static constexpr uint64_t all_lanes = 0xFFFF;
  static simdutf_really_inline vec splat(uint8_t c) {
    return _mm_set1_epi8(char(c));
  }
  static simdutf_really_inline vec load(const char *p) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
  }
  // p is 16-byte aligned: SSE can fold the load into the compare
  static simdutf_really_inline vec load_aligned(const char *p) {
    return _mm_load_si128(reinterpret_cast<const __m128i *>(p));
  }
  static simdutf_really_inline bvec eq(vec a, vec b) {
    return _mm_cmpeq_epi8(a, b);
  }
  static simdutf_really_inline bvec and_(bvec a, bvec b) {
    return _mm_and_si128(a, b);
  }
  static simdutf_really_inline bvec or_(bvec a, bvec b) {
    return _mm_or_si128(a, b);
  }
  static simdutf_really_inline bool any(bvec v) {
    return _mm_movemask_epi8(v) != 0;
  }
  static simdutf_really_inline uint64_t mask(bvec v) {
    return uint32_t(_mm_movemask_epi8(v));
  }
};
