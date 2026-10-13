// Primitives for the Needle-Hammer search (src/generic/search.h): 32 lanes,
// one mask bit per lane.
struct search_ops {
  using vec = __m256i;
  using bvec = __m256i;
  static constexpr size_t W = 32;
  // blocks tested at once while no survivor turns up (measured on Emerald
  // Rapids)
  static constexpr size_t skip = 3;
  static constexpr int lane_shift = 0;
  static constexpr uint64_t all_lanes = 0xFFFFFFFF;
  static simdutf_really_inline vec splat(uint8_t c) {
    return _mm256_set1_epi8(char(c));
  }
  static simdutf_really_inline vec load(const char *p) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
  }
  static simdutf_really_inline vec load_aligned(const char *p) {
    return _mm256_load_si256(reinterpret_cast<const __m256i *>(p));
  }
  static simdutf_really_inline bvec eq(vec a, vec b) {
    return _mm256_cmpeq_epi8(a, b);
  }
  static simdutf_really_inline bvec and_(bvec a, bvec b) {
    return _mm256_and_si256(a, b);
  }
  static simdutf_really_inline bvec or_(bvec a, bvec b) {
    return _mm256_or_si256(a, b);
  }
  static simdutf_really_inline bool any(bvec v) {
    return _mm256_movemask_epi8(v) != 0;
  }
  static simdutf_really_inline uint64_t mask(bvec v) {
    return uint32_t(_mm256_movemask_epi8(v));
  }
};
