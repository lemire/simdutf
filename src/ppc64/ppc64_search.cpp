// Primitives for the Needle-Hammer search (src/generic/search.h): 16 lanes,
// one mask bit per lane, through the simd8 wrappers.
struct search_ops {
  using vec = simd8<uint8_t>;
  using bvec = simd8<bool>;
  static constexpr size_t W = 16;
  // blocks tested at once while no survivor turns up (not measured)
  static constexpr size_t skip = 2;
  static constexpr int lane_shift = 0;
  static constexpr uint64_t all_lanes = 0xFFFF;
  static simdutf_really_inline vec splat(uint8_t c) { return vec::splat(c); }
  static simdutf_really_inline vec load(const char *p) {
    return vec::load(reinterpret_cast<const uint8_t *>(p));
  }
  static simdutf_really_inline vec load_aligned(const char *p) {
    return load(p);
  }
  static simdutf_really_inline bvec eq(vec a, vec b) { return a == b; }
  static simdutf_really_inline bvec and_(bvec a, bvec b) { return a & b; }
  static simdutf_really_inline bvec or_(bvec a, bvec b) { return a | b; }
  static simdutf_really_inline bool any(bvec v) { return v.any(); }
  static simdutf_really_inline uint64_t mask(bvec v) { return v.to_bitmask(); }
};
