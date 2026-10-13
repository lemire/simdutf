// Primitives for the Needle-Hammer search (src/generic/search.h): 16 lanes,
// one mask bit per lane.
struct search_ops {
  using vec = __m128i;
  using bvec = __m128i;
  static constexpr size_t W = 16;
  // blocks tested at once while no survivor turns up (not measured)
  static constexpr size_t skip = 2;
  static constexpr int lane_shift = 0;
  static constexpr uint64_t all_lanes = 0xFFFF;
  static simdutf_really_inline vec splat(uint8_t c) {
    return __lsx_vreplgr2vr_b(c);
  }
  static simdutf_really_inline vec load(const char *p) {
    return __lsx_vld(reinterpret_cast<const void *>(p), 0);
  }
  static simdutf_really_inline vec load_aligned(const char *p) {
    return load(p);
  }
  static simdutf_really_inline bvec eq(vec a, vec b) {
    return __lsx_vseq_b(a, b);
  }
  static simdutf_really_inline bvec and_(bvec a, bvec b) {
    return __lsx_vand_v(a, b);
  }
  static simdutf_really_inline bvec or_(bvec a, bvec b) {
    return __lsx_vor_v(a, b);
  }
  static simdutf_really_inline bool any(bvec v) { return __lsx_bnz_v(v); }
  static simdutf_really_inline uint64_t mask(bvec v) {
    return __lsx_vpickve2gr_wu(__lsx_vmsknz_b(v), 0);
  }
};
