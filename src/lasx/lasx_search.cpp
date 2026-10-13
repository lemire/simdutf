// Primitives for the Needle-Hammer search (src/generic/search.h): 32 lanes,
// one mask bit per lane.
struct search_ops {
  using vec = __m256i;
  using bvec = __m256i;
  static constexpr size_t W = 32;
  // blocks tested at once while no survivor turns up (not measured)
  static constexpr size_t skip = 2;
  static constexpr int lane_shift = 0;
  static constexpr uint64_t all_lanes = 0xFFFFFFFF;
  static simdutf_really_inline vec splat(uint8_t c) {
    return __lasx_xvreplgr2vr_b(c);
  }
  static simdutf_really_inline vec load(const char *p) {
    return __lasx_xvld(reinterpret_cast<const void *>(p), 0);
  }
  static simdutf_really_inline vec load_aligned(const char *p) {
    return load(p);
  }
  static simdutf_really_inline bvec eq(vec a, vec b) {
    return __lasx_xvseq_b(a, b);
  }
  static simdutf_really_inline bvec and_(bvec a, bvec b) {
    return __lasx_xvand_v(a, b);
  }
  static simdutf_really_inline bvec or_(bvec a, bvec b) {
    return __lasx_xvor_v(a, b);
  }
  static simdutf_really_inline bool any(bvec v) { return __lasx_xbnz_v(v); }
  static simdutf_really_inline uint64_t mask(bvec v) {
    const __m256i m = __lasx_xvmsknz_b(v);
    return uint64_t(__lasx_xvpickve2gr_wu(m, 0)) |
           (uint64_t(__lasx_xvpickve2gr_wu(m, 4)) << 16);
  }
};
