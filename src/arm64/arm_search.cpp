// Primitives for the Needle-Hammer search (src/generic/search.h). A compare
// result is a 0x00/0xFF byte vector; mask() narrows it to four bits per lane
// with shrn and keeps one of them, so lane indices are ctz / 4.
struct search_ops {
  using vec = uint8x16_t;
  using bvec = uint8x16_t;
  static constexpr size_t W = 16;
  // blocks tested at once while no survivor turns up (measured on an Apple M4)
  static constexpr size_t skip = 2;
  static constexpr int lane_shift = 2;
  static constexpr uint64_t all_lanes = 0x8888888888888888ull;
  static simdutf_really_inline vec splat(uint8_t c) { return vdupq_n_u8(c); }
  static simdutf_really_inline vec load(const char *p) {
    return vld1q_u8(reinterpret_cast<const uint8_t *>(p));
  }
  static simdutf_really_inline vec load_aligned(const char *p) {
    return load(p);
  }
  static simdutf_really_inline bvec eq(vec a, vec b) { return vceqq_u8(a, b); }
  static simdutf_really_inline bvec and_(bvec a, bvec b) {
    return vandq_u8(a, b);
  }
  static simdutf_really_inline bvec or_(bvec a, bvec b) {
    return vorrq_u8(a, b);
  }
  static simdutf_really_inline uint64_t nibbles(bvec v) {
    return vget_lane_u64(
        vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(v), 4)), 0);
  }
  // An integer test: comparing the nibbles as a double would treat a
  // denormal as zero when the FPCR flush-to-zero bit is set.
  static simdutf_really_inline bool any(bvec v) { return nibbles(v) != 0; }
  static simdutf_really_inline uint64_t mask(bvec v) {
    return nibbles(v) & all_lanes;
  }
};
