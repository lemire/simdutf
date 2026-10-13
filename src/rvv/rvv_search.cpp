// Substring search: every candidate position is filtered on two needle bytes
// (the first and the last, each walked inward to avoid repeated values),
// vl positions per step, and the first survivor is verified byte by byte.
// The bytes compared in verification are charged to a budget proportional
// to the haystack; once it is spent the two-way search takes over, so the
// search is linear whatever the input.
const char *implementation::search(const char *start, const char *end,
                                   const char *needle_start,
                                   const char *needle_end) const noexcept {
  const size_t n = start < end ? size_t(end - start) : 0;
  const size_t m =
      needle_start < needle_end ? size_t(needle_end - needle_start) : 0;
  if (m == 0) {
    return start;
  }
  if (n < m) {
    return end;
  }
  if (m == 1) {
    return find(start, end, needle_start[0]);
  }
  const uint8_t *hay = reinterpret_cast<const uint8_t *>(start);
  const uint8_t *nd = reinterpret_cast<const uint8_t *>(needle_start);
  size_t o[4];
  scalar::search::positional(nd, m, o);
  const size_t a0 = o[0], a1 = o[3];
  const uint8_t c0 = nd[a0], c1 = nd[a1];
  const size_t positions = n - m + 1;
  const size_t budget = n / 4 + 64;
  size_t spent = 0;
  size_t i = 0;
  while (i < positions) {
    const size_t vl = __riscv_vsetvl_e8m4(positions - i);
    const vuint8m4_t x0 = __riscv_vle8_v_u8m4(hay + i + a0, vl);
    const vuint8m4_t x1 = __riscv_vle8_v_u8m4(hay + i + a1, vl);
    const vbool2_t f =
        __riscv_vmand_mm_b2(__riscv_vmseq_vx_u8m4_b2(x0, c0, vl),
                            __riscv_vmseq_vx_u8m4_b2(x1, c1, vl), vl);
    const long k = __riscv_vfirst_m_b2(f, vl);
    if (k < 0) {
      i += vl;
      continue;
    }
    const size_t p = i + size_t(k);
    size_t l = 0;
    while (l < m && hay[p + l] == nd[l]) {
      ++l;
    }
    if (l == m) {
      return start + p;
    }
    spent += l + 1;
    if (spent > budget) {
      const size_t r = scalar::search::twoway(start, n, needle_start, m, p);
      return r == scalar::search::npos ? end : start + r;
    }
    i = p + 1;
  }
  return end;
}
