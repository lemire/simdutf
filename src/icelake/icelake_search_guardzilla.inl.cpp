// Guardzilla substring search, AVX-512 (BW): StringZilla's search with an
// aligned scan and a work guard.
//
// Three needle bytes (the anchors: the first, middle and last, moved to
// distinct values and off UTF-8 lead bytes) are tested at 64 positions per
// step, in steps that start on 64-byte boundaries of the haystack. Each
// position that passes is compared against the needle's first (up to) 64
// bytes in one masked compare. The comparisons beyond those 64 bytes are
// counted; once they exceed four per 64 bytes scanned, plus a slack of
// m/8 + 64, the search finishes with the two-way algorithm from the current
// position. Every search therefore takes O(n + m) time. Masked loads cover
// the ends of the haystack, so no byte outside the haystack or the needle is
// read.
//
// Credits: the anchor choice is StringZilla's sz_locate_needle_anomalies_
// (Ash Vardanian, Apache-2.0). Guardzilla is described in "Guardzilla:
// Linear-Time AVX-512 Substring Search from StringZilla" (D. Lemire); the
// reference implementation is github.com/lemire/simdsearch.
namespace search_guardzilla {

using scalar::search::npos;

simdutf_really_inline __mmask64 low_mask(size_t k) {
  return k >= 64 ? ~__mmask64(0) : (__mmask64(1) << k) - 1;
}

// The number of leading bytes on which a[0..len) and b[0..len) agree.
simdutf_really_inline size_t same_prefix(const uint8_t *a, const uint8_t *b,
                                         size_t len) {
  for (size_t k = 0; k < len; k += 64) {
    const __mmask64 lm = low_mask(len - k);
    const __mmask64 ne =
        _mm512_mask_cmpneq_epi8_mask(lm, _mm512_maskz_loadu_epi8(lm, a + k),
                                     _mm512_maskz_loadu_epi8(lm, b + k));
    if (ne) {
      return k + size_t(_tzcnt_u64(ne));
    }
  }
  return len;
}

// The number of trailing bytes on which a[0..len) and b[0..len) agree.
simdutf_really_inline size_t same_suffix(const uint8_t *a, const uint8_t *b,
                                         size_t len) {
  for (size_t k = len; k > 0;) {
    const size_t c = k < 64 ? k : 64;
    k -= c;
    const __mmask64 lm = low_mask(c);
    const __mmask64 ne =
        _mm512_mask_cmpneq_epi8_mask(lm, _mm512_maskz_loadu_epi8(lm, a + k),
                                     _mm512_maskz_loadu_epi8(lm, b + k));
    if (ne) {
      return len - k - size_t(64 - _lzcnt_u64(ne));
    }
  }
  return len;
}

// Crochemore-Perrin two-way search of p[0..m) in t[from..n): linear time,
// constant space. Up to four bytes are compared one at a time, where most
// windows end, then 64 at a time. Returns the first match at or after
// `from`, or n.
inline size_t twoway(const uint8_t *t, size_t n, const uint8_t *p, size_t m,
                     size_t from) {
  // Critical factorization: the later of the two maximal suffixes, under the
  // byte order and under its reverse.
  auto max_suffix = [&](bool reverse, size_t *period) {
    size_t ms = SIZE_MAX, j = 0, k = 1, q = 1; // ms + k wraps to k - 1
    while (j + k < m) {
      const uint8_t a = p[j + k], b = p[ms + k];
      if (reverse ? a > b : a < b) {
        j += k;
        k = 1;
        q = j - ms;
      } else if (a == b) {
        if (k == q) {
          j += q;
          k = 1;
        } else {
          ++k;
        }
      } else {
        ms = j++;
        k = q = 1;
      }
    }
    *period = q;
    return ms + 1;
  };
  size_t per1, per2;
  const size_t c1 = max_suffix(false, &per1), c2 = max_suffix(true, &per2);
  const size_t crit = c1 > c2 ? c1 : c2;
  size_t period = c1 > c2 ? per1 : per2;
  const bool periodic = std::memcmp(p, p + period, crit) == 0;
  if (!periodic) {
    period = (crit > m - crit ? crit : m - crit) + 1;
  }
  size_t memory = 0;
  for (size_t j = from; j + m <= n;) {
    size_t i = crit > memory ? crit : memory;
    const size_t lim = i + 4 < m ? i + 4 : m;
    while (i < lim && p[i] == t[j + i]) {
      ++i;
    }
    if (i == lim) {
      i += same_prefix(p + i, t + j + i, m - i);
    }
    if (i < m) {
      j += i - crit + 1;
      memory = 0;
      continue;
    }
    size_t k = crit;
    const size_t low = memory + 4 < crit ? crit - 4 : memory;
    while (k > low && p[k - 1] == t[j + k - 1]) {
      --k;
    }
    if (k == low) {
      k -= same_suffix(p + memory, t + j + memory, k - memory);
    }
    if (k <= memory) {
      return j;
    }
    j += period;
    memory = periodic ? m - period : 0;
  }
  return n;
}

// StringZilla's three anchors: the first, middle and last bytes, moved to
// distinct values and, past 8 bytes, off UTF-8 lead bytes (>= 0xC0).
simdutf_really_inline void anchors(const uint8_t *s, size_t m, size_t o[3]) {
  size_t a = 0, b = m / 2, c = m - 1;
  if (m > 3 && (s[a] == s[b] || s[a] == s[c] || s[b] == s[c])) {
    while (s[b] == s[a] && b + 1 < c) {
      ++b;
    }
    while ((s[c] == s[b] || s[c] == s[a]) && c > b + 1) {
      --c;
    }
  }
  if (m > 8) {
    size_t va = a, vb = b;
    while ((s[vb] > 191 || s[vb] == s[c]) && vb + 1 < c) {
      ++vb;
    }
    if (s[vb] < 191) {
      b = vb;
    } else {
      vb = b;
    }
    while ((s[va] > 191 || s[va] == s[vb] || s[va] == s[c]) && va + 1 < vb) {
      ++va;
    }
    if (s[va] < 191) {
      a = va;
    }
  }
  o[0] = a, o[1] = b, o[2] = c;
}

// Whether a[64..m) == b[64..m), counting the 64-byte compares in *work. Out
// of line: it is rarely called, and inlined it slows the scan.
simdutf_never_inline bool rest_matches(const uint8_t *a, const uint8_t *b,
                                       size_t m, size_t *work) {
  for (size_t d = 64; d < m; d += 64) {
    const __mmask64 lm = low_mask(m - d);
    ++*work;
    if (_mm512_mask_cmpneq_epi8_mask(lm, _mm512_maskz_loadu_epi8(lm, a + d),
                                     _mm512_maskz_loadu_epi8(lm, b + d))) {
      return false;
    }
  }
  return true;
}

// The first match among candidates [0, n - m], n if there is none, or
// n + 1 + q if the work budget ran out at candidate q. Requires m >= 2.
inline size_t scan(const uint8_t *t, size_t n, const uint8_t *p, size_t m) {
  size_t o[3];
  if (m >= 3) {
    anchors(p, m, o);
  } else {
    o[0] = 0, o[1] = o[2] = 1;
  }
  __m512i v[3];
  for (int j = 0; j < 3; ++j) {
    v[j] = _mm512_set1_epi8(char(p[o[j]]));
  }
  const __mmask64 head_mask = low_mask(m);
  const __m512i head = _mm512_maskz_loadu_epi8(head_mask, p);
  const size_t last = n - m, slack = m / 8 + 64;
  size_t work = 0; // 64-byte compares beyond the first 64 needle bytes

  // Candidates in [i, i + 64) among `live`, with masked loads.
  auto filter = [&](size_t i, __mmask64 live) {
    for (int j = 0; j < 3; ++j) {
      live &= _mm512_mask_cmpeq_epi8_mask(
          live, _mm512_maskz_loadu_epi8(live, t + i + o[j]), v[j]);
    }
    return live;
  };
  // Checks the candidates w at i: 0 if none matches, else 1 + the result.
  auto verify = [&](size_t i, __mmask64 w) -> size_t {
    for (; w; w &= w - 1) {
      const size_t q = i + size_t(_tzcnt_u64(w));
      if (_mm512_mask_cmpneq_epi8_mask(
              head_mask, _mm512_maskz_loadu_epi8(head_mask, t + q), head)) {
        continue;
      }
      if (m <= 64 || rest_matches(t + q, p, m, &work)) {
        return q + 1;
      }
      if (work > 4 * (q / 64) + slack) {
        return n + 2 + q;
      }
    }
    return 0;
  };

  size_t r = 0;
  // A masked head up to the first 64-byte boundary of the haystack.
  size_t i = (64 - (uintptr_t(t) & 63)) & 63;
  if (i > last + 1) {
    i = last + 1;
  }
  if (i && (r = verify(0, filter(0, low_mask(i))))) {
    return r - 1;
  }
  for (; i + 63 <= last; i += 64) {
    __mmask64 s = ~__mmask64(0);
    for (int j = 0; j < 3; ++j) {
      s &= _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(t + i + o[j]), v[j]);
    }
    if (s && (r = verify(i, s))) {
      return r - 1;
    }
  }
  if (i <= last && (r = verify(i, filter(i, low_mask(last - i + 1))))) {
    return r - 1;
  }
  return n;
}

simdutf_really_inline const char *search_range(const char *start,
                                               const char *end,
                                               const char *needle_start,
                                               const char *needle_end) {
  const size_t n = start < end ? size_t(end - start) : 0;
  const size_t m =
      needle_start < needle_end ? size_t(needle_end - needle_start) : 0;
  if (m == 0) {
    return start;
  }
  if (n < m) {
    return end;
  }
  if (m == 1) { // a byte search
    const size_t r = search::short_search<1>(start, n, needle_start);
    return r == npos ? end : start + r;
  }
  const uint8_t *t = reinterpret_cast<const uint8_t *>(start);
  const uint8_t *p = reinterpret_cast<const uint8_t *>(needle_start);
  size_t r = scan(t, n, p, m);
  if (r > n) { // the budget ran out
    r = twoway(t, n, p, m, r - n - 1);
  }
  return r < n ? start + r : end;
}

} // namespace search_guardzilla
