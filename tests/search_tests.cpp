#include "simdutf.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef __linux__
  #include <sys/mman.h>
  #include <unistd.h>
#endif

#include <tests/helpers/fixed_string.h>
#include <tests/helpers/test.h>

namespace {

const char *reference(const char *start, const char *end, const char *ns,
                      const char *ne) {
  return std::search(start, end, ns, ne);
}

// The first `size` bytes of a haystack drawn from an alphabet of `sigma`
// letters starting at `base`.
std::string random_string(std::mt19937_64 &gen, size_t size, int sigma,
                          int base = 'a') {
  std::uniform_int_distribution<int> letter(0, sigma - 1);
  std::string s(size, '\0');
  for (auto &c : s) {
    c = char(base + letter(gen));
  }
  return s;
}

template <class impl>
void check(impl &implementation, const std::string &hay,
           const std::string &needle) {
  const char *start = hay.data(), *end = hay.data() + hay.size();
  const char *ns = needle.data(), *ne = needle.data() + needle.size();
  const char *expected = reference(start, end, ns, ne);
  const char *got = implementation.search(start, end, ns, ne);
  if (got != expected) {
    printf("haystack size %zu, needle size %zu: expected %zd, got %zd\n",
           hay.size(), needle.size(), expected - start, got - start);
  }
  ASSERT_TRUE(got == expected);
}

} // namespace

TEST(search_empty) {
  const char hay[] = "abc";
  const char *empty = hay + 3;
  // an empty needle occurs at the start, even of an empty haystack
  ASSERT_TRUE(implementation.search(hay, hay + 3, empty, empty) == hay);
  ASSERT_TRUE(implementation.search(empty, empty, empty, empty) == empty);
  // a non-empty needle does not occur in an empty haystack
  ASSERT_TRUE(implementation.search(empty, empty, hay, hay + 1) == empty);
  // nor in a haystack shorter than itself
  ASSERT_TRUE(implementation.search(hay, hay + 2, hay, hay + 3) == hay + 2);
  ASSERT_TRUE(implementation.search(hay, hay + 3, hay, hay + 3) == hay);
}

TEST(search_random_substrings) {
  std::mt19937_64 gen(0x5eed);
  for (int sigma : {1, 2, 4, 26, 256}) {
    for (size_t size :
         {size_t(0), size_t(1), size_t(15), size_t(63), size_t(64), size_t(200),
          size_t(1000), size_t(1500), size_t(5000)}) {
      const std::string hay =
          random_string(gen, size, sigma, sigma == 256 ? 0 : 'a');
      for (size_t m = 1; m <= 300 && m <= size; m += (m < 40 ? 1 : 37)) {
        std::uniform_int_distribution<size_t> pos(0, size - m);
        for (int trial = 0; trial < 4; ++trial) {
          std::string needle = hay.substr(pos(gen), m);
          check(implementation, hay, needle);
          // the same needle with one byte changed is usually absent
          needle[gen() % m] ^= 1;
          check(implementation, hay, needle);
        }
      }
    }
  }
}

TEST(search_random_needles) {
  std::mt19937_64 gen(1234);
  for (int sigma : {2, 3, 4}) {
    for (int trial = 0; trial < 2000; ++trial) {
      const std::string hay = random_string(gen, gen() % 3000, sigma);
      const std::string needle = random_string(gen, 1 + gen() % 12, sigma);
      check(implementation, hay, needle);
    }
  }
}

TEST(search_needle_at_every_position) {
  std::mt19937_64 gen(42);
  for (size_t size : {size_t(100), size_t(700), size_t(2100)}) {
    for (size_t m : {size_t(1), size_t(2), size_t(3), size_t(4), size_t(5),
                     size_t(9), size_t(17), size_t(33), size_t(65)}) {
      if (m > size) {
        continue;
      }
      const std::string needle = random_string(gen, m, 26, 'A');
      for (size_t p = 0; p + m <= size; ++p) {
        std::string hay(size, 'x');
        hay.replace(p, m, needle);
        check(implementation, hay, needle);
      }
    }
  }
}

// Needles and haystacks built to defeat a filter on a few needle bytes:
// every position survives the anchors, so the search must narrow, and on
// long needles the work budget runs out and the two-way search takes over.
TEST(search_adversarial) {
  for (size_t m :
       {size_t(4), size_t(8), size_t(16), size_t(37), size_t(64), size_t(100),
        size_t(255), size_t(256), size_t(257), size_t(1000)}) {
    for (size_t n :
         {size_t(m), size_t(2 * m + 3), size_t(3000), size_t(40000)}) {
      // all-'a' haystack, needle a^(m-1) b: absent, then at the end
      std::string hay(n, 'a');
      std::string needle(m - 1, 'a');
      needle += 'b';
      check(implementation, hay, needle);
      hay[n - 1] = 'b';
      check(implementation, hay, needle);
      // (a^(m-1) b)* against a^m: absent, then at the end
      std::string block;
      while (block.size() < n) {
        block += std::string(m - 1, 'a') + 'b';
      }
      block.resize(n);
      check(implementation, block, std::string(m, 'a'));
      block.replace(n - m, m, std::string(m, 'a'));
      check(implementation, block, std::string(m, 'a'));
      // a needle whose odd byte sits in the middle
      std::string mid(m, 'a');
      mid[m / 2] = 'b';
      check(implementation, hay, mid);
      // a near-match: the needle's period repeated, with a late byte changed
      std::string period = std::string(m > 1 ? m - 1 : 1, 'a') + 'b';
      std::string near;
      while (near.size() < n) {
        near += period;
      }
      near.resize(n);
      std::string late = near.substr(0, m < near.size() ? m : near.size());
      late[late.size() - 1] = 'c';
      check(implementation, near, late);
    }
  }
}

TEST(search_all_alignments) {
  std::mt19937_64 gen(7);
  const std::string text = random_string(gen, 3000, 4);
  for (size_t m :
       {size_t(1), size_t(3), size_t(4), size_t(6), size_t(20), size_t(70)}) {
    const std::string needle = text.substr(2500, m);
    for (size_t align = 0; align < 64; ++align) {
      std::vector<char> backing(text.size() + 64 + 64, 0);
      char *buf = backing.data() + align;
      std::memcpy(buf, text.data(), text.size());
      const char *expected =
          reference(buf, buf + text.size(), needle.data(), needle.data() + m);
      const char *got = implementation.search(buf, buf + text.size(),
                                              needle.data(), needle.data() + m);
      ASSERT_TRUE(got == expected);
    }
  }
}

#ifdef __linux__
// A buffer of `size` bytes that ends where an unreadable page begins.
static char *alloc_at_page_end(size_t size) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const size_t pages_needed = (size + page - 1) / page + 1;
  const size_t total = pages_needed * page;
  char *base = static_cast<char *>(mmap(nullptr, total, PROT_READ | PROT_WRITE,
                                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (base == MAP_FAILED) {
    return nullptr;
  }
  mmap(base + total - page, page, PROT_NONE,
       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  return base + total - page - size;
}

static void free_at_page_end(char *buf, size_t size) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const size_t pages_needed = (size + page - 1) / page + 1;
  const size_t total = pages_needed * page;
  munmap(buf + size + page - total, total);
}

// No byte past either buffer is read, whatever the sizes.
TEST(search_guard_page) {
  std::mt19937_64 gen(99);
  for (size_t n : {size_t(1), size_t(5), size_t(16), size_t(63), size_t(64),
                   size_t(65), size_t(300), size_t(1100), size_t(5000)}) {
    for (size_t m : {size_t(1), size_t(2), size_t(3), size_t(4), size_t(7),
                     size_t(16), size_t(33), size_t(64), size_t(200)}) {
      if (m > n) {
        continue;
      }
      char *hay = alloc_at_page_end(n);
      char *needle = alloc_at_page_end(m);
      ASSERT_TRUE(hay != nullptr && needle != nullptr);
      const std::string text = random_string(gen, n, 2);
      std::memcpy(hay, text.data(), n);
      // absent: the needle's last byte is outside the haystack's alphabet
      std::memcpy(needle, text.data() + n - m, m);
      needle[m - 1] = 'z';
      ASSERT_TRUE(implementation.search(hay, hay + n, needle, needle + m) ==
                  hay + n);
      // present at the very end
      needle[m - 1] = text[n - 1];
      const char *expected = reference(hay, hay + n, needle, needle + m);
      ASSERT_TRUE(implementation.search(hay, hay + n, needle, needle + m) ==
                  expected);
      free_at_page_end(hay, n);
      free_at_page_end(needle, m);
    }
  }
}
#endif

TEST(search_public_api) {
  const std::string hay = "The quick brown fox jumps over the lazy dog";
  const std::string needle = "lazy";
  const char *r = simdutf::search(hay.data(), hay.data() + hay.size(),
                                  needle.data(), needle.data() + needle.size());
  ASSERT_EQUAL(size_t(r - hay.data()), size_t(35));
}

#if SIMDUTF_CPLUSPLUS23
TEST(compile_time_search) {
  using namespace simdutf::tests::helpers;
  constexpr auto s = "ensure search() is constexpr"_latin1;
  constexpr auto n = "constexpr"_latin1;
  constexpr auto loc =
      std::distance(s.data(), simdutf::search(s.data(), s.data() + s.size(),
                                              n.data(), n.data() + n.size()));
  static_assert(loc == 19);
}
#endif

TEST_MAIN
