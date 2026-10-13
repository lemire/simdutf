// Benchmarks simdutf::search, with every implementation the processor
// supports, against the C++ standard library (and memmem where available).
//
// Usage: searchbenchmark [file]
//
// The haystack is the file, or 1 MiB of random lowercase words if none is
// given. For each needle length, the needles are substrings of the haystack
// written backwards and absent from it, so that every search scans the whole
// haystack; the throughput is haystack bytes per second.
#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "findbenchmarker.h"

#include "simdutf.h"

namespace {

void pretty_print(const std::string &name, size_t bytes, size_t searches,
                  event_aggregate agg) {
  printf("%-36s : ", name.c_str());
  printf(" %6.2f GB/s ", double(bytes) * double(searches) / agg.elapsed_ns());
  if (collector.has_events()) {
    printf(" %5.2f GHz ", agg.cycles() / agg.elapsed_ns());
    printf(" %6.3f c/b ", agg.cycles() / double(bytes * searches));
    printf(" %6.3f i/b ", agg.instructions() / double(bytes * searches));
  }
  printf("\n");
}

std::string random_words(size_t length) {
  std::mt19937 gen(1234);
  std::uniform_int_distribution<int> letter('a', 'z');
  std::uniform_int_distribution<int> word(1, 10);
  std::string s;
  s.reserve(length);
  while (s.size() < length) {
    for (int k = word(gen); k > 0; --k) {
      s += char(letter(gen));
    }
    s += ' ';
  }
  s.resize(length);
  return s;
}

std::vector<std::string> absent_needles(const std::string &hay, size_t m,
                                        size_t count) {
  std::mt19937_64 gen(m);
  std::uniform_int_distribution<size_t> pos(0, hay.size() - m);
  std::vector<std::string> needles;
  for (size_t tries = 0; needles.size() < count && tries < 100 * count;
       ++tries) {
    std::string nd = hay.substr(pos(gen), m);
    std::reverse(nd.begin(), nd.end());
    if (std::string_view(hay).find(nd) == std::string_view::npos) {
      needles.push_back(nd);
    }
  }
  return needles;
}

} // namespace

int main(int argc, char **argv) {
  std::string hay;
  if (argc > 1) {
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) {
      std::cerr << "cannot read " << argv[1] << std::endl;
      return EXIT_FAILURE;
    }
    hay.assign(std::istreambuf_iterator<char>(in),
               std::istreambuf_iterator<char>());
  } else {
    hay = random_words(size_t(1) << 20);
  }
  printf("haystack: %zu bytes\n", hay.size());
  volatile size_t counter = 0;
  for (size_t m : {4, 8, 16, 32, 64, 256, 1024}) {
    if (m > hay.size()) {
      break;
    }
    const std::vector<std::string> needles = absent_needles(hay, m, 10);
    if (needles.empty()) {
      continue;
    }
    printf("\nneedle length %zu (%zu needles, absent)\n", m, needles.size());
    for (const auto &impl : simdutf::get_available_implementations()) {
      if (!impl->supported_by_runtime_system()) {
        continue;
      }
      pretty_print(
          "simdutf::search (" + std::string(impl->name()) + ")", hay.size(),
          needles.size(), bench([&]() {
            for (const auto &nd : needles) {
              counter = counter +
                        size_t(impl->search(hay.data(), hay.data() + hay.size(),
                                            nd.data(), nd.data() + nd.size()) -
                               hay.data());
            }
          }));
    }
    pretty_print("std::string_view::find", hay.size(), needles.size(),
                 bench([&]() {
                   for (const auto &nd : needles) {
                     counter = counter + std::string_view(hay).find(nd);
                   }
                 }));
    pretty_print("std::boyer_moore_horspool_searcher", hay.size(),
                 needles.size(), bench([&]() {
                   for (const auto &nd : needles) {
                     auto it = std::search(hay.begin(), hay.end(),
                                           std::boyer_moore_horspool_searcher(
                                               nd.begin(), nd.end()));
                     counter = counter + size_t(it - hay.begin());
                   }
                 }));
#if defined(__linux__) || defined(__APPLE__)
    pretty_print("memmem", hay.size(), needles.size(), bench([&]() {
                   for (const auto &nd : needles) {
                     const void *p =
                         memmem(hay.data(), hay.size(), nd.data(), nd.size());
                     counter = counter + (p == nullptr ? 0 : 1);
                   }
                 }));
#endif
  }
  return EXIT_SUCCESS;
}
