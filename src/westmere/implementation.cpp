#include "simdutf/westmere/begin.h"

namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {
namespace {

using namespace simd;

simdutf_really_inline bool is_ascii(const simd8x64<uint8_t>& input) {
  return input.reduce_or().is_ascii();
}

simdutf_unused simdutf_really_inline simd8<bool> must_be_continuation(const simd8<uint8_t> prev1, const simd8<uint8_t> prev2, const simd8<uint8_t> prev3) {
  simd8<uint8_t> is_second_byte = prev1.saturating_sub(0b11000000u-1); // Only 11______ will be > 0
  simd8<uint8_t> is_third_byte  = prev2.saturating_sub(0b11100000u-1); // Only 111_____ will be > 0
  simd8<uint8_t> is_fourth_byte = prev3.saturating_sub(0b11110000u-1); // Only 1111____ will be > 0
  // Caller requires a bool (all 1's). All values resulting from the subtraction will be <= 64, so signed comparison is fine.
  return simd8<int8_t>(is_second_byte | is_third_byte | is_fourth_byte) > int8_t(0);
}

simdutf_really_inline simd8<bool> must_be_2_3_continuation(const simd8<uint8_t> prev2, const simd8<uint8_t> prev3) {
  simd8<uint8_t> is_third_byte  = prev2.saturating_sub(0b11100000u-1); // Only 111_____ will be > 0
  simd8<uint8_t> is_fourth_byte = prev3.saturating_sub(0b11110000u-1); // Only 1111____ will be > 0
  // Caller requires a bool (all 1's). All values resulting from the subtraction will be <= 64, so signed comparison is fine.
  return simd8<int8_t>(is_third_byte | is_fourth_byte) > int8_t(0);
}

} // unnamed namespace
} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf

#include "generic/buf_block_reader.h"
#include "generic/utf8_validation/utf8_lookup4_algorithm.h"
#include "generic/utf8_validation/utf8_validator.h"
#include "generic/utf16_validation/utf16_scalar_validator.h" // Daniel: This should go in the fallback kernel TODO
// transcoding from UTF-16 to UTF-8
#include "generic/utf16_to_utf8/valid_utf16_to_utf8.h"
#include "generic/utf16_to_utf8/utf16_to_utf8.h"
// transcoding from UTF-8 to UTF-16
#include "generic/utf8_to_utf16/valid_utf8_to_utf16.h"
#include "generic/utf8_to_utf16/utf8_to_utf16.h"
//
// Implementation-specific overrides
//

namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {

simdutf_warn_unused bool implementation::validate_utf8(const char *buf, size_t len) const noexcept {
  return westmere::utf8_validation::generic_validate_utf8(buf, len);
}

simdutf_warn_unused bool implementation::validate_utf16(const char *buf, size_t len) const noexcept {
  return westmere::utf16_validation::scalar_validate_utf16(buf, len);
}

simdutf_warn_unused size_t implementation::convert_utf8_to_utf16(const char* /*buf*/, size_t /*len*/, char* /*utf16_output*/) const noexcept {
  return 0; // stub
}

simdutf_warn_unused size_t implementation::convert_valid_utf8_to_utf16(const char* /*buf*/, size_t /*len*/, char* /*utf16_output*/) const noexcept {
  size_t pos = 0;
  uint16_t *start = output;
  while (pos + 16 <= size) {
    const __m128i in = _mm_loadu_si128((__m128i *)(input + pos));
    const uint16_t non_ascii_chars = uint16_t(_mm_movemask_epi8(in));
    // ASCII is likely common in many cases, we want a fast path.
    if(non_ascii_chars == 0) {
        // could use _mm_cvtepi8_epi16
        const __m128i out1 = _mm_unpacklo_epi8(in, _mm_setzero_si128());// order of parameter determines endianness
        _mm_storeu_si128((__m128i*)output, out1);
        output += 8;
        const __m128i out2 = _mm_unpackhi_epi8(in, _mm_setzero_si128());
        _mm_storeu_si128((__m128i*)output, out2);
        output += 8;
        pos += 16;
        continue;
    }
    // 
    // Otherwise, we use an approach where we try to process up to 12 input bytes.
    // Why 12 input bytes and not 16? Because we are concerned with the size of the lookup tables.
    // Also 12 is nicely divisible by two and three.
    //
    // The most significant 4 bits (high nibble) can be used to classify the bytes, as
    // to whether they start a new code point.
    const __m128i hn = _mm_set1_epi8(uint8_t(0xF));
    const __m128i in_high_nibbles = _mm_and_si128(hn, _mm_srli_epi16(in, 4));
    // We could remove the UTF8 headers with one instruction:
    const uint16_t start_of_code_point = uint16_t(_mm_movemask_epi8(_mm_shuffle_epi8(_mm_set_epi8(0xff, 0xff, 0xff, 0xff, 0x0, 0x0, 0x0, 0x0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff), in_high_nibbles)));
    /////////////
    // ASCII characters are the start and the end of a code point 
    // and anything immediately before the start of a code point 
    // must be the end of a start of code point.
    ///////////
    const uint16_t end_of_code_point = uint16_t((start_of_code_point >> 1) | (! non_ascii_chars));
    const uint16_t twelvebits_mask = 0xFFF;
    const uint16_t twelvebits_end_of_code_point = end_of_code_point&twelvebits_mask;
    const uint8_t idx = utf8bigindex[twelvebits_end_of_code_point][0];
    const uint8_t consumed = utf8bigindex[twelvebits_end_of_code_point][1];
    //
    // Let us first try to see if we are in the easy two-byte scenario
    //
    if(idx < 64) {
        // this is a relatively easy scenario
        // we process SIX (6) input code-words. The max length in bytes of six code words
        // spanning between 1 and 2 bytes each is 12 bytes.
        // On processors where pdep/pext is fast, we might be able to use a small lookup table.
        const __m128i sh = _mm_loadu_si128((const __m128i *)shufutf8[idx]);
        const __m128i perm = _mm_shuffle_epi8(in, sh);
        const __m128i ascii = _mm_and_si128(perm,_mm_set1_epi16(0x7f));
        const __m128i highbyte = _mm_and_si128(perm,_mm_set1_epi16(0x1f00));
        const __m128i composed = _mm_or_si128(ascii,_mm_srli_epi16(highbyte,2));
        _mm_storeu_si128((__m128i*)output, composed);
        output += 6;
        pos += consumed;
    } else if (idx < 145) {
        const __m128i sh = _mm_loadu_si128((const __m128i *)shufutf8[idx]);
        const __m128i perm = _mm_shuffle_epi8(in, sh);
        const __m128i ascii = _mm_and_si128(perm,_mm_set1_epi32(0x7f)); // 7 or 6 bits
        const __m128i middlebyte = _mm_and_si128(perm,_mm_set1_epi32(0x3f00)); // 5 or 6 bits
        const __m128i middlebyte_shifted = _mm_srli_epi32(middlebyte,2);
        const __m128i highbyte = _mm_and_si128(perm,_mm_set1_epi32(0x0f0000)); // 4 bits
        const __m128i highbyte_shifted = _mm_srli_epi32(highbyte,4);
        const __m128i composed = _mm_or_si128(_mm_or_si128(ascii,middlebyte_shifted),highbyte_shifted);
        const __m128i composed_repacked = _mm_packus_epi32(composed,composed);
        _mm_storeu_si128((__m128i*)output,  composed_repacked);
        output += 4;
        pos += consumed;
    } else if(idx < 209) {
        const __m128i sh = _mm_loadu_si128((const __m128i *)shufutf8[idx]);
        const __m128i perm = _mm_shuffle_epi8(in, sh);
        const __m128i ascii = _mm_and_si128(perm,_mm_set1_epi32(0x7f)); 
        const __m128i middlebyte = _mm_and_si128(perm,_mm_set1_epi32(0x3f00));
        const __m128i middlebyte_shifted = _mm_srli_epi32(middlebyte,2);
        __m128i middlehighbyte = _mm_and_si128(perm,_mm_set1_epi32(0x3f0000));//_mm_xor_si128(_mm_and_si128(perm,_mm_set1_epi32(0x3f0000)), );
        // correct for spurious high bit
        const __m128i correct = _mm_srli_epi32(_mm_and_si128(perm,_mm_set1_epi32(0x400000)),1);
        middlehighbyte = _mm_xor_si128(correct, middlehighbyte);
        const __m128i middlehighbyte_shifted = _mm_srli_epi32(middlehighbyte,4);
        const __m128i highbyte = _mm_and_si128(perm,_mm_set1_epi32(0x07000000));
        const __m128i highbyte_shifted = _mm_srli_epi32(highbyte,6);
        const __m128i composed = _mm_or_si128(_mm_or_si128(ascii,middlebyte_shifted),_mm_or_si128(highbyte_shifted,middlehighbyte_shifted));
        const __m128i composedminus = _mm_sub_epi32(composed,_mm_set1_epi32(0x10000));
        const __m128i lowtenbits = _mm_and_si128(composedminus,_mm_set1_epi32(0x3ff));
        const __m128i hightenbits = _mm_srli_epi32(composedminus,10);
        const __m128i lowtenbitsadd = _mm_add_epi32(lowtenbits,_mm_set1_epi32(0xDC00));
        const __m128i hightenbitsadd = _mm_add_epi32(hightenbits,_mm_set1_epi32(0xD800));
        const __m128i lowtenbitsaddshifted = _mm_slli_epi32(lowtenbitsadd,16);
        const __m128i surrogates = _mm_or_si128(hightenbitsadd,lowtenbitsaddshifted);
        uint32_t basic_buffer[4];
        _mm_storeu_si128((__m128i*)basic_buffer,  composed);
        uint32_t surrogate_buffer[4];
        _mm_storeu_si128((__m128i*)surrogate_buffer,  surrogates);
        for(size_t i = 0; i < 3; i++) {
            if(basic_buffer[i]<65536) {
                *output = uint16_t(basic_buffer[i]);
                output++;
            } else {
                output[0] = uint16_t(surrogate_buffer[i] &0xFFFF);
                output[1] = uint16_t(surrogate_buffer[i] >> 16);
                output += 2;
            }
        }
        pos += consumed;        
    } else {
        // here we know that there is an error but we do not handle errors
    }
  }
  size_t len = strlen_utf8_to_utf16_with_length(input + pos, size - pos);
  finisher_functions::utf8_to_utf16_with_length(input + pos, size - pos, output);
  output += len;
  return output - start;
}

simdutf_warn_unused size_t implementation::convert_utf16_to_utf8(const char* /*buf*/, size_t /*len*/, char* /*utf8_output*/) const noexcept {
  return 0; // stub
}

simdutf_warn_unused size_t implementation::convert_valid_utf16_to_utf8(const char* /*buf*/, size_t /*len*/, char* /*utf8_output*/) const noexcept {
  return 0; // stub
}

} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf

#include "simdutf/westmere/end.h"
