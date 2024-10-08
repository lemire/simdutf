#ifndef SIMDUTF_VALID_LATIN1_TO_UTF16_H
#define SIMDUTF_VALID_LATIN1_TO_UTF16_H

namespace simdutf {
namespace scalar {
namespace {
namespace latin1_to_utf16 {

template <endianness big_endian>
inline size_t convert_valid(const char* buf, size_t len, char16_t* utf16_output) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(buf);
    size_t pos = 0;
    char16_t* start{ utf16_output };

    while (pos < len) {
        uint16_t word = uint16_t(data[pos]); // extend Latin-1 char to 16-bit Unicode code point

        *utf16_output++ = match_system(big_endian) ? word : utf16::swap_bytes(word);
        pos++;
    }

    return utf16_output - start;
}


} // latin1_to_utf16 namespace
} // unnamed namespace
} // namespace scalar
} // namespace simdutf

#endif