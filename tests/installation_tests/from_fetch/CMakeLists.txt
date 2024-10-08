
cmake_minimum_required(VERSION 3.15)

project(simdutf_demo_install VERSION 0.1.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)

FetchContent_Declare(
  simdutf
  GIT_REPOSITORY git@github.com:simdutf/simdutf.git
  GIT_SHALLOW TRUE
  GIT_TAG tags/v3.2.7
  )
FetchContent_MakeAvailable(simdutf)

file(WRITE main.cpp "
#include <iostream>
#include \"simdutf.h\"
int main(int argc, char *argv[]) {
   const char * ascii = \"1234\";
   bool validutf8 = simdutf::validate_utf8(ascii, 4);
   if(validutf8) {
       std::cout << \"valid UTF-8\" << std::endl;
   } else {
       std::cout << \"invalid UTF-8\" << std::endl;
       return EXIT_FAILURE;
   }
   char16_t utf16_output[4];
   // convert to UTF-16LE
   size_t utf16words = simdutf::convert_utf8_to_utf16le(ascii, 4, utf16_output);
   std::cout << \"wrote \" << utf16words << \" UTF-16 words.\" << std::endl;
   // It wrote utf16words * sizeof(char16_t) bytes.
   //
   // convert it back:
   char buffer[4];
   size_t utf8words = simdutf::convert_utf16le_to_utf8(utf16_output, utf16words, buffer);
   std::cout << \"wrote \" << utf8words << \" UTF-8 words.\" << std::endl;
   return EXIT_SUCCESS;
}")



add_executable(repro main.cpp)
target_link_libraries(repro PUBLIC simdutf::simdutf)