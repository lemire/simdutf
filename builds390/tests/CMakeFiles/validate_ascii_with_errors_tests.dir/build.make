# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/leorio/simdutf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/leorio/simdutf/builds390

# Include any dependencies generated for this target.
include tests/CMakeFiles/validate_ascii_with_errors_tests.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/validate_ascii_with_errors_tests.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/validate_ascii_with_errors_tests.dir/flags.make

tests/CMakeFiles/validate_ascii_with_errors_tests.dir/validate_ascii_with_errors_tests.cpp.o: tests/CMakeFiles/validate_ascii_with_errors_tests.dir/flags.make
tests/CMakeFiles/validate_ascii_with_errors_tests.dir/validate_ascii_with_errors_tests.cpp.o: ../tests/validate_ascii_with_errors_tests.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/leorio/simdutf/builds390/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/validate_ascii_with_errors_tests.dir/validate_ascii_with_errors_tests.cpp.o"
	cd /home/leorio/simdutf/builds390/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/validate_ascii_with_errors_tests.dir/validate_ascii_with_errors_tests.cpp.o -c /home/leorio/simdutf/tests/validate_ascii_with_errors_tests.cpp

tests/CMakeFiles/validate_ascii_with_errors_tests.dir/validate_ascii_with_errors_tests.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/validate_ascii_with_errors_tests.dir/validate_ascii_with_errors_tests.cpp.i"
	cd /home/leorio/simdutf/builds390/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/leorio/simdutf/tests/validate_ascii_with_errors_tests.cpp > CMakeFiles/validate_ascii_with_errors_tests.dir/validate_ascii_with_errors_tests.cpp.i

tests/CMakeFiles/validate_ascii_with_errors_tests.dir/validate_ascii_with_errors_tests.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/validate_ascii_with_errors_tests.dir/validate_ascii_with_errors_tests.cpp.s"
	cd /home/leorio/simdutf/builds390/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/leorio/simdutf/tests/validate_ascii_with_errors_tests.cpp -o CMakeFiles/validate_ascii_with_errors_tests.dir/validate_ascii_with_errors_tests.cpp.s

# Object files for target validate_ascii_with_errors_tests
validate_ascii_with_errors_tests_OBJECTS = \
"CMakeFiles/validate_ascii_with_errors_tests.dir/validate_ascii_with_errors_tests.cpp.o"

# External object files for target validate_ascii_with_errors_tests
validate_ascii_with_errors_tests_EXTERNAL_OBJECTS =

tests/validate_ascii_with_errors_tests: tests/CMakeFiles/validate_ascii_with_errors_tests.dir/validate_ascii_with_errors_tests.cpp.o
tests/validate_ascii_with_errors_tests: tests/CMakeFiles/validate_ascii_with_errors_tests.dir/build.make
tests/validate_ascii_with_errors_tests: src/libsimdutf.a
tests/validate_ascii_with_errors_tests: tests/helpers/libsimdutf_tests_helpers.a
tests/validate_ascii_with_errors_tests: tests/reference/libsimdutf_tests_reference.a
tests/validate_ascii_with_errors_tests: src/libsimdutf.a
tests/validate_ascii_with_errors_tests: tests/CMakeFiles/validate_ascii_with_errors_tests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/leorio/simdutf/builds390/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable validate_ascii_with_errors_tests"
	cd /home/leorio/simdutf/builds390/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/validate_ascii_with_errors_tests.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/validate_ascii_with_errors_tests.dir/build: tests/validate_ascii_with_errors_tests

.PHONY : tests/CMakeFiles/validate_ascii_with_errors_tests.dir/build

tests/CMakeFiles/validate_ascii_with_errors_tests.dir/clean:
	cd /home/leorio/simdutf/builds390/tests && $(CMAKE_COMMAND) -P CMakeFiles/validate_ascii_with_errors_tests.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/validate_ascii_with_errors_tests.dir/clean

tests/CMakeFiles/validate_ascii_with_errors_tests.dir/depend:
	cd /home/leorio/simdutf/builds390 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/leorio/simdutf /home/leorio/simdutf/tests /home/leorio/simdutf/builds390 /home/leorio/simdutf/builds390/tests /home/leorio/simdutf/builds390/tests/CMakeFiles/validate_ascii_with_errors_tests.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/validate_ascii_with_errors_tests.dir/depend

