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
include tools/CMakeFiles/sutf.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/sutf.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/sutf.dir/flags.make

tools/CMakeFiles/sutf.dir/sutf.cpp.o: tools/CMakeFiles/sutf.dir/flags.make
tools/CMakeFiles/sutf.dir/sutf.cpp.o: ../tools/sutf.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/leorio/simdutf/builds390/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/CMakeFiles/sutf.dir/sutf.cpp.o"
	cd /home/leorio/simdutf/builds390/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sutf.dir/sutf.cpp.o -c /home/leorio/simdutf/tools/sutf.cpp

tools/CMakeFiles/sutf.dir/sutf.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sutf.dir/sutf.cpp.i"
	cd /home/leorio/simdutf/builds390/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/leorio/simdutf/tools/sutf.cpp > CMakeFiles/sutf.dir/sutf.cpp.i

tools/CMakeFiles/sutf.dir/sutf.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sutf.dir/sutf.cpp.s"
	cd /home/leorio/simdutf/builds390/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/leorio/simdutf/tools/sutf.cpp -o CMakeFiles/sutf.dir/sutf.cpp.s

# Object files for target sutf
sutf_OBJECTS = \
"CMakeFiles/sutf.dir/sutf.cpp.o"

# External object files for target sutf
sutf_EXTERNAL_OBJECTS =

tools/sutf: tools/CMakeFiles/sutf.dir/sutf.cpp.o
tools/sutf: tools/CMakeFiles/sutf.dir/build.make
tools/sutf: src/libsimdutf.a
tools/sutf: tools/CMakeFiles/sutf.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/leorio/simdutf/builds390/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable sutf"
	cd /home/leorio/simdutf/builds390/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sutf.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/sutf.dir/build: tools/sutf

.PHONY : tools/CMakeFiles/sutf.dir/build

tools/CMakeFiles/sutf.dir/clean:
	cd /home/leorio/simdutf/builds390/tools && $(CMAKE_COMMAND) -P CMakeFiles/sutf.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/sutf.dir/clean

tools/CMakeFiles/sutf.dir/depend:
	cd /home/leorio/simdutf/builds390 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/leorio/simdutf /home/leorio/simdutf/tools /home/leorio/simdutf/builds390 /home/leorio/simdutf/builds390/tools /home/leorio/simdutf/builds390/tools/CMakeFiles/sutf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/sutf.dir/depend

