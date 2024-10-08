# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/leorio/simdutf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/leorio/simdutf/buildclang

# Include any dependencies generated for this target.
include singleheader/CMakeFiles/amalgamation_demo.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include singleheader/CMakeFiles/amalgamation_demo.dir/compiler_depend.make

# Include the progress variables for this target.
include singleheader/CMakeFiles/amalgamation_demo.dir/progress.make

# Include the compile flags for this target's objects.
include singleheader/CMakeFiles/amalgamation_demo.dir/flags.make

singleheader/simdutf.cpp: ../singleheader/amalgamate.py
singleheader/simdutf.cpp: src/libsimdutf.a
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/leorio/simdutf/buildclang/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating simdutf.cpp, simdutf.h, amalgamation_demo.cpp, README.md"
	cd /home/leorio/simdutf/buildclang/singleheader && /usr/bin/cmake -E env AMALGAMATE_SOURCE_PATH=/home/leorio/simdutf/src AMALGAMATE_INPUT_PATH=/home/leorio/simdutf/include AMALGAMATE_OUTPUT_PATH=/home/leorio/simdutf/buildclang/singleheader /usr/bin/python3.9 /home/leorio/simdutf/singleheader/amalgamate.py

singleheader/simdutf.h: singleheader/simdutf.cpp
	@$(CMAKE_COMMAND) -E touch_nocreate singleheader/simdutf.h

singleheader/amalgamation_demo.cpp: singleheader/simdutf.cpp
	@$(CMAKE_COMMAND) -E touch_nocreate singleheader/amalgamation_demo.cpp

singleheader/README.md: singleheader/simdutf.cpp
	@$(CMAKE_COMMAND) -E touch_nocreate singleheader/README.md

singleheader/CMakeFiles/amalgamation_demo.dir/amalgamation_demo.cpp.o: singleheader/CMakeFiles/amalgamation_demo.dir/flags.make
singleheader/CMakeFiles/amalgamation_demo.dir/amalgamation_demo.cpp.o: singleheader/amalgamation_demo.cpp
singleheader/CMakeFiles/amalgamation_demo.dir/amalgamation_demo.cpp.o: singleheader/CMakeFiles/amalgamation_demo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/leorio/simdutf/buildclang/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object singleheader/CMakeFiles/amalgamation_demo.dir/amalgamation_demo.cpp.o"
	cd /home/leorio/simdutf/buildclang/singleheader && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT singleheader/CMakeFiles/amalgamation_demo.dir/amalgamation_demo.cpp.o -MF CMakeFiles/amalgamation_demo.dir/amalgamation_demo.cpp.o.d -o CMakeFiles/amalgamation_demo.dir/amalgamation_demo.cpp.o -c /home/leorio/simdutf/buildclang/singleheader/amalgamation_demo.cpp

singleheader/CMakeFiles/amalgamation_demo.dir/amalgamation_demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/amalgamation_demo.dir/amalgamation_demo.cpp.i"
	cd /home/leorio/simdutf/buildclang/singleheader && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/leorio/simdutf/buildclang/singleheader/amalgamation_demo.cpp > CMakeFiles/amalgamation_demo.dir/amalgamation_demo.cpp.i

singleheader/CMakeFiles/amalgamation_demo.dir/amalgamation_demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/amalgamation_demo.dir/amalgamation_demo.cpp.s"
	cd /home/leorio/simdutf/buildclang/singleheader && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/leorio/simdutf/buildclang/singleheader/amalgamation_demo.cpp -o CMakeFiles/amalgamation_demo.dir/amalgamation_demo.cpp.s

# Object files for target amalgamation_demo
amalgamation_demo_OBJECTS = \
"CMakeFiles/amalgamation_demo.dir/amalgamation_demo.cpp.o"

# External object files for target amalgamation_demo
amalgamation_demo_EXTERNAL_OBJECTS =

singleheader/amalgamation_demo: singleheader/CMakeFiles/amalgamation_demo.dir/amalgamation_demo.cpp.o
singleheader/amalgamation_demo: singleheader/CMakeFiles/amalgamation_demo.dir/build.make
singleheader/amalgamation_demo: singleheader/CMakeFiles/amalgamation_demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/leorio/simdutf/buildclang/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable amalgamation_demo"
	cd /home/leorio/simdutf/buildclang/singleheader && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/amalgamation_demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
singleheader/CMakeFiles/amalgamation_demo.dir/build: singleheader/amalgamation_demo
.PHONY : singleheader/CMakeFiles/amalgamation_demo.dir/build

singleheader/CMakeFiles/amalgamation_demo.dir/clean:
	cd /home/leorio/simdutf/buildclang/singleheader && $(CMAKE_COMMAND) -P CMakeFiles/amalgamation_demo.dir/cmake_clean.cmake
.PHONY : singleheader/CMakeFiles/amalgamation_demo.dir/clean

singleheader/CMakeFiles/amalgamation_demo.dir/depend: singleheader/README.md
singleheader/CMakeFiles/amalgamation_demo.dir/depend: singleheader/amalgamation_demo.cpp
singleheader/CMakeFiles/amalgamation_demo.dir/depend: singleheader/simdutf.cpp
singleheader/CMakeFiles/amalgamation_demo.dir/depend: singleheader/simdutf.h
	cd /home/leorio/simdutf/buildclang && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/leorio/simdutf /home/leorio/simdutf/singleheader /home/leorio/simdutf/buildclang /home/leorio/simdutf/buildclang/singleheader /home/leorio/simdutf/buildclang/singleheader/CMakeFiles/amalgamation_demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : singleheader/CMakeFiles/amalgamation_demo.dir/depend

