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
CMAKE_COMMAND = /home/dimon/clion-2020.1/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/dimon/clion-2020.1/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dimon/VectorGraphLibrary

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dimon/VectorGraphLibrary/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/VectorGraphLibrary.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/VectorGraphLibrary.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/VectorGraphLibrary.dir/flags.make

CMakeFiles/VectorGraphLibrary.dir/apps/analyse_graph.cpp.o: CMakeFiles/VectorGraphLibrary.dir/flags.make
CMakeFiles/VectorGraphLibrary.dir/apps/analyse_graph.cpp.o: ../apps/analyse_graph.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimon/VectorGraphLibrary/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/VectorGraphLibrary.dir/apps/analyse_graph.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VectorGraphLibrary.dir/apps/analyse_graph.cpp.o -c /home/dimon/VectorGraphLibrary/apps/analyse_graph.cpp

CMakeFiles/VectorGraphLibrary.dir/apps/analyse_graph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VectorGraphLibrary.dir/apps/analyse_graph.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimon/VectorGraphLibrary/apps/analyse_graph.cpp > CMakeFiles/VectorGraphLibrary.dir/apps/analyse_graph.cpp.i

CMakeFiles/VectorGraphLibrary.dir/apps/analyse_graph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VectorGraphLibrary.dir/apps/analyse_graph.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimon/VectorGraphLibrary/apps/analyse_graph.cpp -o CMakeFiles/VectorGraphLibrary.dir/apps/analyse_graph.cpp.s

CMakeFiles/VectorGraphLibrary.dir/apps/bfs.cpp.o: CMakeFiles/VectorGraphLibrary.dir/flags.make
CMakeFiles/VectorGraphLibrary.dir/apps/bfs.cpp.o: ../apps/bfs.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimon/VectorGraphLibrary/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/VectorGraphLibrary.dir/apps/bfs.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VectorGraphLibrary.dir/apps/bfs.cpp.o -c /home/dimon/VectorGraphLibrary/apps/bfs.cpp

CMakeFiles/VectorGraphLibrary.dir/apps/bfs.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VectorGraphLibrary.dir/apps/bfs.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimon/VectorGraphLibrary/apps/bfs.cpp > CMakeFiles/VectorGraphLibrary.dir/apps/bfs.cpp.i

CMakeFiles/VectorGraphLibrary.dir/apps/bfs.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VectorGraphLibrary.dir/apps/bfs.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimon/VectorGraphLibrary/apps/bfs.cpp -o CMakeFiles/VectorGraphLibrary.dir/apps/bfs.cpp.s

CMakeFiles/VectorGraphLibrary.dir/apps/cc.cpp.o: CMakeFiles/VectorGraphLibrary.dir/flags.make
CMakeFiles/VectorGraphLibrary.dir/apps/cc.cpp.o: ../apps/cc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimon/VectorGraphLibrary/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/VectorGraphLibrary.dir/apps/cc.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VectorGraphLibrary.dir/apps/cc.cpp.o -c /home/dimon/VectorGraphLibrary/apps/cc.cpp

CMakeFiles/VectorGraphLibrary.dir/apps/cc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VectorGraphLibrary.dir/apps/cc.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimon/VectorGraphLibrary/apps/cc.cpp > CMakeFiles/VectorGraphLibrary.dir/apps/cc.cpp.i

CMakeFiles/VectorGraphLibrary.dir/apps/cc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VectorGraphLibrary.dir/apps/cc.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimon/VectorGraphLibrary/apps/cc.cpp -o CMakeFiles/VectorGraphLibrary.dir/apps/cc.cpp.s

CMakeFiles/VectorGraphLibrary.dir/apps/generate_test_data.cpp.o: CMakeFiles/VectorGraphLibrary.dir/flags.make
CMakeFiles/VectorGraphLibrary.dir/apps/generate_test_data.cpp.o: ../apps/generate_test_data.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimon/VectorGraphLibrary/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/VectorGraphLibrary.dir/apps/generate_test_data.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VectorGraphLibrary.dir/apps/generate_test_data.cpp.o -c /home/dimon/VectorGraphLibrary/apps/generate_test_data.cpp

CMakeFiles/VectorGraphLibrary.dir/apps/generate_test_data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VectorGraphLibrary.dir/apps/generate_test_data.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimon/VectorGraphLibrary/apps/generate_test_data.cpp > CMakeFiles/VectorGraphLibrary.dir/apps/generate_test_data.cpp.i

CMakeFiles/VectorGraphLibrary.dir/apps/generate_test_data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VectorGraphLibrary.dir/apps/generate_test_data.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimon/VectorGraphLibrary/apps/generate_test_data.cpp -o CMakeFiles/VectorGraphLibrary.dir/apps/generate_test_data.cpp.s

CMakeFiles/VectorGraphLibrary.dir/apps/kcore.cpp.o: CMakeFiles/VectorGraphLibrary.dir/flags.make
CMakeFiles/VectorGraphLibrary.dir/apps/kcore.cpp.o: ../apps/kcore.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimon/VectorGraphLibrary/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/VectorGraphLibrary.dir/apps/kcore.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VectorGraphLibrary.dir/apps/kcore.cpp.o -c /home/dimon/VectorGraphLibrary/apps/kcore.cpp

CMakeFiles/VectorGraphLibrary.dir/apps/kcore.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VectorGraphLibrary.dir/apps/kcore.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimon/VectorGraphLibrary/apps/kcore.cpp > CMakeFiles/VectorGraphLibrary.dir/apps/kcore.cpp.i

CMakeFiles/VectorGraphLibrary.dir/apps/kcore.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VectorGraphLibrary.dir/apps/kcore.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimon/VectorGraphLibrary/apps/kcore.cpp -o CMakeFiles/VectorGraphLibrary.dir/apps/kcore.cpp.s

CMakeFiles/VectorGraphLibrary.dir/apps/pr.cpp.o: CMakeFiles/VectorGraphLibrary.dir/flags.make
CMakeFiles/VectorGraphLibrary.dir/apps/pr.cpp.o: ../apps/pr.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimon/VectorGraphLibrary/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/VectorGraphLibrary.dir/apps/pr.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VectorGraphLibrary.dir/apps/pr.cpp.o -c /home/dimon/VectorGraphLibrary/apps/pr.cpp

CMakeFiles/VectorGraphLibrary.dir/apps/pr.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VectorGraphLibrary.dir/apps/pr.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimon/VectorGraphLibrary/apps/pr.cpp > CMakeFiles/VectorGraphLibrary.dir/apps/pr.cpp.i

CMakeFiles/VectorGraphLibrary.dir/apps/pr.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VectorGraphLibrary.dir/apps/pr.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimon/VectorGraphLibrary/apps/pr.cpp -o CMakeFiles/VectorGraphLibrary.dir/apps/pr.cpp.s

CMakeFiles/VectorGraphLibrary.dir/apps/sssp.cpp.o: CMakeFiles/VectorGraphLibrary.dir/flags.make
CMakeFiles/VectorGraphLibrary.dir/apps/sssp.cpp.o: ../apps/sssp.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimon/VectorGraphLibrary/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/VectorGraphLibrary.dir/apps/sssp.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VectorGraphLibrary.dir/apps/sssp.cpp.o -c /home/dimon/VectorGraphLibrary/apps/sssp.cpp

CMakeFiles/VectorGraphLibrary.dir/apps/sssp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VectorGraphLibrary.dir/apps/sssp.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimon/VectorGraphLibrary/apps/sssp.cpp > CMakeFiles/VectorGraphLibrary.dir/apps/sssp.cpp.i

CMakeFiles/VectorGraphLibrary.dir/apps/sssp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VectorGraphLibrary.dir/apps/sssp.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimon/VectorGraphLibrary/apps/sssp.cpp -o CMakeFiles/VectorGraphLibrary.dir/apps/sssp.cpp.s

CMakeFiles/VectorGraphLibrary.dir/apps/sswp.cpp.o: CMakeFiles/VectorGraphLibrary.dir/flags.make
CMakeFiles/VectorGraphLibrary.dir/apps/sswp.cpp.o: ../apps/sswp.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimon/VectorGraphLibrary/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/VectorGraphLibrary.dir/apps/sswp.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VectorGraphLibrary.dir/apps/sswp.cpp.o -c /home/dimon/VectorGraphLibrary/apps/sswp.cpp

CMakeFiles/VectorGraphLibrary.dir/apps/sswp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VectorGraphLibrary.dir/apps/sswp.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimon/VectorGraphLibrary/apps/sswp.cpp > CMakeFiles/VectorGraphLibrary.dir/apps/sswp.cpp.i

CMakeFiles/VectorGraphLibrary.dir/apps/sswp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VectorGraphLibrary.dir/apps/sswp.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimon/VectorGraphLibrary/apps/sswp.cpp -o CMakeFiles/VectorGraphLibrary.dir/apps/sswp.cpp.s

CMakeFiles/VectorGraphLibrary.dir/misc/custom_test.cpp.o: CMakeFiles/VectorGraphLibrary.dir/flags.make
CMakeFiles/VectorGraphLibrary.dir/misc/custom_test.cpp.o: ../misc/custom_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimon/VectorGraphLibrary/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/VectorGraphLibrary.dir/misc/custom_test.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VectorGraphLibrary.dir/misc/custom_test.cpp.o -c /home/dimon/VectorGraphLibrary/misc/custom_test.cpp

CMakeFiles/VectorGraphLibrary.dir/misc/custom_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VectorGraphLibrary.dir/misc/custom_test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimon/VectorGraphLibrary/misc/custom_test.cpp > CMakeFiles/VectorGraphLibrary.dir/misc/custom_test.cpp.i

CMakeFiles/VectorGraphLibrary.dir/misc/custom_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VectorGraphLibrary.dir/misc/custom_test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimon/VectorGraphLibrary/misc/custom_test.cpp -o CMakeFiles/VectorGraphLibrary.dir/misc/custom_test.cpp.s

CMakeFiles/VectorGraphLibrary.dir/misc/generate_memory_traces.cpp.o: CMakeFiles/VectorGraphLibrary.dir/flags.make
CMakeFiles/VectorGraphLibrary.dir/misc/generate_memory_traces.cpp.o: ../misc/generate_memory_traces.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimon/VectorGraphLibrary/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/VectorGraphLibrary.dir/misc/generate_memory_traces.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VectorGraphLibrary.dir/misc/generate_memory_traces.cpp.o -c /home/dimon/VectorGraphLibrary/misc/generate_memory_traces.cpp

CMakeFiles/VectorGraphLibrary.dir/misc/generate_memory_traces.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VectorGraphLibrary.dir/misc/generate_memory_traces.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimon/VectorGraphLibrary/misc/generate_memory_traces.cpp > CMakeFiles/VectorGraphLibrary.dir/misc/generate_memory_traces.cpp.i

CMakeFiles/VectorGraphLibrary.dir/misc/generate_memory_traces.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VectorGraphLibrary.dir/misc/generate_memory_traces.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimon/VectorGraphLibrary/misc/generate_memory_traces.cpp -o CMakeFiles/VectorGraphLibrary.dir/misc/generate_memory_traces.cpp.s

CMakeFiles/VectorGraphLibrary.dir/misc/lambda_tests.cpp.o: CMakeFiles/VectorGraphLibrary.dir/flags.make
CMakeFiles/VectorGraphLibrary.dir/misc/lambda_tests.cpp.o: ../misc/lambda_tests.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimon/VectorGraphLibrary/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/VectorGraphLibrary.dir/misc/lambda_tests.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VectorGraphLibrary.dir/misc/lambda_tests.cpp.o -c /home/dimon/VectorGraphLibrary/misc/lambda_tests.cpp

CMakeFiles/VectorGraphLibrary.dir/misc/lambda_tests.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VectorGraphLibrary.dir/misc/lambda_tests.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimon/VectorGraphLibrary/misc/lambda_tests.cpp > CMakeFiles/VectorGraphLibrary.dir/misc/lambda_tests.cpp.i

CMakeFiles/VectorGraphLibrary.dir/misc/lambda_tests.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VectorGraphLibrary.dir/misc/lambda_tests.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimon/VectorGraphLibrary/misc/lambda_tests.cpp -o CMakeFiles/VectorGraphLibrary.dir/misc/lambda_tests.cpp.s

CMakeFiles/VectorGraphLibrary.dir/misc/nvgraph_comparison.cpp.o: CMakeFiles/VectorGraphLibrary.dir/flags.make
CMakeFiles/VectorGraphLibrary.dir/misc/nvgraph_comparison.cpp.o: ../misc/nvgraph_comparison.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimon/VectorGraphLibrary/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/VectorGraphLibrary.dir/misc/nvgraph_comparison.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VectorGraphLibrary.dir/misc/nvgraph_comparison.cpp.o -c /home/dimon/VectorGraphLibrary/misc/nvgraph_comparison.cpp

CMakeFiles/VectorGraphLibrary.dir/misc/nvgraph_comparison.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VectorGraphLibrary.dir/misc/nvgraph_comparison.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimon/VectorGraphLibrary/misc/nvgraph_comparison.cpp > CMakeFiles/VectorGraphLibrary.dir/misc/nvgraph_comparison.cpp.i

CMakeFiles/VectorGraphLibrary.dir/misc/nvgraph_comparison.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VectorGraphLibrary.dir/misc/nvgraph_comparison.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimon/VectorGraphLibrary/misc/nvgraph_comparison.cpp -o CMakeFiles/VectorGraphLibrary.dir/misc/nvgraph_comparison.cpp.s

# Object files for target VectorGraphLibrary
VectorGraphLibrary_OBJECTS = \
"CMakeFiles/VectorGraphLibrary.dir/apps/analyse_graph.cpp.o" \
"CMakeFiles/VectorGraphLibrary.dir/apps/bfs.cpp.o" \
"CMakeFiles/VectorGraphLibrary.dir/apps/cc.cpp.o" \
"CMakeFiles/VectorGraphLibrary.dir/apps/generate_test_data.cpp.o" \
"CMakeFiles/VectorGraphLibrary.dir/apps/kcore.cpp.o" \
"CMakeFiles/VectorGraphLibrary.dir/apps/pr.cpp.o" \
"CMakeFiles/VectorGraphLibrary.dir/apps/sssp.cpp.o" \
"CMakeFiles/VectorGraphLibrary.dir/apps/sswp.cpp.o" \
"CMakeFiles/VectorGraphLibrary.dir/misc/custom_test.cpp.o" \
"CMakeFiles/VectorGraphLibrary.dir/misc/generate_memory_traces.cpp.o" \
"CMakeFiles/VectorGraphLibrary.dir/misc/lambda_tests.cpp.o" \
"CMakeFiles/VectorGraphLibrary.dir/misc/nvgraph_comparison.cpp.o"

# External object files for target VectorGraphLibrary
VectorGraphLibrary_EXTERNAL_OBJECTS =

VectorGraphLibrary: CMakeFiles/VectorGraphLibrary.dir/apps/analyse_graph.cpp.o
VectorGraphLibrary: CMakeFiles/VectorGraphLibrary.dir/apps/bfs.cpp.o
VectorGraphLibrary: CMakeFiles/VectorGraphLibrary.dir/apps/cc.cpp.o
VectorGraphLibrary: CMakeFiles/VectorGraphLibrary.dir/apps/generate_test_data.cpp.o
VectorGraphLibrary: CMakeFiles/VectorGraphLibrary.dir/apps/kcore.cpp.o
VectorGraphLibrary: CMakeFiles/VectorGraphLibrary.dir/apps/pr.cpp.o
VectorGraphLibrary: CMakeFiles/VectorGraphLibrary.dir/apps/sssp.cpp.o
VectorGraphLibrary: CMakeFiles/VectorGraphLibrary.dir/apps/sswp.cpp.o
VectorGraphLibrary: CMakeFiles/VectorGraphLibrary.dir/misc/custom_test.cpp.o
VectorGraphLibrary: CMakeFiles/VectorGraphLibrary.dir/misc/generate_memory_traces.cpp.o
VectorGraphLibrary: CMakeFiles/VectorGraphLibrary.dir/misc/lambda_tests.cpp.o
VectorGraphLibrary: CMakeFiles/VectorGraphLibrary.dir/misc/nvgraph_comparison.cpp.o
VectorGraphLibrary: CMakeFiles/VectorGraphLibrary.dir/build.make
VectorGraphLibrary: CMakeFiles/VectorGraphLibrary.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dimon/VectorGraphLibrary/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Linking CXX executable VectorGraphLibrary"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/VectorGraphLibrary.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/VectorGraphLibrary.dir/build: VectorGraphLibrary

.PHONY : CMakeFiles/VectorGraphLibrary.dir/build

CMakeFiles/VectorGraphLibrary.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/VectorGraphLibrary.dir/cmake_clean.cmake
.PHONY : CMakeFiles/VectorGraphLibrary.dir/clean

CMakeFiles/VectorGraphLibrary.dir/depend:
	cd /home/dimon/VectorGraphLibrary/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dimon/VectorGraphLibrary /home/dimon/VectorGraphLibrary /home/dimon/VectorGraphLibrary/cmake-build-debug /home/dimon/VectorGraphLibrary/cmake-build-debug /home/dimon/VectorGraphLibrary/cmake-build-debug/CMakeFiles/VectorGraphLibrary.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/VectorGraphLibrary.dir/depend

