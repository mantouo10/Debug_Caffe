# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

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
CMAKE_COMMAND = /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/usrp/lstmcaffe

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/usrp/lstmcaffe/cmake-build-debug

# Include any dependencies generated for this target.
include tools/CMakeFiles/net_speed_benchmark.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/net_speed_benchmark.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/net_speed_benchmark.dir/flags.make

tools/CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.o: tools/CMakeFiles/net_speed_benchmark.dir/flags.make
tools/CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.o: ../tools/net_speed_benchmark.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/usrp/lstmcaffe/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.o"
	cd /home/usrp/lstmcaffe/cmake-build-debug/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.o -c /home/usrp/lstmcaffe/tools/net_speed_benchmark.cpp

tools/CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.i"
	cd /home/usrp/lstmcaffe/cmake-build-debug/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/usrp/lstmcaffe/tools/net_speed_benchmark.cpp > CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.i

tools/CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.s"
	cd /home/usrp/lstmcaffe/cmake-build-debug/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/usrp/lstmcaffe/tools/net_speed_benchmark.cpp -o CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.s

tools/CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.o.requires:

.PHONY : tools/CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.o.requires

tools/CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.o.provides: tools/CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.o.requires
	$(MAKE) -f tools/CMakeFiles/net_speed_benchmark.dir/build.make tools/CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.o.provides.build
.PHONY : tools/CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.o.provides

tools/CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.o.provides.build: tools/CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.o


# Object files for target net_speed_benchmark
net_speed_benchmark_OBJECTS = \
"CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.o"

# External object files for target net_speed_benchmark
net_speed_benchmark_EXTERNAL_OBJECTS =

tools/net_speed_benchmark-d: tools/CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.o
tools/net_speed_benchmark-d: tools/CMakeFiles/net_speed_benchmark.dir/build.make
tools/net_speed_benchmark-d: lib/libcaffe-d.so.1.0.0
tools/net_speed_benchmark-d: lib/libcaffeproto-d.a
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libboost_system.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libboost_thread.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libglog.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libprotobuf.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libhdf5_cpp.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libhdf5.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libz.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libdl.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libm.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libhdf5_hl_cpp.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/liblmdb.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libleveldb.so
tools/net_speed_benchmark-d: /usr/local/lib/libopencv_highgui.so.3.1.0
tools/net_speed_benchmark-d: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
tools/net_speed_benchmark-d: /usr/local/lib/libopencv_imgproc.so.3.1.0
tools/net_speed_benchmark-d: /usr/local/lib/libopencv_core.so.3.1.0
tools/net_speed_benchmark-d: /usr/lib/liblapack.so
tools/net_speed_benchmark-d: /usr/lib/libcblas.so
tools/net_speed_benchmark-d: /usr/lib/libatlas.so
tools/net_speed_benchmark-d: /usr/lib/x86_64-linux-gnu/libboost_python.so
tools/net_speed_benchmark-d: tools/CMakeFiles/net_speed_benchmark.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/usrp/lstmcaffe/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable net_speed_benchmark-d"
	cd /home/usrp/lstmcaffe/cmake-build-debug/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/net_speed_benchmark.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/net_speed_benchmark.dir/build: tools/net_speed_benchmark-d

.PHONY : tools/CMakeFiles/net_speed_benchmark.dir/build

tools/CMakeFiles/net_speed_benchmark.dir/requires: tools/CMakeFiles/net_speed_benchmark.dir/net_speed_benchmark.cpp.o.requires

.PHONY : tools/CMakeFiles/net_speed_benchmark.dir/requires

tools/CMakeFiles/net_speed_benchmark.dir/clean:
	cd /home/usrp/lstmcaffe/cmake-build-debug/tools && $(CMAKE_COMMAND) -P CMakeFiles/net_speed_benchmark.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/net_speed_benchmark.dir/clean

tools/CMakeFiles/net_speed_benchmark.dir/depend:
	cd /home/usrp/lstmcaffe/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/usrp/lstmcaffe /home/usrp/lstmcaffe/tools /home/usrp/lstmcaffe/cmake-build-debug /home/usrp/lstmcaffe/cmake-build-debug/tools /home/usrp/lstmcaffe/cmake-build-debug/tools/CMakeFiles/net_speed_benchmark.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/net_speed_benchmark.dir/depend

