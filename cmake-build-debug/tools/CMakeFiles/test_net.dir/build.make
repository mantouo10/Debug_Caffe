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
include tools/CMakeFiles/test_net.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/test_net.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/test_net.dir/flags.make

tools/CMakeFiles/test_net.dir/test_net.cpp.o: tools/CMakeFiles/test_net.dir/flags.make
tools/CMakeFiles/test_net.dir/test_net.cpp.o: ../tools/test_net.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/usrp/lstmcaffe/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/CMakeFiles/test_net.dir/test_net.cpp.o"
	cd /home/usrp/lstmcaffe/cmake-build-debug/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_net.dir/test_net.cpp.o -c /home/usrp/lstmcaffe/tools/test_net.cpp

tools/CMakeFiles/test_net.dir/test_net.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_net.dir/test_net.cpp.i"
	cd /home/usrp/lstmcaffe/cmake-build-debug/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/usrp/lstmcaffe/tools/test_net.cpp > CMakeFiles/test_net.dir/test_net.cpp.i

tools/CMakeFiles/test_net.dir/test_net.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_net.dir/test_net.cpp.s"
	cd /home/usrp/lstmcaffe/cmake-build-debug/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/usrp/lstmcaffe/tools/test_net.cpp -o CMakeFiles/test_net.dir/test_net.cpp.s

tools/CMakeFiles/test_net.dir/test_net.cpp.o.requires:

.PHONY : tools/CMakeFiles/test_net.dir/test_net.cpp.o.requires

tools/CMakeFiles/test_net.dir/test_net.cpp.o.provides: tools/CMakeFiles/test_net.dir/test_net.cpp.o.requires
	$(MAKE) -f tools/CMakeFiles/test_net.dir/build.make tools/CMakeFiles/test_net.dir/test_net.cpp.o.provides.build
.PHONY : tools/CMakeFiles/test_net.dir/test_net.cpp.o.provides

tools/CMakeFiles/test_net.dir/test_net.cpp.o.provides.build: tools/CMakeFiles/test_net.dir/test_net.cpp.o


# Object files for target test_net
test_net_OBJECTS = \
"CMakeFiles/test_net.dir/test_net.cpp.o"

# External object files for target test_net
test_net_EXTERNAL_OBJECTS =

tools/test_net-d: tools/CMakeFiles/test_net.dir/test_net.cpp.o
tools/test_net-d: tools/CMakeFiles/test_net.dir/build.make
tools/test_net-d: lib/libcaffe-d.so.1.0.0
tools/test_net-d: lib/libcaffeproto-d.a
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libboost_system.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libboost_thread.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libglog.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libprotobuf.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libhdf5_cpp.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libhdf5.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libz.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libdl.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libm.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libhdf5_hl_cpp.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/liblmdb.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libleveldb.so
tools/test_net-d: /usr/local/lib/libopencv_highgui.so.3.1.0
tools/test_net-d: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
tools/test_net-d: /usr/local/lib/libopencv_imgproc.so.3.1.0
tools/test_net-d: /usr/local/lib/libopencv_core.so.3.1.0
tools/test_net-d: /usr/lib/liblapack.so
tools/test_net-d: /usr/lib/libcblas.so
tools/test_net-d: /usr/lib/libatlas.so
tools/test_net-d: /usr/lib/x86_64-linux-gnu/libboost_python.so
tools/test_net-d: tools/CMakeFiles/test_net.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/usrp/lstmcaffe/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_net-d"
	cd /home/usrp/lstmcaffe/cmake-build-debug/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_net.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/test_net.dir/build: tools/test_net-d

.PHONY : tools/CMakeFiles/test_net.dir/build

tools/CMakeFiles/test_net.dir/requires: tools/CMakeFiles/test_net.dir/test_net.cpp.o.requires

.PHONY : tools/CMakeFiles/test_net.dir/requires

tools/CMakeFiles/test_net.dir/clean:
	cd /home/usrp/lstmcaffe/cmake-build-debug/tools && $(CMAKE_COMMAND) -P CMakeFiles/test_net.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/test_net.dir/clean

tools/CMakeFiles/test_net.dir/depend:
	cd /home/usrp/lstmcaffe/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/usrp/lstmcaffe /home/usrp/lstmcaffe/tools /home/usrp/lstmcaffe/cmake-build-debug /home/usrp/lstmcaffe/cmake-build-debug/tools /home/usrp/lstmcaffe/cmake-build-debug/tools/CMakeFiles/test_net.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/test_net.dir/depend

