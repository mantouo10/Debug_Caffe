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
include tools/CMakeFiles/convert_imageset.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/convert_imageset.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/convert_imageset.dir/flags.make

tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o: tools/CMakeFiles/convert_imageset.dir/flags.make
tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o: ../tools/convert_imageset.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/usrp/lstmcaffe/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o"
	cd /home/usrp/lstmcaffe/cmake-build-debug/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o -c /home/usrp/lstmcaffe/tools/convert_imageset.cpp

tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/convert_imageset.dir/convert_imageset.cpp.i"
	cd /home/usrp/lstmcaffe/cmake-build-debug/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/usrp/lstmcaffe/tools/convert_imageset.cpp > CMakeFiles/convert_imageset.dir/convert_imageset.cpp.i

tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/convert_imageset.dir/convert_imageset.cpp.s"
	cd /home/usrp/lstmcaffe/cmake-build-debug/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/usrp/lstmcaffe/tools/convert_imageset.cpp -o CMakeFiles/convert_imageset.dir/convert_imageset.cpp.s

tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o.requires:

.PHONY : tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o.requires

tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o.provides: tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o.requires
	$(MAKE) -f tools/CMakeFiles/convert_imageset.dir/build.make tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o.provides.build
.PHONY : tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o.provides

tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o.provides.build: tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o


# Object files for target convert_imageset
convert_imageset_OBJECTS = \
"CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o"

# External object files for target convert_imageset
convert_imageset_EXTERNAL_OBJECTS =

tools/convert_imageset-d: tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o
tools/convert_imageset-d: tools/CMakeFiles/convert_imageset.dir/build.make
tools/convert_imageset-d: lib/libcaffe-d.so.1.0.0
tools/convert_imageset-d: lib/libcaffeproto-d.a
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libboost_system.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libboost_thread.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libglog.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libprotobuf.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libhdf5_cpp.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libhdf5.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libz.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libdl.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libm.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libhdf5_hl_cpp.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/liblmdb.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libleveldb.so
tools/convert_imageset-d: /usr/local/lib/libopencv_highgui.so.3.1.0
tools/convert_imageset-d: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
tools/convert_imageset-d: /usr/local/lib/libopencv_imgproc.so.3.1.0
tools/convert_imageset-d: /usr/local/lib/libopencv_core.so.3.1.0
tools/convert_imageset-d: /usr/lib/liblapack.so
tools/convert_imageset-d: /usr/lib/libcblas.so
tools/convert_imageset-d: /usr/lib/libatlas.so
tools/convert_imageset-d: /usr/lib/x86_64-linux-gnu/libboost_python.so
tools/convert_imageset-d: tools/CMakeFiles/convert_imageset.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/usrp/lstmcaffe/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable convert_imageset-d"
	cd /home/usrp/lstmcaffe/cmake-build-debug/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/convert_imageset.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/convert_imageset.dir/build: tools/convert_imageset-d

.PHONY : tools/CMakeFiles/convert_imageset.dir/build

tools/CMakeFiles/convert_imageset.dir/requires: tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o.requires

.PHONY : tools/CMakeFiles/convert_imageset.dir/requires

tools/CMakeFiles/convert_imageset.dir/clean:
	cd /home/usrp/lstmcaffe/cmake-build-debug/tools && $(CMAKE_COMMAND) -P CMakeFiles/convert_imageset.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/convert_imageset.dir/clean

tools/CMakeFiles/convert_imageset.dir/depend:
	cd /home/usrp/lstmcaffe/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/usrp/lstmcaffe /home/usrp/lstmcaffe/tools /home/usrp/lstmcaffe/cmake-build-debug /home/usrp/lstmcaffe/cmake-build-debug/tools /home/usrp/lstmcaffe/cmake-build-debug/tools/CMakeFiles/convert_imageset.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/convert_imageset.dir/depend

