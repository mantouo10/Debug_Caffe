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

# Utility rule file for docs.

# Include the progress variables for this target.
include docs/CMakeFiles/docs.dir/progress.make

docs/CMakeFiles/docs:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/usrp/lstmcaffe/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Launching doxygen..."
	cd /home/usrp/lstmcaffe && /usr/local/bin/doxygen /home/usrp/lstmcaffe/.Doxyfile

docs: docs/CMakeFiles/docs
docs: docs/CMakeFiles/docs.dir/build.make
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Copying notebook examples/00-classification.ipynb to /home/usrp/lstmcaffe/docs/gathered/examples/00-classification.ipynb"
	cd /home/usrp/lstmcaffe && /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake -E make_directory /home/usrp/lstmcaffe/docs/gathered/examples
	cd /home/usrp/lstmcaffe && /usr/bin/python2.7 scripts/copy_notebook.py examples/00-classification.ipynb /home/usrp/lstmcaffe/docs/gathered/examples/00-classification.ipynb
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Copying notebook examples/01-learning-lenet.ipynb to /home/usrp/lstmcaffe/docs/gathered/examples/01-learning-lenet.ipynb"
	cd /home/usrp/lstmcaffe && /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake -E make_directory /home/usrp/lstmcaffe/docs/gathered/examples
	cd /home/usrp/lstmcaffe && /usr/bin/python2.7 scripts/copy_notebook.py examples/01-learning-lenet.ipynb /home/usrp/lstmcaffe/docs/gathered/examples/01-learning-lenet.ipynb
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Copying notebook examples/02-fine-tuning.ipynb to /home/usrp/lstmcaffe/docs/gathered/examples/02-fine-tuning.ipynb"
	cd /home/usrp/lstmcaffe && /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake -E make_directory /home/usrp/lstmcaffe/docs/gathered/examples
	cd /home/usrp/lstmcaffe && /usr/bin/python2.7 scripts/copy_notebook.py examples/02-fine-tuning.ipynb /home/usrp/lstmcaffe/docs/gathered/examples/02-fine-tuning.ipynb
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Copying notebook examples/brewing-logreg.ipynb to /home/usrp/lstmcaffe/docs/gathered/examples/brewing-logreg.ipynb"
	cd /home/usrp/lstmcaffe && /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake -E make_directory /home/usrp/lstmcaffe/docs/gathered/examples
	cd /home/usrp/lstmcaffe && /usr/bin/python2.7 scripts/copy_notebook.py examples/brewing-logreg.ipynb /home/usrp/lstmcaffe/docs/gathered/examples/brewing-logreg.ipynb
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Copying notebook examples/detection.ipynb to /home/usrp/lstmcaffe/docs/gathered/examples/detection.ipynb"
	cd /home/usrp/lstmcaffe && /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake -E make_directory /home/usrp/lstmcaffe/docs/gathered/examples
	cd /home/usrp/lstmcaffe && /usr/bin/python2.7 scripts/copy_notebook.py examples/detection.ipynb /home/usrp/lstmcaffe/docs/gathered/examples/detection.ipynb
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Copying notebook examples/net_surgery.ipynb to /home/usrp/lstmcaffe/docs/gathered/examples/net_surgery.ipynb"
	cd /home/usrp/lstmcaffe && /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake -E make_directory /home/usrp/lstmcaffe/docs/gathered/examples
	cd /home/usrp/lstmcaffe && /usr/bin/python2.7 scripts/copy_notebook.py examples/net_surgery.ipynb /home/usrp/lstmcaffe/docs/gathered/examples/net_surgery.ipynb
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Copying notebook examples/pascal-multilabel-with-datalayer.ipynb to /home/usrp/lstmcaffe/docs/gathered/examples/pascal-multilabel-with-datalayer.ipynb"
	cd /home/usrp/lstmcaffe && /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake -E make_directory /home/usrp/lstmcaffe/docs/gathered/examples
	cd /home/usrp/lstmcaffe && /usr/bin/python2.7 scripts/copy_notebook.py examples/pascal-multilabel-with-datalayer.ipynb /home/usrp/lstmcaffe/docs/gathered/examples/pascal-multilabel-with-datalayer.ipynb
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Copying notebook examples/siamese/mnist_siamese.ipynb to /home/usrp/lstmcaffe/docs/gathered/examples/siamese/mnist_siamese.ipynb"
	cd /home/usrp/lstmcaffe && /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake -E make_directory /home/usrp/lstmcaffe/docs/gathered/examples/siamese
	cd /home/usrp/lstmcaffe && /usr/bin/python2.7 scripts/copy_notebook.py examples/siamese/mnist_siamese.ipynb /home/usrp/lstmcaffe/docs/gathered/examples/siamese/mnist_siamese.ipynb
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Creating symlink /home/usrp/lstmcaffe/docs/gathered/examples/cifar10.md -> /home/usrp/lstmcaffe/examples/cifar10/readme.md"
	cd /home/usrp/lstmcaffe && /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake -E make_directory /home/usrp/lstmcaffe/docs/gathered/examples
	cd /home/usrp/lstmcaffe && ln -sf /home/usrp/lstmcaffe/examples/cifar10/readme.md /home/usrp/lstmcaffe/docs/gathered/examples/cifar10.md
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Creating symlink /home/usrp/lstmcaffe/docs/gathered/examples/cpp_classification.md -> /home/usrp/lstmcaffe/examples/cpp_classification/readme.md"
	cd /home/usrp/lstmcaffe && /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake -E make_directory /home/usrp/lstmcaffe/docs/gathered/examples
	cd /home/usrp/lstmcaffe && ln -sf /home/usrp/lstmcaffe/examples/cpp_classification/readme.md /home/usrp/lstmcaffe/docs/gathered/examples/cpp_classification.md
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Creating symlink /home/usrp/lstmcaffe/docs/gathered/examples/feature_extraction.md -> /home/usrp/lstmcaffe/examples/feature_extraction/readme.md"
	cd /home/usrp/lstmcaffe && /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake -E make_directory /home/usrp/lstmcaffe/docs/gathered/examples
	cd /home/usrp/lstmcaffe && ln -sf /home/usrp/lstmcaffe/examples/feature_extraction/readme.md /home/usrp/lstmcaffe/docs/gathered/examples/feature_extraction.md
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Creating symlink /home/usrp/lstmcaffe/docs/gathered/examples/finetune_flickr_style.md -> /home/usrp/lstmcaffe/examples/finetune_flickr_style/readme.md"
	cd /home/usrp/lstmcaffe && /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake -E make_directory /home/usrp/lstmcaffe/docs/gathered/examples
	cd /home/usrp/lstmcaffe && ln -sf /home/usrp/lstmcaffe/examples/finetune_flickr_style/readme.md /home/usrp/lstmcaffe/docs/gathered/examples/finetune_flickr_style.md
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Creating symlink /home/usrp/lstmcaffe/docs/gathered/examples/imagenet.md -> /home/usrp/lstmcaffe/examples/imagenet/readme.md"
	cd /home/usrp/lstmcaffe && /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake -E make_directory /home/usrp/lstmcaffe/docs/gathered/examples
	cd /home/usrp/lstmcaffe && ln -sf /home/usrp/lstmcaffe/examples/imagenet/readme.md /home/usrp/lstmcaffe/docs/gathered/examples/imagenet.md
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Creating symlink /home/usrp/lstmcaffe/docs/gathered/examples/mnist.md -> /home/usrp/lstmcaffe/examples/mnist/readme.md"
	cd /home/usrp/lstmcaffe && /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake -E make_directory /home/usrp/lstmcaffe/docs/gathered/examples
	cd /home/usrp/lstmcaffe && ln -sf /home/usrp/lstmcaffe/examples/mnist/readme.md /home/usrp/lstmcaffe/docs/gathered/examples/mnist.md
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Creating symlink /home/usrp/lstmcaffe/docs/gathered/examples/siamese.md -> /home/usrp/lstmcaffe/examples/siamese/readme.md"
	cd /home/usrp/lstmcaffe && /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake -E make_directory /home/usrp/lstmcaffe/docs/gathered/examples
	cd /home/usrp/lstmcaffe && ln -sf /home/usrp/lstmcaffe/examples/siamese/readme.md /home/usrp/lstmcaffe/docs/gathered/examples/siamese.md
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Creating symlink /home/usrp/lstmcaffe/docs/gathered/examples/web_demo.md -> /home/usrp/lstmcaffe/examples/web_demo/readme.md"
	cd /home/usrp/lstmcaffe && /home/usrp/clion-2017.2.3/bin/cmake/bin/cmake -E make_directory /home/usrp/lstmcaffe/docs/gathered/examples
	cd /home/usrp/lstmcaffe && ln -sf /home/usrp/lstmcaffe/examples/web_demo/readme.md /home/usrp/lstmcaffe/docs/gathered/examples/web_demo.md
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Creating symlink /home/usrp/lstmcaffe/docs/doxygen -> /home/usrp/lstmcaffe/doxygen/html"
	cd /home/usrp/lstmcaffe/docs && ln -sfn /home/usrp/lstmcaffe/doxygen/html doxygen
.PHONY : docs

# Rule to build all files generated by this target.
docs/CMakeFiles/docs.dir/build: docs

.PHONY : docs/CMakeFiles/docs.dir/build

docs/CMakeFiles/docs.dir/clean:
	cd /home/usrp/lstmcaffe/cmake-build-debug/docs && $(CMAKE_COMMAND) -P CMakeFiles/docs.dir/cmake_clean.cmake
.PHONY : docs/CMakeFiles/docs.dir/clean

docs/CMakeFiles/docs.dir/depend:
	cd /home/usrp/lstmcaffe/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/usrp/lstmcaffe /home/usrp/lstmcaffe/docs /home/usrp/lstmcaffe/cmake-build-debug /home/usrp/lstmcaffe/cmake-build-debug/docs /home/usrp/lstmcaffe/cmake-build-debug/docs/CMakeFiles/docs.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : docs/CMakeFiles/docs.dir/depend

