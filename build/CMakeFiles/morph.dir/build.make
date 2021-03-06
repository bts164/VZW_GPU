# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

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
CMAKE_SOURCE_DIR = /home/brad/Desktop/VZW_Project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/brad/Desktop/VZW_Project/build

# Include any dependencies generated for this target.
include CMakeFiles/morph.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/morph.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/morph.dir/flags.make

CMakeFiles/morph.dir/morph_generated_yuv2rgb.cu.o: CMakeFiles/morph.dir/morph_generated_yuv2rgb.cu.o.depend
CMakeFiles/morph.dir/morph_generated_yuv2rgb.cu.o: CMakeFiles/morph.dir/morph_generated_yuv2rgb.cu.o.cmake
CMakeFiles/morph.dir/morph_generated_yuv2rgb.cu.o: ../yuv2rgb.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brad/Desktop/VZW_Project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/morph.dir/morph_generated_yuv2rgb.cu.o"
	cd /home/brad/Desktop/VZW_Project/build/CMakeFiles/morph.dir && /usr/bin/cmake -E make_directory /home/brad/Desktop/VZW_Project/build/CMakeFiles/morph.dir//.
	cd /home/brad/Desktop/VZW_Project/build/CMakeFiles/morph.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/brad/Desktop/VZW_Project/build/CMakeFiles/morph.dir//./morph_generated_yuv2rgb.cu.o -D generated_cubin_file:STRING=/home/brad/Desktop/VZW_Project/build/CMakeFiles/morph.dir//./morph_generated_yuv2rgb.cu.o.cubin.txt -P /home/brad/Desktop/VZW_Project/build/CMakeFiles/morph.dir//morph_generated_yuv2rgb.cu.o.cmake

CMakeFiles/morph.dir/morph_generated_morph.cu.o: CMakeFiles/morph.dir/morph_generated_morph.cu.o.depend
CMakeFiles/morph.dir/morph_generated_morph.cu.o: CMakeFiles/morph.dir/morph_generated_morph.cu.o.cmake
CMakeFiles/morph.dir/morph_generated_morph.cu.o: ../morph.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brad/Desktop/VZW_Project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building NVCC (Device) object CMakeFiles/morph.dir/morph_generated_morph.cu.o"
	cd /home/brad/Desktop/VZW_Project/build/CMakeFiles/morph.dir && /usr/bin/cmake -E make_directory /home/brad/Desktop/VZW_Project/build/CMakeFiles/morph.dir//.
	cd /home/brad/Desktop/VZW_Project/build/CMakeFiles/morph.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/brad/Desktop/VZW_Project/build/CMakeFiles/morph.dir//./morph_generated_morph.cu.o -D generated_cubin_file:STRING=/home/brad/Desktop/VZW_Project/build/CMakeFiles/morph.dir//./morph_generated_morph.cu.o.cubin.txt -P /home/brad/Desktop/VZW_Project/build/CMakeFiles/morph.dir//morph_generated_morph.cu.o.cmake

CMakeFiles/morph.dir/main.cpp.o: CMakeFiles/morph.dir/flags.make
CMakeFiles/morph.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/brad/Desktop/VZW_Project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/morph.dir/main.cpp.o"
	/usr/bin/g++-4.9   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/morph.dir/main.cpp.o -c /home/brad/Desktop/VZW_Project/main.cpp

CMakeFiles/morph.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/morph.dir/main.cpp.i"
	/usr/bin/g++-4.9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/brad/Desktop/VZW_Project/main.cpp > CMakeFiles/morph.dir/main.cpp.i

CMakeFiles/morph.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/morph.dir/main.cpp.s"
	/usr/bin/g++-4.9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/brad/Desktop/VZW_Project/main.cpp -o CMakeFiles/morph.dir/main.cpp.s

CMakeFiles/morph.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/morph.dir/main.cpp.o.requires

CMakeFiles/morph.dir/main.cpp.o.provides: CMakeFiles/morph.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/morph.dir/build.make CMakeFiles/morph.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/morph.dir/main.cpp.o.provides

CMakeFiles/morph.dir/main.cpp.o.provides.build: CMakeFiles/morph.dir/main.cpp.o


# Object files for target morph
morph_OBJECTS = \
"CMakeFiles/morph.dir/main.cpp.o"

# External object files for target morph
morph_EXTERNAL_OBJECTS = \
"/home/brad/Desktop/VZW_Project/build/CMakeFiles/morph.dir/morph_generated_yuv2rgb.cu.o" \
"/home/brad/Desktop/VZW_Project/build/CMakeFiles/morph.dir/morph_generated_morph.cu.o"

morph: CMakeFiles/morph.dir/main.cpp.o
morph: CMakeFiles/morph.dir/morph_generated_yuv2rgb.cu.o
morph: CMakeFiles/morph.dir/morph_generated_morph.cu.o
morph: CMakeFiles/morph.dir/build.make
morph: /opt/cuda/lib64/libcudart_static.a
morph: /usr/lib64/librt.so
morph: CMakeFiles/morph.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/brad/Desktop/VZW_Project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable morph"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/morph.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/morph.dir/build: morph

.PHONY : CMakeFiles/morph.dir/build

CMakeFiles/morph.dir/requires: CMakeFiles/morph.dir/main.cpp.o.requires

.PHONY : CMakeFiles/morph.dir/requires

CMakeFiles/morph.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/morph.dir/cmake_clean.cmake
.PHONY : CMakeFiles/morph.dir/clean

CMakeFiles/morph.dir/depend: CMakeFiles/morph.dir/morph_generated_yuv2rgb.cu.o
CMakeFiles/morph.dir/depend: CMakeFiles/morph.dir/morph_generated_morph.cu.o
	cd /home/brad/Desktop/VZW_Project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/brad/Desktop/VZW_Project /home/brad/Desktop/VZW_Project /home/brad/Desktop/VZW_Project/build /home/brad/Desktop/VZW_Project/build /home/brad/Desktop/VZW_Project/build/CMakeFiles/morph.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/morph.dir/depend

