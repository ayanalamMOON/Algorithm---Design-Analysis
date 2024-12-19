#!/bin/bash

# Build script for the Algorithm Design and Analysis project

# Compiler
CXX=g++

# Compiler flags
CXXFLAGS="-std=c++17 -O2"

# Source files
SOURCES="test2.cpp"

# Output executable
OUTPUT="algorithm_project"

# Compile the source files
$CXX $CXXFLAGS -o $OUTPUT $SOURCES
