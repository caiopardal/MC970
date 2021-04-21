#!/bin/sh -l

# Compile PLASMA
export CC=clang
export CXX=clang++

mkdir build
cd build
cmake ..
make -j24

# Execute PLASMA within OmpCluster container
./plasmatest spotri --dim=256 --nb=64
