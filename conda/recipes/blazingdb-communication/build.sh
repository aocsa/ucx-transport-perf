#!/bin/bash
build_type="Release"
buffer_size=128
if [ ! -z $1 ]; then
  buffer_size=$1
fi

if [ ! -d "build" ]; then
  mkdir build
fi
cd build
cmake .. -DCMAKE_BUILD_TYPE=$build_type -DBUFFER_SIZE=$buffer_size
make -j
