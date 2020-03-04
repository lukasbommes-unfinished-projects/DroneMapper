#!/bin/bash

# Install DBow3
CWD=`pwd`
git clone https://github.com/rmsalinas/DBow3 DBow3
cd DBow3
mkdir build
cd build

cmake -DOpenCV_DIR=$CWD/opencv3/build \
      -DBUILD_SHARED_LIBS=OFF \
      -DUSE_CONTRIB=ON \
      -DCMAKE_INSTALL_PREFIX=$CWD/DBow3 \
      -DCMAKE_CXX_FLAGS="-fPIC" \
      -DCMAKE_C_FLAGS="-fPIC" \
      -DBUILD_UTILS=OFF .. && make && make install
cd ../..
