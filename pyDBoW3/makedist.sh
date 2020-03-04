#!/bin/bash


cd build
mkdir dist
cd dist
mkdir pyDBoW3
cp ../pyDBoW3.so pyDBoW3
cp ../../src/__init__.py pyDBoW3
cp ../../src/setup.py .
cp ../../src/MANIFEST.in .

if [[ "$BUILD_PYTHON" == "python3" ]]; then
    python3 setup.py bdist_wheel
else
    python3 setup.py bdist_wheel
fi

cd ../..
