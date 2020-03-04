cd install

call dbow3.bat

cd ..
mkdir build
cd build

cmake -DBUILD_PYTHON3=ON -DBUILD_STATICALLY_LINKED=OFF -DBOOST_ROOT=c:\boost_1_67_0 -DOpenCV_DIR=c:/opencv/build -DDBoW3_DIR=../install/DBow3/build -DDBoW3_INCLUDE_DIRS=../install/DBow3/src -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 15 2017 Win64" ../src

cd ..