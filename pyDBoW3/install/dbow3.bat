# Install DBow3
git clone https://github.com/rmsalinas/DBow3 DBow3
cd DBow3
mkdir build
cd build

cmake -DLIB_INSTALL_DIR=lib -DOpenCV_DIR=c:\opencv\build -DBUILD_SHARED_LIBS=OFF -DUSE_CONTRIB=ON -DBUILD_UTILS=OFF -G "Visual Studio 15 2017 Win64" .. 

cd ..
cd ..