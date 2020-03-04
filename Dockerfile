FROM ubuntu:18.04 AS builder

RUN apt-get update && \
    apt-get install -y \
        wget \
        unzip \
        build-essential \
        cmake \
        git \
        pkg-config \
        autoconf \
        automake \
        git-core \
        python3-dev \
        python3-pip \
        python3-numpy \
        python3-pkgconfig && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s $(which python3) /usr/local/bin/python

RUN apt-get update && \
    apt-get install -y \
        libgtk-3-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libx265-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libatlas-base-dev \
        gfortran \
        openexr \
        libtbb2 \
        libtbb-dev \
        libdc1394-22-dev \
        libeigen3-dev \
        libvorbis-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home
RUN wget -O /home/opencv.zip https://github.com/opencv/opencv/archive/4.1.0.zip && \
    unzip /home/opencv.zip && \
    mv /home/opencv-4.1.0/ /home/opencv/ && \
    rm -rf /home/opencv.zip && \
    wget -O /home/opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.0.zip && \
    unzip /home/opencv_contrib.zip && \
    mv /home/opencv_contrib-4.1.0/ /home/opencv_contrib/ && \
    rm -rf /home/opencv_contrib.zip

WORKDIR /home/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D OPENCV_GENERATE_PKGCONFIG=YES \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/home/opencv_contrib/modules \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D WITH_OPENCL=ON \
        -D BUILD_TIFF=ON \
        -D WITH_CSTRIPES=ON \
        -D WITH_EIGEN=ON \
        -D WITH_IPP=ON \
        -D WITH_V4L=ON \
        -D WITH_OPENMP=ON \
        -D BUILD_opencv_xfeatures2d=ON \
        .. && \
    make -j $(nproc) && \
    make install && \
    ldconfig

# Install DBoW3
WORKDIR /home
RUN git clone https://github.com/LukasBommes/DBow3.git dbow3
WORKDIR /home/dbow3/build
RUN cmake -DOpenCV_DIR=/home/opencv/build \
        -DBUILD_SHARED_LIBS=OFF \
        -DUSE_CONTRIB=ON \
        -DCMAKE_INSTALL_PREFIX=/home/dbow3 \
        -DCMAKE_CXX_FLAGS="-fPIC" \
        -DCMAKE_C_FLAGS="-fPIC" \
        -DBUILD_UTILS=OFF \
        .. && \
        make -j $(nproc) && \
        make install

# Install Boost
RUN apt-get update && \
    apt-get install -y libboost-all-dev

RUN pip3 install pip --upgrade && \
    pip3 install wheel

# Build pyDBoW3
WORKDIR /home/pydbow3/src
COPY ./pyDBoW3/src .

WORKDIR /home/pydbow3/build
RUN cmake -DBUILD_PYTHON3=ON \
    -DBUILD_STATICALLY_LINKED=OFF \
    -DOpenCV_DIR=/home/opencv/build \
    -DDBoW3_DIR=/home/dbow3/build \
    -DDBoW3_INCLUDE_DIRS=/home/dbow3/src \
    -DCMAKE_BUILD_TYPE=Release \
    ../src && \
    make

# Create pyDBoW3 wheel
WORKDIR /home/pydbow3
COPY ./pyDBoW3 .

WORKDIR /home/pydbow3/build/dist
RUN mkdir pyDBoW3 && \
    cp ../pyDBoW3.so pyDBoW3 && \
    cp ../../src/__init__.py pyDBoW3 && \
    cp ../../src/setup.py . && \
    cp ../../src/MANIFEST.in . && \
    python3 setup.py bdist_wheel


FROM ubuntu:18.04 AS production

RUN apt-get update && \
    apt-get install -y \
        python3-dev \
        python3-pip \
        python3-numpy \
        python3-pkgconfig \
        libboost-all-dev && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s $(which python3) /usr/local/bin/python

# copy Python wheel of pyDBoW3
WORKDIR /home/pydbow3
COPY --from=builder /home/pydbow3/build/dist/dist .

# Install pyDBoW3
RUN python3 -m pip install $(ls /home/pydbow3/*.whl)

# copy other libraries
WORKDIR /usr/local/lib
COPY --from=builder /usr/local/lib .
WORKDIR /usr/local/include/opencv4/
COPY --from=builder /usr/local/include/opencv4/ .
WORKDIR /home/opencv/build/lib
COPY --from=builder /home/opencv/build/lib .

##############################################################################
#
#   Pangolin Viewer
#
##############################################################################

# Install pangoling dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        pkg-config \
        libgl1-mesa-dev \
        libglew-dev \
        cmake \
        libpython2.7-dev \
        libegl1-mesa-dev \
        libwayland-dev \
        libxkbcommon-dev \
        wayland-protocols \
        libeigen3-dev \
        doxygen && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install pyopengl Pillow pybind11

WORKDIR /home
RUN git clone https://github.com/LukasBommes/Pangolin.git pangolin

WORKDIR /home/pangolin
RUN git submodule init && git submodule update

WORKDIR /home/pangolin/build
RUN cmake .. && \
    cmake --build . && \
    cmake --build . --target doc

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics


##############################################################################
#
#   Python Packages
#
##############################################################################

# needed for opencv-python and matplotlib
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libsm6 \
        libxext6 \
        libxrender-dev \
        python3-tk \
    	libcanberra-gtk-module \
    	libcanberra-gtk3-module && \
	rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /
RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt

##############################################################################
#
#   Container Startup
#
##############################################################################

WORKDIR /dronemapper

# TODO:
# 1) install pangolin
# 2) Install pyg2o

CMD ["sh", "-c", "tail -f /dev/null"]
