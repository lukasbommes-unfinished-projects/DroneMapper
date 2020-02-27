FROM ubuntu:18.04

RUN apt-get update && \
  apt-get install -y \
    software-properties-common \
    lsb-release \
    mesa-utils \
    wget \
    curl \
    sudo vim \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
  apt-get install -y \
    cmake \
    git \
    build-essential \
    libeigen3-dev \
    libsuitesparse-dev \
    qtdeclarative5-dev \
    qt5-qmake \
    libqglviewer-dev-qt4 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /code
RUN git clone https://github.com/RainerKuemmerle/g2o.git

WORKDIR /code/g2o/build
RUN cmake ../ \
  && make -j12 \
  && make install -j12 \
  && ldconfig


WORKDIR /code/g2o

CMD ["sh", "-c", "tail -f /dev/null"]
