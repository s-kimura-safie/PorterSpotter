FROM ubuntu:20.04

# Define workdir
WORKDIR /workspace

# Use jaist
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install apt packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip python3.8 python3-pip \
    sudo wget python3-dev zip libc++-9-dev libatomic1 \
    build-essential libgflags-dev libeigen3-dev qtbase5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make symbolic link for python3.8 as python
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# build opencv
RUN apt-get update && apt-get install -y ffmpeg --no-install-recommends \
    cmake git pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV OPENCV_VERSION=4.5.5
RUN wget -qO opencv.zip "https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip" \
    && unzip -q opencv.zip \
    && rm opencv.zip \
    && cd opencv-${OPENCV_VERSION} \
    && mkdir build \
    && cd build \
    && cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_opencv_world=OFF \
    -D BUILD_opencv_core=ON \
    -D BUILD_opencv_cnn_3dobj=OFF \
    -D BUILD_opencv_dnn_modern=OFF \
    -D BUILD_DOCS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF .. \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    && make -j $(nproc) \
    && make install \
    && ldconfig

# Set enviroment variables of spne root directory 
ENV SNPE_ROOT=/workspace/snpe

# Add initialization script to .bashrc
RUN echo 'export LD_LIBRARY_PATH=${SNPE_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}' >> ~/.bashrc
