FROM ubuntu:18.04 AS toolchain_expander

# Use jaist
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list

# Install apt packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Expand qualcomm toolchain
RUN mkdir qualcomm_toolchain
COPY cameraToolchains/toolchain/oecore-x86_64-aarch64-toolchain-nodistro.0.sh /qualcomm_toolchain
RUN chmod a+x /qualcomm_toolchain/oecore-x86_64-aarch64-toolchain-nodistro.0.sh \
    && /qualcomm_toolchain/oecore-x86_64-aarch64-toolchain-nodistro.0.sh -y;exit 0


FROM ubuntu:18.04
# Define workdir
WORKDIR /workspace

# Use jaist
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list

# Install apt packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy qualcomm toolchain
COPY --from=toolchain_expander /usr/local/oecore-x86_64 /usr/local/oecore-x86_64

# Set enviroment variables of sdks' root dirs.
# SDKs will be mounted on these directories when executing run.sh.
ENV APP_ROOT=/workspace/EdgePoseEstimation
ENV SNPE_ROOT=/workspace/snpe
ENV ANDROID_NDK_ROOT=/workspace/android_ndk
ENV APP_FW_ROOT=/workspace/SafieEdgeAppFramework

# Add initialization script to .bashrc
RUN echo "source /usr/local/oecore-x86_64/environment-setup-aarch64-oe-linux" >> ~/.bashrc
