#!/bin/bash -eu

# working in PorterSpotter to copy sdk files
cd `dirname $0`

# Download camera tool chain
if [ ! -e cameraToolchains ]; then
    git clone -b dev/qcs610_aarch64 git@github.com:SafieDev/cameraToolchains.git cameraToolchains
fi

docker build -t porterspotter/aarch64:latest .
