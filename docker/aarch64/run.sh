#!/bin/bash -eu

# cd to PorterSpotter root dir
APP_ROOT=$(cd $(dirname $BASH_SOURCE)/../..; pwd)
cd $APP_ROOT

# SafieAppFrameWorkRoot
APP_FW_ROOT=$APP_ROOT/../SafieEdgeAppFramework

# SDK PATH
SDK_DIR=$APP_ROOT/sdk
ANDROID_NDK_ZIP=android-ndk-r17c-linux-x86_64.zip
SNPE_ZIP=snpe-1.68.0.zip


# MEMO: awk command is used to show progress
echo extracting $SDK_DIR/$ANDROID_NDK_ZIP
unzip -n $SDK_DIR/$ANDROID_NDK_ZIP -d $SDK_DIR
echo 

echo extracting $SDK_DIR/$SNPE_ZIP
unzip -n $SDK_DIR/$SNPE_ZIP -d $SDK_DIR
echo 

# Get extraction dir
ANDROID_NDK_DIR=$(find $SDK_DIR -maxdepth 1  -name "android*" -type d)
SNPE_DIR=$(find $SDK_DIR -maxdepth 1  -name "snpe*" -type d)
echo "ANDROID_NDK_DIR=$ANDROID_NDK_DIR"
echo "SNPE_DIR=$SNPE_DIR"

docker run -it --rm \
    --name PorterSpotter_aarch64 \
    -v $APP_ROOT:/workspace/PorterSpotter \
    -v $APP_FW_ROOT:/workspace/SafieEdgeAppFramework \
    -v $ANDROID_NDK_DIR:/workspace/android_ndk \
    -v $SNPE_DIR:/workspace/snpe \
    -w /workspace/PorterSpotter \
    porterspotter/aarch64
