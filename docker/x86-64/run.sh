#!/bin/bash -eu

# cd to Porter Spotter root dir
APP_ROOT=$(cd $(dirname $BASH_SOURCE)/../..; pwd)
cd $APP_ROOT

# SDK PATH
SDK_DIR=$APP_ROOT/sdk
SNPE_ZIP=snpe-1.68.0.zip

echo extracting $SDK_DIR/$SNPE_ZIP
unzip -n $SDK_DIR/$SNPE_ZIP -d $SDK_DIR
echo 

# Get extraction dir
SNPE_DIR=$(find $SDK_DIR -maxdepth 1  -name "snpe*" -type d)
echo "SNPE_DIR=$SNPE_DIR"

docker run -it --rm \
    --name PorterSpotter_x86 \
    -v $APP_ROOT:/workspace/PorterSpotter \
    -v $SNPE_DIR:/workspace/snpe \
    -w /workspace/PorterSpotter \
    porterspotter/x86-64
