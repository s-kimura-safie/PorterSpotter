#!/bin/sh

SCRIPT_DIR=$(cd $(dirname $0); pwd)

# Set Environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SCRIPT_DIR/lib:$SCRIPT_DIR/lib/snpe:$SCRIPT_DIR/lib/dsp
export ADSP_LIBRARY_PATH="$SCRIPT_DIR/lib/dsp;/usr/lib/rfsa/adsp;/dsp"

cd $SCRIPT_DIR

# Set parameters
APP=./bin/OfflineEvkVideoAnalysis
DETECT_MODEL=models/yolov5s.dlc
POSE_MODEL=models/rtmpose.dlc
ACTION_MODEL=models/stgcn.dlc
INPUT_VIDEO=videos/GH019443.MP4
RUNTIME="dsp,cpu"

$APP -d $DETECT_MODEL -p $POSE_MODEL -a $ACTION_MODEL -i $INPUT_VIDEO -r $RUNTIME -f 10.0
