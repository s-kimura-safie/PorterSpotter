#!/bin/bash -eu

SCRIPT_DIR=$(cd $(dirname $0); pwd)
ADB=adb.exe

$ADB pull /data/EdgePoseEstimation/outputs output_evk
