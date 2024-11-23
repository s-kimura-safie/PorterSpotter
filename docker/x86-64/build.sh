#!/bin/bash -eu

# working in EdgeFallDetection to copy sdk files
cd `dirname $0`

docker build -t edgefalldetection/x86-64:latest .
