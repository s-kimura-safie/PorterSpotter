#!/bin/bash -eu

# working in PorterSpotter to copy sdk files
cd `dirname $0`

docker build -t porterspotter/x86-64:latest .
