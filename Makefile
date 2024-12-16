# Copyright (c) 2024 Safie Inc. All rights reserved.
#
# NOTICE: No part of this file may be reproduced, stored
# in a retrieval system, or transmitted, in any form, or by any means,
# electronic, mechanical, photocopying, recording, or otherwise,
# without the prior consent of Safie Inc.


# TARGET = "x86-64"(default) or "aarch64"
TARGET=x86-64

CXXFLAGS        += -std=c++11 -fPIC -march=$(MARCH)

ifeq ($(TARGET),x86-64)
    LDFLAGS 	+= -L $(SNPE_ROOT)/lib/x86_64-linux-clang -L /usr/local/lib
    MARCH		:= x86-64
    INCLUDES	+= -I /usr/include/eigen3 -I /usr/local/include/opencv4
endif

# Include paths
INCLUDES += \
    -I $(SNPE_ROOT)/include/zdl \
    -I $(CURDIR)/src \
    -I $(CURDIR)/src/utils

# Specify the link libraries
LDLIBS += -lSNPE
ifeq ($(TARGET),x86-64)

    LDLIBS += -lgflags_nothreads
    LDLIBS += -lopencv_calib3d
    LDLIBS += -lopencv_core
    LDLIBS += -lopencv_dnn
    LDLIBS += -lopencv_features2d
    LDLIBS += -lopencv_flann
    LDLIBS += -lopencv_gapi
    LDLIBS += -lopencv_highgui
    LDLIBS += -lopencv_imgcodecs
    LDLIBS += -lopencv_imgproc
    LDLIBS += -lopencv_ml
    LDLIBS += -lopencv_objdetect
    LDLIBS += -lopencv_photo
    LDLIBS += -lopencv_stitching
    LDLIBS += -lopencv_video
    LDLIBS += -lopencv_videoio
endif

# Specify the target
OUT_ROOT   		:= $(CURDIR)/bin
OUT_DIR			:= $(OUT_ROOT)/$(TARGET)

OBJ_ROOT		:= $(CURDIR)/obj
OBJ_DIR			:= $(OBJ_ROOT)/$(TARGET)

# Exclude framework depending files
# For standalone cpp files
SA_SRC_DIR      := $(CURDIR)/src
SA_SRCS         := $(wildcard $(SA_SRC_DIR)/*.cpp) \
                    $(wildcard $(SA_SRC_DIR)/hold_detecion/*.cpp) \
                    $(wildcard $(SA_SRC_DIR)/object_detection/*.cpp) \
					$(wildcard $(SA_SRC_DIR)/pipeline/*.cpp) \
                    $(wildcard $(SA_SRC_DIR)/pose_estimation/*.cpp) \
                    $(wildcard $(SA_SRC_DIR)/tracking/*.cpp) \
                    $(wildcard $(SA_SRC_DIR)/utils/*.cpp)

SA_OBJ_DIR      := $(OBJ_DIR)
SA_OBJS         := $(SA_SRCS:$(SA_SRC_DIR)/%.cpp=$(SA_OBJ_DIR)/%.o)

# List .cpp files which contains main
MAIN_SRCS		:= OfflinePoseEstimator.cpp OfflinePorterSpotter.cpp OfflinePorterSpotterV.cpp
MAIN_OBJS		:= $(MAIN_SRCS:%.cpp=$(SA_OBJ_DIR)/%.o)
SA_OBJS_WO_MAIN	:= $(filter-out $(MAIN_OBJS), $(SA_OBJS))

# ex) src/OfflineDetector.cpp -> bin/x86-64/OfflineDetector
PROGRAMS		:= $(MAIN_SRCS:%.cpp=$(OUT_DIR)/%)
.PHONY: all clean

all: $(PROGRAMS)

# compile standalone
$(SA_OBJ_DIR)/%.o: $(SA_SRC_DIR)/%.cpp
	@if [ ! -e `dirname $@` ]; then mkdir -p `dirname $@`; fi
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $< -o $@

$(PROGRAMS): $(SA_OBJS_WO_MAIN) $(MAIN_OBJS)
# Object file which contains main
	$(eval MAIN_OBJ := $(@:$(OUT_DIR)/%=$(SA_OBJ_DIR)/%.o))
	@if [ ! -e `dirname $@` ]; then mkdir -p `dirname $@`; fi
	$(CXX) $(LDFLAGS) $(SA_OBJS_WO_MAIN) $(MAIN_OBJ) $(LDLIBS) -o $@

clean:
	-rm -vr $(OBJ_ROOT)
	-rm -vr $(OUT_ROOT)
