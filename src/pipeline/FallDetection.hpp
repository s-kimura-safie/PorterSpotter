/*
 * (c) 2024 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */
#pragma once

#include <chrono>
#include <iostream>
#include <string>

#include "tracking/Byte.hpp"
#include "detect/Yolov5.hpp"
#include "pose/PoseEstimator.hpp"
#include "action_recognition/STGCN.hpp"

/// @brief 人物の体を検出し、追跡を行い、ポーズを推定し、時系列のポーズデータを解析し、転倒を検知するクラス
class FallDetection
{
private:
    Yolov5 yolov5;
    Byte byte;
    PoseEstimator poseEstimator;
    STGCN stgcn;
    
    bool isDetectionModelReady;
    bool isPoseEstimatorModelReady;
    bool isActionRecognitionModelReady;

public:
    FallDetection();
    ~FallDetection();

    bool InitializeDetection(const uint8_t *buffer, const size_t size, const std::vector<std::string> &runtimes);
    bool InitializePoseEstimator(const uint8_t *buffer, const size_t size, const std::vector<std::string> &runtimes);
    bool InitializeActionRecognition(const uint8_t *buffer, const size_t size, const std::vector<std::string> &runtimes);

    void Run(const cv::Mat &image, std::vector<TrackedBbox> &tracks);
};
