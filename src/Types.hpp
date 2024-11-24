/// @brief このファイルは、プロジェクト全体で使用されるドメインオブジェクトを定義します。
/*
 * (c) 2021 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */
#pragma once

#include <opencv2/opencv.hpp>
#include <string>

#include "pose_estimation/PoseUtils.hpp"


/// @brief Bounding box in corner format
struct BboxXyxy
{
    double x0;
    double y0;
    double x1;
    double y1;
    double confidence;

    inline double x_center() const { return (x0 + x1) / 2.0; }
    inline double y_center() const { return (y0 + y1) / 2.0; }

    BboxXyxy() : x0(0.0), y0(0.0), x1(0.0), y1(0.0), confidence(0.0){};
    BboxXyxy(const double x0, const double y0, const double x1, const double y1, const double confidence = 0.0)
        : x0(x0), y0(y0), x1(x1), y1(y1), confidence(confidence){};
};

/// @brief Bounding box in UVSR format
struct BboxUvsr
{
    double u; // 中心座標
    double v; // 中心座標
    double s; // 面積（scale）
    double r; // Aspect ratio
};


/// @brief 追跡結果
struct TrackedBbox
{
    unsigned int id;
    cv::Vec2d velocity;
    BboxXyxy bodyBbox;
    bool isBodyDetected{false}; // トラッカーの推定値が検出結果と関連付けされているとき true
    std::vector<PosePoint> poseKeypoints;
    std::string action;
    

    TrackedBbox() : id(0), velocity(0, 0), bodyBbox(0.0, 0.0, 0.0, 0.0){};
    // TODO: double id -> unsigned int id
    TrackedBbox(const double id, const BboxXyxy &bodyBbox) : id(id), bodyBbox(bodyBbox){};
    TrackedBbox(const double id, const BboxXyxy &bodyBbox, const cv::Vec2d &velocity)
        : id(id), velocity(velocity), bodyBbox(bodyBbox){};
    TrackedBbox(const double id, const BboxXyxy &bodyBbox, const cv::Vec2d &velocity, const bool isBodyDetected)
        : id(id), velocity(velocity), bodyBbox(bodyBbox), isBodyDetected(isBodyDetected){};

    void AddPoseKeypoints(const std::vector<PosePoint> &poseKeypoints)
    {
        this->poseKeypoints = poseKeypoints;
    }

    void ClearPoseKeypoints()
    {
        this->poseKeypoints.clear();
    }
};

