/*
 * (c) 2024 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#pragma once
#include <algorithm>
#include "MathUtil.hpp"
#include "Types.hpp"
#include "pose_estimation/PoseUtils.hpp"
#include <opencv2/opencv.hpp>

/// @brief 可視化に関連するユーティリティ
namespace visualization_util
{
    // Standart 80 color palette for COCO visualization
    extern const float color_list[80][3];

    // Keypoint color palette
    extern const std::vector<cv::Scalar> keypointColors;

    void DrawLabel(cv::Mat &image, cv::Point2i point, const std::string &text, const cv::Scalar &color);

    // キーポイントを画像に描画する関数
    void drawSkeleton(const std::vector<PosePoint> &points, cv::Mat &image);
    void drawTracksSkeleton(const std::vector<TrackedBbox> &tracks, cv::Mat &image);

    // 人物のバウンディングボックスを描画する関数
    void drawPersonBbox(const std::vector<TrackedBbox> &tracks, cv::Mat &image);
    void drawObjectBbox(const std::vector<BboxXyxy> &objectDetections, cv::Mat &image);
}
