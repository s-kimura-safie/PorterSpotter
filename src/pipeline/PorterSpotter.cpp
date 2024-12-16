/*
 * (c) 2024 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#include "PorterSpotter.hpp"

// 人物のKEYPOINTの手の位置と物体のBBOXの中心から一定の距離以内であれば、tracksのisHoldingObjectフラグをtrueにする
void checkObjectHolding(std::vector<TrackedBbox> &tracks, std::vector<BboxXyxy> &objectDetections)
{
    for (TrackedBbox &track : tracks)
    {
        PosePoint rightHandPoint = track.poseKeypoints[5]; // 右手
        PosePoint leftHandPoint = track.poseKeypoints[6];  // 左手

        for (BboxXyxy &objectDetection : objectDetections)
        {
            double objectCenterX = (objectDetection.x0 + objectDetection.x1) / 2;
            double objectCenterY = (objectDetection.y0 + objectDetection.y1) / 2;
            double distanceFromRight =
                sqrt(pow(rightHandPoint.x - objectCenterX, 2) + pow(rightHandPoint.y - objectCenterY, 2));
            double distanceFromLeft =
                sqrt(pow(leftHandPoint.x - objectCenterX, 2) + pow(leftHandPoint.y - objectCenterY, 2));

            double distance = std::min(distanceFromRight, distanceFromLeft);
            double distanceThreshold = (track.bodyBbox.x1 - track.bodyBbox.x0) / 2;
            if (distance < distanceThreshold)
            {
                track.isHoldingObject = true;
                break;
            }
        }
    }
}

PorterSpotter::PorterSpotter()
{
    isDetectionModelReady = false;
    isPoseEstimatorModelReady = false;

    const bool isSortOn = false;
    const double confidenceThreshold = 0.35;
    byte.SetSortOn(isSortOn);
    byte.SetConfidenceThreshold(confidenceThreshold);
}

PorterSpotter::~PorterSpotter() {}

bool PorterSpotter::InitializeDetection(const uint8_t *buffer, const size_t size, const std::vector<std::string> &runtimes)
{
    if (yolov8.CreateNetwork(buffer, size, runtimes))
    {
        isDetectionModelReady = true;
        return true;
    }
    else
    {
        return false;
    }
}

bool PorterSpotter::InitializePoseEstimator(const uint8_t *buffer, const size_t size,
                                            const std::vector<std::string> &runtimes)
{
    if (poseEstimator.CreateNetwork(buffer, size, runtimes))
    {
        isPoseEstimatorModelReady = true;
        return true;
    }
    else
    {
        return false;
    }
}

void PorterSpotter::ResetTracker() { byte.Reset(); }

void PorterSpotter::Run(const cv::Mat &rgbImage, std::vector<TrackedBbox> &tracks, std::vector<BboxXyxy> &objectDetections)
{
    // 物体検出
    std::vector<std::vector<BboxXyxy>> multiclassDetections;
    yolov8.Infer(rgbImage, multiclassDetections);

    // 追跡
    const std::vector<BboxXyxy> &personDetections = multiclassDetections[0];
    byte.Exec(personDetections, tracks);

    // 姿勢推定
    poseEstimator.Exec(rgbImage, tracks);

    // 対象物を持っているかどうかの判定
    objectDetections = multiclassDetections[1];
    checkObjectHolding(tracks, objectDetections);
}
