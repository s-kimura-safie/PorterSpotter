/*
 * (c) 2024 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#include "FallDetection.hpp"

FallDetection::FallDetection()
{
    isDetectionModelReady = false;
    isPoseEstimatorModelReady = false;
    isActionRecognitionModelReady = false;
    
    const bool isSortOn = false;
    const double confidenceThreshold = 0.35;
    byte.SetSortOn(isSortOn);
    byte.SetConfidenceThreshold(confidenceThreshold);
}

FallDetection::~FallDetection() {}

bool FallDetection::InitializeDetection(const uint8_t *buffer, const size_t size, const std::vector<std::string> &runtimes)
{
    if (yolov5.CreateNetwork(buffer, size, runtimes))
    {
        isDetectionModelReady = true;
        return true;
    }
    else
    {
        return false;
    }
}

bool FallDetection::InitializePoseEstimator(const uint8_t *buffer, const size_t size, const std::vector<std::string> &runtimes)
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

bool FallDetection::InitializeActionRecognition(const uint8_t *buffer, const size_t size, const std::vector<std::string> &runtimes)
{
    if (stgcn.CreateNetwork(buffer, size, runtimes))
    {
        isActionRecognitionModelReady = true;
        return true;
    }
    else
    {
        return false;
    }
}

void FallDetection::Run(const cv::Mat &rgbImage, std::vector<TrackedBbox> &tracks)
{
    // detection
    std::vector<BboxXyxy> detectedObjects;
    yolov5.Detect(rgbImage, detectedObjects);

    // tracking
    byte.Exec(detectedObjects, tracks);

    // pose estimation
    poseEstimator.Exec(rgbImage, tracks);

    // action recognition
    for (TrackedBbox &track : tracks)
    {   
        int trackId = track.id;
        const SequentialPoseKeypoints sequentialPoseKeypoints = poseEstimator.sequentialPoseKeypointsByTrackId.at(trackId);
        if (sequentialPoseKeypoints.size() == 10)  stgcn.Exec(sequentialPoseKeypoints, track);
    }

}
