/*
 * (c) 2024 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#include "PorterSpotter.hpp"

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

bool PorterSpotter::InitializePoseEstimator(const uint8_t *buffer, const size_t size, const std::vector<std::string> &runtimes)
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


void PorterSpotter::Run(const cv::Mat &rgbImage, std::vector<TrackedBbox> &tracks)
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
        // hold detection

    }

}
