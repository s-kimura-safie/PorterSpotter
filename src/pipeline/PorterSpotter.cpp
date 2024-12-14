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

void PorterSpotter::Run(const cv::Mat &rgbImage, std::vector<TrackedBbox> &tracks)
{
    std::cout << "Start Porter Spotter" << std::endl;
    // detection
    std::vector<std::vector<BboxXyxy>> multiclassDetections;

    yolov8.Infer(rgbImage, multiclassDetections);
    std::cout << "size:" << multiclassDetections.size() << std::endl;

    // tracking
    const std::vector<BboxXyxy>& personDetections = multiclassDetections[0];
    byte.Exec(personDetections, tracks);

    // pose estimation
    poseEstimator.Exec(rgbImage, tracks);

    // action recognition
    for (TrackedBbox &track : tracks)
    {
        const std::vector<BboxXyxy>& objectDetections = multiclassDetections[1];
        // hold detection
    }
}
