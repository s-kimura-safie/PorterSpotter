/*
 * (c) 2024 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */
#pragma once

#include "DlSystem/RuntimeList.hpp"
#include "SNPE/SNPE.hpp"
#include "Types.hpp"
#include "pose_estimation/PoseUtils.hpp"
#include <opencv2/opencv.hpp>

using SequentialPoseKeypoints = std::deque<std::vector<PosePoint>>;
const size_t POINTMAXSIZE = 10;

class PoseEstimator
{
private:
    unsigned int featureSize;

    bool isNetworkReady;
    std::unique_ptr<zdl::SNPE::SNPE> network;

    static void makeFloatImg(const cv::Mat &input, cv::Mat &output);

    std::pair<cv::Mat, cv::Mat> cropImageByDetectBox(const cv::Mat &input_image, const BboxXyxy &box);

    void decodeOutput(const zdl::DlSystem::TensorMap &tensorMap) const;
    void addPoseKeypoints(const int trackId, const std::vector<PosePoint> &poseKeypoints);
    void clearDisappearedTracks(const std::vector<int> &tracks);

public:
    std::map<int, SequentialPoseKeypoints> sequentialPoseKeypointsByTrackId;
    PoseEstimator();
    ~PoseEstimator();

    bool CreateNetwork(const uint8_t *buffer, const size_t size, const std::vector<std::string> &runtimes);
    std::vector<PosePoint> Inference(const cv::Mat &input_mat, const BboxXyxy &box);
    void Exec(const cv::Mat &input_image, std::vector<TrackedBbox> &tracks);
};
