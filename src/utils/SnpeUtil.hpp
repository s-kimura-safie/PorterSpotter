
/*
 * (c) 2024 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <string>

#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPE.hpp"

#include "pose_estimation/PoseEstimator.hpp"

namespace SnpeUtil
{
    std::unique_ptr<zdl::DlContainer::IDlContainer> loadContainerFromFile(std::string containerPath);
    std::unique_ptr<zdl::DlContainer::IDlContainer> loadContainerFromBuffer(const uint8_t *buffer, const size_t size);

    typedef unsigned int GLuint;
    std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensor(std::unique_ptr<zdl::SNPE::SNPE> &snpe, cv::Mat inputImage);
    std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensor(std::unique_ptr<zdl::SNPE::SNPE> &snpe,
                                                            SequentialPoseKeypoints &poseKeypoints);
}
