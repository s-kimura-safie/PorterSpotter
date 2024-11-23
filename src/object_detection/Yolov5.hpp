/*
 * (c) 2022 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */
#pragma once

#include "DlSystem/RuntimeList.hpp"
#include "SNPE/SNPE.hpp"
#include "Yolov5Util.hpp"

/// @brief YOLOv5-s 物体検出器 (SNPE 版)
class Yolov5 final
{
private:
    float scoreThreshold;
    float iouThreshold;
    int nmsTopK;

    bool isNetworkFedBgr;
    bool isNetworkReady;

    std::unique_ptr<zdl::SNPE::SNPE> network;

    /// @brief Transform CV_8UC3 to CV_32FC3 with [0, 1] normalization.
    static void makeFloatImg(const cv::Mat &input, cv::Mat &output);

    /// @brief Save feature map for debug purpose
    static void saveFeatureMap(const zdl::DlSystem::TensorMap &tensorMap, const int w_img, const int h_img);

    void decode_output(const zdl::DlSystem::TensorMap &tensorMap, std::vector<Yolov5Util::Object> &objects,
                       const float scale, const cv::Vec2i &delta, const int w_img, const int h_img) const;

public:
    Yolov5();
    ~Yolov5();

    void SetScoreThreshold(const float scoreThreshold) { this->scoreThreshold = scoreThreshold; }
    void SetIoUThreshold(const float iouThreshold) { this->iouThreshold = iouThreshold; }
    void SetNmsTopK(const unsigned int nmsTopK) { this->nmsTopK = nmsTopK; }
    bool IsNetworkFedBgr() const { return isNetworkFedBgr; }
    bool IsNetworkReady() const { return isNetworkReady; }

    float GetScoreThreshold() { return scoreThreshold; }

    bool CreateNetwork(const uint8_t *buffer, const size_t size, const std::vector<std::string> &runtimes);
    // Yolov5が推論して出力するbboxは画面内 [0, 1] x [0, 1] に制限されない
    bool Detect(const cv::Mat &inputBitmap, std::vector<BboxXyxy> &objectList);
};
