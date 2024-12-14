/*
 * (c) 2024 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#pragma once

#include "IMultiClassDetector.hpp"
#include "SNPE/SNPE.hpp"
#include "Types.hpp"

class Yolov8 : IMultiClassDetector
{
public:
    struct BoundingBox; // FIXME: 消す

private:
    float ratio;
    int paddingWidthIdx;
    int paddingHeightIdx;

    std::unique_ptr<zdl::SNPE::SNPE> network;

    void preprocess(const cv::Mat &rawImg, cv::Mat &resizedImg);
    void postprocess(const std::vector<std::vector<BoundingBox>> &decoded, std::vector<std::vector<BboxXyxy>> &result);

public:
    Yolov8(){};
    ~Yolov8(){};

    bool CreateNetwork(const uint8_t *buffer, const size_t size, const std::vector<std::string> &runtimes);

    /// @brief 人物、頭、顔を検出する
    /// @param image 任意のアスペクト比とサイズの画像（例: 1280 x 720）
    /// @param result 入力画像のピクセル座標系のBBOX
    bool Infer(const cv::Mat &image, std::vector<std::vector<BboxXyxy>> &result) override;
};
