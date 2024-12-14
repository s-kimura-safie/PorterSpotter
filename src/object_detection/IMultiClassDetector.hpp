/*
 * (c) 2024 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */
#pragma once

#include "Types.hpp"

// TODO: IDetectorとの共通化を検討
/// @brief 人物と顔クラスの物体検出器のインターフェース
class IMultiClassDetector
{
public:
    virtual ~IMultiClassDetector(){};

    /// @brief 検出するBBOXは画面内に制限されない
    virtual bool Infer(const cv::Mat &image, std::vector<std::vector<BboxXyxy>> &result) = 0;
};
