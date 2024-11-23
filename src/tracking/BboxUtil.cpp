/*
 * (c) 2021 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#include "BboxUtil.hpp"
#include <cmath>
#include <iostream>

/// @brief コーナーフォーマットのバウンディングボックスをUVSRフォーマットに変換する
BboxUvsr BboxUtil::Xyxy2Uvsr(const BboxXyxy input)
{
    double w, h;
    w = input.x1 - input.x0;
    h = input.y1 - input.y0;
    BboxUvsr ret;

    ret.u = input.x0 + w / 2;
    ret.v = input.y0 + h / 2;
    ret.s = w * h;
    ret.r = w / h;
    return ret;
}

/// @brief UVSRフォーマットのバウンディングボックスをコーナーフォーマットに変換する
BboxXyxy BboxUtil::Uvsr2Xyxy(const BboxUvsr input)
{
    double w, h;
    BboxXyxy ret;

    w = std::sqrt(input.s * input.r);
    h = input.s / w;
    w = w;
    h = h;
    ret.x0 = input.u - w / 2;
    ret.y0 = input.v - h / 2;
    ret.x1 = input.u + w / 2;
    ret.y1 = input.v + h / 2;

    return ret;
}
