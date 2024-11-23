/*
 * (c) 2021 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */
#pragma once

#include "Types.hpp"

/// @brief Bboxに関連するユーティリティ関数を提供します。
namespace BboxUtil
{
    BboxUvsr Xyxy2Uvsr(const BboxXyxy input);
    BboxXyxy Uvsr2Xyxy(const BboxUvsr input);
}
