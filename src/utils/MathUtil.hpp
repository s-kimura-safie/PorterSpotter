/*
 * (c) 2022 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

/// @brief 数学に関連するユーティリティ
namespace MathUtil
{
    template <class T> inline T Clamp(T value, T min, T max) { return std::max(min, std::min(value, max)); }
}
