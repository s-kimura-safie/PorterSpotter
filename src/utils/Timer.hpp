/*
 * (c) 2021 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#pragma once

#include <chrono>
#include <cmath>
#include <string>

/// @brief レイテンシ測定のためのタイマークラスです。Start() と End() で挟まれたコードの実行時間を計測します。
/// 平均、標準偏差、合計を取得できます。
class Timer
{
private:
    std::string name;

    std::chrono::_V2::system_clock::time_point startTime;
    std::chrono::_V2::system_clock::time_point endTime;
    std::chrono::duration<double> accumulated;
    double square_sum;
    unsigned int count;

public:
    Timer(const std::string &name = "Timer");
    void Start();
    void End(const bool print = false, const std::string &endString = "\n");
    int Count() const { return count; }
    double Accumulated() const { return accumulated.count(); }
    double Average() const { return accumulated.count() / count; }

    /// @brief Returns standard deviation
    double Stdev() const
    {
        if (count < 2) return -1.0; // Return -1.0 if number of samples is not sufficient

        const double avarage = accumulated.count() / count;

        // Standard deviation of sample
        const double s = std::sqrt(square_sum / count - std::pow(avarage, 2));

        // Return the standard deviation of population estimated from sample
        return std::sqrt(count / (double)(count - 1.0)) * s;
    }

    std::string ResultString(const bool reset = false);
};
