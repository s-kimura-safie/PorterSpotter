/*
 * (c) 2021 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#include "Timer.hpp"
#include <iostream>

Timer::Timer(const std::string &name)
{
    this->name = name;
    square_sum = 0.0;
    count = 0;
}

void Timer::Start() { startTime = std::chrono::high_resolution_clock::now(); }

void Timer::End(const bool print, const std::string &endString)
{
    endTime = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = endTime - startTime;
    accumulated += elapsed;
    square_sum += std::pow(elapsed.count(), 2.0);
    count++;
    if (print)
    {
        std::cout << name + ": " + std::to_string(elapsed.count()) + endString;
    }
}

std::string Timer::ResultString(const bool reset)
{
    using namespace std;

    const string ret = "# count: " + to_string(Count()) + ", Accumulated: " + to_string(Accumulated()) +
                       " [sec], Avarage: " + to_string(Average() * 1000) + " [msec], Stdev: " + to_string(Stdev() * 1000) +
                       " [msec] (" + to_string(Stdev() / Average() * 100) + "%)";

    if (reset)
    {
        accumulated = std::chrono::duration<double>(0.0);
        square_sum = 0.0;
        count = 0;
    }

    return ret;
}
