/*
 * (c) 2021 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */
#pragma once

#include <list>
#include <opencv2/opencv.hpp>
#include <vector>
/// @brief 線形割当 (Hungarian Algorithm) の結果を行番号列番号で格納する構造体
struct RowCol
{
    int row;
    int col;
};

/// @brief Hungarian algorithm を用いて linear sum assignment を行う
class LinearSumAssignment
{
private:
    cv::Mat mat, dist, starMat, primeMat;

    int nrows, ncols, minDim, currRow, currCol;

    std::vector<bool> coveredCols, coveredRows;

    bool isRowBigThanCol;

    void putStMat2Vec(std::list<RowCol> &res) const;
    void step1();
    void step2(const bool isFirst);
    bool step3() const;
    bool step4();
    void step5();
    void step6();
    int findInMat(const cv::Mat m, const int c, const bool isRow);

public:
    LinearSumAssignment(const cv::Mat m);
    ~LinearSumAssignment(){};

    void ComputeAssociation(std::list<RowCol> &association);
};
