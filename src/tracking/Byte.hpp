/*
 * (c) 2023 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */
#pragma once

#include "LinearSumAssignment.hpp"
#include "ObjectTracker.hpp"

/// @brief Byte トラッキングアルゴリズムを実行する。クリーンアーキテクチャにおけるユースケース層に所属する。
class Byte
{
private:
    std::list<ObjectTracker> trackers;

    int currFrame; // 現在のフレーム
    int currId;    // 現在のID

    int maxAge;
    int numInitialFrame;     // 連続検出の条件をスキップする初期フレーム数
    int minFrameSustained;   // 何フレーム以上連続で検出された場合に可視化状態になる
    double iouThresholdHigh; // used for first association
    double iouThresholdLow;  // used for second association
    double confidenceThreshold;
    bool isSortOn;

    static double calcIou(const BboxXyxy d1, const BboxXyxy d2);
    static bool isMatchedUniquely(const cv::Mat iou_bigger_flag);
    static cv::Mat calcIouMatrix(const std::vector<BboxXyxy> &srcs, const std::vector<BboxXyxy> &tgts);

    void getBboxesXyxy(const ObjectTracker &tracker, BboxXyxy &xyxy) const;
    void associateDetectionsToTrackers(const std::vector<BboxXyxy> &detections,
                                       const std::vector<BboxXyxy> &trackedBboxesXyxy, const double iouThreshold,
                                       std::list<RowCol> &associations, std::vector<int> &unmatchedDetectionIdcs) const;
    void makeMatchedIdcs(const cv::Mat iouMatrix, std::list<RowCol> &matchedIdcs, const double iouThreshold) const;
    void cleanTrackers();

public:
    Byte();
    Byte(const int maxAge, const int numInitialFrame, const int minFrameSustained, const double iouThresholdHigh,
         const double iouThresholdLow, const double confidenceThreshold, const bool isSortOn);
    ~Byte(){};

    void SetMaxAge(const int maxAge);
    void SetNumInitialFrame(const int numIntialFrame);
    void SetMinFrameSustained(const int minFrameSustained);
    void SetIouThresholdHigh(const double iouThresholdHigh);
    void SetIouThresholdLow(const double iouThresholdLow);
    void SetConfidenceThreshold(const double confidenceThreshold);
    void SetSortOn(const bool isSortOn);

    double GetConfidenceThreshold() const { return confidenceThreshold; }

    void Reset();

    /// @brief Execute Byte algorithm
    /// @param detections Input detections
    /// @param visibleTracks Tracks that have association to a detection
    void Exec(const std::vector<BboxXyxy> &detectedBBox, std::vector<TrackedBbox> &visibleTracks);
    // TODO: TrackedBboxがスタンプ結合になっているので必要な変数だけ返却するようにする
};
