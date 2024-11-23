/*
 * (c) 2023 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#include "Byte.hpp"
#include "MathUtil.hpp"

double Byte::calcIou(const BboxXyxy b1, const BboxXyxy b2)
{
    const double overlapX1 = std::max(b1.x0, b2.x0);
    const double overlapY1 = std::max(b1.y0, b2.y0);
    const double overlapX2 = std::min(b1.x1, b2.x1);
    const double overlapY2 = std::min(b1.y1, b2.y1);

    const double overlapArea = std::max(0., overlapX2 - overlapX1) * std::max(0., overlapY2 - overlapY1);
    const double unionArea = ((b1.x1 - b1.x0) * (b1.y1 - b1.y0) + (b2.x1 - b2.x0) * (b2.y1 - b2.y0) - overlapArea);

    const double iou = overlapArea / unionArea;
    return iou;
}

/// @brief iouBiggerFlg行列から、割当が一意にきまるかどうかを判定
/// @param iouBiggerFlg IOU行列のうちiouThresholdより大きい要素には1、小さい要素には0が格納された行列
/// @retval 判定結果
bool Byte::isMatchedUniquely(const cv::Mat iouBiggerFlg)
{
    // 行列で各行各列の総和が１かどうかを判定
    // 1だった場合は割当が一意に決まる
    cv::Mat iflagRow, iflagCol;

    cv::reduce(iouBiggerFlg, iflagRow, 0, cv::REDUCE_SUM);
    cv::reduce(iouBiggerFlg, iflagCol, 1, cv::REDUCE_SUM);

    for (int i = 0; i < iflagRow.cols; i++)
    {
        if (iflagRow.at<double>(0, i) > 1)
        {
            return false;
        }
    }
    for (int i = 0; i < iflagCol.rows; i++)
    {
        if (iflagCol.at<double>(i, 0) > 1)
        {
            return false;
        }
    }
    return true;
}

/// @brief trackersのbboxを取得
void Byte::getBboxesXyxy(const ObjectTracker &tracker, BboxXyxy &xyxy) const
{
    const BboxUvsr uvsr = tracker.GetBbox();
    xyxy = BboxUtil::Uvsr2Xyxy(uvsr);
    xyxy.x0 = MathUtil::Clamp<double>(xyxy.x0, 0.0, 1.0);
    xyxy.y0 = MathUtil::Clamp<double>(xyxy.y0, 0.0, 1.0);
    xyxy.x1 = MathUtil::Clamp<double>(xyxy.x1, 0.0, 1.0);
    xyxy.y1 = MathUtil::Clamp<double>(xyxy.y1, 0.0, 1.0);
    xyxy.confidence = tracker.GetConfidence();
}

/// @brief IOU行列から適切な割当indexを作成する関数
/// @param iouMatrix IOU行列
/// @param iouThreshold
void Byte::makeMatchedIdcs(const cv::Mat iouMatrix, std::list<RowCol> &associations, const double iouThreshold) const
{
    // 以下マッチングの作成

    // IOU行列のうち閾値以上のものをフラグ化
    cv::Mat iouBiggerFlg = (iouMatrix > iouThreshold) / 255;
    iouBiggerFlg.convertTo(iouBiggerFlg, CV_64F, 1);
    // 一意に決まる場合
    if (isMatchedUniquely(iouBiggerFlg))
    {
        // 一意に決まっているものをmatchedIndexLisに格納
        for (int j = 0; j < iouBiggerFlg.rows; j++)
        {
            for (int i = 0; i < iouBiggerFlg.cols; i++)
            {
                if (iouBiggerFlg.at<double>(j, i) == 1)
                {
                    RowCol t_m;
                    t_m.row = j;
                    t_m.col = i;
                    associations.push_back(t_m);
                }
            }
        }
    }
    else
    {
        // 線形割当(Hungarian Algorithm)
        LinearSumAssignment lsa(iouMatrix * (-1));
        lsa.ComputeAssociation(associations);
        // matchedIndexLisの要素のうち、iouThreshold未満のものは削除
        for (auto itr_a = associations.begin(); itr_a != associations.end();)
        {
            if (iouMatrix.at<double>(itr_a->row, itr_a->col) < iouThreshold)
            {
                itr_a = associations.erase(itr_a);
            }
            else
            {
                itr_a++;
            }
        }
    }
}

/// @brief 検出結果とトラッキングデータの割当
void Byte::associateDetectionsToTrackers(const std::vector<BboxXyxy> &detections,
                                         const std::vector<BboxXyxy> &trackedBboxesXyxy, const double iouThreshold,
                                         std::list<RowCol> &associations, std::vector<int> &unmatchedDetectionIdcs) const
{
    // IOU行列を計算
    cv::Mat iouMatrix = calcIouMatrix(detections, trackedBboxesXyxy);
    if (std::min(detections.size(), trackedBboxesXyxy.size()) > 0)
    {
        makeMatchedIdcs(iouMatrix, associations, iouThreshold);
    }
    else
    {
        associations = {};
    }

    // 検出結果(detectionBboxVec)の中からマッチングしていないものをunmatchedHighDetectionIdcsに格納
    for (int d_i = 0; d_i < (int)detections.size(); d_i++)
    {
        bool isMatched = false;
        for (const RowCol &matchedIndex : associations)
        {
            if (matchedIndex.row == d_i)
            {
                isMatched = true;
                break;
            }
        }
        if (!isMatched)
        {
            unmatchedDetectionIdcs.push_back(d_i);
        }
    }

    return;
}

/// @brief 検出結果のリストとトラッキングデータのリストからIOUの行列を作成する関数
cv::Mat Byte::calcIouMatrix(const std::vector<BboxXyxy> &detections, const std::vector<BboxXyxy> &trackedBboxesXyxy)
{
    cv::Mat iouMatrix((int)detections.size(), (int)trackedBboxesXyxy.size(), CV_64F);
    for (int i = 0; i < (int)detections.size(); i++)
    {
        for (int j = 0; j < (int)trackedBboxesXyxy.size(); j++)
        {
            iouMatrix.at<double>(i, j) = calcIou(trackedBboxesXyxy[j], detections[i]);
        }
    }
    return iouMatrix;
}

/// @brief 使用済みデータのクリア。リストのクリアと、maxAge以上のトラッキングデータの削除。
void Byte::cleanTrackers()
{
    for (auto trackrtItr = trackers.begin(); trackrtItr != trackers.end();)
    {
        if (trackrtItr->NumFrameDropped() > maxAge)
        {
            trackrtItr = trackers.erase(trackrtItr);
        }
        else
        {
            trackrtItr++;
        }
    }
}

Byte::Byte()
    : currFrame(0), currId(1), maxAge(5), numInitialFrame(3), minFrameSustained(3), iouThresholdHigh(0.3),
      iouThresholdLow(0.3), confidenceThreshold(0.35), isSortOn(false)
{
}

Byte::Byte(const int maxAge, const int numInitialFrame, const int minFrameSustained, const double iouThresholdHigh,
           const double iouThresholdLow, const double confidenceThreshold, const bool isSortOn)
    : currFrame(0), currId(1), maxAge(maxAge), numInitialFrame(numInitialFrame), minFrameSustained(minFrameSustained),
      iouThresholdHigh(iouThresholdHigh), iouThresholdLow(iouThresholdLow), confidenceThreshold(confidenceThreshold),
      isSortOn(isSortOn)
{
}

void Byte::SetMaxAge(const int maxAge) { this->maxAge = maxAge; }

void Byte::SetNumInitialFrame(const int numInitialFrame) { this->numInitialFrame = numInitialFrame; }

void Byte::SetMinFrameSustained(const int minFrameSustained) { this->minFrameSustained = minFrameSustained; }

void Byte::SetIouThresholdHigh(const double iouThresholdHigh) { this->iouThresholdHigh = iouThresholdHigh; }

void Byte::SetIouThresholdLow(const double iouThresholdLow) { this->iouThresholdLow = iouThresholdLow; }

void Byte::SetConfidenceThreshold(const double confidenceThreshold) { this->confidenceThreshold = confidenceThreshold; }

void Byte::SetSortOn(const bool isSortOn) { this->isSortOn = isSortOn; }

void Byte::Reset()
{
    currFrame = 0;
    currId = 1;
    trackers.clear();
}

void Byte::Exec(const std::vector<BboxXyxy> &detections, std::vector<TrackedBbox> &visibleTracks)
{
    currFrame += 1;

    // Trackerを現フレームの状態に更新する
    for (auto trackerItr = trackers.begin(); trackerItr != trackers.end();)
    {
        trackerItr->Predict();
        const BboxUvsr uvsr = trackerItr->GetBbox();
        const BboxXyxy xyxy = BboxUtil::Uvsr2Xyxy(uvsr);
        if (std::isnan(xyxy.x0) || std::isnan(xyxy.y0) || std::isnan(xyxy.x1) || std::isnan(xyxy.y1))
        {
            // 結果がnanなものは取り除く
            trackerItr = trackers.erase(trackerItr);
        }
        else
        {
            trackerItr++;
        }
    }

    std::vector<BboxXyxy> highConfidenceDetections;
    std::vector<BboxXyxy> lowConfidenceDetections;

    if (isSortOn)
    {
        // Sortの場合、detectionsを全部highConfidenceDetectionsに格納
        highConfidenceDetections = detections;
    }
    else
    {
        // Byteの場合、マッチング用のdetectionを2種類計算
        for (const BboxXyxy &detection : detections)
        {
            if (detection.confidence >= confidenceThreshold)
            {
                highConfidenceDetections.push_back(detection);
            }
            else
            {
                lowConfidenceDetections.push_back(detection);
            }
        }
    }

    // First association between all trackers and detections with high confidence
    std::list<RowCol> highAssociations;          // matched index of first association
    std::vector<int> unmatchedHighDetectionIdcs; // unmatched detections' ids after first association
    std::vector<BboxXyxy> trackedBboxesXyxy;
    for (const ObjectTracker &tracker : trackers)
    {
        BboxXyxy xyxy;
        getBboxesXyxy(tracker, xyxy);
        trackedBboxesXyxy.push_back(xyxy);
    }
    associateDetectionsToTrackers(highConfidenceDetections, trackedBboxesXyxy, iouThresholdHigh, highAssociations,
                                  unmatchedHighDetectionIdcs);

    // 一回目にマッチングしたdetectionでtrackerを更新する
    for (const RowCol &matchedIndex : highAssociations)
    {
        const BboxUvsr detectionUvsr = BboxUtil::Xyxy2Uvsr(highConfidenceDetections[matchedIndex.row]);
        auto trackerItr = trackers.begin();
        std::advance(trackerItr, matchedIndex.col);
        trackerItr->Update(detectionUvsr);
        trackerItr->UpdateConfidence(highConfidenceDetections[matchedIndex.row].confidence);
    }

    std::list<ObjectTracker> remainedTrackers;

    // Update trackers and initialize remained trackers
    for (auto trackerItr = trackers.begin(); trackerItr != trackers.end();)
    {
        if (trackerItr->IsUpdated())
        {
            trackerItr++;
        }
        else
        {
            remainedTrackers.push_back(*trackerItr); // Push back umatched trackers to remained trackers
            trackerItr = trackers.erase(trackerItr);
        }
    }

    // Second association from the detections with low confidence to remained trackers
    std::list<RowCol> lowAssociations;
    std::vector<int> unmatchedLowDetectionIdcs;
    std::vector<BboxXyxy> remainedTrackedBboxesXyxy;
    for (const ObjectTracker &tracker : remainedTrackers)
    {
        BboxXyxy xyxy;
        getBboxesXyxy(tracker, xyxy);
        remainedTrackedBboxesXyxy.push_back(xyxy);
    }
    associateDetectionsToTrackers(lowConfidenceDetections, remainedTrackedBboxesXyxy, iouThresholdLow, lowAssociations,
                                  unmatchedLowDetectionIdcs);

    // 二回目にマッチングしたdetectionでtrackerを更新する
    for (const RowCol &matchedIndex : lowAssociations)
    {
        const BboxUvsr detectionUvsr = BboxUtil::Xyxy2Uvsr(lowConfidenceDetections[matchedIndex.row]);
        auto trackerItr = remainedTrackers.begin();
        std::advance(trackerItr, matchedIndex.col);
        trackerItr->Update(detectionUvsr);
        trackerItr->UpdateConfidence(lowConfidenceDetections[matchedIndex.row].confidence);
    }

    // Merge unmatched trakers into trackers
    trackers.splice(trackers.end(), remainedTrackers);

    cleanTrackers();

    // 一回目にマッチングしていないdetectionをtrackerに追加する
    for (const int &detectionId : unmatchedHighDetectionIdcs)
    {
        const BboxUvsr detection = BboxUtil::Xyxy2Uvsr(highConfidenceDetections[detectionId]);
        ObjectTracker tracker(detection, numInitialFrame, minFrameSustained, currId,
                              highConfidenceDetections[detectionId].confidence);
        currId++;
        trackers.push_back(tracker);
    }

    // Before numInitialFrame, add unmatched detections with low confidence to trackers
    if (currFrame < numInitialFrame)
    {
        for (const int &detectionId : unmatchedLowDetectionIdcs)
        {
            const BboxUvsr detection = BboxUtil::Xyxy2Uvsr(lowConfidenceDetections[detectionId]);
            ObjectTracker tracker(detection, numInitialFrame, minFrameSustained, currId,
                                  lowConfidenceDetections[detectionId].confidence);
            currId++;
            trackers.push_back(tracker);
        }
    }

    // 可視状態のtrackerを保存する
    for (ObjectTracker &tracker : trackers)
    {
        if (tracker.IsMatchedTrackVisible(currFrame))
        {
            tracker.Reveal();
            BboxXyxy bodyXyxy;
            getBboxesXyxy(tracker, bodyXyxy);
            const cv::Vec2d v = tracker.GetVelocity();
            bool isBodyDetected = true;
            const TrackedBbox t(tracker.GetId(), bodyXyxy, v, isBodyDetected);
            visibleTracks.push_back(t);
        }
    }

    // Detectionがない場合は、一回可視化されたtrackerの予測結果を保存する
    for (const ObjectTracker &tracker : trackers)
    {
        if (tracker.NumFrameDropped() > 0 && tracker.NumFrameDropped() <= maxAge && tracker.IsVisibleSoFar())
        {
            BboxXyxy bodyXyxy;
            getBboxesXyxy(tracker, bodyXyxy);
            const cv::Vec2d v = tracker.GetVelocity();
            const TrackedBbox t(tracker.GetId(), bodyXyxy, v);
            visibleTracks.push_back(t);
        }
    }
}
