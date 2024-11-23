/*
 * (c) 2021 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */
#pragma once

#include <deque>

#include "BboxUtil.hpp"


/// @brief カルマンフィルタを用いたオブジェクトトラッカー
class ObjectTracker
{
private:
    cv::Mat x, z;
    cv::Mat F, H, R, Q, P;
    cv::Mat e, K;

    int numFrameDropped;   // 検出できていないフレーム数
    int numFrameSustained; // 連続で検出できているフレーム
    int numInitialFrame;   // 連続検出の条件をスキップする初期フレーム数
    int minFrameSustained; // 何フレーム以上連続で検出された場合に可視化状態になる
    int id;
    bool isVisibleSoFar; // 一度でもvisibleになったか
    bool isHidden;       // 公開していないトラックレット
    double confidence;

    void adjustScale();

public:
    ObjectTracker(){};
    ObjectTracker(const BboxUvsr bbox, const int numInitialFrame, const int minFrameSustained, const int id);
    ObjectTracker(const BboxUvsr bbox, const int numInitialFrame, const int minFrameSustained, const int id,
                  const double confidence);
    ~ObjectTracker(){};

    // Getter
    BboxUvsr GetBbox() const;
    double GetSpeed() const;
    double GetConfidence() const { return confidence; };
    cv::Vec2d GetVelocity() const;
    int GetId() const;
    int NumFrameSustained() const;
    int NumFrameDropped() const;
    bool IsMatchedTrackVisible(const int currFrame);
    bool IsVisibleSoFar() const;

    void Reveal();
    void Update(const BboxUvsr bbox);
    void UpdateConfidence(const double confidence) { this->confidence = confidence; }
    void Predict();
    bool IsUpdated() const;
};
