/*
 * (c) 2021 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */
#include "ObjectTracker.hpp"

/// @brief スケールが小さい場合に値を編集する
void ObjectTracker::adjustScale()
{
    // スケールはマイナスになるとbbox_x1y1x2y2に変換時にnanが発生するので0に。
    // (特に微分値をケア)。
    if (x.at<double>(6, 0) + x.at<double>(2, 0) <= 0)
    {
        x.at<double>(6, 0) *= 0.0;
    }
}

/// @brief コンストラクタ
/// @param bbox バウンディングボックス
/// @param detectionThreshold_ 検出の閾値
/// @param id_ トラッキングデータに割り当てるID
ObjectTracker::ObjectTracker(const BboxUvsr bbox, const int numInitialFrame, const int minFrameSustained, const int id)
    : numInitialFrame(numInitialFrame), minFrameSustained(minFrameSustained), id(id)
{
    // Kalman Filter パラメータ
    // 時間遷移を表す行列
    F = (cv::Mat_<double>(7, 7) << 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1);

    // 状態変数と観測変数の変換を表す行列
    H = (cv::Mat_<double>(4, 7) << 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0);

    // 観測ノイズの共分散行列
    R = (cv::Mat_<double>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 10, 0, 0, 0, 0, 10);

    // ノイズの共分散行列
    Q = (cv::Mat_<double>(7, 7) << 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
         0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0.0001);

    // 誤差の共分散行列
    // トラッキングデータごとに内部で変化・ここでは初期値のみ指定
    P = (cv::Mat_<double>(7, 7) << 10, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0,
         0, 0, 0, 10000, 0, 0, 0, 0, 0, 0, 0, 10000, 0, 0, 0, 0, 0, 0, 0, 10000);

    // 状態変数
    x = (cv::Mat_<double>(7, 1) << bbox.u, bbox.v, bbox.s, bbox.r, 0, 0, 0);

    numFrameDropped = 0;
    numFrameSustained = 0;
    confidence = 0.0;
    isVisibleSoFar = false;
    isHidden = true;
}

ObjectTracker::ObjectTracker(const BboxUvsr bbox, const int numInitialFrame, const int minFrameSustained, const int id,
                             const double confidence)
    : ObjectTracker(bbox, numInitialFrame, minFrameSustained, id)
{
    this->confidence = confidence;
}

/// @brief Bboxを取得する
BboxUvsr ObjectTracker::GetBbox() const
{
    BboxUvsr ret;
    ret.u = x.at<double>(0, 0);
    ret.v = x.at<double>(1, 0);
    ret.s = x.at<double>(2, 0);
    ret.r = x.at<double>(3, 0);
    return ret;
}

double ObjectTracker::GetSpeed() const
{
    const double delta_u = x.at<double>(4, 0);
    const double delta_v = x.at<double>(5, 0);

    // Returns speed without considering the aspect ratio
    return std::sqrt(std::pow(delta_u, 2) + std::pow(delta_v, 2));
}

cv::Vec2d ObjectTracker::GetVelocity() const
{
    const double delta_u = x.at<double>(4, 0);
    const double delta_v = x.at<double>(5, 0);
    return cv::Vec2d(delta_u, delta_v);
}

/// @brief tracked BboxのIDを取得する
int ObjectTracker::GetId() const { return id; }

/// @brief 検出され続けているフレームの数
int ObjectTracker::NumFrameSustained() const { return numFrameSustained; }

/// @brief 更新されていないフレームの数
int ObjectTracker::NumFrameDropped() const { return numFrameDropped; }

/// @brief マッチングしたトラックレットの可視状態
/// @param currFrame 現在のフレーム
bool ObjectTracker::IsMatchedTrackVisible(const int currFrame)
{
    if (numFrameDropped >= 1) // マッチングしていないときは判定対象外としてfalseを返す
    {
        return false;
    }

    if (!isHidden) // 連続検出フレーム数の閾値を一度でも超えたトラックレット
    {
        isVisibleSoFar = true;
        return true;
    }
    else if (numFrameSustained >= minFrameSustained) // 連続検出フレーム数が閾値以上なら可視
    {
        isVisibleSoFar = true;
        return true;
    }
    else if (currFrame <= numInitialFrame) // 追跡を開始してから最初の数フレームは可視
    {
        isVisibleSoFar = true;
        return true;
    }
    else
    {
        return false;
    }
}

bool ObjectTracker::IsVisibleSoFar() const { return isVisibleSoFar; }

void ObjectTracker::Reveal() { isHidden = false; }

/// @brief Kalman Filterを更新する
/// @param bbox 更新に使うbbox
void ObjectTracker::Update(const BboxUvsr bbox)
{
    numFrameDropped = 0;
    numFrameSustained++;

    // 以下Kalman Filter演算
    z = (cv::Mat_<double>(4, 1) << bbox.u, bbox.v, bbox.s, bbox.r);
    e = z - H * x;
    K = P * H.t() * (R + H * P * H.t()).inv(cv::DECOMP_SVD);
    x = x + K * e;
    P = (cv::Mat::eye(7, 7, CV_64F) - K * H) * P;
}

/// @brief Kalman filterで次のフレームのtracked Bboxを計算する
void ObjectTracker::Predict()
{
    adjustScale();

    // Kalman Filter演算
    x = F * x;
    P = F * P * F.t() + Q;

    // 更新されていないようならばdetectFrameを0にする
    if (numFrameDropped > 0)
    {
        numFrameSustained = 0;
    }
    numFrameDropped++;
}

bool ObjectTracker::IsUpdated() const { return (numFrameDropped == 0); }
