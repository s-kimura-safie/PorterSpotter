/*
 * (c) 2024 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#include "VisualizationUtil.hpp"

const float visualization_util::color_list[80][3] = {
    {0.741, 0.447, 0.000}, {0.098, 0.325, 0.850}, {0.125, 0.694, 0.929}, {0.556, 0.184, 0.494}, {0.188, 0.674, 0.466},
    {0.933, 0.745, 0.301}, {0.184, 0.078, 0.635}, {0.300, 0.300, 0.300}, {0.600, 0.600, 0.600}, {0.000, 0.000, 1.000},
    {0.000, 0.500, 1.000}, {0.000, 0.749, 0.749}, {0.000, 1.000, 0.000}, {1.000, 0.000, 0.000}, {1.000, 0.000, 0.667},
    {0.000, 0.333, 0.333}, {0.000, 0.667, 0.333}, {0.000, 1.000, 0.333}, {0.000, 0.333, 0.667}, {0.000, 0.667, 0.667},
    {0.000, 1.000, 0.667}, {0.000, 0.333, 1.000}, {0.000, 0.667, 1.000}, {0.000, 1.000, 1.000}, {0.500, 0.333, 0.000},
    {0.500, 0.667, 0.000}, {0.500, 1.000, 0.000}, {0.500, 0.000, 0.333}, {0.500, 0.333, 0.333}, {0.500, 0.667, 0.333},
    {0.500, 1.000, 0.333}, {0.500, 0.000, 0.667}, {0.500, 0.333, 0.667}, {0.500, 0.667, 0.667}, {0.500, 1.000, 0.667},
    {0.500, 0.000, 1.000}, {0.500, 0.333, 1.000}, {0.500, 0.667, 1.000}, {0.500, 1.000, 1.000}, {1.000, 0.333, 0.000},
    {1.000, 0.667, 0.000}, {1.000, 1.000, 0.000}, {1.000, 0.000, 0.333}, {1.000, 0.333, 0.333}, {1.000, 0.667, 0.333},
    {1.000, 1.000, 0.333}, {1.000, 0.000, 0.667}, {1.000, 0.333, 0.667}, {1.000, 0.667, 0.667}, {1.000, 1.000, 0.667},
    {1.000, 0.000, 1.000}, {1.000, 0.333, 1.000}, {1.000, 0.667, 1.000}, {0.000, 0.000, 0.333}, {0.000, 0.000, 0.500},
    {0.000, 0.000, 0.667}, {0.000, 0.000, 0.833}, {0.000, 0.000, 1.000}, {0.000, 0.167, 0.000}, {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000}, {0.000, 0.667, 0.000}, {0.000, 0.833, 0.000}, {0.000, 1.000, 0.000}, {0.167, 0.000, 0.000},
    {0.333, 0.000, 0.000}, {0.500, 0.000, 0.000}, {0.667, 0.000, 0.000}, {0.833, 0.000, 0.000}, {1.000, 0.000, 0.000},
    {0.000, 0.000, 0.000}, {0.143, 0.143, 0.143}, {0.286, 0.286, 0.286}, {0.429, 0.429, 0.429}, {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714}, {0.857, 0.857, 0.857}, {0.741, 0.447, 0.000}, {0.741, 0.717, 0.314}, {0.000, 0.500, 0.500}};

const std::vector<cv::Scalar> visualization_util::keypointColors = {
    cv::Scalar(255, 0, 0),     // Blue
    cv::Scalar(0, 255, 0),     // Green
    cv::Scalar(0, 0, 255),     // Red
    cv::Scalar(255, 255, 0),   // Cyan
    cv::Scalar(255, 0, 255),   // Magenta
    cv::Scalar(0, 255, 255),   // Yellow
    cv::Scalar(128, 0, 0),     // Maroon
    cv::Scalar(0, 128, 0),     // Olive
    cv::Scalar(128, 128, 0),   // Lime
    cv::Scalar(0, 0, 128),     // Navy
    cv::Scalar(128, 0, 128),   // Purple
    cv::Scalar(0, 128, 128),   // Teal
    cv::Scalar(128, 128, 128), // Gray
    cv::Scalar(64, 0, 0),      // Brown
    cv::Scalar(0, 64, 0),      // Dark Green
    cv::Scalar(0, 0, 64),      // Dark Blue
    cv::Scalar(64, 64, 64)     // Dark Gray
};

void visualization_util::DrawLabel(cv::Mat &image, cv::Point2i point, const std::string &text, const cv::Scalar &color)
{
    // Calculate text color according to the label color
    const float c_mean = cv::mean(color)[0];
    cv::Scalar txt_color;
    if (c_mean > 0.5)
    {
        txt_color = cv::Scalar(0, 0, 0);
    }
    else
    {
        txt_color = cv::Scalar(255, 255, 255);
    }

    // Write multi-line string
    std::stringstream ss(text);
    std::string line;
    while (std::getline(ss, line, '\n'))
    {
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
        cv::Scalar txt_bk_color = color * 0.7 * 255;

        cv::rectangle(image, cv::Rect(point, cv::Size(label_size.width, label_size.height + baseLine)), txt_bk_color, -1);

        cv::putText(image, line, cv::Point(point.x, point.y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color,
                    1, cv::LineTypes::LINE_AA);

        point = cv::Point(point.x, point.y + label_size.height + baseLine + 2);
    }
}

int calcBodyLength(const std::vector<PosePoint> &points)
{
    // Keypoints : https://github.com/open-mmlab/mmpose/tree/1.x/projects/rtmpose#body-2d
    const PosePoint &point6 = points[6];
    const PosePoint &point12 = points[12];
    double distanceLeft = std::sqrt(std::pow(point12.x - point6.x, 2) + std::pow(point12.y - point6.y, 2));
    const PosePoint &point5 = points[5];
    const PosePoint &point11 = points[11];
    double distanceRight = std::sqrt(std::pow(point11.x - point5.x, 2) + std::pow(point11.y - point5.y, 2));
    double bodyLength = std::max(distanceLeft, distanceRight);
    return bodyLength;
}

// キーポイントを入力し、画像に描画する関数
void visualization_util::drawSkeleton(const std::vector<PosePoint> &points, cv::Mat &image)
{
    // 骨格を定義（キーポイントのインデックスペア）
    const std::vector<std::pair<int, int>> skeleton = {{0, 1},   {0, 2},   {1, 3},   {2, 4},  {5, 6},  {5, 7},
                                                       {7, 9},   {6, 8},   {8, 10},  {5, 11}, {6, 12}, {11, 12},
                                                       {11, 13}, {13, 15}, {12, 14}, {14, 16}};

    // 画像サイズ
    int width = image.cols;
    int height = image.rows;

    // キーポイントの大きさと骨線の太さ
    // double bodyLength = calcBodyLength(points); // FIXME:BBOXに応じて大きさを変える
    const int radius = 10;
    const int thickness = 3;

    // 骨格を描画
    for (const auto &bone : skeleton)
    {
        const int startIdx = bone.first;
        const int endIdx = bone.second;
        const cv::Point startPoint = cv::Point(points[startIdx].x * width, points[startIdx].y * height);
        const cv::Point endPoint = cv::Point(points[endIdx].x * width, points[endIdx].y * height);
        cv::line(image, startPoint, endPoint, cv::Scalar(0, 0, 255), thickness); // 赤色の線
    }
    // キーポイントを描画
    for (size_t keyPointIdx = 0; keyPointIdx < points.size(); keyPointIdx++)
    {
        PosePoint point = points[keyPointIdx];
        // # circle(画像, 中心座標, 半径, 色, 線幅, 連結)
        cv::circle(image, cv::Point(point.x * width, point.y * height), radius, keypointColors[keyPointIdx], cv::FILLED);
    }
}

// 検出した全人物のトラックを入力し、キーポイントを画像に描画する関数
void visualization_util::drawTracksSkeleton(const std::vector<TrackedBbox> &tracks, cv::Mat &image)
{
    // 骨格を定義（キーポイントのインデックスペア）
    const std::vector<std::pair<int, int>> skeleton = {{1, 2}, {1, 3}, {3, 5}, {2, 4},  {4, 6},  {1, 7},
                                                       {2, 8}, {7, 8}, {7, 9}, {9, 11}, {8, 10}, {10, 12}};
    // const std::vector<std::pair<int, int>> skeleton = {{0, 1},   {0, 2},   {1, 3},   {2, 4},  {5, 6},  {5, 7},
    //                                                    {7, 9},   {6, 8},   {8, 10},  {5, 11}, {6, 12}, {11, 12},
    //                                                    {11, 13}, {13, 15}, {12, 14}, {14, 16}};
    // キーポイントの大きさと色
    const int radius = 4;

    for (const TrackedBbox &track : tracks)
    {
        std::vector<PosePoint> points = track.poseKeypoints;
        // 骨格を描画
        for (const auto &bone : skeleton)
        {
            const int startIdx = bone.first;
            const int endIdx = bone.second;
            const cv::Point startPoint = cv::Point(static_cast<int>(points[startIdx].x * image.cols),
                                                   static_cast<int>(points[startIdx].y * image.rows));
            const cv::Point endPoint =
                cv::Point(static_cast<int>(points[endIdx].x * image.cols), static_cast<int>(points[endIdx].y * image.rows));
            cv::line(image, startPoint, endPoint, cv::Scalar(0, 0, 255), 2); // 赤色の線
        }
        // キーポイントを描画
        for (size_t keyPointIdx = 0; keyPointIdx < points.size(); keyPointIdx++)
        {
            PosePoint point = points[keyPointIdx];
            // # circle(画像, 中心座標, 半径, 色, 線幅, 連結)
            cv::circle(image, cv::Point(static_cast<int>(point.x * image.cols), static_cast<int>(point.y * image.rows)),
                       radius, keypointColors[keyPointIdx], cv::FILLED);
        }
    }
}

void visualization_util::drawPersonBbox(cv::Mat &image, const std::vector<TrackedBbox> &tracks)
{
    for (size_t i = 0; i < tracks.size(); i++)
    {
        const TrackedBbox &trackedBbox = tracks[i];

        const int colorIdx = (int)trackedBbox.id % 80;
        cv::Scalar color =
            cv::Scalar(visualization_util::color_list[colorIdx][0], visualization_util::color_list[colorIdx][1],
                       visualization_util::color_list[colorIdx][2]);

        const int personX0 = MathUtil::Clamp<int>(image.cols * trackedBbox.bodyBbox.x0, 0, image.cols - 1);
        const int personY0 = MathUtil::Clamp<int>(image.rows * trackedBbox.bodyBbox.y0, 0, image.rows - 1);
        const int personX1 = MathUtil::Clamp<int>(image.cols * trackedBbox.bodyBbox.x1, 0, image.cols - 1);
        const int personY1 = MathUtil::Clamp<int>(image.rows * trackedBbox.bodyBbox.y1, 0, image.rows - 1);
        cv::Rect personRect = cv::Rect(cv::Point2i(personX0, personY0), cv::Point2i(personX1, personY1));
        cv::rectangle(image, personRect, color * 255, 2);

        cv::putText(image, trackedBbox.action, cv::Point(personX0, personY0 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
}
