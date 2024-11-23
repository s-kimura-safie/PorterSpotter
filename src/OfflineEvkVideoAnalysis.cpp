/*
 * (c) 2024 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#include <fstream>
#include <getopt.h>
#include <iostream>
#include <sys/stat.h>

#include "DlSystem/DlError.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "nlohmann/json.hpp"

#include "detect/Yolov5.hpp"
#include "pose/PoseEstimator.hpp"
#include "pipeline/FallDetection.hpp"
#include "action_recognition/STGCN.hpp"
#include "Timer.hpp"
#include "Types.hpp"
#include "KeypointColorPalette.hpp"
#include "MathUtil.hpp"
#include "VisualizationUtil.hpp"

void split(std::vector<std::string> &output, const std::string &input, char delimiter)
{
    output.clear();
    std::istringstream iss(input);
    std::string item;
    while (std::getline(iss, item, delimiter))
    {
        output.push_back(item);
    }
}

// キーポイントを画像に描画する関数
void drawSkeleton(cv::Mat &image, const std::vector<TrackedBbox> &tracks)
{
    // 骨格を定義（キーポイントのインデックスペア）
    const std::vector<std::pair<int, int>> skeleton = {{1, 2},  {1, 3},
                                                       {3, 5},   {2, 4},   {4, 6},  {1, 7}, {2, 8}, {7, 8},
                                                       {7, 9}, {9, 11}, {8, 10}, {10, 12}};
    // const std::vector<std::pair<int, int>> skeleton = {{0, 1},   {0, 2},   {1, 3},   {2, 4},  {5, 6},  {5, 7},
    //                                                    {7, 9},   {6, 8},   {8, 10},  {5, 11}, {6, 12}, {11, 12},
    //                                                    {11, 13}, {13, 15}, {12, 14}, {14, 16}};
    // キーポイントの大きさと色
    const int radius = 4;
    const std::vector<cv::Scalar> keypointColors = KeypointColorPalette::keypointColors;

    for (const TrackedBbox &track : tracks)
    {   
        std::vector<PosePoint> points = track.poseKeypoints;
        // 骨格を描画
        for (const auto &bone : skeleton)
        {
            const int startIdx = bone.first;
            const int endIdx = bone.second;
            const cv::Point startPoint = cv::Point(static_cast<int>(points[startIdx].x * image.cols), static_cast<int>(points[startIdx].y * image.rows));
            const cv::Point endPoint = cv::Point(static_cast<int>(points[endIdx].x * image.cols), static_cast<int>(points[endIdx].y * image.rows));
            cv::line(image, startPoint, endPoint, cv::Scalar(0, 0, 255), 2); // 赤色の線
        }
        // キーポイントを描画
        for (size_t keyPointIdx = 0; keyPointIdx < points.size(); keyPointIdx++)
        {
            PosePoint point = points[keyPointIdx];
            // # circle(画像, 中心座標, 半径, 色, 線幅, 連結)
            cv::circle(image, cv::Point(static_cast<int>(point.x * image.cols), static_cast<int>(point.y * image.rows)), radius, keypointColors[keyPointIdx], cv::FILLED);
        }
    }
}

void drawPersonBbox(cv::Mat &image, const std::vector<TrackedBbox> &tracks)
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

bool processFrame(FallDetection &fallDetection, cv::Mat &image, const int frameCnt, const int utcMsec, cv::VideoWriter &videoWriter, Timer &t_detection, Timer &t_poseEstimation, Timer &t_actionRecognition)
{

    cv::Mat rgbImage;
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);

    std::vector<TrackedBbox> tracks;
    fallDetection.Run(rgbImage, tracks, t_detection, t_poseEstimation, t_actionRecognition);

    drawSkeleton(image, tracks);
    drawPersonBbox(image, tracks);
    videoWriter << image;

    for (const TrackedBbox &track : tracks)
    {
        // std::cout << "Frame: " << frameCnt << "," << utcMsec << "," << track.id  << std::endl;
        // for (const PosePoint &point : track.poseKeypoints)
        // {
        //     std::cout << point.x << "," << point.y << ",";
        // }
    }

    return true;
}

bool runVideoAnalysis(FallDetection &fallDetection, const std::string &filePath, const double execFps)
{
    const size_t periodIdx = filePath.find_last_of(".");
    size_t slashIdx = filePath.find_last_of("/");
    slashIdx = (slashIdx == std::string::npos) ? 0 : slashIdx + 1;
    const std::string basename = filePath.substr(slashIdx, periodIdx - slashIdx);

    cv::VideoCapture videoCapture;
    videoCapture.open(filePath);
    if (!videoCapture.isOpened())
    {
        std::cout << "Couldn't read video: " << filePath << std::endl;
        return false;
    }
    else
    {
        std::cout << "Read video: " << filePath << std::endl;
    }

    std::cout << "Running network..." << std::endl;

    const double readFps = videoCapture.get(cv::CAP_PROP_FPS);

    cv::VideoWriter videoWriter;
    const std::string outputVideoFile = "./outputs/output_" + basename + ".mp4";
    const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    const int width = (int)videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
    const int height = (int)videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
    videoWriter.open(outputVideoFile, fourcc, execFps, cv::Size(width, height));

    Timer t_detection("Detection");
    Timer t_poseEstimation("Pose Estimation");
    Timer t_actionRecognition("Action Recognition");

    const double readIntervalSec = 1 / readFps;
    const double execIntervalSec = 1 / execFps;
    int frameCnt = 0;
    double secondPassed = 0;
    while (1)
    {
        cv::Mat image;
        videoCapture >> image; // videoからimageへ1フレームを取り込む
        if (image.empty() == true)
        {
            break; // 画像が読み込めなかったとき、無限ループを抜ける
        }
        if (frameCnt == 0 || secondPassed >= execIntervalSec)
        {
            processFrame(fallDetection, image, frameCnt, frameCnt / execFps * 1000, videoWriter, t_detection, t_poseEstimation, t_actionRecognition);
            secondPassed -= execIntervalSec;
            frameCnt++;
        }
        secondPassed += readIntervalSec;
    }

    std::cout << "Detection: " << t_detection.ResultString(true) << std::endl;
    std::cout << "Pose Estimation: " << t_poseEstimation.ResultString(true) << std::endl;
    std::cout << "Action Recognition: " << t_actionRecognition.ResultString(true) << std::endl;

    videoCapture.release();
    return true;
}

bool initModel(FallDetection &fallDetection, const std::string &modelType, const std::string &dlcPath,
               const std::vector<std::string> &runtimes)
{
    // Get model file size using stat
    struct stat sb;
    if (stat(dlcPath.c_str(), &sb))
    {
        std::cout << "DLC file doesn't exist: " << dlcPath << std::endl;
        return false;
    }
    auto fileSize = sb.st_size;

    // Allocate buffer to read whole file
    std::vector<char> dlcBuff(fileSize);
    std::ifstream fin(dlcPath, std::ios::in | std::ios::binary);
    if (!fin)
    {
        std::cout << "Couldn't open the DLC file" << dlcPath << std::endl;
        return false;
    }
    fin.read(dlcBuff.data(), fileSize);

    std::cout << "Initializing model " << std::endl;
    // Create network with selected runtime
    if (modelType == "detection")
    {
        if (!fallDetection.InitializeDetection((const uint8_t *)dlcBuff.data(), fileSize, runtimes))
        {
            std::cout << "Couldn't create network instance." << std::endl;
            return false;
        }
    }
    else if (modelType == "pose")
    {
        if (!fallDetection.InitializePoseEstimator((const uint8_t *)dlcBuff.data(), fileSize, runtimes))
        {
            std::cout << "Couldn't create network instance." << std::endl;
            return false;
        }
    }
    else if (modelType == "action")
    {
        if (!fallDetection.InitializeActionRecognition((const uint8_t *)dlcBuff.data(), fileSize, runtimes))
        {
            std::cout << "Couldn't create network instance." << std::endl;
            return false;
        }
    }
    else
    {
        std::cout << "Invalid model type: " << modelType << std::endl;
        return false;
    }
    std::cout << "Network initialized" << std::endl;

    return true;
}

int main(int argc, char **argv)
{
    std::string detectDlcPath = "./models/yolov5s_exp19_new_quantized.dlc";
    std::string poseDlcPath = "./models/rtmpose.dlc";
    std::string actionDlcPath = "./models/stgcn.dlc";
    std::string inputVideoPath = "./videos/GH019443.MP4";
    std::vector<std::string> runtimes = {"cpu"};
    const std::string modelType1 = "detection";
    const std::string modelType2 = "pose";
    const std::string modelType3 = "action";
    double execFps = 10.0;

    int opt = 0;
    while ((opt = getopt(argc, argv, "d:p:a:i:r:f:")) != -1)
    {
        switch (opt)
        {
        case 'd':
            detectDlcPath = optarg;
            break;
        case 'p':
            poseDlcPath = optarg;
            break;
        case 'a':
            actionDlcPath = optarg;
            break;
        case 'i':
            inputVideoPath = optarg;
            break;
        case 'r':
        {
            const std::string runtimeStrings = optarg;
            split(runtimes, runtimeStrings, ',');
            break;
        }
        case 'f':
            execFps = atof(optarg);
            break;
        default:
            std::cerr << "Invalid parameter specified. Please run snpe-sample with the -h flag to see required arguments"
                      << std::endl;
            std::exit(EXIT_FAILURE);
            break;
        }
    }

    FallDetection fallDetection;
    if (!initModel(fallDetection, modelType1, detectDlcPath, runtimes))
    {
        std::cout << "Failed to initialize detection model" << std::endl;
        return false;
    }
    if (!initModel(fallDetection, modelType2, poseDlcPath, runtimes))
    {
        std::cout << "Failed to initialize opse estimation model" << std::endl;
        return false;
    }
    if (!initModel(fallDetection, modelType3, actionDlcPath, runtimes))
    {
        std::cout << "Failed to initialize action recognition model" << std::endl;
        return false;
    }

    // Run analysis
    if (runVideoAnalysis(fallDetection, inputVideoPath, execFps))
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}
