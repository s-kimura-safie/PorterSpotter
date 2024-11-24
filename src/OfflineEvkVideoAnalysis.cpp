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

#include "KeypointColorPalette.hpp"
#include "MathUtil.hpp"
#include "Timer.hpp"
#include "Types.hpp"
#include "VisualizationUtil.hpp"
#include "detect/Yolov5.hpp"
#include "pipeline/FallDetection.hpp"
#include "pose/PoseEstimator.hpp"

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

bool processFrame(FallDetection &fallDetection, cv::Mat &image, const int frameCnt, const int utcMsec,
                  cv::VideoWriter &videoWriter, Timer &t_detection, Timer &t_poseEstimation, Timer &t_actionRecognition)
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
            processFrame(fallDetection, image, frameCnt, frameCnt / execFps * 1000, videoWriter, t_detection,
                         t_poseEstimation, t_actionRecognition);
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
