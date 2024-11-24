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
#include <gflags/gflags.h>

#include "DlSystem/DlError.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "nlohmann/json.hpp"

#include "object_detection/Yolov5.hpp"
#include "pose_estimation/PoseEstimator.hpp"
#include "pipeline/PorterSpotter.hpp"
#include "Timer.hpp"
#include "Types.hpp"
#include "VisualizationUtil.hpp"

// Define and parser command line arguments
DEFINE_string(d, "./models/yolov5s_exp19_new_quantized.dlc", "Path to detection model DLC file");
DEFINE_string(h, "./models/rtmpose.dlc", "Path to pose estimation model DLC file");
DEFINE_string(input_video, "videos/sample.mp4", "Path to input video file. e.g. sample.mp4");
DEFINE_string(output_dir, "outputs", "Path to output dir");
DEFINE_bool(output_video, false, "Save track as annotated video");
DEFINE_bool(person_box, false, "Draw person bbox in video");
DEFINE_bool(skeleton, false, "Draw skeleton in video");
DEFINE_bool(txt, false, "Save track as txt");


void writeTrack(std::ofstream &trackOfs, const std::vector<TrackedBbox> &tracks, const int frameCnt, const int utcMsec)
{
    for (size_t i = 0; i < tracks.size(); i++)
    {
        const TrackedBbox &trackedBbox = tracks[i];
        if (trackOfs.is_open())
        {
            trackOfs << frameCnt << "," << utcMsec << "," << trackedBbox.id << std::endl;
            for (const PosePoint &point : trackedBbox.poseKeypoints)
            {
                trackOfs << point.x << "," << point.y << ",";
            }
        }
    }
}


bool processFrame(PorterSpotter &porterSpotter, cv::Mat &image, const int frameCnt,
                  const int utcMsec, cv::VideoWriter &videoWriter, std::ofstream &outputTrackers, 
                  const bool isSaveVideo, const bool isDrawPersonBbox, const bool isDrawSkeleton, const bool isSaveTxt)
{

    cv::Mat rgbImage;
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);

    std::vector<TrackedBbox> tracks;
    porterSpotter.Run(rgbImage, tracks);

    if (isSaveTxt) writeTrack(outputTrackers, tracks, frameCnt, utcMsec);
    if (isDrawSkeleton) visualization_util::drawTracksSkeleton(tracks, image);
    if (isDrawPersonBbox) visualization_util::drawPersonBbox(image, tracks);
    if (isSaveVideo) videoWriter << image;

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

bool runVideoAnalysis(PorterSpotter &porterSpotter, const std::string &filePath, const std::string &outDir, 
                      const bool isSaveVideo, const bool isDrawPersonBbox, const bool isDrawSkeleton, const bool isSaveTxt)
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

    std::ofstream outputTrackers;
    if (isSaveTxt)
    {
        outputTrackers.open(outDir + "/" + "output_" + basename + "_trackers.txt", std::ios_base::out);
    }

    const double readFps = videoCapture.get(cv::CAP_PROP_FPS);
    const double execFps = 5; 

    cv::VideoWriter videoWriter;
    if (isSaveVideo)
    {
        const std::string outputVideoFile = outDir + "/" + "output_" + basename + ".mp4";
        const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        const int width = (int)videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
        const int height = (int)videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
        videoWriter.open(outputVideoFile, fourcc, execFps, cv::Size(width, height));
    }

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
            processFrame(porterSpotter, image, frameCnt, frameCnt / execFps * 1000, videoWriter,
                         outputTrackers, isSaveVideo, isDrawPersonBbox, isDrawSkeleton, isSaveTxt);
            secondPassed -= execIntervalSec;
            frameCnt++;
        }
        secondPassed += readIntervalSec;
    }
    outputTrackers.close();
    videoCapture.release();
    return true;
}

bool initModel(PorterSpotter &porterSpotter, const std::string &modelType, const std::string &dlcPath,
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
        if (!porterSpotter.InitializeDetection((const uint8_t *)dlcBuff.data(), fileSize, runtimes))
        {
            std::cout << "Couldn't create network instance." << std::endl;
            return false;
        }
    }
    else if (modelType == "pose")
    {
        if (!porterSpotter.InitializePoseEstimator((const uint8_t *)dlcBuff.data(), fileSize, runtimes))
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
    gflags::SetUsageMessage("Offline analysis program for fall detection.");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    PorterSpotter porterSpotter;
    std::string modelType1 = "detection";
    std::string modelType2 = "pose";
    std::string modelType3 = "action";
    const std::vector<std::string> runtimes = {"cpu"};
    if (!initModel(porterSpotter, modelType1, FLAGS_d, runtimes))
    {
        std::cout << "Failed to initialize detection model" << std::endl;
        return false;
    }
    if (!initModel(porterSpotter, modelType2, FLAGS_h, runtimes))
    {
        std::cout << "Failed to initialize opse estimation model" << std::endl;
        return false;
    }

    // Run analysis
    if (runVideoAnalysis(porterSpotter, FLAGS_input_video, FLAGS_output_dir, FLAGS_output_video,
                         FLAGS_person_box, FLAGS_skeleton, FLAGS_txt))
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}
