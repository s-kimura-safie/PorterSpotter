/*
 * (c) 2024 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <gflags/gflags.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "DlSystem/DlError.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "nlohmann/json.hpp"

#include "Timer.hpp"
#include "Types.hpp"
#include "VisualizationUtil.hpp"
#include "object_detection/Yolov5.hpp"
#include "pipeline/PorterSpotter.hpp"
#include "pose_estimation/PoseEstimator.hpp"

// Define and parser command line arguments
DEFINE_string(d, "./models/yolov5s_exp19_new_quantized.dlc", "Path to detection model DLC file");
DEFINE_string(h, "./models/rtmpose.dlc", "Path to pose estimation model DLC file");
DEFINE_string(input_file, "videos/sample.mp4", "Path to input video file. e.g. sample.mp4");
DEFINE_string(output_dir, "outputs", "Path to output dir");
DEFINE_bool(person_box, false, "Draw person bbox in video");
DEFINE_bool(skeleton, false, "Draw skeleton in video");
DEFINE_bool(txt, false, "Save track as txt");

std::string getStem(const std::string &filePath)
{
    // strip path of directory and extension
    // ex) "dir1/dir2/stem.ext" -> "stem"

    // strip extension
    size_t pos = filePath.rfind('.');
    const std::string baseName = filePath.substr(0, pos);

    // strip dir name
    pos = baseName.rfind('/');
    if (pos == std::string::npos)
    {
        return baseName;
    }
    else
    {
        return baseName.substr(pos + 1);
    }
}

std::vector<std::string> getAllFiles(const std::string &directoryPath)
{
    std::vector<std::string> files;
    DIR *dir;
    struct dirent *ent;
    struct stat st;

    dir = opendir(directoryPath.c_str());
    if (dir != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            const std::string file_name = ent->d_name;
            const std::string full_file_name = directoryPath + "/" + file_name;

            if (stat(full_file_name.c_str(), &st) == 0)
            {
                if (S_ISREG(st.st_mode))
                {
                    files.push_back(full_file_name);
                }
            }
        }
        closedir(dir);
    }

    return files;
}

void writeTrack(std::ofstream &trackOfs, const std::vector<TrackedBbox> &tracks)
{
    for (size_t i = 0; i < tracks.size(); i++)
    {
        const TrackedBbox &trackedBbox = tracks[i];
        if (trackOfs.is_open())
        {
            for (const PosePoint &point : trackedBbox.poseKeypoints)
            {
                trackOfs << point.x << "," << point.y << ",";
            }
        }
    }
}

bool processFrame(PorterSpotter &porterSpotter, cv::Mat &image, std::ofstream &outputTrackers, const bool isDrawPersonBbox,
                  const bool isDrawSkeleton, const bool isSaveTxt, cv::Mat outImage)
{

    cv::Mat rgbImage;
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);

    std::vector<TrackedBbox> tracks;
    porterSpotter.Run(rgbImage, tracks);

    if (isSaveTxt) writeTrack(outputTrackers, tracks);
    if (isDrawSkeleton) visualization_util::drawTracksSkeleton(tracks, outImage);
    if (isDrawPersonBbox) visualization_util::drawPersonBbox(tracks, outImage);

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

bool runImageAnalysis(PorterSpotter &porterSpotter, const std::string &directoryPath, const std::string &outDir,
                      const bool isDrawPersonBbox, const bool isDrawSkeleton, const bool isSaveTxt)
{
    // Get images from path
    std::vector<std::string> inputFiles = getAllFiles(directoryPath);

    std::cout << "Running network..." << std::endl;

    for (const std::string &inputFile : inputFiles)
    {
        cv::Mat inputImage = cv::imread(inputFile);
        cv::Mat outImage = inputImage.clone();
        std::ofstream outputTrackers;
        std::string outputImageFile = outDir + "/" + getStem(inputFile) + "_output.jpg";
        if (isSaveTxt) outputTrackers.open(outDir + "/" + getStem(inputFile) + "output_trackers.txt", std::ios_base::out);

        // inputImage is BGR order
        processFrame(porterSpotter, inputImage, outputTrackers, isDrawPersonBbox, isDrawSkeleton, isSaveTxt, outImage);
        cv::imwrite(outputImageFile, outImage);
        porterSpotter.ResetTracker(); //画像が連続の場合はコメントアウト
        outputTrackers.close();
    }

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
    if (runImageAnalysis(porterSpotter, FLAGS_input_file, FLAGS_output_dir, FLAGS_person_box, FLAGS_skeleton, FLAGS_txt))
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}
