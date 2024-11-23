/*
 * (c) 2021 Safie Inc.
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
#include "KeypointColorPalette.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "Timer.hpp"
#include "Types.hpp"
#include "detect/Yolov5.hpp"
#include "nlohmann/json.hpp"
#include "pose/PoseEstimator.hpp"

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

std::vector<std::string> getAvailableRuntimes()
{
    using zdl::DlSystem::Runtime_t;
    Runtime_t runtimes[] = {Runtime_t::CPU_FLOAT32, Runtime_t::GPU_FLOAT32_16_HYBRID, Runtime_t::DSP_FIXED8_TF,
                            Runtime_t::GPU_FLOAT16, Runtime_t::AIP_FIXED8_TF};
    std::vector<std::string> ret;
    for (const auto &runtime : runtimes)
    {
        if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime))
        {
            ret.push_back(zdl::DlSystem::RuntimeList::runtimeToString(runtime));
        }
    }
    return ret;
}

__off_t getFileSize(const std::string &dlcPath)
{
    struct stat sb;
    if (stat(dlcPath.c_str(), &sb))
    {
        std::cout << "Couldn't get the filesize of DLC: " << dlcPath << std::endl;
        return EXIT_FAILURE;
    }
    return sb.st_size;
}

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
void drawSkeleton(const std::vector<PosePoint> &points, cv::Mat &image)
{
    // 骨格を定義（キーポイントのインデックスペア）
    const std::vector<std::pair<int, int>> skeleton = {{0, 1},   {0, 2},   {1, 3},   {2, 4},  {5, 6},  {5, 7},
                                                       {7, 9},   {6, 8},   {8, 10},  {5, 11}, {6, 12}, {11, 12},
                                                       {11, 13}, {13, 15}, {12, 14}, {14, 16}};
    // キーポイントの大きさと色
    const int radius = 10;
    const std::vector<cv::Scalar> keypointColors = KeypointColorPalette::keypointColors;

    // 骨格を描画
    for (const auto &bone : skeleton)
    {
        const int startIdx = bone.first;
        const int endIdx = bone.second;
        const cv::Point startPoint = cv::Point(points[startIdx].x, points[startIdx].y);
        const cv::Point endPoint = cv::Point(points[endIdx].x, points[endIdx].y);
        cv::line(image, startPoint, endPoint, cv::Scalar(0, 0, 255), 2); // 赤色の線
    }
    // キーポイントを描画
    for (size_t keyPointIdx = 0; keyPointIdx < points.size(); keyPointIdx++)
    {
        PosePoint point = points[keyPointIdx];
        // # circle(画像, 中心座標, 半径, 色, 線幅, 連結)
        cv::circle(image, cv::Point(point.x, point.y), radius, keypointColors[keyPointIdx], cv::FILLED);
    }
}

int main(int argc, char **argv)
{
    // Variables for command line arguments
    std::string detectDlcPath = "";
    std::string poseDlcPath = "";
    float scoreThreshold = 0.37;
    float iouThreshold = 0.35;
    std::vector<std::string> runtimes = {"cpu"};
    std::string outputDirectory = "outputs";

    // Process command line arguments
    int opt = 0;
    while ((opt = getopt(argc, argv, "hd:p:t:i:r:o:")) != -1)
    {
        switch (opt)
        {
        case 'h':
        {

            std::cout << "Offile object detection using SNPE C++ API\n"
                      << "Usage:\n"
                      << "  -d <FILE> DLC file for detecction (Required)\n"
                      << "  -p <FILE> DLC file for pose estimation (Required)\n"
                      << "  -t <THRESHOLD> Score threshold (Default: 0.25)\n"
                      << "  -i <IOU_THRESHOLD> IoU threshold (Default: 0.45)\n"
                      << "  -r <RUNTIME>, <RUNTIME> ... Runtime order. [cpu, dsp, gpu] (Default: cpu)\n"
                      << "  -o <OUTPUT_DIR> Output directory (Default: outputs)\n"
                      << "  inputfile ... Input image files\n"
                      << std::endl;
            std::cout << "Library Version: " << zdl::SNPE::SNPEFactory::getLibraryVersion().toString() << std::endl;

            // Display available runtimes
            const std::vector<std::string> availableRuntimes = getAvailableRuntimes();
            std::stringstream ss;
            ss << "Available runtimes = [";
            for (const auto &runtime : availableRuntimes)
            {
                ss << "\"" << runtime << "\",";
            }
            // Remove last ","
            ss.seekp(-1, std::ios_base::end);
            ss << "]" << std::endl;
            std::cout << ss.str();

            std::exit(EXIT_SUCCESS);
            break;
        }
        case 'd':
            detectDlcPath = optarg;
            break;
        case 'p':
            poseDlcPath = optarg;
            break;
        case 't':
            scoreThreshold = atof(optarg);
            break;
        case 'i':
            iouThreshold = atof(optarg);
            break;
        case 'r':
        {
            const std::string runtimeStrings = optarg;
            split(runtimes, runtimeStrings, ',');
            break;
        }
        case 'o':
            outputDirectory = optarg;
            break;
        default:
            std::cerr << "Invalid parameter specified. Please run snpe-sample with the -h flag to see required arguments"
                      << std::endl;
            std::exit(EXIT_FAILURE);
            break;
        }
    }

    // Get input files
    std::vector<std::string> inputFiles;
    for (int i = optind; i < argc; i++)
    {
        inputFiles.push_back(argv[i]);
    }

    // Check if output directory exists
    {
        // Memo: Cannot make more thant 2 depth directory
        int ret = mkdir(outputDirectory.c_str(),
                        S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IWGRP | S_IXGRP | S_IROTH | S_IXOTH | S_IXOTH);
        if (ret == 0)
        {
            std::cout << "Create output directory: " << outputDirectory << std::endl;
        }
        else if (errno == EEXIST)
            std::cout << "Output directory already exist: " << outputDirectory << std::endl;
        else
        {
            std::cerr << "Could not create output directory: " << outputDirectory << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    // Create model instance
    // Detector
    Yolov5 detector;

    // Load network for detector
    // Get file size using stat
    auto detectFileSize = getFileSize(detectDlcPath);

    // Allocate buffer to read whole file
    std::vector<char> detectDlcBuff(detectFileSize);
    std::ifstream detectFin(detectDlcPath, std::ios::in | std::ios::binary);
    if (!detectFin)
    {
        std::cout << "Couldn't open Detection DLC file: " << detectDlcPath << std::endl;
        return EXIT_FAILURE;
    }
    detectFin.read(detectDlcBuff.data(), detectFileSize);
    detectFin.close();

    for (const std::string &runtimeStr : runtimes)
    {
        std::cout << "Runtime is " << runtimeStr << std::endl;
    }

    // Create network
    if (!detector.CreateNetwork((const uint8_t *)detectDlcBuff.data(), detectFileSize, runtimes))
    {
        std::cout << "Couldn't create detection network instance" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Detection network initialized" << std::endl;

    // Set params
    detector.SetIoUThreshold(iouThreshold);
    detector.SetScoreThreshold(scoreThreshold);

    // Load network for Pose Estimator
    PoseEstimator poseEstimator;

    // Get file size using stat
    auto poseFileSize = getFileSize(poseDlcPath);

    // Allocate buffer to read whole file
    std::vector<char> poseDlcBuff(poseFileSize);
    std::ifstream poseFin(poseDlcPath, std::ios::in | std::ios::binary);
    if (!poseFin)
    {
        std::cout << "Couldn't open the Pose DLC file" << poseDlcPath << std::endl;
        return EXIT_FAILURE;
    }
    poseFin.read(poseDlcBuff.data(), poseFileSize);
    poseFin.close();

    for (const std::string &runtimeStr : runtimes)
    {
        std::cout << "Runtime is " << runtimeStr << std::endl;
    }

    if (!poseEstimator.CreateNetwork((const uint8_t *)poseDlcBuff.data(), poseFileSize, runtimes))
    {
        std::cout << "Couldn't create Pose network instance." << std::endl;
        return EXIT_FAILURE;
    }

    Timer t_detection("Detection");
    Timer t_poseEstimation("Pose Estimation");
    std::cout << "Running network..." << std::endl;

    for (const std::string &inputFile : inputFiles)
    {
        // inputImage is BGR order
        cv::Mat inputImage = cv::imread(inputFile);
        std::string outputImageFile = outputDirectory + "/" + getStem(inputFile) + "_output.jpg";

        std::cout << "Processing: " << inputFile << std::endl;

        // BGR to RGB
        cv::Mat rgbImage;
        cv::cvtColor(inputImage, rgbImage, cv::COLOR_BGR2RGB);

        // Detect objects
        std::vector<BboxXyxy> objectList;
        std::cout << "Detecting objects..." << std::endl;
        t_detection.Start();
        detector.Detect(rgbImage, objectList);
        t_detection.End();

        std::cout << "Detected objects: ";
        std::cout << objectList.size() << std::endl;

        // Pose Estimation
        std::cout << "Estimating pose..." << std::endl;
        std::vector<PosePoint> posePoints;
        for (const BboxXyxy &bbox : objectList)
        {
            t_poseEstimation.Start();
            posePoints = poseEstimator.Inference(inputImage, bbox);
            t_poseEstimation.End();

            drawSkeleton(posePoints, inputImage);
        }

        std::cout << "Pose estimation done" << std::endl;
        cv::imwrite("outputs/" + getStem(inputFile) + ".jpg", inputImage);
    }

    std::cout << "# Detection: " << t_detection.Count() << ", Accumulated: " << t_detection.Accumulated()
              << ", Avarage: " << t_detection.Average() << ", Stdev: " << t_detection.Stdev() << " ("
              << t_detection.Stdev() / t_detection.Average() * 100 << "%)" << std::endl;

    std::cout << "# PoseEstimation: " << t_poseEstimation.Count() << ", Accumulated: " << t_poseEstimation.Accumulated()
              << ", Avarage: " << t_poseEstimation.Average() << ", Stdev: " << t_poseEstimation.Stdev() << " ("
              << t_poseEstimation.Stdev() / t_poseEstimation.Average() * 100 << "%)" << std::endl;

    return EXIT_SUCCESS;
}
