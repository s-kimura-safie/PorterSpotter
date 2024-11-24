/*
 * (c) 2024 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#include "PoseEstimator.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SnpeUtil.hpp"

#include <cmath>

// TODO:適切な関数名に変える。NormalizeImageとか。
void PoseEstimator::makeFloatImg(const cv::Mat &input, cv::Mat &output)
{
    cv::Mat convertedInput;
    input.convertTo(convertedInput, CV_32FC3, 1.0 / 255.0);

    const int h_img = input.rows;
    const int w_img = input.cols;
    cv::Mat mean(h_img, w_img, CV_32FC3, {0.485, 0.456, 0.406});
    cv::Mat std(h_img, w_img, CV_32FC3, {0.229, 0.224, 0.225});
    output = (convertedInput - mean) / std;
}

// TODO: 役割をクロップとアフィン変換に分けた関数を作る。
std::pair<cv::Mat, cv::Mat> PoseEstimator::cropImageByDetectBox(const cv::Mat &inputImage, const BboxXyxy &box)
{
    std::pair<cv::Mat, cv::Mat> resultPair;

    if (!inputImage.data)
    {
        return resultPair;
    }

    // deep copy
    cv::Mat inputMatCopy;
    inputImage.copyTo(inputMatCopy);

    // calculate the width, height and center points of the human detection box
    float inputWidth = inputImage.cols;
    float inputHeight = inputImage.rows;
    int x0 = box.x0 * inputWidth;
    int x1 = box.x1 * inputWidth;
    int y0 = box.y0 * inputHeight;
    int y1 = box.y1 * inputHeight;
    int box_width = x1 - x0;
    int box_height = y1 - y0;
    int box_center_x = x0 + box_width / 2;
    int box_center_y = y0 + box_height / 2;

    float aspect_ratio = 192.0 / 256.0;

    // adjust the width and height ratio of the size of the picture in the RTMPOSE input
    if (box_width > (aspect_ratio * box_height))
    {
        box_height = box_width / aspect_ratio;
    }
    else if (box_width < (aspect_ratio * box_height))
    {
        box_width = box_height * aspect_ratio;
    }

    float scale_image_width = box_width * 1.2;
    float scale_image_height = box_height * 1.2;

    // get the affine matrix
    cv::Mat affine_transform =
        GetAffineTransform(box_center_x, box_center_y, scale_image_width, scale_image_height, 192, 256);

    cv::Mat affine_transform_reverse =
        GetAffineTransform(box_center_x, box_center_y, scale_image_width, scale_image_height, 192, 256, true);

    // affine transform
    cv::Mat affine_image;
    cv::warpAffine(inputMatCopy, affine_image, affine_transform, cv::Size(192, 256), cv::INTER_LINEAR);
    resultPair = std::make_pair(affine_image, affine_transform_reverse);

    return resultPair;
}

void PoseEstimator::decodeOutput(const zdl::DlSystem::TensorMap &tensorMap) const
{
    const std::string layerName = "output";
    zdl::DlSystem::ITensor *output = tensorMap.getTensor(layerName.c_str());
    auto it = output->cbegin();
}

void removeEyesAndEars(const std::vector<PosePoint> &poseKeypoints, std::vector<PosePoint> &poseKeypoints_removed)
{
    poseKeypoints_removed = poseKeypoints;
    poseKeypoints_removed.erase(poseKeypoints_removed.begin() + 4); // left ear
    poseKeypoints_removed.erase(poseKeypoints_removed.begin() + 3); // right ear
    poseKeypoints_removed.erase(poseKeypoints_removed.begin() + 2); // left eye
    poseKeypoints_removed.erase(poseKeypoints_removed.begin() + 1); // right eye
}

void PoseEstimator::addPoseKeypoints(const int trackId, const std::vector<PosePoint> &poseKeypoints)
{
    if (sequentialPoseKeypointsByTrackId.find(trackId) == sequentialPoseKeypointsByTrackId.end())
    {
        sequentialPoseKeypointsByTrackId[trackId] = SequentialPoseKeypoints();
    }

    // std::vector<PosePoint> poseKeypoints_removed;
    // removeEyesAndEars(poseKeypoints, poseKeypoints_removed);

    sequentialPoseKeypointsByTrackId[trackId].push_back(poseKeypoints);
    if (sequentialPoseKeypointsByTrackId[trackId].size() > POINTMAXSIZE)
    {
        sequentialPoseKeypointsByTrackId[trackId].pop_front();
    }
}

void PoseEstimator::clearDisappearedTracks(const std::vector<int> &trackIds)
{
    for (auto it = sequentialPoseKeypointsByTrackId.begin(); it != sequentialPoseKeypointsByTrackId.end();)
    {
        if (std::find(trackIds.begin(), trackIds.end(), it->first) == trackIds.end())
        {
            it = sequentialPoseKeypointsByTrackId.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

PoseEstimator::PoseEstimator() {}

PoseEstimator::~PoseEstimator() {}

bool PoseEstimator::CreateNetwork(const uint8_t *buffer, const size_t size, const std::vector<std::string> &runtimes)
{
    this->featureSize = featureSize;
    if (buffer == nullptr)
    {
        std::cout << "Dlc data was not found/valid." << std::endl;
        return false;
    }

    if (runtimes.empty())
    {
        std::cout << "Runtime was not specified" << std::endl;
        return false;
    }

    zdl::DlSystem::RuntimeList runtimeList;
    for (const std::string &runtime_str : runtimes)
    {
        const zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::RuntimeList::stringToRuntime(runtime_str.c_str());
        if (runtime == zdl::DlSystem::Runtime_t::UNSET) return false;
        runtimeList.add(runtime);
    }

    zdl::DlSystem::StringList outputTensorNames; // Three layers from network
    outputTensorNames.append("simcc_x");
    outputTensorNames.append("simcc_y");

    // Create SNPE object
    const zdl::DlSystem::PlatformConfig platformConfig;
    std::unique_ptr<zdl::DlContainer::IDlContainer> container = SnpeUtil::loadContainerFromBuffer(buffer, size);
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    network = snpeBuilder.setOutputTensors(outputTensorNames)
                  .setRuntimeProcessorOrder(runtimeList)
                  .setUseUserSuppliedBuffers(false)
                  .setPlatformConfig(platformConfig)
                  .setInitCacheMode(false)
                  .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::BALANCED)
                  .build();

    if (network == nullptr)
    {
        std::cerr << "Error while building SNPE object" << std::endl;
        isNetworkReady = false;
        return false;
    }

    isNetworkReady = true;
    return true;
}

std::vector<PosePoint> PoseEstimator::Inference(const cv::Mat &input_mat, const BboxXyxy &box)
{
    std::vector<PosePoint> pose_result;

    // 人物がCropされてアフィン変換やスケーリングがされた画像と、逆変換する変換マップのペアを作成
    std::pair<cv::Mat, cv::Mat> crop_resultPair = cropImageByDetectBox(input_mat, box);
    cv::Mat crop_mat = crop_resultPair.first;
    cv::Mat affine_transform_reverse = crop_resultPair.second;

    // deep copy
    cv::Mat crop_mat_copy;
    crop_mat.copyTo(crop_mat_copy);

    // TODO: preprocess() で関数化
    // BGR to RGB
    cv::Mat input_mat_copy_rgb;
    cv::cvtColor(crop_mat_copy, input_mat_copy_rgb, cv::COLOR_BGR2RGB);

    // Standardization
    cv::Mat input_mat_copy_rgb_std;
    makeFloatImg(input_mat_copy_rgb, input_mat_copy_rgb_std);

    // image data, HWC->CHW, image_data - mean / std normalize
    int image_channels = input_mat_copy_rgb_std.channels();
    int image_height = input_mat_copy_rgb_std.rows;
    int image_width = input_mat_copy_rgb_std.cols;

    // inference
    std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = SnpeUtil::loadInputTensor(network, input_mat_copy_rgb_std);
    zdl::DlSystem::TensorMap output_tensors;
    if (!network->execute(inputTensor.get(), output_tensors))
    {
        std::cerr << "Error while executing the network." << std::endl;
        return pose_result;
    }

    // postprocess
    // TODO: postprocess() で関数化
    const std::string layerName_x = "simcc_x";
    const std::string layerName_y = "simcc_y";
    zdl::DlSystem::ITensor *simcc_x = output_tensors.getTensor(layerName_x.c_str());
    zdl::DlSystem::ITensor *simcc_y = output_tensors.getTensor(layerName_y.c_str());

    zdl::DlSystem::TensorShape simcc_x_dims = simcc_x->getShape();
    zdl::DlSystem::TensorShape simcc_y_dims = simcc_y->getShape();

    int batch_size = 0;
    if (simcc_x_dims[0] == simcc_y_dims[0])
    {
        batch_size = simcc_x_dims[0]; // batch_size: 1
    }

    int joint_num = 0;
    if (simcc_x_dims[1] == simcc_y_dims[1])
    {
        joint_num = simcc_x_dims[1]; // joint_num: 17
    }

    int extend_width = simcc_x_dims[2];  // extend_width: 384
    int extend_height = simcc_y_dims[2]; // extend_width: 512

    auto ptrX = simcc_x->cbegin();
    auto ptrY = simcc_y->cbegin();
    const float *simcc_x_result = &(*ptrX);
    const float *simcc_y_result = &(*ptrY);

    for (int i = 0; i < joint_num; i++)
    {
        // find the maximum and maximum indexes in the value of each Extend_width length
        auto x_biggest_iter =
            std::max_element(simcc_x_result + i * extend_width, simcc_x_result + i * extend_width + extend_width);
        int max_x_pos = std::distance(simcc_x_result + i * extend_width, x_biggest_iter);
        float pose_x = static_cast<float>(max_x_pos) / 2.0f;
        float score_x = *x_biggest_iter;

        // find the maximum and maximum indexes in the value of each exten_height length
        auto y_biggest_iter =
            std::max_element(simcc_y_result + i * extend_height, simcc_y_result + i * extend_height + extend_height);
        int max_y_pos = std::distance(simcc_y_result + i * extend_height, y_biggest_iter);
        float pose_y = static_cast<float>(max_y_pos) / 2.0f;
        float score_y = *y_biggest_iter;

        // float score = (score_x + score_y) / 2;
        float score = std::max(score_x, score_y);

        PosePoint temp_point;
        temp_point.x = pose_x;
        temp_point.y = pose_y;
        temp_point.score = score;
        pose_result.emplace_back(temp_point);
    }

    // anti affine transformation to obtain the coordinates on the original picture
    for (int i = 0; i < pose_result.size(); ++i)
    {
        cv::Mat origin_point_Mat = cv::Mat::ones(3, 1, CV_64FC1);
        origin_point_Mat.at<double>(0, 0) = pose_result[i].x;
        origin_point_Mat.at<double>(1, 0) = pose_result[i].y;

        cv::Mat temp_result_mat = affine_transform_reverse * origin_point_Mat;

        pose_result[i].x = temp_result_mat.at<double>(0, 0);
        pose_result[i].y = temp_result_mat.at<double>(1, 0);
    }

    // Normalize scale points in image with size of image to (0-1)
    for (int i = 0; i < pose_result.size(); ++i)
    {
        pose_result[i].x = pose_result[i].x / input_mat.cols;
        pose_result[i].y = pose_result[i].y / input_mat.rows;
    }

    return pose_result;
}

void PoseEstimator::Exec(const cv::Mat &input_image, std::vector<TrackedBbox> &tracks)
{
    std::vector<int> trackIds;
    for (TrackedBbox &track : tracks)
    {
        trackIds.push_back(track.id);
        std::vector<PosePoint> posePoints;
        posePoints = Inference(input_image, track.bodyBbox);
        if (!posePoints.empty())
        {
            std::vector<PosePoint> poseKeypoints_removed;
            removeEyesAndEars(posePoints, poseKeypoints_removed);
            addPoseKeypoints(track.id, poseKeypoints_removed);
            track.AddPoseKeypoints(poseKeypoints_removed);
        }
    }

    clearDisappearedTracks(trackIds);

    for (const auto &pair : sequentialPoseKeypointsByTrackId)
    {
        int trackId = pair.first;
        const SequentialPoseKeypoints &keypoints = pair.second;

        std::cout << "Track ID: " << trackId << ", Number of Pose Sequences: " << keypoints.size() << std::endl;
    }
}
