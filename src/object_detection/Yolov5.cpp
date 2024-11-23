/*
 * (c) 2022 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#include "Yolov5.hpp"
#include "MathUtil.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SnpeUtil.hpp"

// transfered from DetectorDebugUtil.hpp
using Vecd = std::vector<double>;

void Yolov5::makeFloatImg(const cv::Mat &input, cv::Mat &output)
{
    constexpr int channels = 3;
    const int h_img = input.rows;
    const int w_img = input.cols;

    output = cv::Mat(h_img, w_img, CV_32FC3);

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < h_img; h++)
        {
            for (int w = 0; w < w_img; w++)
            {
                output.at<cv::Vec3f>(h, w)[c] = (float)input.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
    output = output / 255.0f; // Normalize
}

void Yolov5::decode_output(const zdl::DlSystem::TensorMap &tensorMap, std::vector<Yolov5Util::Object> &objects,
                           const float scale, const cv::Vec2i &delta, const int w_img, const int h_img) const
{
    // 3 layers from the network:
    // 376_Reshape_257.ncs, 395_Reshape_272.ncs, and 414_Reshape_287.ncs,
    // which has 1x18x40x40, 1x18x20x20, and 1x18x10x10 resolution, respectively.
    const std::string layer_names[] = {"onnx::Reshape_380.ncs", "onnx::Reshape_418.ncs", "onnx::Reshape_456.ncs"};
    constexpr int num_layers = 3;
    constexpr int num_anchors = 3;
    constexpr int grids[num_layers] = {40, 20, 10};
    constexpr int strides[num_layers] = {8, 16, 32};
    constexpr int x_anchors[num_layers][num_anchors] = {{3, 6, 10}, {17, 22, 39}, {51, 97, 173}};
    constexpr int y_anchors[num_layers][num_anchors] = {{8, 14, 25}, {33, 57, 65}, {120, 168, 196}};
    std::vector<Yolov5Util::Object> proposals;
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++)
    {
        zdl::DlSystem::ITensor *feat_ptr = tensorMap.getTensor(layer_names[layer_idx].c_str());
        auto it = feat_ptr->cbegin();
        const int nx = grids[layer_idx];
        const int ny = grids[layer_idx];
        for (int iy = 0; iy < ny; iy++)
        {
            for (int ix = 0; ix < nx; ix++)
            {
                for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
                {
                    // 0: cx, 1: cy, 2: w, 3: h, 4: conf, 5: pred_class
                    Yolov5Util::Object obj;
                    constexpr int num_categories = 1;
                    const int num_grid = nx * ny;
                    const int anchor_start = anchor_idx * (num_categories + 5) * num_grid;
                    const int base_idx = anchor_start + ix + nx * iy;
                    const int w_stride = strides[layer_idx];
                    const int h_stride = strides[layer_idx];
                    const float x_center =
                        (2.0f * Yolov5Util::sigmoid(it[base_idx + 0 * num_grid]) - 0.5f + (float)ix) * (float)w_stride;
                    const float y_center =
                        (2.0f * Yolov5Util::sigmoid(it[base_idx + 1 * num_grid]) - 0.5f + (float)iy) * (float)h_stride;
                    const float w = std::pow(Yolov5Util::sigmoid(it[base_idx + 2 * num_grid]) * 2.0f, 2.0f) *
                                    (float)x_anchors[layer_idx][anchor_idx];
                    const float h = std::pow(Yolov5Util::sigmoid(it[base_idx + 3 * num_grid]) * 2.0f, 2.0f) *
                                    (float)y_anchors[layer_idx][anchor_idx];
                    obj.rect = cv::Rect2f(x_center - w / 2.0f, y_center - h / 2.0f, w, h);

                    const float objectness = Yolov5Util::sigmoid(it[base_idx + 4 * num_grid]);
                    const float score = Yolov5Util::sigmoid(it[base_idx + 5 * num_grid]); // Score for person
                    obj.prob = objectness * score;

                    if (obj.prob > scoreThreshold) proposals.push_back(obj);
                }
            }
        }
    }

    // NMS
    Yolov5Util::qsort_descent_inplace(proposals);
    std::vector<int> picked;
    Yolov5Util::nms_sorted_bboxes(proposals, picked, iouThreshold);

    // Scaling
    const int num_picked = (int)picked.size();
    objects.resize(num_picked);
    for (int i = 0; i < num_picked; i++)
    {
        objects[i] = proposals[picked[i]];

        // Adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (float)delta[0]) / scale;
        float y0 = (objects[i].rect.y - (float)delta[1]) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (float)delta[0]) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (float)delta[1]) / scale;

        // Clamp
        x0 = MathUtil::Clamp<float>(x0, 0.f, (float)(w_img - 1));
        y0 = MathUtil::Clamp<float>(y0, 0.f, (float)(h_img - 1));
        x1 = MathUtil::Clamp<float>(x1, 0.f, (float)(w_img - 1));
        y1 = MathUtil::Clamp<float>(y1, 0.f, (float)(h_img - 1));

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
}

void Yolov5::saveFeatureMap(const zdl::DlSystem::TensorMap &tensorMap, const int w_img, const int h_img)
{
    constexpr int num_anchors = 3;
    const std::string layer_name = "395_Reshape_272.ncs";
    zdl::DlSystem::ITensor *feat_ptr = tensorMap.getTensor(layer_name.c_str());
    auto it = feat_ptr->cbegin();
    std::vector<std::vector<Vecd>> featureMap;
    const int ny = 20;
    for (int iy = 0; iy < ny; iy++)
    {
        std::vector<Vecd> row;
        const int nx = 20;
        for (int ix = 0; ix < nx; ix++)
        {
            Vecd channels; // 6 x 3 channels
            for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
            {
                constexpr int num_categories = 1;
                const int num_grid = nx * ny;
                const int anchor_start = anchor_idx * (num_categories + 5) * num_grid;
                const int base_idx = anchor_start + ix + nx * iy;
                channels.push_back(it[base_idx + 0 * num_grid]);                      // x-center
                channels.push_back(it[base_idx + 1 * num_grid]);                      // y-center
                channels.push_back(it[base_idx + 2 * num_grid]);                      // w
                channels.push_back(it[base_idx + 3 * num_grid]);                      // h
                channels.push_back(Yolov5Util::sigmoid(it[base_idx + 4 * num_grid])); // Objectness
                channels.push_back(Yolov5Util::sigmoid(it[base_idx + 5 * num_grid])); // Score
            }
            row.push_back(channels);
        }
        featureMap.push_back(row);
    }
}
Yolov5::Yolov5() : isNetworkReady(false)
{
    scoreThreshold = 0.35;
    iouThreshold = 0.35;
    nmsTopK = 100;
    isNetworkFedBgr = false;
}

Yolov5::~Yolov5() {}
bool Yolov5::CreateNetwork(const uint8_t *buffer, const size_t size, const std::vector<std::string> &runtimes)
{
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
    outputTensorNames.append("onnx::Reshape_380.ncs");
    outputTensorNames.append("onnx::Reshape_418.ncs");
    outputTensorNames.append("onnx::Reshape_456.ncs");

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

bool Yolov5::Detect(const cv::Mat &rgbImage, std::vector<BboxXyxy> &detectedObjects)
{
    // Preprocess
    cv::Mat resized; // Resized image
    float scale;     // Scaling factor for resize
    cv::Vec2i delta; // Padding for x and y directions
    Yolov5Util::resize(rgbImage, cv::Size(320, 320), resized, scale, delta);

    cv::Mat processed;
    makeFloatImg(resized, processed);

    // Inference
    std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = SnpeUtil::loadInputTensor(network, processed);
    zdl::DlSystem::TensorMap outputTensorMap;
    if (!network->execute(inputTensor.get(), outputTensorMap))
    {
        std::cerr << "Error while executing the network." << std::endl;
        return false;
    }

    // Post-process
    std::vector<Yolov5Util::Object> objects;
    decode_output(outputTensorMap, objects, scale, delta, rgbImage.cols, rgbImage.rows);
    Yolov5Util::generateDetectedObject(objects, detectedObjects, rgbImage.cols, rgbImage.rows);

#if 0 // For debug use
    saveFeatureMap(outputTensorMap, rgbImage.cols, rgbImage.rows);

    cv::Mat bgrImage;
    cv::cvtColor(rgbImage, bgrImage, cv::COLOR_RGB2BGR);
    PredictorDebugUtil::SaveBboxes(bgrImage, detectedObjects);
#endif

    return true;
}
