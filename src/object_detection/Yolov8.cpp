/*
 * (c) 2024 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#include "Yolov8.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SnpeUtil.hpp"
#include "Types.hpp"

struct Yolov8::BoundingBox
{

    float x1{0.0};
    float y1{0.0};
    float x2{0.0};
    float y2{0.0};
    float w{0.0};
    float h{0.0};
    float score{-1.0};
    int label{0};

    BoundingBox()
        : x1(0), y1(0), x2(0), y2(0), w(0), h(0), score(-1), label(0){}; // 0 is unknown in output label list of model

    BoundingBox(float x1_, float y1_, float w_, float h_, float score_, int label_)
    {
        x1 = x1_;
        y1 = y1_;
        w = w_;
        h = h_;
        x2 = x1 + w;
        y2 = y1 + h;
        score = score_;
        label = label_;
    }
};

struct BoxLimit
{
    int hMin = 0;
    int hMax = std::numeric_limits<int>::max();
    int wMin = 0;
    int wMax = std::numeric_limits<int>::max();
};

static void scaleCoords(std::vector<Yolov8::BoundingBox> &vBoundingBoxs, const float ratio, const int pad_w, const int pad_h)
{
    for (auto &it : vBoundingBoxs)
    {
        it.x1 = (it.x1 - (float)pad_w) / ratio;
        it.y1 = (it.y1 - (float)pad_h) / ratio;
        it.x2 = (it.x2 - (float)pad_w) / ratio;
        it.y2 = (it.y2 - (float)pad_h) / ratio;
        it.w = it.x2 - it.x1;
        it.h = it.y2 - it.y1;
    }
}

static void decodeOutput(const zdl::DlSystem::TensorMap &outputTensorMap,
                         std::vector<std::vector<Yolov8::BoundingBox>> &decoded)
{
    std::vector<std::shared_ptr<float[]>> output_result;
    std::vector<std::vector<int>> output_shape;

    const std::vector<std::string> outputNodeName = {"/model.22/Mul_2_output_0", "/model.22/Sigmoid_output_0"};
    for (const auto &nodeName : outputNodeName)
    {
        zdl::DlSystem::ITensor *tensorPtr = outputTensorMap.getTensor(nodeName.c_str());

        // Copy shape into output_shape
        const zdl::DlSystem::TensorShape shape = tensorPtr->getShape();
        std::vector<int> data_shape;
        for (unsigned int i = 0; i < shape.rank(); i++)
        {
            // Get the size of each dimension and convert it to an int, then push it to the data_shape vector
            data_shape.push_back(static_cast<int>(shape.getDimensions()[i]));
        }

        // Push the shape into the shape vector
        output_shape.push_back(data_shape); // Store shape information

        // Copy result into output_result

        // Get the size using getSize()
        const std::size_t tensorSize = tensorPtr->getSize();

        // Create a shared pointer to a new float array with the size of the tensor
        std::shared_ptr<float[]> tensorData(new float[tensorSize], std::default_delete<float[]>());

        // Copy data from the tensor into the shared float array
        std::copy(tensorPtr->cbegin(), tensorPtr->cend(), tensorData.get());

        // Push the shared pointer into the result vector
        output_result.push_back(tensorData);
    }
    // to decode raw model output into BoundingBox object
    // input : vector<float> from model output
    // output : BoundingBox object

    // get logits shapes
    std::vector<int> anchor_shape = output_shape[0];
    std::vector<int> conf_shape = output_shape[1];

    // save shape into yolov8 format (rows & column)
    const int num_class = 80;
    int num_ele_per_row = 4 + num_class; // 4 for bounding box coordinates x,y,w,h
    int num_rows = anchor_shape[2];

    // get anchor data
    std::shared_ptr<float[]> raw_anchor_result = output_result[0];
    std::vector<float> anchor_result(raw_anchor_result.get(), raw_anchor_result.get() + num_ele_per_row * num_rows);

    std::shared_ptr<float[]> raw_conf_result = output_result[1];
    std::vector<float> conf_result(raw_conf_result.get(), raw_conf_result.get() + num_ele_per_row * num_rows);

    // 検出対象のクラスIDを定義
    const std::vector<int> targetClassIds = {0, 67}; // 0: person, 39: bottle, 67: cell phone
    const std::map<int, float> scoreThresholdMap = {{targetClassIds[0], 0.25}, {targetClassIds[1], 0.015}};
    const std::map<int, int> objectIndexMap = {{targetClassIds[0], 0}, {targetClassIds[1], 1}};

    for (int row = 0; row < num_rows; ++row)
    {
        std::vector<float> row_data;
        // Add first 4 elements from anchor_result
        for (int ele = 0; ele < 4; ++ele)
        {
            row_data.push_back(anchor_result[ele * num_rows + row]);
        }
        // Add elements from conf_result
        for (int ele = 0; ele < num_class; ++ele)
        {
            row_data.push_back(conf_result[ele * num_rows + row]);
        }

        const int max_ele = (int)std::distance(row_data.begin() + 4, std::max_element(row_data.begin() + 4, row_data.end()));
        float max_score = row_data[max_ele + 4];

        // 対象クラスIDかどうかをチェック
        bool is_target_class = std::find(targetClassIds.begin(), targetClassIds.end(), max_ele) != targetClassIds.end();
        if (!is_target_class) continue;

        // スコアの閾値を設定
        if (max_score >= scoreThresholdMap.at(max_ele)) // 条件追加
        {
            float x = row_data[0];
            float y = row_data[1];
            float w = row_data[2];
            float h = row_data[3];

            float left = (x - 0.5f * w);
            float top = (y - 0.5f * h);
            float width = w;
            float height = h;

            Yolov8::BoundingBox box(left, top, width, height, max_score, max_ele);
            decoded[objectIndexMap.at(max_ele)].push_back(box);
        }
    }
}

static double calculateIou(const cv::Rect2f &b1, const cv::Rect2f &b2)
{
    // calculate the IOU between two boxes
    // input: bounding boxes
    // output: IOU value

    float intersection_area = (b1 & b2).area();
    float union_area = b1.area() + b2.area() - intersection_area;

    if (union_area < DBL_EPSILON) return 0;

    return (double)(intersection_area / union_area);
}

static void nms(std::vector<Yolov8::BoundingBox> &input_boxes, const double &nms_threshold)
{
    // non maximum suppression used to keep best out of many overlapping detections
    // input: bounding box vector, nms iou threshold
    // output: filtered bounding boxes list

    std::sort(input_boxes.begin(), input_boxes.end(),
              [](const Yolov8::BoundingBox &a, const Yolov8::BoundingBox &b) { return a.score > b.score; });
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        cv::Rect2f rect_box_i(input_boxes[i].x1, input_boxes[i].y1, input_boxes[i].x2 - input_boxes[i].x1,
                              input_boxes[i].y2 - input_boxes[i].y1);
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            cv::Rect2f rect_box_j(input_boxes[j].x1, input_boxes[j].y1, input_boxes[j].x2 - input_boxes[j].x1,
                                  input_boxes[j].y2 - input_boxes[j].y1);
            float ovr = (float)calculateIou(rect_box_i, rect_box_j);
            if (ovr >= (float)nms_threshold)
            {
                input_boxes.erase(input_boxes.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

static void filterBySize(std::vector<Yolov8::BoundingBox> &input_boxes, const int min_height, const int max_height,
                         const int min_width, const int max_width)
{
    for (int i = 0; i < (int)input_boxes.size(); i++)
    {
        Yolov8::BoundingBox bbox = input_boxes[i];
        if (!(bbox.w >= (float)min_width && bbox.w <= (float)max_width && bbox.h >= (float)min_height &&
              bbox.h <= (float)max_height))
        {
            input_boxes.erase(input_boxes.begin() + i);
        }
    }
}

void Yolov8::preprocess(const cv::Mat &img, cv::Mat &processed)
{
    // resize by keeping aspect ratio
    const int model_input_width = 640;
    const int model_input_height = 640;
    ratio = std::min(1.0f * (float)model_input_width / (float)image_width,
                     1.0f * (float)model_input_height / (float)image_height);

    int resizedHeight = int((float)image_height * ratio);
    int resizedWidth = int((float)image_width * ratio);

    // odd number->pad size error
    if (resizedHeight % 2 != 0) resizedHeight -= 1;
    if (resizedWidth % 2 != 0) resizedWidth -= 1;

    paddingWidthIdx = (model_input_width - resizedWidth) / 2;
    paddingHeightIdx = (model_input_height - resizedHeight) / 2;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resizedWidth, resizedHeight), 0, 0, cv::INTER_LINEAR);

    // pad the gap between resized image and model input size
    const cv::Scalar paddingColor = cv::Scalar(128, 128, 128);
    cv::copyMakeBorder(resized, resized, paddingHeightIdx, paddingHeightIdx, paddingWidthIdx, paddingWidthIdx,
                       cv::BORDER_CONSTANT, paddingColor);

    // color channel swap
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    cv::Mat normalized(model_input_height, model_input_width, CV_32FC3);
    resized.convertTo(normalized, CV_32F);
    normalized = normalized / 255.0f;
    processed = normalized;
}

void Yolov8::postprocess(const std::vector<std::vector<BoundingBox>> &input, std::vector<std::vector<BboxXyxy>> &result)
{
    std::vector<std::vector<BoundingBox>> processed = input;
    // unifyChildAdult(input, processed);

    // NMSの適用と座標のスケーリング
    const Vecd ious = {0.35, 0.10};
    for (int classIdx = 0; classIdx < (int)processed.size(); classIdx++)
    {
        nms(processed[classIdx], ious[classIdx]);

        // ネットワーク入力層のピクセル座標系（例: 640 x 384）から、元の画像のピクセル座標系（例: 1280 x 720）に変換
        scaleCoords(processed[classIdx], ratio, paddingWidthIdx, paddingHeightIdx);
    }

    // Bboxの大きさでフィルタリング
    for (int classIdx = 0; classIdx < (int)processed.size(); classIdx++)
    {
        const BoxLimit limit;
        filterBySize(processed[0], limit.hMin, limit.hMax, limit.wMin, limit.wMax);
    }

    // BoundingBox を BboxXyxy に変換
    result.resize(processed.size());
    for (size_t classIdx = 0; classIdx < processed.size(); classIdx++)
    {
        for (size_t boxIdx = 0; boxIdx < processed[classIdx].size(); boxIdx++)
        {
            const BoundingBox &box = processed[classIdx][boxIdx];
            result[classIdx].push_back(BboxXyxy(box.x1 / image_width, box.y1 / image_height, box.x2 / image_width,
                                                box.y2 / image_height, box.score));
        }
    }
}

bool Yolov8::CreateNetwork(const uint8_t *buffer, const size_t size, const std::vector<std::string> &runtimes)
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

    zdl::DlSystem::StringList outputTensorNames; // Two layers from network
    outputTensorNames.append("/model.22/Mul_2");
    outputTensorNames.append("/model.22/Sigmoid");

    std::unique_ptr<zdl::DlContainer::IDlContainer> container = SnpeUtil::loadContainerFromBuffer(buffer, size);
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    network = snpeBuilder.setOutputLayers(outputTensorNames)
                  .setRuntimeProcessorOrder(runtimeList)
                  .setUseUserSuppliedBuffers(false)
                  .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
                  .setProfilingLevel(zdl::DlSystem::ProfilingLevel_t::OFF)
                  .build();

    if (network == nullptr)
    {
        std::cerr << "Error while building SNPE object" << std::endl;
        return false;
    }
    return true;
}

bool Yolov8::Infer(const cv::Mat &image, std::vector<std::vector<BboxXyxy>> &result)
{
    image_width = image.cols;
    image_height = image.rows;

    cv::Mat processed;
    preprocess(image, processed);
    std::cout << "Preprocess done" << std::endl;

    std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = SnpeUtil::loadInputTensor(network, processed);
    zdl::DlSystem::TensorMap outputTensorMap;
    network->execute(inputTensor.get(), outputTensorMap);
    std::cout << "Inference done" << std::endl;

    std::vector<std::vector<BoundingBox>> decoded(80);
    decodeOutput(outputTensorMap, decoded);
    postprocess(decoded, result);
    std::cout << "Postprocess done" << std::endl;
    return true;
}
