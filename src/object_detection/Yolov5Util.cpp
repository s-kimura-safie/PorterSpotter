/*
 * (c) 2023 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#include "Yolov5Util.hpp"

struct DetectBox
{
    int left;
    int top;
    int right;
    int bottom;
    float score;
    int label;

    DetectBox()
    {
        left = -1;
        top = -1;
        right = -1;
        bottom = -1;
        score = -1.0;
        label = -1;
    }

    bool IsValid() const { return left != -1 && top != -1 && right != -1 && bottom != -1 && score != -1.0 && label != -1; }
};

static bool BoxCompare(const DetectBox &a, const DetectBox &b) { return a.score > b.score; }

void Yolov5Util::resize(const cv::Mat &input, const cv::Size &target_shape, cv::Mat &output, float &scale, cv::Vec2i &delta)
{
    const int w_input = input.cols;
    const int h_input = input.rows;

    const int w_target = target_shape.width;
    const int h_target = target_shape.height;

    // Calculate size of unpadded image
    scale = std::min((float)w_target / (float)w_input, (float)h_target / (float)h_input);
    cv::Size size_unpad;
    size_unpad.width = (int)std::round((float)w_input * scale);
    size_unpad.height = (int)std::round((float)h_input * scale);

    cv::Mat resized;
    cv::resize(input, resized, size_unpad, 0, 0, cv::INTER_CUBIC);

    delta[0] = (w_target - size_unpad.width) / 2;
    delta[1] = (h_target - size_unpad.height) / 2;

    output = cv::Mat(h_target, w_target, CV_8UC3, cv::Scalar(128, 128, 128));
    resized.copyTo(output(cv::Rect(delta[0], delta[1], size_unpad.width, size_unpad.height)));
}

void Yolov5Util::qsort_descent_inplace(std::vector<Object> &objects)
{
    if (objects.empty()) return;

    qsort_descent_inplace(objects, 0, (int)objects.size() - 1);
}

void Yolov5Util::qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    {
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void Yolov5Util::nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();

    const int n = (int)faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold) keep = 0;
        }

        if (keep) picked.push_back(i);
    }
}

void Yolov5Util::generateDetectedObject(const std::vector<Object> &src, std::vector<BboxXyxy> &tgt, const int w_img,
                                        const int h_img)
{
    tgt.resize(src.size());
    for (size_t i = 0; i < src.size(); i++)
    {
        tgt[i].x0 = src[i].rect.x / (float)w_img;
        tgt[i].y0 = src[i].rect.y / (float)h_img;
        tgt[i].x1 = (src[i].rect.x + src[i].rect.width) / (float)w_img;
        tgt[i].y1 = (src[i].rect.y + src[i].rect.height) / (float)h_img;
        tgt[i].confidence = src[i].prob;
    }
}
