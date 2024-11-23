/*
 * (c) 2023 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#include "Types.hpp"

/// @brief Yolov5 のユーティリティ関数
namespace Yolov5Util
{

    struct Object
    {
        cv::Rect2f rect;
        float prob; // prob = objectness * score
    };

    inline float sigmoid(const float x)
    {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }

    inline float intersection_area(const Object &a, const Object &b)
    {
        cv::Rect2f inter = a.rect & b.rect;
        return inter.area();
    }

    /// @brief Resize with padding. Keep the input aspect ratio of unpadded image.
    void resize(const cv::Mat &input, const cv::Size &target_shape, cv::Mat &output, float &scale, cv::Vec2i &delta);

    /// @brief Sort object by probability.
    void qsort_descent_inplace(std::vector<Object> &objects);

    /// @brief Sort object by probability.
    void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right);

    /// @brief Non-maximum supression with sorted object by probability.
    void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, float nms_threshold);

    /// @brief Transform from Object to DetectedObject type.
    void generateDetectedObject(const std::vector<Object> &src, std::vector<BboxXyxy> &tgt, const int w_img,
                                const int h_img);
}
