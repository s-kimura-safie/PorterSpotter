/*
 * (c) 2024 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#include "VisualizationUtil.hpp"

void visualization_util::DrawLabel(cv::Mat &image, cv::Point2i point, const std::string &text, const cv::Scalar &color)
{
    // Calculate text color according to the label color
    const float c_mean = cv::mean(color)[0];
    cv::Scalar txt_color;
    if (c_mean > 0.5)
    {
        txt_color = cv::Scalar(0, 0, 0);
    }
    else
    {
        txt_color = cv::Scalar(255, 255, 255);
    }

    // Write multi-line string
    std::stringstream ss(text);
    std::string line;
    while (std::getline(ss, line, '\n'))
    {
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
        cv::Scalar txt_bk_color = color * 0.7 * 255;

        cv::rectangle(image, cv::Rect(point, cv::Size(label_size.width, label_size.height + baseLine)), txt_bk_color, -1);

        cv::putText(image, line, cv::Point(point.x, point.y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color,
                    1, cv::LineTypes::LINE_AA);

        point = cv::Point(point.x, point.y + label_size.height + baseLine + 2);
    }
}
