
#include <vector>
#include <opencv2/opencv.hpp>


namespace KeypointColorPalette{
    std::vector<cv::Scalar> keypointColors = {
    cv::Scalar(255, 0, 0),   // Blue
    cv::Scalar(0, 255, 0),   // Green
    cv::Scalar(0, 0, 255),   // Red
    cv::Scalar(255, 255, 0), // Cyan
    cv::Scalar(255, 0, 255), // Magenta
    cv::Scalar(0, 255, 255), // Yellow
    cv::Scalar(128, 0, 0),   // Maroon
    cv::Scalar(0, 128, 0),   // Olive
    cv::Scalar(128, 128, 0), // Lime
    cv::Scalar(0, 0, 128),   // Navy
    cv::Scalar(128, 0, 128), // Purple
    cv::Scalar(0, 128, 128), // Teal
    cv::Scalar(128, 128, 128), // Gray
    cv::Scalar(64, 0, 0),    // Brown
    cv::Scalar(0, 64, 0),    // Dark Green
    cv::Scalar(0, 0, 64),    // Dark Blue
    cv::Scalar(64, 64, 64)   // Dark Gray
};
}
