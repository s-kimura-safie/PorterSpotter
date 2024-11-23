
/*
 * (c) 2024 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#include "SnpeUtil.hpp"

#include "DlSystem/ITensorFactory.hpp"
#include "SNPE/SNPEFactory.hpp"

// This method is based on the SNPE sample: $SNPE_ROOT/examples/SNPE/NativeCpp/SampleCode/jni/LoadContainer.cpp
std::unique_ptr<zdl::DlContainer::IDlContainer> SnpeUtil::loadContainerFromFile(std::string containerPath)
{
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(containerPath.c_str()));
    return container;
}

// This method is based on the SNPE sample: $SNPE_ROOT/examples/SNPE/NativeCpp/SampleCode/jni/LoadContainer.cpp
std::unique_ptr<zdl::DlContainer::IDlContainer> SnpeUtil::loadContainerFromBuffer(const uint8_t *buffer, const size_t size)
{
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    container = zdl::DlContainer::IDlContainer::open(buffer, size);
    return container;
}

// This method is based on the SNPE sample: $SNPE_ROOT/examples/SNPE/NativeCpp/SampleCode/jni/LoadInputTensor.cpp
std::unique_ptr<zdl::DlSystem::ITensor> SnpeUtil::loadInputTensor(std::unique_ptr<zdl::SNPE::SNPE> &snpe, cv::Mat inputImage)
{
    std::unique_ptr<zdl::DlSystem::ITensor> input;
    const auto &strList_opt = snpe->getInputTensorNames();
    if (!strList_opt) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &strList = *strList_opt;
    // Make sure the network requires only a single input
    assert(strList.size() == 1);

    // If the network has a single input, each line represents the input file to be loaded for that input
    std::vector<float> inputVec;

    /* Create an input tensor that is correctly sized to hold the input of the network. Dimensions that have no fixed size
     * will be represented with a value of 0. */
    const auto &inputDims_opt = snpe->getInputDimensions(strList.at(0));
    const auto &inputShape = *inputDims_opt;

    /* Calculate the total number of elements that can be stored in the tensor so that we can check that the input contains
       the expected number of elements. With the input dimensions computed create a tensor to convey the input into the
       network. */
    input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);
    // Padding the input vector so as to make the size of the vector to equal to an integer multiple of the batch size
    zdl::DlSystem::TensorShape tensorShape = snpe->getInputDimensions();
    int net_height = (int)tensorShape.getDimensions()[1]; // FIXME: Narrowing conversion
    int net_width = (int)tensorShape.getDimensions()[2];  // FIXME: Narrowing conversion
    int net_ch = (int)tensorShape.getDimensions()[3];     // FIXME: Narrowing conversion

    int mat_height = inputImage.rows;
    int mat_width = inputImage.cols;
    int mat_ch = inputImage.channels();

    if ((net_height != mat_height) || (net_width != mat_width) || (net_ch != mat_ch))
    {
        std::cerr << "Size of image does not match network input.\n"
                  << "Expecting: " << net_height << "," << net_width << "," << net_ch << "\n"
                  << "Got: " << mat_height << "," << mat_width << "," << mat_ch << "\n";
        return nullptr;
    }

    for (int y = 0; y < mat_height; y++)
    {
        for (int x = 0; x < mat_width; x++)
        {
            for (int z = 0; z < mat_ch; z++)
            {
                float value = inputImage.at<cv::Vec3f>(y, x)[z];
                inputVec.push_back(value);
            }
        }
    }

    /* Copy the loaded input file contents into the networks input tensor. SNPE's ITensor supports C++ STL functions like
     * std::copy() */
    std::copy(inputVec.begin(), inputVec.end(), input->begin());
    return input;
}

std::unique_ptr<zdl::DlSystem::ITensor> SnpeUtil::loadInputTensor(std::unique_ptr<zdl::SNPE::SNPE> &snpe, SequentialPoseKeypoints &poseKeypoints)
{
    std::unique_ptr<zdl::DlSystem::ITensor> input;
    const auto &strList_opt = snpe->getInputTensorNames();
    if (!strList_opt) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &strList = *strList_opt;
    // Make sure the network requires only a single input
    assert(strList.size() == 1);

    // If the network has a single input, each line represents the input file to be loaded for that input
    std::vector<float> inputVec;

    /* Create an input tensor that is correctly sized to hold the input of the network. Dimensions that have no fixed size
     * will be represented with a value of 0. */
    const auto &inputDims_opt = snpe->getInputDimensions(strList.at(0));
    const auto &inputShape = *inputDims_opt;

    /* Calculate the total number of elements that can be stored in the tensor so that we can check that the input contains
       the expected number of elements. With the input dimensions computed create a tensor to convey the input into the
       network. */
    input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);
    // Padding the input vector so as to make the size of the vector to equal to an integer multiple of the batch size
    zdl::DlSystem::TensorShape tensorShape = snpe->getInputDimensions();
    int net_height = (int)tensorShape.getDimensions()[1]; // FIXME: Narrowing conversion
    int net_width = (int)tensorShape.getDimensions()[2];  // FIXME: Narrowing conversion
    int net_ch = (int)tensorShape.getDimensions()[3];     // FIXME: Narrowing conversion

    for (int i = 0; i < poseKeypoints.size(); i++)
    {    
        for (int j = 0; j < poseKeypoints[i].size(); j++)
        {
            float value = poseKeypoints[i][j].x;
            inputVec.push_back(value);
            value = poseKeypoints[i][j].y;
            inputVec.push_back(value);
            value = poseKeypoints[i][j].score;
            inputVec.push_back(value);
        }
    }

    /* Copy the loaded input file contents into the networks input tensor. SNPE's ITensor supports C++ STL functions like
     * std::copy() */
    std::copy(inputVec.begin(), inputVec.end(), input->begin());
    return input;
}
