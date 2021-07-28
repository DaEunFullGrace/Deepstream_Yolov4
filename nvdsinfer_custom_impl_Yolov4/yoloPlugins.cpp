/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "yoloPlugins.h"
#include "NvInferPlugin.h"
#include <cassert>
#include <iostream>
#include <memory>

namespace {
template <typename T>
void write(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void read(const char*& buffer, T& val)
{
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
}
} //namespace

// Forward declaration of cuda kernels
cudaError_t cudaYoloLayer (
    const void* input, void* output, const uint& batchSize,
    const uint& gridH, const uint& gridW, const uint& numOutputClasses,
    const uint& numBBoxes, uint64_t outputSize, cudaStream_t stream, 
    const uint coords, const float scale, const uint type);

YoloLayer::YoloLayer (const void* data, size_t length)
{
    const char *d = static_cast<const char*>(data);
    read(d, m_NumBoxes);
    read(d, m_NumClasses);
    read(d, m_GridH);
    read(d, m_GridW);
    read(d, m_OutputSize);
    read(d, m_Coords);
    read(d, m_Scale);
    read(d, m_Type);
};

YoloLayer::YoloLayer (
    const uint& numBoxes, const uint& numClasses, 
    const uint& gridH, const uint& gridW, 
    const uint coordinates, const float scale, const uint type) :
    m_NumBoxes(numBoxes),
    m_NumClasses(numClasses),
    m_GridH(gridH),
    m_GridW(gridW),
    m_Coords(coordinates),
    m_Scale(scale),
    m_Type(type)
{
    assert(m_NumBoxes > 0);
    assert(m_NumClasses > 0);
    assert(m_GridH > 0);
    assert(m_GridW > 0);

    m_OutputSize = m_GridH * m_GridW * (m_NumBoxes * (4 + 1 + m_NumClasses));
};

nvinfer1::Dims
YoloLayer::getOutputDimensions(
    int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
    assert(index == 0);
    assert(nbInputDims == 1);
    return inputs[0];
}

bool YoloLayer::supportsFormat (
    nvinfer1::DataType type, nvinfer1::PluginFormat format) const {
    return (type == nvinfer1::DataType::kFLOAT &&
            format == nvinfer1::PluginFormat::kNCHW);
}

void
YoloLayer::configureWithFormat (
    const nvinfer1::Dims* inputDims, int nbInputs,
    const nvinfer1::Dims* outputDims, int nbOutputs,
    nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize)
{
    assert(nbInputs == 1);
    assert(format == nvinfer1::PluginFormat::kNCHW);
    assert(inputDims != nullptr);
}

int YoloLayer::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace,
    cudaStream_t stream)
{
    CHECK(cudaYoloLayer(
              inputs[0], outputs[0], batchSize, m_GridH, m_GridW, m_NumClasses, 
              m_NumBoxes, m_OutputSize, stream, m_Coords, m_Scale, m_Type));
    return 0;
}

size_t YoloLayer::getSerializationSize() const
{
    return sizeof(m_NumBoxes) + sizeof(m_NumClasses) + sizeof(m_GridH) 
            + sizeof(m_GridW) + sizeof(m_OutputSize) + sizeof(m_Coords) 
            + sizeof(m_Scale) + sizeof(m_Type);
}

void YoloLayer::serialize(void* buffer) const
{
    char *d = static_cast<char*>(buffer);
    write(d, m_NumBoxes);
    write(d, m_NumClasses);
    write(d, m_GridH);
    write(d, m_GridW);
    write(d, m_OutputSize);
    write(d, m_Coords);
    write(d, m_Scale);
    write(d, m_Type);
}

nvinfer1::IPluginV2* YoloLayer::clone() const
{
    return new YoloLayer (m_NumBoxes, m_NumClasses, m_GridH, 
        m_GridW, m_Coords, m_Scale, m_Type);
}