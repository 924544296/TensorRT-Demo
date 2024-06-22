#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <random>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>
//#include <torch/torch.h>




class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override;
};


void export_parser();


std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);


nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, std::string lname, float eps);


nvinfer1::IActivationLayer* block(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int out_channels, int kernel_size, int stride, int padding, int id_layer);


void export_api();


// 加载模型文件
std::vector<unsigned char> load_engine_file(const std::string& file_name);


void inference_parser();


void inference_api();