#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>




class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override;
};


void export_engine();


int argmax(const float(&rst)[10]);


// 加载模型文件
std::vector<unsigned char> load_engine_file(const std::string& file_name);


void inference();