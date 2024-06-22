#include "Generation.h"


void Logger::log(Severity severity, const char* msg) noexcept
{
    // 抑制信息级别的消息
    if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
}


void export_parser()
{
    // 实例化ILogger
    Logger logger;

    // 创建builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));

    // 创建网络(显性batch)
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));

    // 创建ONNX解析器：parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    // 读取文件
    const char* file_path = "D:\\model\\acgan\\net_g.onnx";
    parser->parseFromFile(file_path, static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    // 创建构建配置，用来指定trt如何优化模型
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    // 设定配置
    // 工作空间大小
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);
    // 设置精度
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // 创建引擎
    auto engine = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));

    //序列化保存engine
    std::ofstream engine_file("D:\\model\\acgan\\net_g_parser.engine", std::ios::binary);
    engine_file.write((char*)engine->data(), engine->size());
    engine_file.close();

    std::cout << "Engine build success!" << std::endl;
}


std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        nvinfer1::Weights wt{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}


nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, std::string lname, float eps) {
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    //std::cout << "len " << len << std::endl;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights scale{ nvinfer1::DataType::kFLOAT, scval, len };

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights shift{ nvinfer1::DataType::kFLOAT, shval, len };

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    nvinfer1::Weights power{ nvinfer1::DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    nvinfer1::IScaleLayer* scale_1 = network->addScale(input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}


nvinfer1::IActivationLayer* block(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int out_channels, int kernel_size, int stride, int padding, int id_layer) {
    //
    nvinfer1::Weights emptywts{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
    nvinfer1::IDeconvolutionLayer* deconv = network->addDeconvolutionNd(input, out_channels, nvinfer1::DimsHW{ kernel_size, kernel_size }, weightMap["layers." + std::to_string(id_layer) + ".weight"], emptywts);
    assert(deconv);
    deconv->setStrideNd(nvinfer1::DimsHW{ stride, stride });
    deconv->setPaddingNd(nvinfer1::DimsHW{ padding, padding });
    //
    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *deconv->getOutput(0), "layers." + std::to_string(id_layer+1), 1e-5);
    //
    nvinfer1::IActivationLayer* relu = network->addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu);

    return relu;
}


void export_api()
{
    // 实例化ILogger
    Logger logger;

    // 创建builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

    // 创建网络(显性batch)
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);

    // 向network中添加网络层
    //
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights("D:/model/acgan/net_g.wts");
    nvinfer1::Weights emptywts{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
    //
    nvinfer1::ITensor* noise = network->addInput("noise", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{-1, 100, 1, 1});
    assert(noise);
    nvinfer1::ITensor* label = network->addInput("label", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{ -1, 10, 1, 1 });
    assert(label); 
    //
    std::vector<nvinfer1::ITensor*> inputs = { noise, label };
    nvinfer1::IConcatenationLayer* concat = network->addConcatenation(inputs.data(), inputs.size());
    concat->setAxis(1);
    assert(concat);
    //
    nvinfer1::IActivationLayer* x = block(network, weightMap, *concat->getOutput(0), 512, 3, 1, 0, 0);
    x = block(network, weightMap, *x->getOutput(0), 256, 3, 1, 0, 3);
    x = block(network, weightMap, *x->getOutput(0), 128, 3, 1, 0, 6);
    x = block(network, weightMap, *x->getOutput(0), 64, 4, 2, 1, 9);
    //
    nvinfer1::IDeconvolutionLayer* deconv = network->addDeconvolutionNd(*x->getOutput(0), 1, nvinfer1::DimsHW{ 4, 4 }, weightMap["layers.12.weight"], emptywts);
    assert(deconv);
    deconv->setStrideNd(nvinfer1::DimsHW{ 2, 2 });
    deconv->setPaddingNd(nvinfer1::DimsHW{ 1, 1 });
    //
    nvinfer1::IActivationLayer* tanh = network->addActivation(*deconv->getOutput(0), nvinfer1::ActivationType::kTANH);
    assert(tanh); 
    tanh->getOutput(0)->setName("image");
    network->markOutput(*tanh->getOutput(0));


    // 创建构建配置，用来指定trt如何优化模型
    std::cout << "Creating builder config..." << std::endl;
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    // 设定配置
    // 工作空间大小
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);
    // 设置精度
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    //
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("noise", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ 1, 100, 1, 1 });
    profile->setDimensions("noise", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ 10, 100, 1, 1 });
    profile->setDimensions("noise", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ 100, 100, 1, 1 });
    profile->setDimensions("label", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ 1, 10, 1, 1 });
    profile->setDimensions("label", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ 10, 10, 1, 1 });
    profile->setDimensions("label", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ 100, 10, 1, 1 });
    config->addOptimizationProfile(profile);

    // 创建引擎
    //auto engine = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    std::cout << "Building engine..." << std::endl;
    nvinfer1::IHostMemory* engine = builder->buildSerializedNetwork(*network, *config);

    //序列化保存engine
    std::ofstream engine_file("D:\\model\\acgan\\net_g_api.engine", std::ios::binary);
    engine_file.write((char*)engine->data(), engine->size());
    engine_file.close();


    // 释放资源
    delete engine;
    delete config;
    delete network;
    delete builder;
    std::cout << "Engine build success!" << std::endl;
}


// 加载模型文件
std::vector<unsigned char> load_engine_file(const std::string& file_name)
{
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(file_name, std::ios::binary);
    assert(engine_file.is_open() && "Unable to load engine file.");
    engine_file.seekg(0, engine_file.end);
    int length = engine_file.tellg();
    engine_data.resize(length);
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(reinterpret_cast<char*>(engine_data.data()), length);
    return engine_data;
}


void inference_parser()
{
    // 实例化ILogger
    Logger logger;

    // 创建runtime
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

    // 读取engine,反序列化
    std::string file_path = "D:\\model\\acgan\\net_g_parser.engine";
    std::vector<unsigned char> plan = load_engine_file(file_path);
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));

    // 创建执行上下文
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    nvinfer1::Dims idims0 = engine->getTensorShape("onnx::Concat_0");
    nvinfer1::Dims idims1 = engine->getTensorShape("onnx::Concat_1");
    nvinfer1::Dims odims = engine->getTensorShape("41");
    std::cout << "input0 dims: " << idims0.d[0] << " " << idims0.d[1] << " " << idims0.d[2] << " " << idims0.d[3] << std::endl;
    std::cout << "input1 dims: " << idims1.d[0] << " " << idims1.d[1] << " " << idims1.d[2] << " " << idims1.d[3] << std::endl;
    std::cout << "output dims: " << odims.d[0] << " " << odims.d[1] << " " << odims.d[2] << " " << odims.d[3] << std::endl;
    nvinfer1::Dims4 inputDims0 = { 10, idims0.d[1], idims0.d[2], idims0.d[3] };
    nvinfer1::Dims4 inputDims1 = { 10, idims1.d[1], idims1.d[2], idims1.d[3] };
    context->setInputShape("onnx::Concat_0", inputDims0);
    context->setInputShape("onnx::Concat_1", inputDims1);

    void* buffers[3]{};
    const int inputIndex0 = 0;
    const int inputIndex1 = 1;
    const int outputIndex = 2;

    cudaMalloc(&buffers[inputIndex0], 10 * 100 * 1 * 1 * sizeof(float));
    cudaMalloc(&buffers[inputIndex1], 10 * 10 * 1 * 1 * sizeof(float));
    cudaMalloc(&buffers[outputIndex], 10 * 1 * 28 * 28 * sizeof(float));

    // 设定数据地址
    context->setTensorAddress("onnx::Concat_0", buffers[inputIndex0]);
    context->setTensorAddress("onnx::Concat_1", buffers[inputIndex1]);
    context->setTensorAddress("41", buffers[outputIndex]);

    // 创建cuda流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 读取文件执行推理
    for (int i = 0; i < 10; i += 1)
    {
        torch::Tensor noise = torch::randn({ 1, 100, 1, 1 });
        torch::Tensor label = torch::zeros({ 1, 10, 1, 1 });
        label[0][i][0][0] = 1.0;

        // 将图像拷贝到GPU
        cudaMemcpyAsync(buffers[inputIndex0], noise.data_ptr(), 100 * 1 * 1 * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(buffers[inputIndex1], label.data_ptr(), 10 * 1 * 1 * sizeof(float), cudaMemcpyHostToDevice, stream);

        //执行推理
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);

        float image_data[1 * 28 * 28];
        cudaMemcpyAsync(&image_data, buffers[outputIndex], 1 * 28 * 28 * sizeof(float), cudaMemcpyDeviceToHost, stream);

        // 将数据还原到0-255范围（归一化）
        for (int m = 0; m < 28; ++m)
        {
            for (int n = 0; n < 28; ++n)
            {
                image_data[m*28+n] = image_data[m*28+n] * 127.5 + 127.5;
            }
        }

        cv::Mat image(28, 28, CV_32FC1, image_data);

        cv::imwrite("D:\\model\\acgan\\parser_" + std::to_string(i) + ".jpg", image);
    }


    //torch::Tensor noise = torch::randn({ 10, 100, 1, 1 });
    //torch::Tensor label = torch::zeros({ 10, 10, 1, 1 });
    //for (int i = 0; i < 10; i += 1)
    //{
    //    label[i][i][0][0] = 1.0;
    //}

    //// 将图像拷贝到GPU
    //cudaMemcpyAsync(buffers[inputIndex0], noise.data_ptr(), 10 * 100 * 1 * 1 * sizeof(float), cudaMemcpyHostToDevice, stream);
    //cudaMemcpyAsync(buffers[inputIndex1], label.data_ptr(), 10 * 10 * 1 * 1 * sizeof(float), cudaMemcpyHostToDevice, stream);

    ////执行推理
    //context->enqueueV3(stream);
    //cudaStreamSynchronize(stream);

    //float image_data[10 * 1 * 28 * 28];
    //cudaMemcpyAsync(&image_data, buffers[outputIndex], 10 * 1 * 28 * 28 * sizeof(float), cudaMemcpyDeviceToHost, stream);

    //torch::Tensor image_tensor = torch::from_blob(image_data, { 10, 28, 28 }, torch::kFloat32);
    //image_tensor = image_tensor * 127.5 + 127.5;
    //for (int i = 0; i < 10; ++i)
    //{
    //    cv::Mat image(28, 28, CV_32FC1, image_tensor.index({i}).data_ptr());
    //    cv::imwrite("D:\\model\\acgan\\" + std::to_string(i) + ".jpg", image);
    //}
    

    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex0]);
    cudaFree(buffers[inputIndex1]);
    cudaFree(buffers[outputIndex]);
}


void inference_api()
{
    // 实例化ILogger
    Logger logger;

    // 创建runtime
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

    // 读取engine,反序列化
    std::string file_path = "D:\\model\\acgan\\net_g_api.engine";
    std::vector<unsigned char> plan = load_engine_file(file_path);
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));

    // 创建执行上下文
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    nvinfer1::Dims idims0 = engine->getTensorShape("noise");
    nvinfer1::Dims idims1 = engine->getTensorShape("label");
    nvinfer1::Dims odims = engine->getTensorShape("image");
    std::cout << "input0 dims: " << idims0.d[0] << " " << idims0.d[1] << " " << idims0.d[2] << " " << idims0.d[3] << std::endl;
    std::cout << "input1 dims: " << idims1.d[0] << " " << idims1.d[1] << " " << idims1.d[2] << " " << idims1.d[3] << std::endl;
    std::cout << "output dims: " << odims.d[0] << " " << odims.d[1] << " " << odims.d[2] << " " << odims.d[3] << std::endl;
    nvinfer1::Dims4 inputDims0 = { 10, idims0.d[1], idims0.d[2], idims0.d[3] };
    nvinfer1::Dims4 inputDims1 = { 10, idims1.d[1], idims1.d[2], idims1.d[3] };
    context->setInputShape("noise", inputDims0);
    context->setInputShape("label", inputDims1);

    void* buffers[3]{};
    const int inputIndex0 = 0;
    const int inputIndex1 = 1;
    const int outputIndex = 2;

    cudaMalloc(&buffers[inputIndex0], 10 * 100 * 1 * 1 * sizeof(float));
    cudaMalloc(&buffers[inputIndex1], 10 * 10 * 1 * 1 * sizeof(float));
    cudaMalloc(&buffers[outputIndex], 10 * 1 * 28 * 28 * sizeof(float));

    // 设定数据地址
    context->setTensorAddress("noise", buffers[inputIndex0]);
    context->setTensorAddress("label", buffers[inputIndex1]);
    context->setTensorAddress("image", buffers[outputIndex]);

    // 创建cuda流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 读取文件执行推理
    torch::Tensor noise = torch::randn({ 10, 100, 1, 1 });
    torch::Tensor label = torch::zeros({ 10, 10, 1, 1 });
    for (int i = 0; i < 10; i += 1)
    {
        label[i][i][0][0] = 1.0;
    }

    // 将图像拷贝到GPU
    cudaMemcpyAsync(buffers[inputIndex0], noise.data_ptr(), 10 * 100 * 1 * 1 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[inputIndex1], label.data_ptr(), 10 * 10 * 1 * 1 * sizeof(float), cudaMemcpyHostToDevice, stream);

    //执行推理
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    float image_data[10 * 1 * 28 * 28];
    cudaMemcpyAsync(&image_data, buffers[outputIndex], 10 * 1 * 28 * 28 * sizeof(float), cudaMemcpyDeviceToHost, stream);

    torch::Tensor image_tensor = torch::from_blob(image_data, { 10, 28, 28 }, torch::kFloat32);
    image_tensor = image_tensor * 127.5 + 127.5;
    for (int i = 0; i < 10; ++i)
    {
        cv::Mat image(28, 28, CV_32FC1, image_tensor.index({i}).data_ptr());
        cv::imwrite("D:\\model\\acgan\\api_" + std::to_string(i) + ".jpg", image);
    }

    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex0]);
    cudaFree(buffers[inputIndex1]);
    cudaFree(buffers[outputIndex]);
}

