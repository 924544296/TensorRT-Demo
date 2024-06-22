#include "Classification.h"




void Logger::log(Severity severity, const char* msg) noexcept
{
    // 抑制信息级别的消息
    if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
}


void export_engine()
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
    const char* file_path = "D:\\BaiduSyncdisk\\代码\\日联\\MNIST.onnx";
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
    std::ofstream engine_file("D:\\BaiduSyncdisk\\代码\\日联\\MNIST.engine", std::ios::binary);
    engine_file.write((char*)engine->data(), engine->size());
    engine_file.close();

    std::cout << "Engine build success!" << std::endl;
}


int argmax(const float(&rst)[10]) {
    float cache = rst[0];
    int idx = 0;
    for (int i = 1; i < 10; i += 1)
    {
        if (rst[i] > cache)
        {
            cache = rst[i];
            idx = i;
        };
    };
    return idx;
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


void inference()
{
    // 实例化ILogger
    Logger logger;

    // 创建runtime
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

    // 读取engine,反序列化
    std::string file_path = "D:\\BaiduSyncdisk\\代码\\日联\\MNIST.engine";
    std::vector<unsigned char> plan = load_engine_file(file_path);
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));

    // 创建执行上下文
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    nvinfer1::Dims idims = engine->getTensorShape("input.1");// 这里的名字可以在导出时修改
    nvinfer1::Dims odims = engine->getTensorShape("23");
    std::cout << "input dims: " << idims.d[0] << " " << idims.d[1] << " " << idims.d[2] << " " << idims.d[3] << std::endl;
    nvinfer1::Dims4 inputDims = { 1, idims.d[1], idims.d[2], idims.d[3] };
    nvinfer1::Dims2 outputDims = { 1, 10 };
    context->setInputShape("input.1", inputDims);

    void* buffers[2]{};
    const int inputIndex = 0;
    const int outputIndex = 1;

    cudaMalloc(&buffers[inputIndex], 1 * 28 * 28 * sizeof(float));
    cudaMalloc(&buffers[outputIndex], 10 * sizeof(float));

    // 设定数据地址
    context->setTensorAddress("input.1", buffers[inputIndex]);
    context->setTensorAddress("23", buffers[outputIndex]);

    // 创建cuda流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 读取文件执行推理
    for (int i = 0; i < 10; i += 1)
    {
        // 读取图片
        cv::Mat img0;
        std::string file_name = "D:\\BaiduSyncdisk\\代码\\日联\\img\\" + std::to_string(i) + ".png";
        img0 = cv::imread(file_name, 0);// 0为灰度图片
        cv::Mat img;
        img0.convertTo(img, CV_32F);

        // 将图像拷贝到GPU
        cudaMemcpyAsync(buffers[inputIndex], img.data, 1 * 28 * 28 * sizeof(float), cudaMemcpyHostToDevice, stream);

        //执行推理
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);

        float rst[10];
        cudaMemcpyAsync(&rst, buffers[outputIndex], 1 * 10 * sizeof(float), cudaMemcpyDeviceToHost, stream);

        std::cout << file_name << " 推理结果: " << argmax(rst) << std::endl;
    }

    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
}