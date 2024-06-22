# CMakeList.txt: Classification 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("Classification")


###
# opencv
#set(OpenCV_DIR "D:/software/opencv/opencv490/build")
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("D:/software/opencv/opencv490/build/include"
                    "D:/software/opencv/opencv490/build/include/opencv2")
link_directories("D:/software/opencv/opencv490/build/x64/vc16/lib")
# libtorch
set(Torch_DIR "D:/software/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
#include_directories(${Torch_INCLUDE_DIRS})
# CUDA
include_directories("D:\\software\\CUDA12_4\\include")
link_directories("D:\\software\\CUDA12_4\\lib\\x64")
# TensorRT
#include_directories("D:\\software\\TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0\\TensorRT-8.6.1.6\\include")
#link_directories("D:\\software\\TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0\\TensorRT-8.6.1.6\\lib")
include_directories("D:/software/TensorRT-10.0.0.6.Windows10.win10.cuda-12.4/TensorRT-10.0.0.6/include")
link_directories("D:/software/TensorRT-10.0.0.6.Windows10.win10.cuda-12.4/TensorRT-10.0.0.6/lib")


# 将源代码添加到此项目的可执行文件。
add_executable (Classification "main.cpp" 
							   "Classification.h" "Classification.cpp")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Classification PROPERTY CXX_STANDARD 20)
endif()

# TODO: 如有需要，请添加测试并安装目标。
# opencv
#target_link_libraries(Classification ${OpenCV_LIBS})
target_link_libraries(Classification opencv_world490d)
# libtorch
target_link_libraries(Classification ${TORCH_LIBRARIES})
# CUDA
target_link_libraries(Classification cudart)
# TensorRT
target_link_libraries(Classification nvinfer nvinfer_plugin nvonnxparser)