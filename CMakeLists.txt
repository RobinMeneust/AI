cmake_minimum_required(VERSION 3.16)

project(CPP_AI_Project LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package( OpenCV REQUIRED )
find_package( OpenMP REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} OpenMP::OpenMP_CXX)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GTK_CFLAGS} -fopenmp")

add_executable(${PROJECT_NAME}
        src/main.cpp
        src/NeuralNetwork.cpp
        src/DenseLayer.cpp
        src/LayersList.cpp
        src/Sigmoid.cpp
        src/Softmax.cpp
        src/Relu.cpp
        include/Instance.h
        include/LeakyRelu.h
        src/LeakyRelu.cpp
        include/Identity.h
        src/Identity.cpp
        include/Tensor.h
        src/Tensor.cpp
        src/Layer.cpp
        src/Instance.cpp
        include/Batch.h
        src/Batch.cpp
        include/LayerType.h
        src/FlattenLayer.cpp
        src/Conv2DLayer.cpp
        include/Conv2DLayer.h
        src/MaxPoolingLayer.cpp
        include/MaxPoolingLayer.h
)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )