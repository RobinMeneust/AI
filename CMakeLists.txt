cmake_minimum_required(VERSION 3.16)

project(CPP_AI_Project LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} src/main.cpp src/NeuralNetwork.cpp src/DenseLayer.cpp src/LayersList.cpp src/Sigmoid.cpp src/Softmax.cpp src/Relu.cpp
        include/Instance.h
        include/LeakyRelu.h
        src/LeakyRelu.cpp
        include/Identity.h
        src/Identity.cpp
        include/Tensor.h
        src/Tensor.cpp
        src/Layer.cpp
        src/Instance.cpp
        src/Batch.cpp
        src/Batch.h)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )