cmake_minimum_required(VERSION 3.0)
project(simple)

aux_source_directory(./src SRC)

set(CMAKE_BUILD_TYPE Debug)

#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)

find_package (OpenCV REQUIRED)

include_directories(/usr/include/aarch64-linux-gnu/)
link_directories(/usr/lib/aarch64-linux-gnu/)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(a ${SRC})

target_link_libraries(a nvinfer cudart  ${OpenCV_LIBS})

