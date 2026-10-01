cmake_minimum_required(VERSION 3.16)
project(nerf-interceptor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(Threads REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${OpenVINO_INCLUDE_DIRS}
    include
)

add_executable(interceptor src/interceptor.cpp)

target_link_libraries(interceptor PRIVATE
    ${OpenCV_LIBS}
    openvino::runtime
    Threads::Threads
)
