cmake_minimum_required(VERSION 3.16)
project(NerfInterceptor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# venv içindeki OpenVINO CMake yolunu otomatik ekle
list(APPEND CMAKE_PREFIX_PATH 
    "${CMAKE_CURRENT_SOURCE_DIR}/venv/lib64/python3.14/site-packages/openvino"
    "${CMAKE_CURRENT_SOURCE_DIR}/venv/lib/python3.14/site-packages/openvino"
)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

add_executable(interceptor src/interceptor.cpp)

target_link_libraries(interceptor
    PRIVATE
    ${OpenCV_LIBS}
    openvino::runtime
    pthread
)
