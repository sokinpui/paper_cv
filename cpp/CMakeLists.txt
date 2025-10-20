cmake_minimum_required(VERSION 3.12)
project(ImageComparator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_GPU "Enable GPU support" ON)

add_executable(image_comparator src/main.cpp
                              src/BmpImage.cpp
                              src/Color.cpp
                              src/Tiler.cpp
                              src/Comparator.cpp)

target_include_directories(image_comparator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(Threads REQUIRED)
target_link_libraries(image_comparator PRIVATE Threads::Threads)

if(ENABLE_GPU)
    message(STATUS "GPU support is enabled")
    target_sources(image_comparator PRIVATE src/GpuProcessor.cu)
    target_compile_definitions(image_comparator PRIVATE USE_GPU)
    find_package(CUDA REQUIRED)
    target_link_libraries(image_comparator PRIVATE CUDA::cudart)
else()
    message(STATUS "GPU support is disabled. Building in CPU-only mode.")
endif()
