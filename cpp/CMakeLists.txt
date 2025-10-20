cmake_minimum_required(VERSION 3.18)
project(ImageComparator LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Define executable and sources ---

add_executable(image_comparator
    src/main.cpp
    src/BmpImage.cpp
    src/Color.cpp
    src/Tiler.cpp
    src/Comparator.cpp
)

target_include_directories(image_comparator PUBLIC src)

# --- Find and configure optional dependencies ---

# Find CUDA for NVIDIA GPUs
find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "CUDA found, enabling GPU support via CUDA.")
    enable_language(CUDA)
    target_sources(image_comparator PRIVATE src/GpuProcessor.cu)
    target_compile_definitions(image_comparator PRIVATE USE_CUDA)
else()
    message(STATUS "CUDA not found. Building without CUDA support.")
endif()

# Find Metal for Apple GPUs
if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    if(METAL_FRAMEWORK)
        message(STATUS "Metal found, enabling GPU support via Metal.")
        target_sources(image_comparator PRIVATE src/MetalProcessor.mm)
        set_source_files_properties(src/MetalProcessor.mm PROPERTIES LANGUAGE OBJCXX)
        target_compile_definitions(image_comparator PRIVATE USE_METAL)
        target_link_libraries(image_comparator PRIVATE "-framework Foundation" "-framework Metal" "-framework CoreGraphics")
    else()
        message(STATUS "Metal not found. Building without Metal support.")
    endif()
endif()

# Link pthreads for std::thread if not on Apple/Windows
if(NOT APPLE AND NOT WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(image_comparator PRIVATE Threads::Threads)
endif()

# --- Installation ---
install(TARGETS image_comparator DESTINATION bin)
