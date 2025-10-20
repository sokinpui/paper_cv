#include "MetalProcessor.h"

#ifdef USE_METAL

#import <Metal/Metal.h>
#include <iostream>

namespace {
const char* metal_kernel_source = R"(
#include <metal_stdlib>
using namespace metal;

struct RGB {
    uchar r, g, b;
};

struct Lab {
    float l, a, b;
};

struct Tile {
    uint id;
    uint x;
    uint y;
    uint width;
    uint height;
    ulong source_image; // Placeholder to match C++ struct layout
};

// sRGB to Linear RGB
float srgbToLinear(float c) {
    c /= 255.0f;
    if (c <= 0.04045f) {
        return c / 12.92f;
    }
    return pow((c + 0.055f) / 1.055f, 2.4f);
}

// XYZ to CIELAB conversion helper
float xyzToLabHelper(float t) {
    if (t > 0.008856f) {
        return cbrt(t);
    }
    return (7.787f * t) + (16.0f / 116.0f);
}

kernel void calculateAveragesKernel(
    device const RGB* all_pixels [[buffer(0)]],
    device const uint* image_width_ptr [[buffer(1)]],
    device const Tile* tiles [[buffer(2)]],
    device Lab* average_colors [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    uint image_width = *image_width_ptr;
    const device Tile& tile = tiles[idx];
    double total_l = 0, total_a = 0, total_b = 0;

    for (uint y = 0; y < tile.height; ++y) {
        for (uint x = 0; x < tile.width; ++x) {
            uint pixel_index = (tile.y + y) * image_width + (tile.x + x);
            const device RGB& rgb = all_pixels[pixel_index];

            float r_linear = srgbToLinear(rgb.r);
            float g_linear = srgbToLinear(rgb.g);
            float b_linear = srgbToLinear(rgb.b);

            float x_val = r_linear * 0.4124f + g_linear * 0.3576f + b_linear * 0.1805f;
            float y_val = r_linear * 0.2126f + g_linear * 0.7152f + b_linear * 0.0722f;
            float z_val = r_linear * 0.0193f + g_linear * 0.1192f + b_linear * 0.9505f;

            x_val *= 100.0f;
            y_val *= 100.0f;
            z_val *= 100.0f;

            float fx = xyzToLabHelper(x_val / 95.047f);
            float fy = xyzToLabHelper(y_val / 100.000f);
            float fz = xyzToLabHelper(z_val / 108.883f);

            total_l += 116.0f * fy - 16.0f;
            total_a += 500.0f * (fx - fy);
            total_b += 200.0f * (fy - fz);
        }
    }
    
    uint num_pixels = tile.width * tile.height;
    if (num_pixels > 0) {
        average_colors[idx] = {
            static_cast<float>(total_l / num_pixels),
            static_cast<float>(total_a / num_pixels),
            static_cast<float>(total_b / num_pixels)
        };
    }
}
)";
}

void MetalProcessor::calculateAverageColors(const std::vector<Tile>& tiles, std::vector<Lab>& average_colors) {
    if (tiles.empty()) return;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Error: Metal is not supported on this device." << std::endl;
            return;
        }

        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:metal_kernel_source] options:nil error:&error];
        if (!library) {
            std::cerr << "Error: Failed to create Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
            return;
        }

        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"calculateAveragesKernel"];
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (!pipelineState) {
            std::cerr << "Error: Failed to create Metal pipeline state: " << [[error localizedDescription] UTF8String] << std::endl;
            return;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        const BmpImage* image = tiles[0].source_image;
        const std::vector<RGB>& all_pixels_vec = image->getPixels();
        uint32_t image_width = image->getWidth();

        id<MTLBuffer> pixelsBuffer = [device newBufferWithBytes:all_pixels_vec.data() length:all_pixels_vec.size() * sizeof(RGB) options:MTLResourceStorageModeShared];
        id<MTLBuffer> imageWidthBuffer = [device newBufferWithBytes:&image_width length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> tilesBuffer = [device newBufferWithBytes:tiles.data() length:tiles.size() * sizeof(Tile) options:MTLResourceStorageModeShared];
        id<MTLBuffer> avgColorsBuffer = [device newBufferWithLength:average_colors.size() * sizeof(Lab) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];

        [commandEncoder setComputePipelineState:pipelineState];
        [commandEncoder setBuffer:pixelsBuffer offset:0 atIndex:0];
        [commandEncoder setBuffer:imageWidthBuffer offset:0 atIndex:1];
        [commandEncoder setBuffer:tilesBuffer offset:0 atIndex:2];
        [commandEncoder setBuffer:avgColorsBuffer offset:0 atIndex:3];

        MTLSize gridSize = MTLSizeMake(tiles.size(), 1, 1);
        NSUInteger threadGroupSize = [pipelineState maxTotalThreadsPerThreadgroup];
        if (threadGroupSize > tiles.size()) {
            threadGroupSize = tiles.size();
        }
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

        [commandEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        memcpy(average_colors.data(), [avgColorsBuffer contents], average_colors.size() * sizeof(Lab));
    }
}

#endif // USE_METAL
