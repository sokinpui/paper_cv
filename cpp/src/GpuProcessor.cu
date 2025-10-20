#include "GpuProcessor.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

namespace {
    __device__ float srgbToLinear(float c) {
        c /= 255.0f;
        if (c <= 0.04045f) {
            return c / 12.92f;
        }
        return powf((c + 0.055f) / 1.055f, 2.4f);
    }

    __device__ float xyzToLabHelper(float t) {
        if (t > 0.008856f) {
            return cbrtf(t);
        }
        return (7.787f * t) + (16.0f / 116.0f);
    }

    __global__ void calculateAveragesKernel(const RGB* all_pixels, uint32_t image_width, 
                                            const Tile* tiles, Lab* average_colors, int num_tiles) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_tiles) return;

        const Tile& tile = tiles[idx];
        double total_l = 0, total_a = 0, total_b = 0;

        for (uint32_t y = 0; y < tile.height; ++y) {
            for (uint32_t x = 0; x < tile.width; ++x) {
                uint32_t pixel_index = (tile.y + y) * image_width + (tile.x + x);
                const RGB& rgb = all_pixels[pixel_index];

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
        
        uint32_t num_pixels = tile.width * tile.height;
        average_colors[idx] = {
            static_cast<float>(total_l / num_pixels),
            static_cast<float>(total_a / num_pixels),
            static_cast<float>(total_b / num_pixels)
        };
    }
}

void GpuProcessor::calculateAverageColors(const std::vector<Tile>& tiles, std::vector<Lab>& average_colors) {
    if (tiles.empty()) return;

    const BmpImage* image = tiles[0].source_image;
    const std::vector<RGB>& all_pixels_vec = image->getPixels();

    RGB* d_all_pixels;
    Tile* d_tiles;
    Lab* d_average_colors;

    cudaMalloc(&d_all_pixels, all_pixels_vec.size() * sizeof(RGB));
    cudaMalloc(&d_tiles, tiles.size() * sizeof(Tile));
    cudaMalloc(&d_average_colors, average_colors.size() * sizeof(Lab));

    cudaMemcpy(d_all_pixels, all_pixels_vec.data(), all_pixels_vec.size() * sizeof(RGB), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tiles, tiles.data(), tiles.size() * sizeof(Tile), cudaMemcpyHostToDevice);

    int num_tiles = tiles.size();
    int threads_per_block = 256;
    int blocks_per_grid = (num_tiles + threads_per_block - 1) / threads_per_block;

    calculateAveragesKernel<<<blocks_per_grid, threads_per_block>>>(d_all_pixels, image->getWidth(), d_tiles, d_average_colors, num_tiles);
    
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(average_colors.data(), d_average_colors, average_colors.size() * sizeof(Lab), cudaMemcpyDeviceToHost);

    cudaFree(d_all_pixels);
    cudaFree(d_tiles);
    cudaFree(d_average_colors);
}

#endif // USE_CUDA
