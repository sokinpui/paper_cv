#include "Comparator.h"
#ifdef USE_CUDA
#include "GpuProcessor.h"
#endif
#ifdef USE_METAL
#include "MetalProcessor.h"
#endif
#include <iostream>
#include <numeric>
#include <thread>
#include <algorithm>

Comparator::Comparator(const std::vector<Tile>& tiles, bool use_gpu)
    : image_tiles(tiles), use_gpu_acceleration(use_gpu) {
    average_colors.resize(tiles.size());
}

void Comparator::computeAverageColors() {
#if defined(USE_CUDA) || defined(USE_METAL)
    if (use_gpu_acceleration) {
        std::cout << "Computing average colors on GPU..." << std::endl;
#if defined(USE_METAL)
        MetalProcessor::calculateAverageColors(image_tiles, average_colors);
#else // USE_CUDA
        GpuProcessor::calculateAverageColors(image_tiles, average_colors);
#endif
        return;
    }
#endif
    std::cout << "Computing average colors on CPU..." << std::endl;
    for (size_t i = 0; i < image_tiles.size(); ++i) {
        const auto& tile = image_tiles[i];
        const auto* image = tile.source_image;
        const auto& pixels = image->getPixels();
        
        double total_l = 0, total_a = 0, total_b = 0;
        for (uint32_t y = 0; y < tile.height; ++y) {
            for (uint32_t x = 0; x < tile.width; ++x) {
                uint32_t pixel_index = (tile.y + y) * image->getWidth() + (tile.x + x);
                Lab lab = rgbToLab(pixels[pixel_index]);
                total_l += lab.l;
                total_a += lab.a;
                total_b += lab.b;
            }
        }
        uint32_t num_pixels = tile.width * tile.height;
        average_colors[i] = {
            static_cast<float>(total_l / num_pixels),
            static_cast<float>(total_a / num_pixels),
            static_cast<float>(total_b / num_pixels)
        };
    }
}

void Comparator::worker(unsigned int worker_id, 
                        const std::vector<std::pair<size_t, size_t>>& pairs,
                        std::vector<ComparisonResult>& results) {
    for (const auto& pair : pairs) {
        float diff = deltaE(average_colors[pair.first], average_colors[pair.second]);
        results.push_back({image_tiles[pair.first].id, image_tiles[pair.second].id, diff});
    }
}

std::vector<ComparisonResult> Comparator::run(unsigned int num_threads) {
    computeAverageColors();

    std::vector<std::pair<size_t, size_t>> all_pairs;
    if (image_tiles.size() > 1) {
        for (size_t i = 0; i < image_tiles.size() - 1; ++i) {
            for (size_t j = i + 1; j < image_tiles.size(); ++j) {
                all_pairs.emplace_back(i, j);
            }
        }
    }
    
    std::cout << "Comparing " << all_pairs.size() << " pairs using " << num_threads << " threads..." << std::endl;

    std::vector<std::thread> threads;
    std::vector<std::vector<ComparisonResult>> thread_results(num_threads);
    std::vector<std::vector<std::pair<size_t, size_t>>> thread_workloads(num_threads);

    for (size_t i = 0; i < all_pairs.size(); ++i) {
        thread_workloads[i % num_threads].push_back(all_pairs[i]);
    }

    for (unsigned int i = 0; i < num_threads; ++i) {
        threads.emplace_back(&Comparator::worker, this, i, std::ref(thread_workloads[i]), std::ref(thread_results[i]));
    }

    for (auto& t : threads) {
        t.join();
    }

    std::vector<ComparisonResult> final_results;
    for (const auto& result_vec : thread_results) {
        final_results.insert(final_results.end(), result_vec.begin(), result_vec.end());
    }

    return final_results;
}
