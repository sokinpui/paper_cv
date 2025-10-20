#ifndef COMPARATOR_H
#define COMPARATOR_H

#include "Tiler.h"
#include "Color.h"
#include <vector>
#include <utility>

struct ComparisonResult {
    uint32_t tile1_id;
    uint32_t tile2_id;
    float color_difference;
};

class Comparator {
public:
    Comparator(const std::vector<Tile>& tiles, bool use_gpu);
    std::vector<ComparisonResult> run(unsigned int num_threads);

private:
    void computeAverageColors();
    void worker(unsigned int worker_id, 
                const std::vector<std::pair<size_t, size_t>>& pairs,
                std::vector<ComparisonResult>& results);

    const std::vector<Tile>& image_tiles;
    std::vector<Lab> average_colors;
    bool use_gpu_acceleration;
};

#endif // COMPARATOR_H
