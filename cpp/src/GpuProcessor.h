#ifndef GPU_PROCESSOR_H
#define GPU_PROCESSOR_H

#ifdef USE_GPU

#include "Tiler.h"
#include "Color.h"
#include <vector>

namespace GpuProcessor {
    void calculateAverageColors(const std::vector<Tile>& tiles, std::vector<Lab>& average_colors);
}

#endif // USE_GPU
#endif // GPU_PROCESSOR_H
