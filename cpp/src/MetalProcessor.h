#ifndef METAL_PROCESSOR_H
#define METAL_PROCESSOR_H

#ifdef USE_METAL

#include "Tiler.h"
#include "Color.h"
#include <vector>

namespace MetalProcessor {
    void calculateAverageColors(const std::vector<Tile>& tiles, std::vector<Lab>& average_colors);
}

#endif // USE_METAL
#endif // METAL_PROCESSOR_H
