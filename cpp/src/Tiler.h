#ifndef TILER_H
#define TILER_H

#include "BmpImage.h"
#include <vector>
#include <memory>

struct Tile {
    uint32_t id;
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
    const BmpImage* source_image;
};

class Tiler {
public:
    static std::vector<Tile> createTiles(const BmpImage& image, uint32_t tile_size);
};

#endif // TILER_H
