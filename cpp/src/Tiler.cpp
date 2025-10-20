#include "Tiler.h"

std::vector<Tile> Tiler::createTiles(const BmpImage& image, uint32_t tile_size) {
    std::vector<Tile> tiles;
    uint32_t id_counter = 0;
    for (uint32_t y = 0; y < image.getHeight(); y += tile_size) {
        for (uint32_t x = 0; x < image.getWidth(); x += tile_size) {
            
            uint32_t current_tile_width = std::min(tile_size, image.getWidth() - x);
            uint32_t current_tile_height = std::min(tile_size, image.getHeight() - y);

            tiles.push_back({id_counter++, x, y, current_tile_width, current_tile_height, &image});
        }
    }
    return tiles;
}
