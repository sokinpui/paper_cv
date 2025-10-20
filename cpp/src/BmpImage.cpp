#include "BmpImage.h"
#include <fstream>
#include <iostream>

bool BmpImage::load(const std::string& filename) {
    std::ifstream file_stream(filename, std::ios::binary);
    if (!file_stream) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    file_stream.read(reinterpret_cast<char*>(&file_header), sizeof(file_header));
    file_stream.read(reinterpret_cast<char*>(&info_header), sizeof(info_header));

    if (!checkHeaders(file_stream)) {
        return false;
    }

    file_stream.seekg(file_header.offset_data, file_stream.beg);

    pixel_data.resize(info_header.width * info_header.height);
    
    const uint32_t padding = (4 - (info_header.width * 3) % 4) % 4;

    for (int32_t y = info_header.height - 1; y >= 0; --y) {
        for (int32_t x = 0; x < info_header.width; ++x) {
            unsigned char color[3];
            file_stream.read(reinterpret_cast<char*>(color), 3);
            pixel_data[y * info_header.width + x] = {color[2], color[1], color[0]};
        }
        file_stream.ignore(padding);
    }

    return true;
}

bool BmpImage::checkHeaders(std::ifstream& file_stream) const {
    if (file_header.file_type != 0x4D42) {
        std::cerr << "Error: Not a valid BMP file." << std::endl;
        return false;
    }
    if (info_header.bit_count != 24) {
        std::cerr << "Error: Only 24-bit BMP files are supported." << std::endl;
        return false;
    }
    if (info_header.compression != 0) {
        std::cerr << "Error: Only uncompressed BMP files are supported." << std::endl;
        return false;
    }
    return true;
}

uint32_t BmpImage::getWidth() const {
    return info_header.width;
}

uint32_t BmpImage::getHeight() const {
    return info_header.height;
}

const std::vector<RGB>& BmpImage::getPixels() const {
    return pixel_data;
}
