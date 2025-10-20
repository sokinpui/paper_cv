#ifndef BMP_IMAGE_H
#define BMP_IMAGE_H

#include <vector>
#include <string>
#include <cstdint>
#include "Color.h"

class BmpImage {
public:
    BmpImage() = default;

    bool load(const std::string& filename);

    uint32_t getWidth() const;
    uint32_t getHeight() const;
    const std::vector<RGB>& getPixels() const;

private:
    #pragma pack(push, 1)
    struct BmpFileHeader {
        uint16_t file_type{0x4D42};
        uint32_t file_size{0};
        uint16_t reserved1{0};
        uint16_t reserved2{0};
        uint32_t offset_data{0};
    };

    struct BmpInfoHeader {
        uint32_t size{0};
        int32_t  width{0};
        int32_t  height{0};
        uint16_t planes{1};
        uint16_t bit_count{0};
        uint32_t compression{0};
        uint32_t size_image{0};
        int32_t  x_pixels_per_meter{0};
        int32_t  y_pixels_per_meter{0};
        uint32_t colors_used{0};
        uint32_t colors_important{0};
    };
    #pragma pack(pop)

    BmpFileHeader file_header;
    BmpInfoHeader info_header;
    std::vector<RGB> pixel_data;

    bool checkHeaders(std::ifstream& file_stream) const;
};

#endif // BMP_IMAGE_H
