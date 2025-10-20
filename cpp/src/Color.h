#ifndef COLOR_H
#define COLOR_H

#include <cstdint>

struct RGB {
    uint8_t r, g, b;
};

struct Lab {
    float l, a, b;
};

Lab rgbToLab(const RGB& rgb);
float deltaE(const Lab& lab1, const Lab& lab2);

#endif // COLOR_H
