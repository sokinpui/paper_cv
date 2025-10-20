#include "Color.h"
#include <cmath>

namespace {
    // sRGB to Linear RGB
    float srgbToLinear(float c) {
        c /= 255.0f;
        if (c <= 0.04045f) {
            return c / 12.92f;
        }
        return std::pow((c + 0.055f) / 1.055f, 2.4f);
    }

    // XYZ D65 reference white
    constexpr float REF_X = 95.047f;
    constexpr float REF_Y = 100.000f;
    constexpr float REF_Z = 108.883f;

    // XYZ to CIELAB conversion helper
    float xyzToLabHelper(float t) {
        if (t > 0.008856f) {
            return std::cbrt(t);
        }
        return (7.787f * t) + (16.0f / 116.0f);
    }
}

Lab rgbToLab(const RGB& rgb) {
    float r_linear = srgbToLinear(rgb.r);
    float g_linear = srgbToLinear(rgb.g);
    float b_linear = srgbToLinear(rgb.b);

    float x = r_linear * 0.4124f + g_linear * 0.3576f + b_linear * 0.1805f;
    float y = r_linear * 0.2126f + g_linear * 0.7152f + b_linear * 0.0722f;
    float z = r_linear * 0.0193f + g_linear * 0.1192f + b_linear * 0.9505f;
    
    x *= 100.0f;
    y *= 100.0f;
    z *= 100.0f;

    float fx = xyzToLabHelper(x / REF_X);
    float fy = xyzToLabHelper(y / REF_Y);
    float fz = xyzToLabHelper(z / REF_Z);

    float l = 116.0f * fy - 16.0f;
    float a = 500.0f * (fx - fy);
    float b = 200.0f * (fy - fz);

    return {l, a, b};
}

float deltaE(const Lab& lab1, const Lab& lab2) {
    float dL = lab1.l - lab2.l;
    float da = lab1.a - lab2.a;
    float db = lab1.b - lab2.b;
    return std::sqrt(dL * dL + da * da + db * db);
}
