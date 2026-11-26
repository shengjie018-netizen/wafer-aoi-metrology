#include "wafer_aoi/image.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

namespace wafer_aoi {

ImageF32 to_float(const ImageU8& src) {
    ImageF32 dst(src.width(), src.height(), src.channels());
    const uint8_t* s = src.data();
    float* d = dst.data();
    const size_t n = src.size();
    for (size_t i = 0; i < n; ++i) {
        d[i] = static_cast<float>(s[i]);
    }
    return dst;
}

ImageU8 to_u8_clamped(const ImageF32& src) {
    ImageU8 dst(src.width(), src.height(), src.channels());
    const float* s = src.data();
    uint8_t* d = dst.data();
    const size_t n = src.size();
    for (size_t i = 0; i < n; ++i) {
        float v = std::clamp(s[i], 0.0f, 255.0f);
        d[i] = static_cast<uint8_t>(std::lround(v));
    }
    return dst;
}

ImageU8 to_u8_normalized(const ImageF32& src) {
    ImageU8 dst(src.width(), src.height(), src.channels());
    if (src.empty()) return dst;
    float mn = std::numeric_limits<float>::max();
    float mx = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < src.size(); ++i) {
        mn = std::min(mn, src.data()[i]);
        mx = std::max(mx, src.data()[i]);
    }
    float range = mx - mn;
    if (range < 1e-6f) {
        dst.fill(0);
        return dst;
    }
    for (size_t i = 0; i < src.size(); ++i) {
        float v = (src.data()[i] - mn) / range * 255.0f;
        dst.data()[i] = static_cast<uint8_t>(std::clamp(std::lround(v), 0L, 255L));
    }
    return dst;
}

static std::string next_pgm_token(std::ifstream& ifs) {
    std::string token;
    while (ifs >> token) {
        if (!token.empty() && token[0] == '#') {
            std::string dummy;
            std::getline(ifs, dummy);
            continue;
        }
        return token;
    }
    return "";
}

ImageU8 load_pgm(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open PGM file: " + path);
    }
    std::string magic = next_pgm_token(ifs);
    if (magic != "P5") {
        throw std::runtime_error("Unsupported PGM format (expected P5): " + magic);
    }
    int width = std::stoi(next_pgm_token(ifs));
    int height = std::stoi(next_pgm_token(ifs));
    int maxval = std::stoi(next_pgm_token(ifs));
    if (width <= 0 || height <= 0 || maxval <= 0 || maxval > 255) {
        throw std::runtime_error("Invalid PGM header parameters in: " + path);
    }
    // Consume single whitespace byte after maxval
    ifs.get();
    ImageU8 img(width, height, 1);
    ifs.read(reinterpret_cast<char*>(img.data()), static_cast<std::streamsize>(img.size()));
    if (!ifs) {
        throw std::runtime_error("Truncated PGM pixel data in: " + path);
    }
    return img;
}

void save_pgm(const std::string& path, const ImageU8& img) {
    if (img.channels() != 1) {
        throw std::invalid_argument("save_pgm requires 1-channel image");
    }
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open output PGM file: " + path);
    }
    ofs << "P5\n" << img.width() << " " << img.height() << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(img.data()),
              static_cast<std::streamsize>(img.size()));
}

void save_ppm(const std::string& path, const ImageU8& rgb) {
    if (rgb.channels() != 3) {
        throw std::invalid_argument("save_ppm requires 3-channel RGB image");
    }
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open output PPM file: " + path);
    }
    ofs << "P6\n" << rgb.width() << " " << rgb.height() << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(rgb.data()),
              static_cast<std::streamsize>(rgb.size()));
}

ImageU8 gray_to_rgb(const ImageU8& gray) {
    ImageU8 rgb(gray.width(), gray.height(), 3);
    for (int y = 0; y < gray.height(); ++y) {
        const uint8_t* src = gray.row_ptr(y);
        uint8_t* dst = rgb.row_ptr(y);
        for (int x = 0; x < gray.width(); ++x) {
            uint8_t v = src[x];
            dst[3 * x + 0] = v;
            dst[3 * x + 1] = v;
            dst[3 * x + 2] = v;
        }
    }
    return rgb;
}

void draw_pixel(ImageU8& rgb, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || y < 0 || x >= rgb.width() || y >= rgb.height()) return;
    rgb.at(x, y, 0) = r;
    rgb.at(x, y, 1) = g;
    rgb.at(x, y, 2) = b;
}

void draw_line(ImageU8& rgb, int x0, int y0, int x1, int y1,
               uint8_t r, uint8_t g, uint8_t b) {
    int dx = std::abs(x1 - x0);
    int dy = -std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;
    while (true) {
        draw_pixel(rgb, x0, y0, r, g, b);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

void draw_rect(ImageU8& rgb, int x, int y, int w, int h,
               uint8_t r, uint8_t g, uint8_t b, int thickness) {
    for (int t = 0; t < thickness; ++t) {
        int x0 = x - t;
        int y0 = y - t;
        int x1 = x + w - 1 + t;
        int y1 = y + h - 1 + t;
        draw_line(rgb, x0, y0, x1, y0, r, g, b);
        draw_line(rgb, x1, y0, x1, y1, r, g, b);
        draw_line(rgb, x1, y1, x0, y1, r, g, b);
        draw_line(rgb, x0, y1, x0, y0, r, g, b);
    }
}

void draw_circle(ImageU8& rgb, float cx, float cy, float radius,
                 uint8_t r, uint8_t g, uint8_t b) {
    int steps = std::max(36, static_cast<int>(std::ceil(2.0 * 3.14159265 * radius * 2.0)));
    for (int i = 0; i < steps; ++i) {
        double a = 2.0 * 3.141592653589793 * i / steps;
        int x = static_cast<int>(std::lround(cx + radius * std::cos(a)));
        int y = static_cast<int>(std::lround(cy + radius * std::sin(a)));
        draw_pixel(rgb, x, y, r, g, b);
    }
}

void draw_cross(ImageU8& rgb, float cx, float cy, int size,
                uint8_t r, uint8_t g, uint8_t b) {
    int ix = static_cast<int>(std::lround(cx));
    int iy = static_cast<int>(std::lround(cy));
    draw_line(rgb, ix - size, iy, ix + size, iy, r, g, b);
    draw_line(rgb, ix, iy - size, ix, iy + size, r, g, b);
}

}  // namespace wafer_aoi
