#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace wafer_aoi {

template <typename T>
class Image {
public:
    Image() = default;

    Image(int width, int height, int channels = 1, T init_val = T{})
        : width_(width),
          height_(height),
          channels_(channels),
          data_(static_cast<size_t>(width) * static_cast<size_t>(height) *
                    static_cast<size_t>(channels),
                init_val) {
        if (width < 0 || height < 0 || channels <= 0) {
            throw std::invalid_argument("Invalid image dimensions");
        }
    }

    int width() const { return width_; }
    int height() const { return height_; }
    int channels() const { return channels_; }
    bool empty() const { return data_.empty() || width_ == 0 || height_ == 0; }
    size_t size() const { return data_.size(); }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    T* row_ptr(int y) {
        return data_.data() + static_cast<size_t>(y) * static_cast<size_t>(width_) *
                                  static_cast<size_t>(channels_);
    }

    const T* row_ptr(int y) const {
        return data_.data() + static_cast<size_t>(y) * static_cast<size_t>(width_) *
                                  static_cast<size_t>(channels_);
    }

    T& at(int x, int y, int c = 0) {
        return data_[(static_cast<size_t>(y) * static_cast<size_t>(width_) +
                      static_cast<size_t>(x)) *
                         static_cast<size_t>(channels_) +
                     static_cast<size_t>(c)];
    }

    const T& at(int x, int y, int c = 0) const {
        return data_[(static_cast<size_t>(y) * static_cast<size_t>(width_) +
                      static_cast<size_t>(x)) *
                         static_cast<size_t>(channels_) +
                     static_cast<size_t>(c)];
    }

    T at_clamped(int x, int y, int c = 0) const {
        int cx = std::clamp(x, 0, width_ - 1);
        int cy = std::clamp(y, 0, height_ - 1);
        return at(cx, cy, c);
    }

    Image<T> clone() const { return *this; }

    Image<T> crop(int x, int y, int w, int h) const {
        if (x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > width_ || y + h > height_) {
            throw std::out_of_range("Crop ROI out of image bounds");
        }
        Image<T> out(w, h, channels_);
        for (int row = 0; row < h; ++row) {
            const T* src = row_ptr(y + row) + static_cast<size_t>(x) * channels_;
            T* dst = out.row_ptr(row);
            std::copy(src, src + static_cast<size_t>(w) * channels_, dst);
        }
        return out;
    }

    void fill(T val) { std::fill(data_.begin(), data_.end(), val); }

private:
    int width_ = 0;
    int height_ = 0;
    int channels_ = 1;
    std::vector<T> data_;
};

using ImageU8 = Image<uint8_t>;
using ImageF32 = Image<float>;
using ImageI32 = Image<int32_t>;

// Type conversions
ImageF32 to_float(const ImageU8& src);
ImageU8 to_u8_clamped(const ImageF32& src);
ImageU8 to_u8_normalized(const ImageF32& src);

// Native lossless binary PGM (P5 grayscale) & PPM (P6 RGB) I/O
ImageU8 load_pgm(const std::string& path);
void save_pgm(const std::string& path, const ImageU8& img);
void save_ppm(const std::string& path, const ImageU8& rgb);

// Simple RGB drawing helpers for zero-dependency visual inspection output
ImageU8 gray_to_rgb(const ImageU8& gray);
void draw_pixel(ImageU8& rgb, int x, int y, uint8_t r, uint8_t g, uint8_t b);
void draw_line(ImageU8& rgb, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b);
void draw_rect(ImageU8& rgb, int x, int y, int w, int h, uint8_t r, uint8_t g, uint8_t b, int thickness = 1);
void draw_circle(ImageU8& rgb, float cx, float cy, float radius, uint8_t r, uint8_t g, uint8_t b);
void draw_cross(ImageU8& rgb, float cx, float cy, int size, uint8_t r, uint8_t g, uint8_t b);

}  // namespace wafer_aoi
