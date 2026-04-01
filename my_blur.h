#ifndef MY_BLUR_H
#define MY_BLUR_H

#include <cmath>
#include <vector>
#include <algorithm> // Thêm cái này để dùng std::min/max
#include <opencv2/opencv.hpp>

// Tự định nghĩa số Pi nếu chưa có
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class MyGaussian {
public:
    // Tạo ma trận trọng số Gaussian (giống hàm gaussian_kernel của Hậu)
    std::vector<std::vector<double>> create_kernel(int ksize, double sigma) {
        int pad = ksize / 2;
        std::vector<std::vector<double>> kernel(ksize, std::vector<double>(ksize));
        double sum = 0.0;

        for (int i = -pad; i <= pad; i++) {
            for (int j = -pad; j <= pad; j++) {
                double val = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
                kernel[i + pad][j + pad] = val;
                sum += val;
            }
        }

        // Chuẩn hóa để tổng các phần tử = 1 (Tránh làm ảnh bị tối hoặc quá sáng)
        for (int i = 0; i < ksize; i++) {
            for (int j = 0; j < ksize; j++) {
                kernel[i][j] /= sum;
            }
        }
        return kernel;
    }

    // Hàm làm mờ tự xây dựng (giống hàm gaussian_blur của Hậu)
    cv::Mat apply_blur(const cv::Mat& input, int ksize, double sigma) {
        auto kernel = create_kernel(ksize, sigma);
        int pad = ksize / 2;
        cv::Mat output = input.clone();

        // Tạo ảnh đệm (Padding) để xử lý phần rìa ảnh
        cv::Mat padded_image;
        cv::copyMakeBorder(input, padded_image, pad, pad, pad, pad, cv::BORDER_REPLICATE);

        // Duyệt từng pixel (Convolution) - Y hệt logic 3 vòng for của Hậu
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                for (int c = 0; c < 3; c++) { // 3 kênh màu BGR
                    double pixel_val = 0.0;
                    for (int u = 0; u < ksize; u++) {
                        for (int v = 0; v < ksize; v++) {
                            pixel_val += kernel[u][v] * padded_image.at<cv::Vec3b>(i + u, j + v)[c];
                        }
                    }
                    output.at<cv::Vec3b>(i, j)[c] = (uchar)std::min(std::max(pixel_val, 0.0), 255.0);
                }
            }
        }
        return output;
    }
};

#endif