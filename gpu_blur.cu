#include "gpu_blur.h"
#include <device_launch_parameters.h>

// KERNEL: Xử lý ảnh 3 kênh màu (BGR)
__global__ void blur_kernel(const unsigned char* input, unsigned char* output, const unsigned char* mask, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int mask_idx = y * width + x;
    int img_idx = (y * width + x) * 3;

    if (mask[mask_idx] < 128)
    {
        float sumB = 0, sumG = 0, sumR = 0;
        float weightSum = 0;

        int radius = 5;
        float sigma = 5.0f;

        for (int ky = -radius; ky <= radius; ky++)
        {
            for (int kx = -radius; kx <= radius; kx++)
            {
                int nx = x + kx;
                int ny = y + ky;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                {
                    float dist = kx * kx + ky * ky;
                    float weight = expf(-dist / (2 * sigma * sigma));

                    int n_idx = (ny * width + nx) * 3;

                    sumB += input[n_idx] * weight;
                    sumG += input[n_idx + 1] * weight;
                    sumR += input[n_idx + 2] * weight;

                    weightSum += weight;
                }
            }
        }

        output[img_idx]     = sumB / weightSum;
        output[img_idx + 1] = sumG / weightSum;
        output[img_idx + 2] = sumR / weightSum;
    }
    else
    {
        output[img_idx]     = input[img_idx];
        output[img_idx + 1] = input[img_idx + 1];
        output[img_idx + 2] = input[img_idx + 2];
    }
}

void launch_gpu_blur(unsigned char* d_input, unsigned char* d_output,
    unsigned char* d_mask, int width, int height)
{
    // Cấu hình 16x16 threads cho mỗi Block (Tối ưu cho hầu hết GPU NVIDIA)
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    blur_kernel<<<gridSize, blockSize>>>(d_input, d_output, d_mask, width, height);
    
    // Đảm bảo GPU hoàn thành trước khi quay lại CPU
    cudaDeviceSynchronize();
}
