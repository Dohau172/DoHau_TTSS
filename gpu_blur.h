#pragma once

#ifndef GPU_BLUR_H
#define GPU_BLUR_H

#include <cuda_runtime.h>

// Hàm này chạy trên CPU để ra lệnh cho GPU "xuất quân"
void launch_gpu_blur(unsigned char* d_input, unsigned char* d_output,
    unsigned char* d_mask, int width, int height);

#endif