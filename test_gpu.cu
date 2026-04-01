#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <sstream>
#include "gpu_blur.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

int main() {
    // ====================== KIỂM TRA GPU ======================
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error Code: " << err << std::endl;
        std::cerr << "CUDA Runtime Error String: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Total CUDA devices found: " << deviceCount << std::endl;

    if (deviceCount == 0) {
        std::cerr << "ERROR: No CUDA-capable GPU found!" << std::endl;
        return -1;
    }

    // 👉 LUÔN dùng GPU 0 (máy bạn chỉ có 1 GPU)
    int gpu_id = 0;
    CUDA_CHECK(cudaSetDevice(gpu_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_id));
    std::cout << "=== USING GPU: " << prop.name << " ===\n" << std::endl;

    // ====================== LOAD MODEL ======================
    cv::dnn::Net net = cv::dnn::readNetFromTFLite(
        "D:\\BCSE2023\\TinhToanSongSong\\ProjectCK\\ProjectCK\\selfie_segmentation.tflite"
    );

    if (net.empty()) {
        std::cerr << "ERROR: Cannot load TFLite model!" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open camera!" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame;
    for (int i = 0; i < 30; i++) cap >> frame;

    if (frame.empty()) {
        std::cerr << "ERROR: Cannot get frame from camera!" << std::endl;
        return -1;
    }

    // ====================== CẤP PHÁT GPU ======================
    unsigned char *d_input = nullptr, *d_output = nullptr, *d_mask = nullptr;

    size_t maxImgSize  = frame.total() * frame.elemSize();
    size_t maxMaskSize = (size_t)frame.rows * frame.cols;

    std::cout << "Allocating memory on GPU... (" << maxImgSize / (1024*1024) << " MB)" << std::endl;

    CUDA_CHECK(cudaMalloc(&d_input,  maxImgSize));
    CUDA_CHECK(cudaMalloc(&d_output, maxImgSize));
    CUDA_CHECK(cudaMalloc(&d_mask,   maxMaskSize));

    std::cout << "CUDA memory allocation successful!" << std::endl;

    // ====================== LOOP ======================
    while (true) {
        cap >> frame;

        if (frame.empty() || frame.data == nullptr) {
            cv::waitKey(1);
            continue;
        }

        cv::flip(frame, frame, 1);

        // ====================== AI SEGMENT ======================
        cv::Mat blob = cv::dnn::blobFromImage(
            frame, 1.0/255.0, cv::Size(256, 144),
            cv::Scalar(0), true, false
        );

        net.setInput(blob);
        cv::Mat output = net.forward();

        if (output.empty()) continue;

        cv::Mat mask_float = output.reshape(1, 144);
        cv::Mat binary_mask;

        cv::resize(mask_float, binary_mask, frame.size());
        cv::threshold(binary_mask, binary_mask, 0.5, 255.0, cv::THRESH_BINARY);
        binary_mask.convertTo(binary_mask, CV_8U);

        // ====================== COPY TO GPU ======================
        size_t imgSize  = frame.total() * frame.elemSize();
        size_t maskSize = (size_t)frame.rows * frame.cols;

        CUDA_CHECK(cudaMemcpy(d_input, frame.data, imgSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mask, binary_mask.data, maskSize, cudaMemcpyHostToDevice));

        // ====================== GPU PROCESS ======================
        int64 t_start = cv::getTickCount();

        launch_gpu_blur(d_input, d_output, d_mask, frame.cols, frame.rows);

        CUDA_CHECK(cudaDeviceSynchronize());

        int64 t_end = cv::getTickCount();
        double gpu_time = (t_end - t_start) * 1000.0 / cv::getTickFrequency();

        // ====================== COPY BACK ======================
        cv::Mat resultGPU = frame.clone();
        CUDA_CHECK(cudaMemcpy(resultGPU.data, d_output, imgSize, cudaMemcpyDeviceToHost));

        // ====================== SHOW ======================
        std::string text = "GPU: " + std::to_string((int)gpu_time) + " ms | RTX 3050";

        cv::putText(resultGPU, text, cv::Point(30, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 1.5,
                    cv::Scalar(0, 255, 0), 2);

        cv::imshow("Project CK - CUDA", resultGPU);

        if (cv::waitKey(1) == 27) break;
    }

    // ====================== FREE ======================
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);

    return 0;
}