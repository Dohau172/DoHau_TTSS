#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include "my_blur.h"

using namespace cv;
using namespace dnn;
using namespace std;

int main(){
	//Load model AI
	
	Net net = readNetFromTFLite("D:\\BCSE2023\\TinhToanSongSong\\ProjectCK\\ProjectCK\\selfie_segmentation.tflite");
	if (net.empty()) {
		cerr << "Failed to load model!" << endl;
		return -1;
	}

	//Mở webcam
	VideoCapture cap(0);
	if(!cap.isOpened()) {
		cerr << "Error opening video stream!" << endl;
		return -1;
	}

	Mat frame, blob;
	cap >> frame;
	if (frame.empty()) {
		cerr << "No frame captured!" << endl;
		return -1;
	}
	MyGaussian customBlur;

	// Khai báo con trỏ GPU
	unsigned char *d_input, *d_output, *d_mask;
	size_t imgSize = frame.total() * frame.elemSize();
	size_t maskSize = frame.total();

	cudaMalloc(&d_input, imgSize);
	cudaMalloc(&d_output, imgSize);
	cudaMalloc(&d_mask, maskSize);

	while (true) {
		cap >> frame;
		if (frame.empty()) {
			cerr << "No frame captured!" << endl;
			break;
		}
		flip(frame, frame, 1); // Lật video ngang

		blobFromImage(frame, blob, 1.0/255.0, Size(256, 144), Scalar(0, 0, 0), true, false);
		net.setInput(blob);

		Mat output = net.forward();
		float* data = output.ptr<float>();

		Mat raw_mask(144, 256, CV_32F, data);

		Mat mask;
		resize(raw_mask, mask, frame.size());

		Mat binary_mask;
		threshold(mask, binary_mask, 0.5, 255, THRESH_BINARY);
		binary_mask.convertTo(binary_mask, CV_8U);
		//imshow("1. Webcam cua Hau", frame);
        //imshow("2. AI Mask", binary_mask);

        // Phần CPU processing giữ nguyên...
        int64 start = getTickCount();

        Mat blurredFrame = customBlur.apply_blur(frame, 15, 3.0); 

        Mat finalResult = frame.clone();

        for (int y = 0; y < frame.rows; y++) {
            for (int x = 0; x < frame.cols; x++) {
                if (binary_mask.at<uchar>(y, x) == 0) {
                    finalResult.at<Vec3b>(y, x) = blurredFrame.at<Vec3b>(y, x);
                }
            }
        }

        int64 end = getTickCount();
        double time_ms = (end - start) * 1000.0 / getTickFrequency();

        string timeText = "CPU Time: " + to_string(int(time_ms)) + " ms";
        putText(finalResult, timeText, Point(30, 50), 
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

        imshow("4. Final Result (CPU - Slow)", finalResult);

        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;




}