// Demo use Opencv

#include <ctime>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "detector.h"

void detect_video() {
	
	std::string weight_path = "yolov5s.onnx";
	std::string img_path = "001.mp4";
	std::string classname_path = "coco.names";
	
	Config config = { 0.3f, 0.3f, weight_path, classname_path, cv::Size(640, 640)};
	std::cout << "Start main process" << std::endl;
	Detector detector(config);

	cv::VideoCapture cap;
	cap.open(img_path);
	if (!cap.isOpened())
	{
		std::cout << "Could not load video ..." << std::endl;
		system("pause");
		return;
	}

	bool stop(false);

	cv::Mat frame;
	std::vector<BoundingBox> bboxes;
	while (!stop)
	{
		// load the next frame
		if (!cap.read(frame))
			break;
		clock_t  start_t = clock();
		// 主要耗时的地方还是检测，占总运行时长的90%以上
		detector.detect(frame, bboxes);
		std::cout << "Detect process finished" << std::endl;
		clock_t end_t = clock();
		std::cout << "Time is:" << (double)(end_t - start_t) / CLOCKS_PER_SEC << "s" << std::endl;
		detector.drawPredection(frame, bboxes);
		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
	}
	cap.release();
	return;
}

void detect_img() {
	std::string weight_path = "yolov5s.onnx";
	std::string img_path = "bus.jpg"; 
	std::string classname_path = "coco.names";

	cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);

	if (img.empty())
	{
		std::cout << "Could not load image ...\n" << std::endl;
		system("pause");
		return;
	}
	Config config = { 0.3f, 0.3f, weight_path, classname_path, cv::Size(640, 640)};
	std::cout << "Start main process" << std::endl;
	Detector detector(config);

	std::vector<BoundingBox> bboxes;
	detector.detect(img, bboxes);
	detector.drawPredection(img, bboxes);
	cv::waitKey(0);
	return;
}

int main(int argc, char *argv[])
{
	// detect video or img
	detect_video();
	cv::destroyAllWindows();
	
}
