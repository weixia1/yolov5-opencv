#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>


struct BoundingBox
{
	cv::Rect box;
	float conf{};
	int classId{};
	std::string label_text;
};

typedef struct
{
	float r;
	int dw;
	int dh;
	int new_unpad_w;
	int new_unpad_h;
	bool flag;
} ScaleParams;

struct Config
{
	float conf_threshold;
	float nms_threshold;
	std::string weight_path;
	std::string classname_path;
	cv::Size size;
};

class Detector
{
public:
	Detector(Config &config);
	void detect(cv::Mat &img, std::vector<BoundingBox> &bboxes);
	void drawPredection(cv::Mat& img, std::vector<BoundingBox>& bboxes);

private:
	std::vector<BoundingBox> postProcess(cv::Mat &img, std::vector<cv::Mat> &outputs, ScaleParams &params);
	void resizeUnscale(const cv::Mat &mat, cv::Mat &mat_rs, int target_height, int target_width, ScaleParams &params);


private:
	float nms_threshold;
	float conf_threshold;
	cv::Size onnx_insize;
	std::vector<std::string> class_names;
	cv::dnn::Net model;
};

