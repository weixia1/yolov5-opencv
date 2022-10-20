#include "detector.h"

Detector::Detector(Config &config) :
	nms_threshold(config.nms_threshold), conf_threshold(config.conf_threshold),
	onnx_insize(config.size)
{
	std::ifstream ifs(config.classname_path);
	if (!ifs.good()) {
		std::cout << "The 'classname_path' is empty..." << std::endl;
		system("pause");
		exit(-1);
	}
	std::string line;
	while (std::getline(ifs, line))
		this->class_names.push_back(line);
	ifs.close();

	std::ifstream ifw(config.weight_path);
	if (!ifw.good()) {
		std::cout << "The 'weight_path' is empty..." << std::endl;
		system("pause");
		exit(-1);
	}

	this->model = cv::dnn::readNetFromONNX(config.weight_path);
	this->model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	this->model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

void Detector::resizeUnscale(const cv::Mat &mat, cv::Mat &mat_rs,
	int target_height, int target_width,
	ScaleParams &scale_params)
{
	if (mat.empty()) return;
	int img_height = static_cast<int>(mat.rows);
	int img_width = static_cast<int>(mat.cols);

	mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
		cv::Scalar(114, 114, 114));
	// scale ratio (new/old) new_shape(h,w)
	float w_r = (float)target_width / (float)img_width;
	float h_r = (float)target_height / (float)img_height;
	float r = std::min(w_r, h_r);
	// compute padding
	int new_unpad_w = static_cast<int>((float)img_width * r);// floor
	int new_unpad_h = static_cast<int>((float)img_height * r); // floor
	int pad_w = target_width - new_unpad_w; // >=0
	int pad_h = target_height - new_unpad_h;// >=0

	int dw = pad_w / 2;
	int dh = pad_h / 2;

	// resize with unscaling
	cv::Mat new_unpad_mat;
	cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
	new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

	// record scale params.
	scale_params.r = r;
	scale_params.dw = dw;
	scale_params.dh = dh;
	scale_params.new_unpad_w = new_unpad_w;
	scale_params.new_unpad_h = new_unpad_h;
}

void Detector::detect(cv::Mat &img, std::vector<BoundingBox> &bboxes)
{
	if (img.empty()) return;
	int img_height = static_cast<int>(img.rows);
	int img_width = static_cast<int>(img.cols);

	cv::Mat img_rs;
	ScaleParams param;
	// resize & unscale
	resizeUnscale(img, img_rs, this->onnx_insize.height, this->onnx_insize.width, param);
	// 1.make input blob
	cv::Mat blob;
	cv::dnn::blobFromImage(img_rs, blob, 1 / 255.0f, this->onnx_insize, cv::Scalar(0, 0, 0), true, false);
	// 2.inference score & boxes.
	std::vector<std::string> outLayerNames = this->model.getUnconnectedOutLayersNames();
	std::vector<cv::Mat> outs;
	this->model.setInput(blob);
	this->model.forward(outs, outLayerNames);
	// 3.remove redundant bboxes
	bboxes = this->postProcess(img_rs, outs, param);
	

}

std::vector<BoundingBox> Detector::postProcess(cv::Mat &img, std::vector<cv::Mat> &outs,
	ScaleParams &params)
{
	std::cout << "Extract output mat from detection" << std::endl;
	cv::Mat out(outs[0].size[1], outs[0].size[2], CV_32F, outs[0].ptr<float>());

	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> indices;
	std::vector<int> classIndexList;

	float r_ = params.r;
	int dw_ = params.dw;
	int dh_ = params.dh;
	for (int r = 0; r < out.rows; r++)
	{
		float cx = out.at<float>(r, 0);
		float cy = out.at<float>(r, 1);
		float w = out.at<float>(r, 2);
		float h = out.at<float>(r, 3);
		float sc = out.at<float>(r, 4);

		cv::Mat confs = out.row(r).colRange(5, out.row(r).cols);
		confs *= sc;
		double minV, maxV;
		cv::Point minI, maxI;
		cv::minMaxLoc(confs, &minV, &maxV, &minI, &maxI);
		scores.push_back(maxV);

		// recover to orignal scale
		float x1 = ((cx - w / 2.f) - (float)dw_) / r_;
		float y1 = ((cy - h / 2.f) - (float)dh_) / r_;
		w = ((cx + w / 2.f) - (float)dw_) / r_ - x1;
		h = ((cy + h / 2.f) - (float)dh_) / r_ - y1;
		boxes.push_back(cv::Rect(x1, y1, w, h));

		indices.push_back(r);
		classIndexList.push_back(maxI.x);
	}

	std::cout << "Do NMS in " << (int)boxes.size() << "  boxes" << std::endl;
	cv::dnn::NMSBoxes(boxes, scores, this->conf_threshold, this->nms_threshold, indices);
	std::cout << "After NMS " << (int)indices.size() << "boxes keeped" << std::endl;

	std::vector<BoundingBox> res;
	for (int i = 0; i < indices.size(); i++)
	{
		BoundingBox bbox;
		bbox.box = boxes[indices[i]];
		bbox.classId = classIndexList[indices[i]];
		bbox.conf = scores[indices[i]];
		bbox.label_text = this->class_names[bbox.classId];
		res.push_back(bbox);
	}
	return res;
}


void Detector::drawPredection(cv::Mat& image, std::vector<BoundingBox>& bboxes)
{
	if (image.empty()) return;
	for (const BoundingBox& bbox : bboxes)
	{
		int x = bbox.box.x;
		int y = bbox.box.y;
		int w = bbox.box.width;
		int h = bbox.box.height;

		int conf = (int)std::round(bbox.conf * 100);
		int classId = bbox.classId;
		std::string label = this->class_names[classId] + " 0." + std::to_string(conf);

		int baseline = 0;

		cv::Size size = cv::getTextSize(label, cv::FONT_ITALIC, 0.8, 2, &baseline);

		// 1.draw bboxes 
		cv::rectangle(image, bbox.box, cv::Scalar(0, 140, 255), 2);
		cv::Point label_rect_pt1, label_rect_pt2, label_text_org;
		// according to the w,h of label txt size, set the label txt coord(up or down)
		if (size.width < w && size.height < h)
		{
			label_rect_pt1 = cv::Point(x, y);
			label_rect_pt2 = cv::Point(x + size.width, y + size.height*1.5);
			label_text_org = cv::Point(x, y + size.height*1.2);
		}
		else
		{
			label_rect_pt1 = cv::Point(x, y - size.height*1.5);
			label_rect_pt2 = cv::Point(x + size.width, y);
			label_text_org = cv::Point(x, y - size.height*0.3);
		}
		// 2.draw label rect
		cv::rectangle(image,
			label_rect_pt1, label_rect_pt2,
			cv::Scalar(0, 140, 255), -1);
		// 3.draw label text
		cv::putText(image, label,
			label_text_org, cv::FONT_ITALIC,
			0.8, cv::Scalar(255, 255, 255), 2);
	}
	cv::imshow("rst", image);
}