#ifndef FACE_DETECTION_HPP
#define FACE_DETECTION_HPP

//#define CPU_ONLY

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace caffe;

class FaceDetection
{
public:
	FaceDetection();
	FaceDetection(const std::vector<std::string> model_file, const std::vector<std::string> train_model);
	~FaceDetection();
	
	void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles);
	void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence);
	void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence, std::vector<std::vector<cv::Point>>& alignment);

	void Preprocess(const cv::Mat& img);
	void P_Net();
	void R_Net();
	void O_Net();
	void detect_net(int i);

	void local_NMS();
	void global_NMS();

	void Predict(const cv::Mat& img, int i);
	void Predict(const std::vector<cv::Mat> imgs, int i);
	void WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels, int i);
	void WrapInputLayer(const std::vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i);
	
	// tools
	float IoU(cv::Rect rect1, cv::Rect rect2);
	float IoM(cv::Rect rect1, cv::Rect rect2);
	void resize_img();
	void GenerateBoxs(cv::Mat img);
	void BoxRegress(std::vector<cv::Rect>& bounding_box, std::vector<cv::Rect> regression_box);
	void Padding(std::vector<cv::Rect>& bounding_box, int img_w, int img_h);
	cv::Mat crop(cv::Mat img, cv::Rect& rect);

	void img_show(cv::Mat img, std::string name);
	void img_show_T(cv::Mat img, std::string name);

	//param for P, R, O, L net
	std::vector<std::shared_ptr<Net<float>>> nets_;
	std::vector<cv::Size> input_size_;
	int num_channels_;

	//variable for the image
	cv::Mat img_;
	std::vector<cv::Mat> img_resized_;
	std::vector<double> scale_;

	//variable for the output of the neural network
	std::vector<float> regression_box_temp_;
	std::vector<cv::Rect> bounding_box_;
	std::vector<float> confidence_;
	std::vector<float> confidence_temp_;
	std::vector<std::vector<cv::Point>> alignment_;
	std::vector<float> alignment_temp_;

	//paramter for the threshold
	int minSize_ = 50;
	float factor_ = 0.709;
	float threshold_[3] = { 0.7f, 0.6f, 0.6f };
	float threshold_NMS_ = 0.7;
};

#endif // FACE_DETECTION_HPP

