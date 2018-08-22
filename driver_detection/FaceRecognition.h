#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>

using namespace caffe;

class MouthDetection {

public:
	MouthDetection();
	MouthDetection(const std::string model_file, const std::string train_model);
	~MouthDetection();

	bool predict(const cv::Mat& img);
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);
private:
	std::shared_ptr<Net<float>> net;
	cv::Size input_size;
	cv::Mat mean_;
	int num_channels_;
};

class FaceRecognition
{
public:
	FaceRecognition();
	~FaceRecognition();
};

