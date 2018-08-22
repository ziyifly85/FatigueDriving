#include "FaceRecognition.h"


MouthDetection::MouthDetection() {}

MouthDetection::MouthDetection(const std::string model_file, const std::string train_model)
{
	net.reset(new Net<float>(model_file, TEST));
	net->CopyTrainedLayersFrom(train_model);
	Blob<float>* input_layer = net->input_blobs()[0];
	input_size = cv::Size(input_layer->width(), input_layer->height());
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 )
		<< "Input layer should have 3 channels.";

}
MouthDetection::~MouthDetection() {}

/* 这个其实是为了获得net_网络的输入层数据的指针，然后后面我们直接把输入图片数据拷贝到这个指针里面*/
void MouthDetection::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{
	Blob<float>* input_layer = net->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}
//图片预处理函数，包括图片缩放、归一化、3通道图片分开存储  
//对于三通道输入CNN，经过该函数返回的是std::vector<cv::Mat>因为是三通道数据，索引用了vector  
void MouthDetection::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels)
{
	/*1、通道处理，因为我们如果是Alexnet网络，那么就应该是三通道输入*/
	cv::Mat sample;
	//如果输入图片是一张彩色图片，但是CNN的输入是一张灰度图像，那么我们需要把彩色图片转换成灰度图片  
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, CV_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, CV_BGRA2GRAY);
	//如果输入图片是灰度图片，或者是4通道图片，而CNN的输入要求是彩色图片，因此我们也需要把它转化成三通道彩色图片  
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, CV_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, CV_GRAY2BGR);
	else
		sample = img;
	/*2、缩放处理，因为我们输入的一张图片如果是任意大小的图片，那么我们就应该把它缩放到227×227*/
	cv::Mat sample_resized;
	if (sample.size() != input_size)
		cv::resize(sample, sample_resized, input_size);
	else
		sample_resized = sample;
	/*3、数据类型处理，因为我们的图片是uchar类型，我们需要把数据转换成float类型*/
	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);
	//均值归一化，为什么没有大小归一化？  
	cv::Mat sample_normalized;
	//cv::subtract(sample_float, mean_, sample_normalized);
	sample_normalized = sample_float * 0.00390625;

	/* 3通道数据分开存储 */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net->input_blobs()[0]->cpu_data()) << "Input channels are not wrapping the input layer of the network.";
}

bool MouthDetection::predict(const cv::Mat& img)
{
	Blob<float>* input_layer = net->input_blobs()[0];
	input_layer->Reshape(1, num_channels_, input_size.height, input_size.width);

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net->Forward();

	//confidence
	Blob<float>* output_layer = net->output_blobs()[0];
	int count = output_layer->count(); //the channel of confidence is two
	const float* confidence_begin = output_layer->cpu_data();
	//std::cout << confidence_begin[0] << " "<< confidence_begin[1] << std::endl;
	if (confidence_begin[0] > confidence_begin[1])
		return true;
	else
		return false;
}

FaceRecognition::FaceRecognition()
{
}


FaceRecognition::~FaceRecognition()
{
}
