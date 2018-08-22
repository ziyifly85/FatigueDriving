#include "FaceDetection.hpp"
#include "LBFRegressor.h"
#include "head.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h> 
#include <iostream>
#include "FaceRecognition.h"

using namespace std;

#define _DEBUG_INFO
//#define _USE_LBFREGRESSOR
#define _USE_DLIB
//#define _USE_DLIB_DETECT

int main(int argc, char* argv[]) {

	//the vector used to input the address of the net model
	std::vector<string> model_file = {"../model/det1.prototxt", "../model/det2.prototxt", "../model/det3.prototxt"};

	//the vector used to input the address of the net parameters
	std::vector<string> trained_file = {"../model/det1.caffemodel", "../model/det2.caffemodel", "../model/det3.caffemodel"};
	
	//use for disabling the console out
	caffe::GlobalInit(&argc, &argv);

	//Step 1: FaceDetection by MTCNN 
	FaceDetection mtcnn(model_file, trained_file);

	MouthDetection mouth("../model/lenet_test.prototxt", "../model/mouth.caffemodel");
	MouthDetection eye("../model/lenet_test_eye.prototxt", "../model/eye.caffemodel");

	cv::VideoCapture cap(0);// ("C:/Users/hzl/Pictures/Camera Roll/WIN_20180228_22_02_39_Pro.mp4");
	//cap.open(1);
	/* use for record the result video 
	VideoCapture cap("../../SuicideSquad.mp4");
	VideoWriter writer;
	writer.open("../result/SuicideSquad.mp4",CV_FOURCC('M', 'J', 'P', 'G'), 25, Size(1280,720), true);
	*/

#ifdef _USE_LBFREGRESSOR
	//Step 2: FaceAlignment by LBFRegressor
	LBFRegressor regressor;
	regressor.Load(modelPath + "LBF.model");
#endif

#ifdef _USE_DLIB
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor ShapePredictor;
	dlib::deserialize("../model/shape_predictor_68_face_landmarks.dat") >> ShapePredictor;
#endif

	cv::Mat img;
	cv::Mat dst;
	int frame_count = 0;
	int fps = 0;
	double _debug_t = 0.0, detect_time=0.0, alignment_time=0.0;
	
	int count1 = 1, count2 = 1, count3 = 1;
	char file_name[100];
	double scale_X = 1.0, scale_Y = 1.0;
	int skip = 0;
	while (cap.read(img))
	{
		/*if (skip++ % 20 != 0)
			continue;
		else
			skip = 1;*/
		if (img.cols > 640 && img.rows > 480) {
			scale_X = img.cols / 640.0;
			scale_Y = img.rows / 480.0;
			cv::resize(img, img, cv::Size(640, 480));
		}
		img.copyTo(dst);
		cv::Mat gray;
		cvtColor(img, gray, CV_BGR2GRAY);
		dlib::cv_image<dlib::bgr_pixel> cimg(img);

		_debug_t = (double)cvGetTickCount();
		std::vector<cv::Rect> rectangles;
		std::vector<float> confidences;
		std::vector<std::vector<cv::Point>> alignment;
		
		mtcnn.detection(img, rectangles, confidences, alignment);

		for (int i = 0; i < rectangles.size(); i++)
		{
			int green = confidences[i] * 255;
			int red = (1 - confidences[i]) * 255;
			cv::rectangle(img, rectangles[i], cv::Scalar(0, green, red), 1);
			for (int j = 0; j < alignment[i].size(); j++)
			{
				cv::circle(img, alignment[i][j], 1, cv::Scalar(255, 255, 0), 1);
			}
			cv::Mat temp;

#define GT(x) x>=0?x:0
			temp = dst(cv::Rect(GT(alignment[i][0].x - 40 / scale_X), GT(alignment[i][0].y - 20 / scale_Y), 30, 20));	//左眼
			cv::putText(img, ((eye.predict(temp) == 1) ? "Close" : "Open"), cvPoint(alignment[i][0].x, alignment[i][0].y), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
			//sprintf_s(file_name, "E:/拍摄视频/自动采集/左眼/%08d.jpg", count1++);
			//cv::imwrite(file_name, temp);
			temp = dst(cv::Rect(GT(alignment[i][1].x - 40 / scale_X), GT(alignment[i][1].y - 20 / scale_Y), 30, 20));	//右眼
			cv::putText(img, ((eye.predict(temp) == 1) ? "Close" : "Open"), cvPoint(alignment[i][1].x, alignment[i][1].y), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
			eye.predict(temp);
			//sprintf_s(file_name, "E:/拍摄视频/自动采集/右眼/%08d.jpg", count2++);
			//cv::imwrite(file_name, temp);
			temp = dst(cv::Rect(alignment[i][3].x, GT(alignment[i][3].y - 25 / scale_Y), GT(alignment[i][4].x - alignment[i][3].x), 80 / scale_Y));	//嘴巴
			cv::putText(img, ((mouth.predict(temp) == 0) ? "Close" : "Open") , cvPoint(alignment[i][4].x, alignment[i][4].y), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
			//sprintf_s(file_name, "E:/拍摄视频/自动采集/嘴巴/%08d.jpg", count3++);
			//cv::imwrite(file_name,temp);
			//imshow("123", temp);
		}

#ifdef  _DEBUG_INFO
		_debug_t = (double)cvGetTickCount() - _debug_t;
		printf("\r");
		detect_time = _debug_t / ((double)cvGetTickFrequency()*1000.);
		printf("detection time = %g ms\t", detect_time);
		_debug_t = (double)cvGetTickCount();
#endif		

#ifdef _USE_LBFREGRESSOR
		for (vector<Rect>::const_iterator r = rectangles.begin(); r != rectangles.end(); r++) {
			Point center;
			BoundingBox boundingbox;

			boundingbox.start_x = r->x;
			boundingbox.start_y = r->y;
			boundingbox.width = (r->width - 1);
			boundingbox.height = (r->height - 1);
			boundingbox.centroid_x = boundingbox.start_x + boundingbox.width / 2.0;
			boundingbox.centroid_y = boundingbox.start_y + boundingbox.height / 2.0;

			Mat_<double> current_shape = regressor.Predict(gray, boundingbox, 1);

			for(int i = 0; i < global_params.landmark_num; i++){
				circle(img,Point2d(current_shape(i,0),current_shape(i,1)),1,Scalar(255,255,255),-1,1,0);
			}
		}
#endif
#ifdef _USE_DLIB_DETECT
		std::vector<dlib::rectangle> dets = detector(cimg);
		cout << "Number of faces detected: " << dets.size() << endl;

		// Now we will go ask the shape_predictor to tell us the pose of
		// each face we detected.
		std::vector<dlib::full_object_detection> shapes;
		for (unsigned long j = 0; j < dets.size(); ++j)
		{
			dlib::full_object_detection shape = ShapePredictor(cimg, dets[j]);
			cout << "number of parts: " << shape.num_parts() << endl;
			cout << "pixel position of first part:  " << shape.part(0) << endl;
			cout << "pixel position of second part: " << shape.part(1) << endl;
			// You get the idea, you can get all the face part locations if
			// you want them.  Here we just store them in shapes so we can
			// put them on the screen.
			shapes.push_back(shape);
			dlib::rectangle& rec = dets[j];
			cv::rectangle(img, cv::Rect(rec.left(), rec.top(), rec.width(), rec.height()), cv::Scalar(0, 255, 0), 1);
		}
		if (!shapes.empty()) {
			for (int i = 0; i < shapes.size(); ++i) {
				for (int j = 0; j < 68; ++j) {
					cv::circle(img, cv::Point(shapes[i].part(j).x(), shapes[i].part(j).y()), 1, cv::Scalar(0, 0, 255), -1);
				}
			}
		}
#endif
#ifdef _USE_DLIB
		dlib::rectangle det;
		std::vector<dlib::full_object_detection> shapes;
		for (std::vector<cv::Rect>::const_iterator r = rectangles.begin(); r != rectangles.end(); ++r) {
			//将opencv检测到的矩形转换为dlib需要的数据结构
			det.set_left(r->x - 10);
			det.set_top(r->y + 20);
			det.set_right(r->x + r->width+10);
			det.set_bottom(r->y + r->height+10);
			shapes.push_back(ShapePredictor(cimg, det));
		}
		if (!shapes.empty()) {
			for (int i = 0; i < shapes.size(); ++i) {
				for (int j = 0; j < 68; ++j) {
					cv::circle(img, cv::Point(shapes[i].part(j).x(), shapes[i].part(j).y()), 1, cv::Scalar(0, 0, 255), -1);
				}
			}
		}
#endif
#ifdef  _DEBUG_INFO
		_debug_t = (double)cvGetTickCount() - _debug_t;
		alignment_time = _debug_t / ((double)cvGetTickFrequency()*1000.);
		printf("alignment time = %g ms", alignment_time);
#endif

		frame_count++;
		fps = (int)(1000.0 / (detect_time + alignment_time));

		cv::putText(img, "fps:" + std::to_string(fps), cvPoint(3, 13), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
		// writer.write(img);
		cv::imshow("Live", img);
		cv::waitKey(100);	
	}
	return 0;
}

//int main() {
//
//    vector<string> model_file = {
//            "../model/det1.prototxt",
//            "../model/det2.prototxt",
//            "../model/det3.prototxt"
////            "../model/det4.prototxt"
//    };
//
//    vector<string> trained_file = {
//            "../model/det1.caffemodel",
//            "../model/det2.caffemodel",
//            "../model/det3.caffemodel"
////            "../model/det4.caffemodel"
//    };
//
//    MTCNN mtcnn(model_file, trained_file);
//
//    vector<Rect> rectangles;
//    string img_path = "../result/trump.jpg";
//    Mat img = imread(img_path);
//
//    mtcnn.detection(img, rectangles);
//
//    std::cout << "Hello, World!" << std::endl;
//    return 0;
//}
