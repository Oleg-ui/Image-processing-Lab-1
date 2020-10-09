#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
#include <algorithm>
#include <ctime>

// use c++ 17

using namespace cv;
using namespace std;

double comparator(double val)
{
	if (val < 0.0)
		return 0.0;
	if (val > 255.0)
		return 255.0;
	return val;
}

double MSE(Mat & img1, Mat & img2)
{
	double MSE = 0;
	int height = img1.rows;
	int width = img1.cols;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			MSE += pow((img1.at<double>(i, j) - img2.at<double>(i, j)),2);

	MSE /= height * width;

	return MSE;
}

double PSNR(Mat & img_src, Mat & img_compressed)
{
	Mat s1;
	absdiff(img_src, img_compressed, s1);   
	s1.convertTo(s1, CV_32F);  
	s1 = s1.mul(s1);           

	Scalar s = sum(s1);        

	double sse = s.val[0] + s.val[1] + s.val[2]; 

	if (sse <= 1e-10)
		return 0;
	else
	{
		double mse = sse / (double)(img_src.channels() * img_src.total());
		double psnr = 10.0*log10((255 * 255) / mse);
		return psnr;
	}
}

void rgb2hsv(double r, double g, double b, Mat& pic, int i, int j);

void RGB2HSV(Mat& RGB, Mat& HSV)
{
	for (int i = 0; i < RGB.rows; i++)
	{
		for (int j = 0; j < RGB.cols; j++)
		{
			rgb2hsv(RGB.at<cv::Vec3b>(i, j)[2], RGB.at<cv::Vec3b>(i, j)[1], RGB.at<cv::Vec3b>(i, j)[0], HSV, i, j);
		}
	}
}

void rgb2hsv(double r, double g, double b, Mat& pic, int i, int j)
{
	double R = r / 255.0, G = g / 255.0, B = b / 255.0;
	double h = 0.0, s = 0.0, v = 0.0;
	v = max(max(r, g), b);
	double t = min(min(r, g), b);
	double delta = v - t;
	if (v != 0.0)
		s = 1 - (t / v);
	else
		s = 0.0;
	if (s == 0.0)
		h = 0.0;
	else
	{
		if (r == v)
			if (g < b)
				h = 60.0 * ((g - b) / delta) + 360.0;
			else
				h = 60.0 * ((g - b) / delta) + 0.0;
		else
			if (g == v)
				h = 60.0 * ((g - b) / delta) + 120.0;
			else
				if (b == v)
					h = 60.0 * ((g - b) / delta) + 240.0;;
		if (h < 0.0)
			h += 360.0;
	}
	pic.at<cv::Vec3b>(i, j)[0] = comparator(h / 360 * 255);
	pic.at<cv::Vec3b>(i, j)[1] = comparator(s * 255.0);
	pic.at<cv::Vec3b>(i, j)[2] = comparator(v);
}

void hsv2rgb(double h, double s, double v, Mat& pic, int i, int j);

void GrayWorld(Mat & RGB, Mat & GRAY);

void HSV2RGB(Mat& HSV, Mat& RGB)
{
	for (int i = 0; i < HSV.rows; i++)
	{
		for (int j = 0; j < HSV.cols; j++)
		{
			hsv2rgb(HSV.at<cv::Vec3b>(i, j)[0], HSV.at<cv::Vec3b>(i, j)[1], HSV.at<cv::Vec3b>(i, j)[2], RGB, i, j);
		}
	}
}

void hsv2rgb(double h, double s, double v, Mat& pic, int i, int j)
{
	int H = h / 255 * 360, S = s / 255 * 100, V = v / 255 * 100;
	double r = 0.0, g = 0.0, b = 0.0;
	double Vmin = 0.0, Vinc = 0.0, Vdec = 0.0, a = 0;

	H = (H / 60) % 6;
	Vmin = ((100 - S)*V) / 100;
	a = (V - Vmin) * ((H % 60) / 60);
	Vinc = Vmin + a;
	Vdec = V - a;

	switch (H)
	{
	case 0:
		r = V;
		g = Vinc;
		b = Vmin;
		break;
	case 1:
		r = Vdec;
		g = V;
		b = Vmin;
		break;
	case 2:
		r = Vmin;
		g = V;
		b = Vinc;
		break;
	case 3:
		r = Vmin;
		g = Vdec;
		b = V;
		break;
	case 4:
		r = Vinc;
		g = Vmin;
		b = V;
		break;
	case 5:
		r = V;
		g = Vmin;
		b = Vdec;
		break;
	}
	pic.at<cv::Vec3b>(i, j)[0] = comparator(r * 2.55);
	pic.at<cv::Vec3b>(i, j)[1] = comparator(g * 2.55);
	pic.at<cv::Vec3b>(i, j)[2] = comparator(b * 2.55);
}

void GrayWorld(Mat& RGB, Mat& GRAY)
{
	int R, G, B, Gray;
	GRAY = RGB.clone();
	for (int i = 0; i < RGB.cols; i++)
		for (int j = 0; j < RGB.rows; j++)
		{
			R = RGB.at<cv::Vec3b>(i, j)[2];
			G = RGB.at<cv::Vec3b>(i, j)[1];
			B = RGB.at<cv::Vec3b>(i, j)[0];
			Gray = comparator(0.2952*R + 0.5547*G + 0.148*B);
			GRAY.at<cv::Vec3b>(i, j)[0] = Gray;
			GRAY.at<cv::Vec3b>(i, j)[1] = Gray;
			GRAY.at<cv::Vec3b>(i, j)[2] = Gray;
		}
}

void inc_brightness(Mat& pic)
{
	for (int i = 0; i < pic.rows; i++)
		for (int j = 0; j < pic.cols; j++)
			for (int k = 0; k < pic.channels(); k++)
				pic.at<cv::Vec3b>(i, j)[k] = comparator(pic.at<cv::Vec3b>(i, j)[k] + 30);
}

int main()
{

	//Task 1
	cout << "Task 1: " << endl;
	Mat src1, src2;

	src1 = imread("1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Original image", src1);
	src2 = imread("2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Ñompressed  image", src2);

	double psnr_val1 = PSNR(src1, src2);
	cout << "PSNR : " << psnr_val1 << endl << endl;
	//

	//Task 2
	cout << "Task 2: " << endl;
	Mat src3 = imread("1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat src4 = imread("1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat gray_src3;
	Mat gray_src4;

	unsigned int start_time0 = clock();
	GrayWorld(src3, gray_src3);
	unsigned int end_time0 = clock();
	unsigned int search_time0 = end_time0 - start_time0;
	imshow("Gray image", gray_src3);
	cout << "Working time of our filter: " << search_time0 << endl;

	unsigned int start_time01 = clock();
	cvtColor(src4, gray_src4, CV_BGR2GRAY);
	unsigned int end_time01 = clock();
	unsigned int search_time01 = end_time01 - start_time01;
	imshow("Gray image OPENCV", gray_src4);
	cout << "Working time of OPENCV filter: " << search_time01 << endl << endl;
	//

	//Task 3
	cout << "Task 3: " << endl;
	Mat RGB_1 = imread("1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat HSV = RGB_1.clone();

	unsigned int start_time02 = clock();
	RGB2HSV(RGB_1, HSV);
	unsigned int end_time02 = clock();
	unsigned int search_time02 = end_time02 - start_time02;
	imshow("RGB->HSV image", HSV);
	cout << "Working time of our conversion: " << search_time02 << endl;

	Mat src5 = imread("1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat result;

	unsigned int start_time03 = clock();
	cvtColor(src5, result, CV_BGR2HSV);
	unsigned int end_time03 = clock();
	unsigned int search_time03 = end_time03 - start_time03;
	imshow("RGB->HSV OPENCV", result);
	cout << "Working time of OPENCV conversion: " << search_time03 << endl << endl;

	double psnr_val3 = PSNR(HSV, result);
	cout << "PSNR HSV: " << psnr_val3 << endl;

	Mat brightness = src5.clone();
	unsigned int start_time1 = clock();
	inc_brightness(brightness);
	unsigned int end_time1 = clock();
	imshow("Increase brightness", brightness);
	unsigned int search_time1 = end_time1 - start_time1;
	cout << "working time of our function: " << search_time1 << endl;

	Mat src6 = imread("1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat img_higher_brightness;
	unsigned int start_time2 = clock();
	src6.convertTo(img_higher_brightness, -1, 1, 30);
	unsigned int end_time2 = clock();
	imshow("Increase brightness OPENCV", img_higher_brightness);
	unsigned int search_time2 = end_time2 - start_time2;
	cout << "working time of OpenCV function: " << search_time2 << endl;

	double psnr_val4 = PSNR(brightness, img_higher_brightness);
	cout << "PSNR HSV: " << psnr_val4 << endl;

	//

	waitKey(0);
	return 0;
}