//
// Created by liao on 24-3-21.
//

#ifndef CH8_MYCODE_LK_DRIVER_H
#define CH8_MYCODE_LK_DRIVER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace cv;
using namespace std;

float GetPixelValue(const Mat &img, float x, float y);
void calculateOpticalFlow(Mat &img1, Mat &img2, vector<KeyPoint> &pt1, vector<KeyPoint> &pt2,vector<uchar> &success,bool init = false);
void calculateOpticalFlowMulti(Mat &img1, Mat &img2, vector<KeyPoint> &pt1, vector<KeyPoint> &pt2,vector<uchar> &success);

#endif //CH8_MYCODE_LK_DRIVER_H
