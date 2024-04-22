//
// Created by liao on 24-3-21.
//
#include <opencv2/opencv.hpp>
#include "LK_driver.h"
using namespace  std;
using namespace  cv;

string file1="/home/liao/桌面/Material/slam14/slam_code/ch8/mycode/LK1.png";
string file2="/home/liao/桌面/Material/slam14/slam_code/ch8/mycode/LK2.png";


int main()
{
    Mat img1 = imread(file1,CV_LOAD_IMAGE_UNCHANGED);
    Mat img2 = imread(file2,CV_LOAD_IMAGE_UNCHANGED);

    vector<KeyPoint> kp1,kp2;
    Ptr<GFTTDetector> detect = GFTTDetector::create(500, 0.01, 20);
    detect->detect(img1,kp1);

    //使用opencv的库
    vector<Point2f> pt1,pt2;
    for(auto &kp:kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    calcOpticalFlowPyrLK(img1,img2,pt1,pt2,status,error);

    //绘制结果
    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i])
        {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(255, 255, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }
    imshow("OpenCV result",img2_CV);

    //使用自己写的单层光流计算法
    vector<uchar> success;
    calculateOpticalFlow(img1,img2,kp1,kp2,success);

    Mat img2_CV2;
    cv::cvtColor(img2, img2_CV2, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2.size(); i++) {
        if (success[i])
        {
            cv::circle(img2_CV2, kp2[i].pt, 2, cv::Scalar(0, 255, 0), 2);
            cv::line(img2_CV2, kp1[i].pt, kp2[i].pt, cv::Scalar(0, 250, 0),1);
        }
    }
    imshow("my single layer result",img2_CV2);

    //使用自己写的多层光流计算法
    vector<uchar> success2;
    calculateOpticalFlowMulti(img1,img2,kp1,kp2,success2);

    Mat img2_CV3;
    cv::cvtColor(img2, img2_CV3, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2.size(); i++) {
        if (success2[i])
        {
            cv::circle(img2_CV3, kp2[i].pt, 2, cv::Scalar(0, 255, 0), 2);
            cv::line(img2_CV3, kp1[i].pt, kp2[i].pt, cv::Scalar(0, 250, 0),1);
        }
    }
    imshow("my multi layer result",img2_CV3);


    waitKey(0);

    return 0;
}