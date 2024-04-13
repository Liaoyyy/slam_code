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


    return 0;
}