#include <iostream>
using namespace std;

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>

string left_file="./steroVision/left.png";
string right_file="./steroVision/right.png";

int main()
{
    //相机内参
    double fx=718.856,fy=718.856,cx=607.1928,cy=185.2157;
    //基线
    double b=0.573;

    cv::Mat left=cv::imread(left_file);
    cv::Mat right=cv::imread(right_file);

    //为
    return 0;
}