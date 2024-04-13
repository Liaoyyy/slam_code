//
// Created by liao on 24-3-21.
//
#include <opencv2/opencv.hpp>

using namespace  cv;
using namespace  std;

int main()
{
    auto img = imread("/home/liao/桌面/Material/slam14/slam_code/ch8/mycode/LK1.png",CV_LOAD_IMAGE_UNCHANGED);
    assert(img.data);
    Mat imgC;
    resize(img,imgC,Size(500,500),1,1);
    imshow("test",imgC);
    imshow("origin",img);
    waitKey(0);
    return 0;
}

