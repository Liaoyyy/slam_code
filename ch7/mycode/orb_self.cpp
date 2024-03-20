#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

using namespace std;
using namespace cv;

string first_file="/home/liao/桌面/Material/slam14/slam_code/ch7/mycode/1.png";
string second_file="/home/liao/桌面/Material/slam14/slam_code/ch7/mycode/2.png";
string first_file_depth="/home/liao/桌面/Material/slam14/slam_code/ch7/mycode//1_depth.png";
string second_file_depth="/home/liao/桌面/Material/slam14/slam_code/ch7/mycode//2_depth.png";

void find_feature_matches(Mat &image1, Mat &image2, 
vector<KeyPoint> &key1, vector<KeyPoint> &key2, vector<DMatch> &matches);

//将像素坐标系坐标转到相机坐标系下归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K); 

int main()
{
    Mat r,t; //旋转矩阵与平移向量
    vector<KeyPoint> keyPoints1,keyPoints2;
    vector<DMatch> matches;
    auto image1 = imread(first_file,CV_LOAD_IMAGE_COLOR);
    auto image2 = imread(second_file,CV_LOAD_IMAGE_COLOR);
    auto image1_depth = imread(first_file_depth,CV_LOAD_IMAGE_UNCHANGED);
    assert(image1.data && image2.data && "can not load images!");

    //相机内参矩阵
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    find_feature_matches(image1,image2,keyPoints1,keyPoints2,matches);

    vector<Point3d> p3;
    vector<Point2d> p2;
    //匹配3D点与2D点
    for(auto m:matches)
    {
        //深度图为16位无符号数
        unsigned short d=image1_depth.ptr<unsigned short>(keyPoints1[m.queryIdx].pt.y)[int(keyPoints1[m.queryIdx].pt.x)];
        //上一行代码解读，ptr指向行的指针，<unsigned short>声明一个像素点大小为16位两个字节(即一个unsigned short长度),该指针每移动一个unsigned short长度即移动一个像素点
        //keypoints为一个vector,m.queryIdx返回一对匹配点的索引点值，即keypoints中匹配点的位置索引，进而访问KeyPoint类中对应的pt(对应图像中的像素位置),.y与.x返回对应x y坐标
        if (d == 0)   // 深度信息有误
            continue;
        float dd = (float)d / 5000.0f;//深度信息
        Point2d p2temp = pixel2cam(keyPoints1[m.queryIdx].pt,K);
        p3.emplace_back(p2temp.x*dd,p2temp.y*dd,dd);
        p2.push_back(keyPoints2[m.trainIdx].pt);
    }

    cout << "3d-2d pairs: " << p3.size() << endl;

    //第四个参数为畸变系数,r为旋转向量，需要使用罗德里格斯公式转化为旋转矩阵
    Mat R;
    solvePnP(p3,p2,K,Mat(),r,t);
    Rodrigues(r,R);

    cout<< "R = " << R << endl;
    cout<< "t = " << t << endl;
    return 0;
}

void find_feature_matches(Mat &image1, Mat &image2, 
vector<KeyPoint> &key1, vector<KeyPoint> &key2, vector<DMatch> &matches)
{
    Mat descriptors_1,descriptors_2;
    Ptr<ORB> test = ORB::create(); //计算关键点与描述子
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming"); //使用汉明距离进行匹配

    test->detect(image1,key1);
    test->detect(image2,key2);
    test->compute(image1,key1,descriptors_1);
    test->compute(image2,key2,descriptors_2);

    vector<DMatch> all_matches;
    matcher->match(descriptors_1,descriptors_2,all_matches);

    Mat matchImg;

    float min_dist=1000;
    for(auto m:all_matches)
    {
        if(m.distance<min_dist)
        {
            min_dist=m.distance;
        }
    }

    cout<<"min distance: " << min_dist << endl;
    cout<<"total matches: "<<all_matches.size()<<endl;
    for(auto & m : all_matches)
    {
        if(m.distance<=max(2*min_dist,30.0f))
        {
            matches.push_back(m);
        }
    }
}

Point2d pixel2cam(const Point2d &p, const Mat &K)
{
    Point2d loc;
    loc.x=(p.x-K.at<double>(0,2))/K.at<double>(0,0);
    loc.y=(p.y-K.at<double>(1,2))/K.at<double>(1,1);
    return loc;
}