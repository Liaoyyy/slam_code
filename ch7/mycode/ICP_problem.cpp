//
// Created by liao on 24-3-20.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

using namespace std;
using namespace cv;

string first_file="/home/liao/桌面/Material/slam14/slam_code/ch7/mycode/1.png";
string second_file="/home/liao/桌面/Material/slam14/slam_code/ch7/mycode/2.png";
string first_file_depth="/home/liao/桌面/Material/slam14/slam_code/ch7/mycode//1_depth.png";
string second_file_depth="/home/liao/桌面/Material/slam14/slam_code/ch7/mycode//2_depth.png";

void find_feature_matches(Mat &image1, Mat &image2,
                          vector<KeyPoint> &key1, vector<KeyPoint> &key2, vector<DMatch> &matches);

//位姿估计
//void posEstimation(vector<Point3d> &pt1, vector<Point3d> &pt2, Mat &R, Mat &t);
void pose_estimation_3d3d(const vector<Point3f> &pts1,
                          const vector<Point3f> &pts2,
                          Mat &R, Mat &t);

//将像素坐标系坐标转到相机坐标系下归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);
int main()
{
    Mat img1 = imread(first_file,CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(second_file,CV_LOAD_IMAGE_COLOR);
    Mat img1_depth = imread(first_file_depth,CV_LOAD_IMAGE_UNCHANGED);
    Mat img2_depth = imread(second_file_depth,CV_LOAD_IMAGE_UNCHANGED);

    assert(img1.data && img2.data && img1_depth.data && img2_depth.data);

    vector<KeyPoint> keyPoints1,keyPoints2;
    vector<DMatch> matches;
    find_feature_matches(img1,img2,keyPoints1,keyPoints2,matches);

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    //获取匹配点对应3D点
    vector<Point3f> point3d_1;
    vector<Point3f> point3d_2;
    for(auto m:matches)
    {
        ushort depth_1 = img1_depth.ptr<unsigned short>(int(keyPoints1[m.queryIdx].pt.y))[int(keyPoints1[m.queryIdx].pt.x)];
        ushort depth_2 = img2_depth.ptr<unsigned short>(int(keyPoints2[m.trainIdx].pt.y))[int(keyPoints2[m.trainIdx].pt.x)];

        if(depth_1==0 || depth_2 ==0)
        {
            continue;
        }
        float dp_1 = (float)depth_1 / 5000.0f;
        float dp_2 = (float)depth_2 / 5000.0f;
        Point2d camPoint_1 = pixel2cam(keyPoints1[m.queryIdx].pt,K);
        Point2d camPoint_2 = pixel2cam(keyPoints2[m.trainIdx].pt,K);

        point3d_1.push_back(Point3f(camPoint_1.x * dp_1, camPoint_1.y * dp_1, dp_1));
        point3d_2.push_back(Point3f(camPoint_2.x * dp_2, camPoint_2.y * dp_2, dp_2));
    }
    cout << "3d-3d pairs: " << point3d_2.size() << endl;

    Mat R,t;
    pose_estimation_3d3d(point3d_1,point3d_2,R,t);
    cout<<"R = "<<R<<endl;
    cout<<"t = "<<t<<endl;

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

//void posEstimation(vector<Point3d> &pt1, vector<Point3d> &pt2, Mat &R, Mat &t)
//{
//    auto N = pt1.size();
//    Point3d p(0,0,0),p_hat(0,0,0);
//    for(int i=0;i<N;i++)
//    {
//        p+=pt1[i];
//        p_hat+=pt2[i];
//    }
//    p = Point3d(p.x/(double)N,p.y/(double)N,p.z/(double)N);
//    p_hat = Point3d(p_hat.x/(double)N,p_hat.y/(double)N,p_hat.z/(double)N);
//
//    //去中心化后点
//    vector<Point3f> qt1(N), qt2(N);
//    for(int i=0;i<N;i++)
//    {
//        qt1[i]=pt1[i]-p;
//        qt2[i]=pt2[i]-p_hat;
//    }
//
//    Eigen::Matrix3d W= Eigen::Matrix3d::Zero();
//    for(int i =0;i<N;i++)
//    {
//        W += Eigen::Vector3d(qt1[i].x,qt1[i].y,qt1[i].z) * Eigen::Vector3d(qt2[i].x,qt2[i].y,qt2[i].z).transpose();
//    }
//    cout<<"W="<<W<<endl;
//
//    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W,Eigen::ComputeFullU | Eigen::ComputeFullV);
//    Eigen::Matrix3d U=svd.matrixU();
//    Eigen::Matrix3d V=svd.matrixV();
//    cout<<"U= "<<U<<endl;
//    cout<<"V= "<<V<<endl;
//
//    Eigen::Matrix3d R_ = U * (V.transpose());
//    if(R_.determinant()<0)
//    {
//        R_ = -R_;
//    }
//
//    Eigen::Vector3d t_ = Eigen::Vector3d(p.x, p.y, p.z) - R_ * Eigen::Vector3d(p_hat.x, p_hat.y, p_hat.z);
//
//    R = (Mat_<double>(3,3) << R_(0,0),R_(0,1),R_(0,2),
//                                        R_(1,0),R_(1,1),R_(1,2),
//                                        R_(2,0),R_(2,1),R_(2,2));
//
//    t=(Mat_<double>(3,1) << t_(0,0),t_(1,0),t_(2,0));
//}

void pose_estimation_3d3d(const vector<Point3f> &pts1,
                          const vector<Point3f> &pts2,
                          Mat &R, Mat &t) {
    Point3f p1, p2;     // center of mass
    int N = pts1.size();
    for (int i = 0; i < N; i++) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3f(Vec3f(p1) / N);
    p2 = Point3f(Vec3f(p2) / N);
    vector<Point3f> q1(N), q2(N); // remove the center
    for (int i = 0; i < N; i++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++) {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    cout << "W=" << W << endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    cout << "U=" << U << endl;
    cout << "V=" << V << endl;

    Eigen::Matrix3d R_ = U * (V.transpose());
    if (R_.determinant() < 0) {
        R_ = -R_;
    }
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
    R = (Mat_<double>(3, 3) <<
                            R_(0, 0), R_(0, 1), R_(0, 2),
            R_(1, 0), R_(1, 1), R_(1, 2),
            R_(2, 0), R_(2, 1), R_(2, 2)
    );
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}