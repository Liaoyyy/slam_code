#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace cv;
using namespace std;

//进行ORB特征点提取并匹配特征点
void findMatchPoints(
        Mat &img1,Mat &img2, vector<DMatch> &matches,
        vector<KeyPoint> &keyPoint1,vector<KeyPoint> &keyPoint2);

//将像素坐标转化为相机坐标系下归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

//使用Ceres库求解PnP
struct ICP_COST
{
    ICP_COST(Point3d _P1, Point3d _P2, Mat _K): P1(_P1),P2(_P2),K(_K){};

    //t为相机位姿向量，为1x6的矩阵，前三位为旋转向量，后三位为平移向量
    template<typename T>
    bool operator()(const T* const r,const T* const t, T *residual) const
    {

        //世界系坐标转化为相机坐标系下归一化坐标
        T Point_1[3],normP_1[3],Point_2[3];
        Point_1[0]=T(P1.x);
        Point_1[1]=T(P1.y);
        Point_1[2]=T(P1.z);

        Point_2[0]=T(P2.x);
        Point_2[1]=T(P2.y);
        Point_2[2]=T(P2.z);

        ceres::AngleAxisRotatePoint(r,Point_1,normP_1);
        normP_1[0]+=t[0];
        normP_1[1]+=t[1];
        normP_1[2]+=t[2];

        residual[0]=T(normP_1[0]-Point_2[0]);
        residual[1]=T(normP_1[1]-Point_2[1]);
        residual[2]=T(normP_1[2]-Point_2[2]);

        return true;
    };


    const Point3d P2;
    const Point3d P1;
    const Mat K;//相机内参矩阵
};


int main()
{
    Mat img1=imread("/home/liao/桌面/Material/slam14/slam_code/ch7/mycode/1.png",CV_LOAD_IMAGE_UNCHANGED);
    Mat img2=imread("/home/liao/桌面/Material/slam14/slam_code/ch7/mycode/2.png",CV_LOAD_IMAGE_UNCHANGED);
    Mat img1_depth = imread("/home/liao/桌面/Material/slam14/slam_code/ch7/mycode/1_depth.png",CV_LOAD_IMAGE_UNCHANGED);
    Mat img2_depth = imread("/home/liao/桌面/Material/slam14/slam_code/ch7/mycode/2_depth.png",CV_LOAD_IMAGE_UNCHANGED);

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<DMatch> matches;
    vector<KeyPoint> keyPoints1,keyPoints2;
    findMatchPoints(img1,img2,matches,keyPoints1,keyPoints2);
    cout<<"There are "<< matches.size() << " pairs of matches" << endl;

    //建立图1中3D点与图2中匹配的像素坐标点 该例子中以图1为世界坐标，求解图像2的位姿到图像1的变换
    vector<Point3d> point3d_1;
    vector<Point3d> point3d_2;
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

        point3d_1.push_back(Point3d(camPoint_1.x * dp_1, camPoint_1.y * dp_1, dp_1));
        point3d_2.push_back(Point3d(camPoint_2.x * dp_2, camPoint_2.y * dp_2, dp_2));
    }
    cout<<"ICP problem pairs: "<<point3d_1.size()<<endl;

    //使用Ceres库求解
    double targetR[3]{0,0,0};
    double targetT[3]{0,0,0};
    cout<<"Use Ceres to solve PnP problem:"<<endl;
    ceres::Problem problem;
    for(int i=0; i<point3d_1.size();i++)
    {
        ceres::CostFunction *costFuc= new ceres::AutoDiffCostFunction<ICP_COST,3,3,3>(new ICP_COST(point3d_1[i],point3d_2[i],K));
        problem.AddResidualBlock(costFuc,nullptr,targetR,targetT);
    }

    //配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;//增量方程求解
    options.minimizer_progress_to_stdout = true; //输出求解过程
    //options.function_tolerance= 0.001;

    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem,&summary);
    Mat rCeres(3,1, CV_64F,targetR);
    Mat tCeres(3,1, CV_64F,targetT);
    Mat RCeres;
    Rodrigues(rCeres,RCeres);
    cout<< "R = " << RCeres << endl;
    cout<<"t = " << tCeres << endl;

    cout<<summary.BriefReport()<<endl;


}

void findMatchPoints(
        Mat &img1,Mat &img2, vector<DMatch> &matches,
        vector<KeyPoint> &keyPoint1,vector<KeyPoint> &keyPoint2)
{
    Ptr<ORB> detect = ORB::create();

    Mat descriptor1,descriptor2;
    detect->detect(img1,keyPoint1);
    detect->detect(img2,keyPoint2);
    detect->compute(img1,keyPoint1,descriptor1);
    detect->compute(img2,keyPoint2,descriptor2);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<DMatch> match;
    matcher->match(descriptor1,descriptor2,match);

    for(auto m:match)
    {
        if(m.distance<=30)
        {
            matches.push_back(m);
        }
    }
}


Point2d pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2d(
            (p.x-K.at<double>(0,2))/K.at<double>(0,0),
            (p.y-K.at<double>(1,2))/K.at<double>(1,1)
    );
}