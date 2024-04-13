#include <Eigen/Core>
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
struct PNP_COST
{
    PNP_COST(Point3d _P, Point2d _u, Mat _K): P(_P),u(_u),K(_K){};

    //t为相机位姿向量，为1x6的矩阵，前三位为旋转向量，后三位为平移向量
    template<typename T>
    bool operator()(const T* const r,const T* const t, T *residual) const
    {

        //世界系坐标转化为相机坐标系下归一化坐标
        T Point[3],normP[3];
        Point[0]=T(P.x);
        Point[1]=T(P.y);
        Point[2]=T(P.z);
        ceres::AngleAxisRotatePoint(r,Point,normP);
        normP[0]+=t[0];
        normP[1]+=t[1];
        normP[2]+=t[2];

        normP[0]=normP[0]/normP[2];
        normP[1]=normP[1]/normP[2];

        //相机系坐标转化为像素坐标系坐标
        T camP[2];
        camP[0]=T(K.at<double>(0,0))*normP[0]+T(K.at<double>(0,2));
        camP[1]=T(K.at<double>(1,1))*normP[1]+T(K.at<double>(1,2));

        residual[0]=T(u.x-camP[0]);
        residual[1]=T(u.y-camP[1]);
    
        return true;
    };


    const Point2d u;
    const Point3d P;
    const Mat K;//相机内参矩阵
};


int main()
{
    Mat img1=imread("/home/liao/桌面/Material/slam14/slam_code/ch7/mycode/1.png",CV_LOAD_IMAGE_UNCHANGED);
    Mat img2=imread("/home/liao/桌面/Material/slam14/slam_code/ch7/mycode/2.png",CV_LOAD_IMAGE_UNCHANGED);
    Mat img1_depth = imread("/home/liao/桌面/Material/slam14/slam_code/ch7/mycode/1_depth.png",CV_LOAD_IMAGE_UNCHANGED);

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<DMatch> matches;
    vector<KeyPoint> keyPoints1,keyPoints2;
    findMatchPoints(img1,img2,matches,keyPoints1,keyPoints2);
    cout<<"There are "<< matches.size() << " pairs of matches" << endl;

    //建立图1中3D点与图2中匹配的像素坐标点 该例子中以图1为世界坐标，求解图像2的位姿到图像1的变换
    vector<Point3d> point3d;
    vector<Point2d> point2d;
    for(auto m:matches)
    {
        ushort depth = img1_depth.ptr<unsigned short>(int(keyPoints1[m.queryIdx].pt.y))[int(keyPoints1[m.queryIdx].pt.x)];
        if(depth==0)
        {
            continue;
        }
        //depth = depth / 5000;
        float dp = depth / 5000.0f;
        Point2d camPoint = pixel2cam(keyPoints1[m.queryIdx].pt,K);

        point3d.push_back(Point3d(camPoint.x * dp, camPoint.y * dp, dp));
        point2d.push_back(keyPoints2[m.trainIdx].pt);
    }

    Mat r,t;
    solvePnP(point3d,point2d,K,Mat(),r,t,false);
    //solvePnP(pts_3d,pts_2d,K,Mat(),r,t,false);
    Mat R;
    Rodrigues(r,R);
    cout<< "R = " << R << endl;
    cout<<"t = " << t << endl;

    //使用Ceres库求解
    double targetR[3]{0,0,0};
    double targetT[3]{0,0,0};
    cout<<"Use Ceres to solve PnP problem:"<<endl;
    ceres::Problem problem;
     for(int i=0; i<point3d.size();i++)
     {
         ceres::CostFunction *costFuc= new ceres::AutoDiffCostFunction<PNP_COST,2,3,3>(new PNP_COST(point3d[i],point2d[i],K));
         problem.AddResidualBlock(costFuc,nullptr,targetR,targetT);
     }

     //配置求解器
     ceres::Solver::Options options;
     options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;//增量方程求解
     options.minimizer_progress_to_stdout = true; //输出求解过程

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