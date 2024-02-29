#include <iostream>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
using namespace std;


//定义代价函数的计算模型
struct CURVE_FITTING_COST
{
    CURVE_FITTING_COST(double x, double y):  _x(x),_y(y) {} //构造函数

    //重载操作符()
    template<typename T>
    bool operator() (const T *const abc, T *residual) const//传入参数为指向常量T的常量指针和非常量版本的指针residual，该函数为常量成员函数
    {
        //T()执行了一次强制类型转换为T类型
        residual[0]=T(_y)- ceres::exp(abc[0]*T(_x)*T(_x)+abc[1]*T(_x)+abc[2]);
        return true;

    }

private:
    const double _x,_y;

};

int main()
{
    double ar=1.0,br=2.0,cr=1.0;
    double ae=2.0,be=-1.0,ce=5.0;

    int N=100;
    double w_sigma=1.0;
    double inv_sigma=1.0/w_sigma;
    cv::RNG rng;

    vector<double> x_data,y_data;
    for(int i=0;i<N;i++)
    {
        double x= i*0.01;
        x_data.push_back(x);
        y_data.push_back(exp(ar*x*x+br*x+cr));
    }

    double abc[3]={ae,be,ce};

    //构造最小二乘问题
    ceres::Problem problem;
    for(int i=0;i<N;i++)
    {
        //添加误差项
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST,1,3>(
                new CURVE_FITTING_COST(x_data[i],y_data[i])
            ),
            nullptr,
            abc
        );
    }

    //配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type=ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout=true; //输出到cout

    ceres::Solver::Summary summary; //优化信息
    ceres::Solve(options,&problem,&summary);

    cout<<summary.BriefReport()<<endl;
    cout<<"estimated a,b,c=";
    for(auto a:abc) cout<<a<<" ";
    cout<<endl;

    return 0;
}