#include<iostream>
using namespace std;

#include<Eigen/Core>
#include<Eigen/Dense>
#include<opencv2/opencv.hpp>
using namespace Eigen;

int main()
{
    //牛顿法对例题函数收敛性太差了
    double a=1,b=2;
    double estimate_a=1,estimate_b=0;

    //随机数生成器
    cv::RNG rng;
    const int N=10;//数据个数
    //存储x,y数据
    Eigen::Matrix<double,2,N> data_mat;

    double sigma=1;
    for(int i=0;i<N;i++)
    {
        double x=0.01*i;
        double y=a*x+b;
        data_mat(0,i)=x;
        data_mat(1,i)=y;
    }

    //迭代次数
    int iterNum=300;
    
    double lambda=0.05;
    double cost=0;//损失函数
    double last_cost=0;
    for(int iter=0;iter<iterNum;iter++)
    {
        cost = 0;
        Vector2d dx=Vector2d::Zero();

        for(int i=0;i<N;i++)
        {
            Vector2d J;

            double xi=data_mat(0,i);
            double yi=data_mat(1,i);
            double error = yi-estimate_a*xi-estimate_b;

            //计算损失，即书中F(x)
            cost+=error*error;

            //计算雅可比矩阵
            J[0] = -xi*error;
            J[1] = -error;


            //将结果加入系数矩阵
            dx -= J;
        }

        estimate_a+=dx(0)/N;
        estimate_b+=dx(1)/N;

        last_cost=cost;
        if(cost<0.1){
            lambda=0.0001;
        }
        else if(cost<1e-3)
        {
            lambda=1e-5;
        }
    }

    cout<<"方程参数为: "<<estimate_a<<" "<<estimate_b<<endl;
    cout<<cost<<endl;

    return 0;
}