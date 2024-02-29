#include<iostream>
using namespace std;

#include<Eigen/Core>
#include<Eigen/Dense>
#include<opencv2/opencv.hpp>
using namespace Eigen;

int main()
{
    //构建一个线性分布函数y=(a*x+b)*exp(c*x)，后来发现这个函数非凸
    double a=1,b=2,c=1;
    double estimate_a=2,estimate_b=-1,estimate_c=5;

    //随机数生成器
    cv::RNG rng;
    const int N=100;//数据个数
    //存储x,y数据
    Eigen::Matrix<double,2,N> data_mat;

    double sigma=1;
    for(int i=0;i<N;i++)
    {
        double x=0.01*i;
        double y=(a*x+b)*exp(c*x)+rng.gaussian(sigma);
        //double y=exp(a*x*x+b*x+c);
        //data_mat(0,i)=x;
        data_mat(1,i)=y;
    }

    //迭代次数
    int iterNum=100;
    
    double lambda=0.01;
    double cost=0;//损失函数
    double last_cost=0;
    for(int iter=0;iter<iterNum;iter++)
    {
        cost = 0;
        Matrix3d H=Matrix3d::Zero();
        Vector3d g=Vector3d::Zero();

        for(int i=0;i<N;i++)
        {
            Vector3d J;

            double xi=data_mat(0,i);
            double yi=data_mat(1,i);
            double error=yi-(estimate_a*xi+estimate_b)*exp(estimate_c*xi);
            //double error=yi-exp(estimate_a*xi*xi+estimate_b*xi+estimate_c);

            //计算损失，即书中F(x)
            cost+=error*error;

            //计算雅可比矩阵
            J[0]=-xi*exp(estimate_c*xi);
            J[1]=-exp(estimate_c*xi);
            J[2]=-(estimate_a*xi+estimate_b)*xi*exp(estimate_c*xi);
            // J[0] = -xi*xi*exp(estimate_a*xi*xi+estimate_b*xi+estimate_c);
            // J[1] = -xi*exp(estimate_a*xi*xi+estimate_b*xi+estimate_c);
            // J[2] = -exp(estimate_a*xi*xi+estimate_b*xi+estimate_c);


            //将结果加入系数矩阵
            H += J*J.transpose();
            g -= J*error;

        }


        if(iter>0 && cost>last_cost)
        {
            //break;
        }

        Vector3d dx=H.inverse()*g;
        

        estimate_a+=dx(0);
        estimate_b+=dx(1);
        estimate_c+=dx(2);

        last_cost=cost;
    }

    cout<<"方程参数为: "<<estimate_a<<" "<<estimate_b<<" "<<estimate_c<<endl;
    cout<<cost<<endl;

    return 0;
}