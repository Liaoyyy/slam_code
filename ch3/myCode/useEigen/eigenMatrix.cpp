#include  <iostream>
using namespace std;

#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>

#define MATRIX_SIZE 50
using namespace Eigen;

int main()
{
    Eigen::Matrix<float,3,3> mat33;
    Eigen::Vector3d v_3d;
    VectorXd v_xd(5);
    v_xd[0]=1;
    v_xd(1)=3;
    cout<<v_xd<<endl;

    Matrix<float,3,1> vd_3d(1,2,3);
    cout<<vd_3d.transpose()<<endl;



    return 0;
}

