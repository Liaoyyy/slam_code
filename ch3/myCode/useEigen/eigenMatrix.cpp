#include  <iostream>
using namespace std;

#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>

#define MATRIX_SIZE 5
using namespace Eigen;

int main()
{
    // Eigen::Matrix<float,3,3> mat33;
    // Eigen::Vector3d v_3d;
    // VectorXd v_xd(5);
    // v_xd[0]=1;
    // v_xd(1)=3;

    // Matrix<float,3,1> vd_3d(1,2,3);

    // Matrix<float,2,3> matrix_23;
    // matrix_23<< 1,2,3,4,5,6;

    // v_3d << 3,2,1;
    // vd_3d << 4,5,6;

    // //矩阵乘法
    // Matrix<double,2,1> result=matrix_23.cast<double>() * v_3d;

    // //随机矩阵
    // Matrix<float,2,2> random_mat =Matrix<float,2,2>::Random();
    // Matrix<float,5,5> zero_mat =Matrix<float,5,5>::Zero();

    // cout<< random_mat<< endl;
    // cout << endl;
    // cout << "sum:"<<random_mat.sum()<<endl;
    // cout << "trace"<< random_mat.trace()<<endl;
    // cout << "reverse:\n"<< random_mat.reverse()<<endl;
    // cout << "determinant:" << random_mat.determinant()<<endl;

    // //特征值与特征向量
    // Matrix<double,MATRIX_SIZE,MATRIX_SIZE> matrix_NN = MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE);
    // matrix_NN = matrix_NN * matrix_NN.transpose();
    // Matrix<double,MATRIX_SIZE,1> v_Nd = MatrixXd::Random(MATRIX_SIZE,1);

    // SelfAdjointEigenSolver<Matrix<double,MATRIX_SIZE,MATRIX_SIZE>> eigenSolver(matrix_NN);
    // cout<< "Eigen values = \n" << eigenSolver.eigenvalues()<<endl;
    // cout<< "Eigen vector = \n" << eigenSolver.eigenvectors()<<endl;

    Matrix3d H=Matrix3d::Random();
    Vector3d g=Vector3d::Random();

    Matrix3d test=H.inverse();
    Matrix<double,3,1> result=H.inverse()*g;


    return 0;
}

