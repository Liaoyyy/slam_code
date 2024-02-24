#include <iostream>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace Eigen;

int main()
{
    Matrix3d rotation_matrix = Matrix3d::Identity();

    AngleAxisd rotation_vector(M_PI/4, Vector3d(0,0,1));
    cout.precision(3);
    rotation_matrix = rotation_vector.matrix();
    //cout<<rotation_matrix<<endl;

    Vector3d loc(1,0,0);
    Vector3d rotate_loc=rotation_matrix*loc;
    Vector3d rotate_loc2= rotation_vector*loc;
    // cout<<rotate_loc<<"\n"<<endl;
    // cout<<rotate_loc2<<endl;
    

    //齐次变换矩阵T4x4
    Isometry3d T=Isometry3d::Identity();
    T.rotate(rotation_matrix);
    T.pretranslate(Vector3d(1,3,4));//平移(1,3,4)
    Isometry3d T2(rotation_vector);

    // cout<<T.matrix()<<endl;
    // cout<<endl;
    // cout<<T2.matrix()<<endl;

    //四元数
    Quaterniond q(rotation_vector);
    cout << "quaternion from rotation vector = "<<q.coeffs().transpose()<<endl; //coeffs为(x,y,z,w)四元数
    rotate_loc= q*loc;
    cout << "(1,0,0) after rotation = "<< rotate_loc.transpose()<<endl;
    cout << q.matrix()<<endl;
    return 0;

}