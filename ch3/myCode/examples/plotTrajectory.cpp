#include<iostream>
using namespace std;

#include<Eigen/Core>
using namespace Eigen;

#include<pangolin/pangolin.h>
#include<unistd.h>

string trajectory_file="./examples/trajectory.txt"; //file path

void DrawTrajectory(vector<Isometry3d,aligned_allocator<Isometry3d>>);

int main()
{
    vector<Isometry3d,aligned_allocator<Isometry3d>> poses;
    //aligned_allocator是动态内存管理器，Isometry3d是元素类型
    ifstream fin(trajectory_file);
    if(!fin)
    {
        cout << "cannot find trajectory file at " << trajectory_file << endl;
        return 1;
    }
    
    while(!fin.eof())
    {
        double time,tx,ty,tz,qx,qy,qz,qw;
        fin>>time>>tx>>ty>>tz>>qx>>qy>>qz>>qw;

        Isometry3d Twr(Quaterniond(qw,qx,qy,qz));
        Twr.pretranslate(Vector3d(tx,ty,tz));

        poses.push_back(Twr);
    }

    cout<< " read total" << poses.size() << "pose entries" << endl;

    DrawTrajectory(poses);

    return 0;
}

void DrawTrajectory(vector<Isometry3d,aligned_allocator<Isometry3d>>)
{
    //创建Pangolin界面
    pangolin::CreateWindowAndBind("Trajectory view",1024,768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,1000),
        pangolin::ModelViewLookAt(0,-0.1,-1.8,0,0,0,0.0,-1.0,0.0)
    );
}
