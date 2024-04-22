//
// Created by liao on 24-3-21.
//

#include "LK_driver.h"

//通过B样条返回(x,y)点的灰度值
float GetPixelValue(const Mat &img, float x, float y)
{
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;

    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);

    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
           + xx * (1 - yy) * img.at<uchar>(y, x_a1)
           + (1 - xx) * yy * img.at<uchar>(y_a1, x)
           + xx * yy * img.at<uchar>(y_a1, x_a1);
}

//计算单层光流计匹配点
void calculateOpticalFlow(Mat &img1, Mat &img2, vector<KeyPoint> &pt1, vector<KeyPoint> &pt2,vector<uchar> &success,bool init)
{
    auto size = pt1.size();
    pt2.resize(size);
    success.resize(size);

    int iterations = 10;

    //考虑特征点附近一个小窗口内的光流变化情况
    int windows_size = 4;

    bool suc=true;

    //对每个特征点分别迭代求dx dy
    for(auto i=0;i<size;i++)
    {
        //保存从img1到img2的点运动的增量
        double dx=0,dy=0;

        //提取特征点坐标
        float x=pt1[i].pt.x;
        float y=pt1[i].pt.y;

        if(init)
        {
            dx = pt2[i].pt.x - x;
            dy = pt2[i].pt.y - y;
        }

        for(auto iter=0;iter<=iterations;iter++)
        {
            //雅可比矩阵和海塞矩阵
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            Eigen::Matrix2d Hessian = Eigen::Matrix2d::Zero();

            for(int step_x=-windows_size;step_x<windows_size;step_x++)
            {
                for(int step_y=-windows_size;step_y<windows_size;step_y++)
                {
                    Eigen::Vector2d Jacobian = Eigen::Vector2d::Zero();
                    float error = GetPixelValue(img1,x+step_x,y+step_y)- GetPixelValue(img2,x+dx+step_x,y+dy+step_y);
                    Jacobian(0) = -0.5*(GetPixelValue(img1,x+step_x+1,y+step_y)-GetPixelValue(img1,x+step_x-1,y+step_y));
                    Jacobian(1) = -0.5*(GetPixelValue(img1,x+step_x,y+step_y+1)-GetPixelValue(img1,x+step_x,y+step_y-1));

                    Hessian += Jacobian * Jacobian.transpose();
                    b += -Jacobian * error;
                }
            }

            auto delta = Hessian.ldlt().solve(b);
            if(isnan(delta(0)))
            {
                std::cout<<" result is nan!"<<endl;
                suc=false;
                break;
            }

            dx+=delta(0);
            dy+=delta(1);
        }

        success[i]=suc;
        pt2[i].pt=pt1[i].pt+Point2f(dx,dy);
    }
}

//计算多层光流计匹配点
void calculateOpticalFlowMulti(Mat &img1, Mat &img2, vector<KeyPoint> &pt1, vector<KeyPoint> &pt2,vector<uchar> &success)
{
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0,0.5,0.25,0.125};

    vector<Mat> pyr1,pyr2;
    for(int i=0;i<pyramids;i++)
    {
        //底层图片直接存入
        if(i==0)
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }
        else
        {
            Mat resize_img1,resize_img2;
            resize(img1,resize_img1,Size(pyr1[i-1].cols * pyramid_scale,pyr1[i-1].rows*pyramid_scale));
            resize(img2,resize_img2,Size(pyr2[i-1].cols * pyramid_scale,pyr2[i-1].rows*pyramid_scale));
            pyr1.push_back(resize_img1);
            pyr2.push_back(resize_img2);
        }
    }

    //提取底层特征点对应金字塔顶层伸缩后位置的特征点
    vector<KeyPoint> kp1_pyr,kp2_pyr;
    for(auto &kp:pt1)
    {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids-1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }


    //开始迭代，从顶层到底层
    for(int level = pyramids - 1;level >=0; level--)
    {
        success.clear();
        calculateOpticalFlow(pyr1[level],pyr2[level],kp1_pyr,kp2_pyr,success, true);

        //放大一下特征点位置
        if(level > 0)
        {
            for(auto &kp:kp1_pyr)
                kp.pt /= pyramid_scale;
            for(auto &kp:kp2_pyr)
                kp.pt /= pyramid_scale;
        }
    }

    //存入结果
    for(auto &kp:kp2_pyr)
        pt2.push_back(kp);
}