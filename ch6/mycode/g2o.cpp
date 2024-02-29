#include<iostream>
#include<g2o/core/g2o_core_api.h>
#include<g2o/core/base_vertex.h>
#include<g2o/core/base_unary_edge.h>
#include<g2o/core/block_solver.h>
#include<g2o/core/optimization_algorithm_levenberg.h>
#include<g2o/core/optimization_algorithm_gauss_newton.h>
#include<g2o/core/optimization_algorithm_dogleg.h>
#include<g2o/solvers/dense/linear_solver_dense.h>
#include<cmath>
#include<Eigen/Core>
#include<opencv2/opencv.hpp>

using namespace std;

// 曲线模型的顶点，模版参数：优化变量维度和数据类型
class CurveFittingVertex:public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;//启用对齐内存分配

    //重置
    virtual void setToOriginImpl() override
    {
        _estimate << 0,0,0;
    }

    //更新
    virtual void oplusImpl(const double *update) override
    {
        _estimate += Eigen::Vector3d(update);
    }

    //存盘读盘
    virtual bool read(istream &in) {return true;}
    virtual bool write(ostream &out) const {return true;}
};

// 误差模板
class CurveFittingEdge : public g2o::BaseUnaryEdge<1,double,CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    CurveFittingEdge(double x):BaseUnaryEdge(),_x(x)     { }

    //计算模型误差
    virtual void computeError() override
    {
        const CurveFittingVertex *v=static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d abc=v->estimate();
        _error(0,0)=_measurement-std::exp(abc(0)*_x*_x+abc(1)*_x+abc(2));
    }

    //计算雅可比矩阵
    virtual void linearizeOplus() override
    {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d abc =v->estimate();
        double y=std::exp(abc(0)*_x*_x+abc(1)*_x+abc(2));
        _jacobianOplusXi[0]=-_x*_x*y;
        _jacobianOplusXi[1]=-_x*y;
        _jacobianOplusXi[2]=-y;
    }

    virtual bool read(istream &in) { return true;}
    virtual bool write(ostream &out) const {return true;}

private:
    double _x;//x值，y值为_measurement
};


int main()
{
    //数据生成
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


    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3,1>> BlockSolverType; //每个误差项优化变量维度为3，误差值维度为1
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; //线性求解器类型

    auto solver=new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    
    g2o::SparseOptimizer optimizer;//图模型
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    //增加顶点
    CurveFittingVertex *v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(ae,be,ce));
    v->setId(0); //设置顶点id

    optimizer.addVertex(v);

    //增加边
    for(int i=0;i<N;i++)
    {
        CurveFittingEdge *edge= new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0,v);//设置连接节点
        edge->setMeasurement(y_data[i]);
        edge->setInformation(Eigen::Matrix<double,1,1>::Identity()*1/(w_sigma*w_sigma));//信息矩阵：协方差矩阵的逆
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);//迭代10次

    //输出结果
    Eigen::Vector3d abc_estimate=v->estimate();
    cout<<abc_estimate<<endl;
    


    return 0;
}