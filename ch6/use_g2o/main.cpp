#include <iostream>
#include <random>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>

//顶点
class Vertex : public g2o::BaseVertex<3, Eigen::Vector3d> // 维度 类型
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW //类中使用Eigen成员 需要加入此句 字节对齐

    //_estimate 代表 Vector3d对应的数据
    virtual void setToOriginImpl(void) //初始化
    {
        _estimate << 0, 0, 0;
    }

    virtual void oplusImpl(const double *update) //更新
    {
        _estimate += Eigen::Vector3d(update);
    }

    // 存盘和读盘：留空
    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &out) const {}
};

//边 BaseUnaryEdge表示单元边
class Edge : public g2o::BaseUnaryEdge<1, double, Vertex> //维度 类型 顶点
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double _x;                // x值， y值为 _measurement
    Edge(double x) : _x(x) {} //传入x值

    // w=y-exp(ax^2+bx+c)
    void computeError()
    {
        // _vertices代表顶点 _error代表误差
        const Vertex *v = static_cast<const Vertex *>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0, 0) = _measurement - exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }

    // 存盘和读盘：留空
    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &out) const {}
    
};

int main(void)
{
    /*
    该代码用g2o求解最小二乘问题
    对于y=exp(ax^2+bx+c)+w
    求解参数a b c使得误差w=y-exp(ax^2+bx+c)最小化
    */

    double miu = 0.0, sigma = 1.0; //高斯噪声均值和方差
    std::default_random_engine engine;
    std::normal_distribution<double> normal(miu, sigma); //高斯分布随机数

    // 数据初始化
    int N = 100; //数据个数
    std::vector<double> x_vec, y_vec;
    double a = 1.0, b = 2.0, c = 1.0;
    for (int i = 0; i < N; i++)
    {
        double x = 1.0 * i / N; //(0-1)
        x_vec.push_back(x);
        y_vec.push_back(exp(a * x * x + b * x + c) + normal(engine));
    }

    // 构建图优化
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> Block;                                //输入维度3（参数） 输出维度1（残差）
    Block::LinearSolverType *linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>(); //线性方程求解器
    Block *solver_ptr = new Block(std::unique_ptr<Block::LinearSolverType>(linearSolver));       //矩阵块求解器

    //梯度下降方法
    //OptimizationAlgorithmLevenberg -> Levenberg LM方法中D=I
    //OptimizationAlgorithmGaussNewton -> Gauss-Newton
    //OptimizationAlgorithmDogleg -> Dogleg
    typedef g2o::OptimizationAlgorithmLevenberg solver_type;
    solver_type *solver = new solver_type(std::unique_ptr<Block>(solver_ptr));
    g2o::SparseOptimizer optimizer; // 图模型
    optimizer.setAlgorithm(solver); // 设置求解器
    optimizer.setVerbose(true);     // 打开调试输出

    // 1个顶点
    Vertex *v = new Vertex();
    v->setId(0);
    optimizer.addVertex(v);

    // N条边
    for (int i = 0; i < N; i++)
    {
        Edge *edge = new Edge(x_vec[i]);
        edge->setId(i);
        edge->setVertex(0, v);
        edge->setMeasurement(y_vec[i]);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (sigma * sigma)); // 信息矩阵：协方差矩阵之逆
        optimizer.addEdge(edge);
    }

    // 执行优化
    optimizer.initializeOptimization(); //初始化
    optimizer.setVerbose(true);         //优化过程输出信息
    optimizer.optimize(100);            //最大迭代次数

    //输出结果
    std::cout << "求解结果：" << std::endl;
    std::cout << v->estimate().transpose() << std::endl;

    return 0;
}