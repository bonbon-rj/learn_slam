#include <iostream>
#include <random>
#include <ceres/ceres.h>

//定义残差怎么计算
struct COST
{
    const double _x, _y;                       //数据
    COST(double x, double y) : _x(x), _y(y) {} //配置new初始化结构体指针参数对应关系

    //重载圆括号运算符
    template <typename T>
    bool operator()(const T *const abc, T *residual) const
    {
        // w=y-exp(ax^2+bx+c)
        residual[0] = T(_y) - exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }
};

int main(void)
{
    /*
    该代码用ceres求解最小二乘问题
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

    //构建最小二乘问题
    ceres::Problem problem;
    double abc[3] = {0}; //参数
    for (int i = 0; i < N; i++)
    {
        problem.AddResidualBlock(
            // 自动求导 传入cost 输出维度（残差） 输入维度（参数）
            new ceres::AutoDiffCostFunction<COST, 1, 3>(
                new COST(x_vec[i], y_vec[i])), //代价函数
            NULL,                              //损失函数 NULL表示损失函数为单位函数
            abc                                //参数模块
        );
    }

    //配置选项以及求解信息
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_QR; //求解方式
    options.minimizer_progress_to_stdout = true;  //输出到cout
    ceres::Solve(options, &problem, &summary);    //求解

    //输出结果
    std::cout << "求解信息：" << std::endl;
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "求解时间：" << std::endl;
    std::cout << summary.total_time_in_seconds << "s" << std::endl;
    std::cout << "求解结果：" << std::endl;
    for (auto d : abc)
        std::cout << d << " ";
    std::cout << "\n";

    return 0;
}
