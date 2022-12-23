#include <iostream>
#include <Eigen/Core>
#include <fstream>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>

#include "pose.h"

typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

// 像四元数 旋转矩阵 齐次变换矩阵等对加法不封闭 需要自定义增量更新
// 例如本次的要求是优化位姿se3 本质上也就是个齐次变换矩阵
// ceres2.2.0版本之前是采用LocalParameterization  2.2.0之后采用Manifold
// 使用参考官方文档：http://ceres-solver.org/nnls_modeling.html#manifolds
class Se3Manifold : public ceres::Manifold
{
public:
    Se3Manifold() {}
    virtual ~Se3Manifold() {}

    // 用李代数左乘更新定义加法
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const
    {
        Sophus::SE3d T = Sophus::SE3d::exp(Vector6d(x));
        Sophus::SE3d delta_T = Sophus::SE3d::exp(Vector6d(delta));

        Vector6d incremental = (delta_T * T).log();
        for (int i = 0; i < 6; i++)
        {
            x_plus_delta[i] = incremental(i);
        }

        return true;
    }
    virtual bool PlusJacobian(const double *x, double *jacobian) const
    {
        ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
        return true;
    }

    virtual int AmbientSize() const { return 6; }; // x的自由度
    virtual int TangentSize() const { return 6; }; // delta x 的自由度

    //  纯虚类 要把所有的纯虚函数都实现一遍才能使用该类生成对象
    virtual bool RightMultiplyByPlusJacobian(const double *x, const int num_rows, const double *ambient_matrix, double *tangent_matrix) const { return 0; };
    virtual bool Minus(const double *y, const double *x, double *y_minus_x) const { return 0; };
    virtual bool MinusJacobian(const double *x, double *jacobian) const { return 0; };
};

class PoseGraphCostFunction : public ceres::SizedCostFunction<6, 6, 6>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    ~PoseGraphCostFunction() {}
    PoseGraphCostFunction(Sophus::SE3d SE3, Matrix6d information_matrix) : _measurment_SE3(SE3), _information_matrix(information_matrix) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {

        // 通过顶点计算delta的预估值
        Sophus::SE3d pose_i = Sophus::SE3d::exp(Vector6d(parameters[0]));
        Sophus::SE3d pose_j = Sophus::SE3d::exp(Vector6d(parameters[1]));
        Sophus::SE3d estimate_SE3 = pose_i.inverse() * pose_j;

        // LLT分解 可以理解为开方
        Matrix6d sqrt_info = _information_matrix.llt().matrixL(); 

        // 计算雅克比
        if (jacobians != NULL)
        {
            // 计算误差e
            Sophus::Vector6d e = (_measurment_SE3.inverse() * estimate_SE3).log(); //[phi rho]
            Eigen::Vector3d phi_e = e.block(0,0,3,1);
            Eigen::Vector3d rho_e = e.block(3,0,3,1);
            
            // 计算 Jr^(-1)
            Matrix6d T; 
            T.block(0, 0, 3, 3) = Sophus::SO3d::hat(phi_e); 
            T.block(0, 3, 3, 3) = Sophus::SO3d::hat(rho_e); 
            T.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero();
            T.block(3, 3, 3, 3) = Sophus::SO3d::hat(phi_e);
            Matrix6d Jr_inv = Matrix6d::Identity() + 0.5 * T;

            if (jacobians[0] != NULL)
            {
                Eigen::Map<Matrix6d> jacobian_i(jacobians[0]);
                jacobian_i = (sqrt_info * (-Jr_inv) * pose_j.inverse().Adj()).transpose();
            }
            if (jacobians[1] != NULL)
            {
                Eigen::Map<Matrix6d> jacobian_j(jacobians[1]);
                jacobian_j = (sqrt_info * Jr_inv * pose_j.inverse().Adj()).transpose();
            }
        }

        // 测量值估计值最小化
        Eigen::Map<Vector6d> residual(residuals);
        residual = sqrt_info * ((_measurment_SE3.inverse() * estimate_SE3).log());

        return true;
    }

private:
    const Sophus::SE3d _measurment_SE3;
    const Matrix6d _information_matrix;
};

int main(int argc, char **argv)
{

    // g2o文件输入
    std::ifstream g2o_input_file("../g2o_files/noise_sphere.g2o");
    if (!g2o_input_file.is_open())
        return -1;

    // g2o文件数据格式如下
    // VERTEX_SE3:QUAT id px py pz qx qy qz qw
    // EDGE_SE3:QUAT idfrom idto px py pz qx qy qz qw 信息矩阵上三角
    // 顶点数据包含其测量位姿  边数据包含两顶点之间的测量位姿 信息矩阵就是目标函数T = 1/2 * e^T * H * e 中的H
    ceres::Manifold *se3_manifold = new Se3Manifold();
    ceres::Problem problem;
    std::vector<Pose> poses;
    while (!g2o_input_file.eof())
    {
        std::string name;
        g2o_input_file >> name;

        if (name == "VERTEX_SE3:QUAT")
        {
            // id
            int id;
            g2o_input_file >> id;

            // data
            double data[7] = {0};
            for (int i = 0; i < 7; i++)
                g2o_input_file >> data[i];

            // 初始化位姿
            Pose pose(data);
            poses.push_back(pose);
        }
        else if (name == "EDGE_SE3:QUAT")
        {
            // id
            int idfrom, idto;
            g2o_input_file >> idfrom >> idto;

            // data
            double data[7];
            for (int i = 0; i < 7; i++)
                g2o_input_file >> data[i];
            Pose delta_pose(data);

            // 信息矩阵
            Matrix6d information_matrix; //
            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    if (i <= j)
                        g2o_input_file >> information_matrix(i, j); // 上三角
                    else
                        information_matrix(i, j) = information_matrix(j, i); // 下三角
                }
            }

            // build problem
            ceres::LossFunction *loss = new ceres::HuberLoss(1.0);
            ceres::CostFunction *costfunc = new PoseGraphCostFunction(delta_pose.SE3, information_matrix);
            problem.AddResidualBlock(costfunc, loss, poses[idfrom].se3.data(), poses[idto].se3.data());

            // 设置流形
            problem.SetManifold(poses[idfrom].se3.data(), se3_manifold);
            problem.SetManifold(poses[idto].se3.data(), se3_manifold);
        }
    }
    g2o_input_file.clear();
    g2o_input_file.seekg(std::ios::beg);

    // 求解
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_linear_solver_iterations = 50;
    options.minimizer_progress_to_stdout = true;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    // 输出 写入顶点
    std::ofstream g2o_output_file("./result.g2o");
    for (int i = 0; i < poses.size(); i++)
    {
        Sophus::SE3d poseSE3 = Sophus::SE3d::exp(poses[i].se3);
        g2o_output_file << "VERTEX_SE3:QUAT" << ' ';
        g2o_output_file << i << ' ';
        g2o_output_file << poseSE3.translation().transpose() << ' ';
        g2o_output_file << poseSE3.unit_quaternion().coeffs().transpose() << ' '; // Eigen 四元数构造是 w x y z  但是内部存储和输出是 x y z w
        g2o_output_file << std::endl;
    }

    // 将原文件的边写入
    while (!g2o_input_file.eof())
    {
        std::string s;
        getline(g2o_input_file, s);
        if (s[0] != 'E')
            continue;
        else
            g2o_output_file << s << std::endl;
    }

    // 关闭文件
    g2o_input_file.close();
    g2o_output_file.close();

    return 0;
}