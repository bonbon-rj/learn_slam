#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "sophus/so3.hpp" //原课程代码是.h
#include "sophus/se3.hpp" //原课程代码是.h

int main(void)
{
    // *关于输出
    // 以SO3为例 
    // 原课程代码重载了Sophus::SO3的<<运算符 使得输出对应的so3
    // 我使用的这版本的代码是没有重载的 
    // 直接.log()获得Eigen::Vector3d对应的so3 
    // 而Eigen的Matrix都是有重载<<的 所以可以直接输出

    // *SO(3)
    //定义旋转
    Eigen::AngleAxisd w(M_PI/6, Eigen::Vector3d (0,0,1));//角轴
    Eigen::Matrix3d R = w.toRotationMatrix(); //旋转矩阵
    Eigen::Quaterniond q = Eigen::Quaterniond (R);//四元数

    //构造李群 
    Sophus::SO3<double> SO3_R(R);  //用旋转矩阵构造
    Sophus::SO3<double> SO3_q(q);  //用四元数构造

    //李代数
    Eigen::Vector3d so3 = SO3_R.log(); //李群对数映射获得李代数
    Eigen::Matrix3d so3_hat = Sophus::SO3<double>::hat(so3); //向量->反对称矩阵
    Eigen::Vector3d so3_vee = Sophus::SO3<double>::vee(so3_hat); //反对称矩阵->向量

    //扰动模型
    Eigen::Vector3d update_so3(1e-4,0,0); //李代数小量
    Sophus::SO3<double> updated_SO3 = Sophus::SO3<double>::exp(update_so3)*SO3_R;//李群乘以小量
    
    // *SE(3)
    //定义平移
    Eigen::Vector3d t(0.2,0.3,0.4); 

    //构造李群
    Sophus::SE3<double> SE3_Rt(R,t); //用旋转矩阵和平移分量构造
    Sophus::SE3<double> SE3_qt(q,t);  //用四元数和平移分量构造

    //李代数
    typedef Eigen::Matrix<double,6,1> Vector6d;
    Vector6d se3 = SE3_Rt.log(); //李群对数映射获得李代数 平移在前 旋转在后
    Eigen::Matrix4d se3_hat = Sophus::SE3<double>::hat(se3); //向量取hat
    Vector6d se3_vee = Sophus::SE3<double>::vee(se3_hat); //矩阵取vee

    //扰动模型
    Vector6d update_se3;
    update_se3<<1e-4,0,0,0,0,0; //李代数小量
    Sophus::SE3<double> updated_SE3 = Sophus::SE3<double>::exp(update_se3)*SE3_Rt;//李群乘以小量

    return 0;
}