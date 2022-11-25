#include "utils.h"

void angleAxis_rotate_point(Eigen::Vector3d angleAxis, Eigen::Vector3d point, Eigen::Vector3d &result)
{
    /*
    假设用角轴 k = [th1 th2 th3]^T  旋转p到p_rot
    可得 th^2 = th1^2 + th2^2 + th3^2

    当th不近0时
    需要先归一化 kn = (1/th) * [th1 th2 th3]^T
    1. 直接求解
    p_rot = p cos(th) + (kn x p)sin(th) + kn (kn . p)(1-cos(th))

    2. 旋转矩阵求解
    Kn = hat(kn)=      0   -th3/th   th2/th
                    th3/th    0     -th1/th
                   -th2/th   th1        0
    R = I + sin(th) Kn + (1-cos(th)) Kn^2
    p_rot = R p

    当th近0时
    一阶泰勒近似 R = I + sin(th) hat(Kn)
    sin(th) ~ th 故有 R = I + th hat(Kn)
    故有 R = I + hat(K)
    左右同时右乘p有 R p = p + hat(K) p
    故有 p_rot = p + k x p
    */

    double th = angleAxis.norm();

    // 比较 th 和 (DBL_EPSILON = 2.22045e-16) 判断是否近0
    // 注意 cout默认输出保留六位有效数字 实际上double 64位 保存的精度是足够下述判断的
    if ((th*th) < std::numeric_limits<double>::epsilon())
    {
        result = point + angleAxis.cross(point);
    }
    else
    {
        Eigen::Vector3d angleAxis_norm = (1 / th) * angleAxis;

        bool useRcalc = true;
        if (useRcalc)
        {
            Eigen::Matrix3d K_norm = Sophus::SO3<double>::hat(angleAxis_norm);
            Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + sin(th) * K_norm + (1 - cos(th)) * K_norm * K_norm;
            result = R * point;
        }
        else
        {
            result = point * cos(th) +
                     (angleAxis_norm.cross(point)) * sin(th) +
                     angleAxis_norm * (angleAxis_norm.dot(point)) * (1 - cos(th));
        }
    }
}

double get_vector_median(std::vector<double> &vec)
{
    int n = vec.size();
    auto mid_point = vec.begin() + n / 2;
    std::nth_element(vec.begin(), mid_point, vec.end());
    return vec[n / 2];
}

double normal_rand(double mu,double sigma)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mu,sigma);
    return distribution(generator);
}