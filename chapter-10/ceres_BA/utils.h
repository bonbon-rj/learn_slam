#pragma once
#include "main.h"

void angleAxis_rotate_point(Eigen::Vector3d angleAxis, Eigen::Vector3d point, Eigen::Vector3d &result);
double get_vector_median(std::vector<double> &vec);
double normal_rand(double mu, double sigma);

// 点积
template <typename T>
inline T dot_product(const T x[3], const T y[3])
{
    return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
}

// 叉积
template <typename T>
inline void cross_product(const T x[3], const T y[3], T result[3])
{
    result[0] = x[1] * y[2] - x[2] * y[1];
    result[1] = x[2] * y[0] - x[0] * y[2];
    result[2] = x[0] * y[1] - x[1] * y[0];
}

// 模板版本 角轴旋转 用于ceres自动求导 
template <typename T>
void angleAxis_rotate_point_template(const T angleAxis[3], const T point[3], T result[3])
{
    T th_square = dot_product(angleAxis, angleAxis);
    if (th_square < std::numeric_limits<double>::epsilon())
    {
        // p_rot = p + k x p
        T cross_result[3];
        cross_product(angleAxis, point, cross_result);
        for (int i = 0; i < 3; i++)
        {
            result[i] = point[i] + cross_result[i];
        }
    }
    else
    {
        T th = sqrt(th_square);

        T angleAxis_norm[3];
        for (int i = 0; i < 3; i++)
        {
            angleAxis_norm[i] = angleAxis[i] / th;
        }

        T cross_result[3];
        cross_product(angleAxis_norm, point, cross_result);
        for (int i = 0; i < 3; i++)
        {
            result[i] = point[i] * cos(th) + cross_result[i] * sin(th) + angleAxis_norm[i] * (dot_product(angleAxis_norm, point)) * (T(1.0) - cos(th));
        }
    }
}