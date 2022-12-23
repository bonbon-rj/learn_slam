#pragma once
#include "main.h"

class ReprojectionError
{
public:
    ReprojectionError(double observation_x, double observation_y) : observed_x(observation_x), observed_y(observation_y) {}

    template <typename T>
    bool operator()(const T *const camera, const T *const point, T *residuals) const
    {
        /*
        先将point从世界坐标系通过-R转到相机坐标系
        然后加上相机坐标系下的平移分量 结果取负 得到点在相机坐标系下的坐标
        */

        // 将点从相机坐标转换到世界坐标
        T pp[3];
        angleAxis_rotate_point_template(camera, point, pp);
        for (int i = 0; i < 3; i++)
        {
            pp[i] += camera[i + 3];
        }

        // 归一化
        T pc[2];
        for (int i = 0; i < 2; i++)
        {
            pc[i] = T(-1.0) * pp[i] / pp[2];
        }

        // 焦距和 畸变
        T f = camera[6];
        T k1 = camera[7];
        T k2 = camera[8];
        T rr = pc[0] * pc[0] + pc[1] * pc[1];
        T distortion = T(1.0) + rr * (k1 + k2 * rr); // 1 + k1*r^2 + k2*r^4

        // 像素预测值
        T predictions[2];
        predictions[0] = f * distortion * pc[0];
        predictions[1] = f * distortion * pc[1];

        // 计算残差
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 9, 3>(new ReprojectionError(observed_x, observed_y)));
    }

private:
    double observed_x;
    double observed_y;
};
