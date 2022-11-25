#pragma once
#include "main.h"

class Camera
{
public:
    cv::Mat K;
    Eigen::Matrix3d K_eigen;

    double depth_scale;
    Camera(cv::Mat K, double depth_scale);
    Camera(double fx, double fy, double cx, double cy, double depth_scale);
    ~Camera();
    
    //编译器认为 传引用不带const是要修改 所以要设为const
    Eigen::Vector3d world2camera(const Eigen::Vector3d &p_w, const Sophus::SE3d &Tcw);
    Eigen::Vector3d camera2world(const Eigen::Vector3d &p_c, const Sophus::SE3d &Tcw);

    Eigen::Vector2d camera2pixel(const Eigen::Vector3d &p_c);
    Eigen::Vector3d pixel2camera(const Eigen::Vector2d &p_p, ushort depth);

    Eigen::Vector3d pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3d &Tcw, ushort depth);
    Eigen::Vector2d world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3d &Tcw);
};
