#include "camera.h"

Camera::Camera(cv::Mat K, double depth_scale)
{
    this->K = K.clone();
    this->depth_scale = depth_scale;

    cv::cv2eigen(K, K_eigen); //类型转换
}
Camera::Camera(double fx, double fy, double cx, double cy, double depth_scale)
{
    this->K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    this->depth_scale = depth_scale;
}
Camera::~Camera()
{
    
}

Eigen::Vector3d Camera::world2camera(const Eigen::Vector3d &p_w, const Sophus::SE3d &Tcw)
{
    return Tcw * p_w;
}

Eigen::Vector3d Camera::camera2world(const Eigen::Vector3d &p_c, const Sophus::SE3d &Tcw)
{
    return Tcw.inverse() * p_c;
}

Eigen::Vector2d Camera::camera2pixel(const Eigen::Vector3d &p_c)
{
    double Z = p_c(2);
    Eigen::Vector3d p = 1.0 / Z * K_eigen * p_c;
    return Eigen::Vector2d(p(0), p(1));
}

Eigen::Vector3d Camera::pixel2camera(const Eigen::Vector2d &p_p, ushort depth)
{
    Eigen::Vector3d p3d = Eigen::Vector3d(p_p(0), p_p(1), 1);
    return K_eigen.inverse() * depth / depth_scale * p3d;
}

Eigen::Vector2d Camera::world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3d &Tcw)
{

    return camera2pixel(world2camera(p_w, Tcw));
}

Eigen::Vector3d Camera::pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3d &Tcw, ushort depth)
{
    return camera2world(pixel2camera(p_p, depth), Tcw);
}
