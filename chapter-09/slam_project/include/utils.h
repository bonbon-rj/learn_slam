#pragma once
#include "main.h"

Sophus::SE3d Mat4d_to_SE3(const Eigen::Matrix4d &Mat4d);
Eigen::Matrix4d cv_to_eigen(cv::Mat &T);
cv::Mat eigen_to_cv(Eigen::Matrix4d &T);