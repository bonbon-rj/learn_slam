#include "utils.h"


Sophus::SE3d Mat4d_to_SE3(const Eigen::Matrix4d &Mat4d)
{
    return Sophus::SE3d(Mat4d.block<3, 3>(0, 0), Mat4d.block<3, 1>(0, 3));
}

Eigen::Matrix4d cv_to_eigen(cv::Mat &T)
{
    Eigen::Matrix4d T_eigen;
    cv::cv2eigen(T, T_eigen);
    return T_eigen;
}

cv::Mat eigen_to_cv(Eigen::Matrix4d &T)
{
    cv::Mat T_mat;
    cv::eigen2cv(T, T_mat);
    return T_mat;
}