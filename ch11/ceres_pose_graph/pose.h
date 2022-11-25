#pragma once
#include <Eigen/Dense>
#include <sophus/se3.hpp>
class Pose
{
public:
    Pose(double *data);
    ~Pose();

    double p[3] = {0}; // point:x y z
    double q[4] = {0}; // quaternion:x y z w

    Sophus::SE3d SE3;
    Sophus::Vector6d se3; 

};