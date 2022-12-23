#include "pose.h"

Pose::Pose(double *data)
{

    // point
    for (int i = 0; i < 3; i++)
        p[i] = *(data + i);

    // quaternion
    for (int i = 0; i < 4; i++)
        q[i] = *(data + 3 + i);

    // se3
    SE3 = Sophus::SE3d(
        Eigen::Quaterniond(q[3], q[0], q[1], q[2]), // Eigen四元数初始化是w x y z
        Eigen::Vector3d(p[0], p[1], p[2]));
    se3 = SE3.log();

}

Pose::~Pose()
{
}
