#pragma once
#include "main.h"

#include "camera.h"
#include "yaml.h"
#include "frame.h"
#include "utils.h"
#include "map.h"

class VisualOdometry
{
public:

    // 自定义类
    Camera *camera;
    Yaml *yaml;
    Frame ref_frame,current_frame;
    Map map;

    // 求解器    
    cv::Ptr<cv::ORB> orb;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    // 求解结果
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    std::vector<cv::DMatch> matches;

    // config文件参数
    int min_inliers;
    int number_of_features;
    double map_point_erase_ratio;
    double scale_factor;
    double level_pyramid; 
    double keyframe_rotation;
    double keyframe_translation;

    void init();
    void init_map(Frame frame);
    bool calc(Frame frame);
    void filter_match_result(std::vector<cv::DMatch>& match_result, std::vector<cv::DMatch>& filter_result);
    bool solve_pose_2d3d(std::vector<cv::Point2f> &p2d,std::vector<cv::Point3f> &p3d, int min_inlier_num, Sophus::SE3<double> &result);
    void add_mapoint(std::vector<cv::DMatch> &match,std::vector<cv::KeyPoint> &keypoint);
    void erase_useless_mappoint();

    VisualOdometry(Yaml *yaml,Camera *camera);
    ~VisualOdometry();
};


//定义残差怎么计算
class CostFunctor : public ceres::SizedCostFunction<2, 6>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    const Eigen::Vector2d _points2d;
    const Eigen::Vector3d _points3d;
    const Eigen::Matrix3d _K;
    CostFunctor(Eigen::Vector2d points2d, Eigen::Vector3d points3d, Eigen::Matrix3d K) : _points2d(points2d), _points3d(points3d), _K(K) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        //对于e=u-1/s*K*exp(xi)*P  求解xi使得误差e最小化

        // parameters是二级指针 *parameters指向待优化的李代数数据头
        // residuals是一级指针 存放残差的数组
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> se3(*parameters); // se3
        Eigen::Vector3d Pp = Sophus::SE3d::exp(se3) * _points3d;        // P'= exp(xi)*P
        Eigen::Vector2d residual = _points2d - (_K * Pp).hnormalized(); // e = u-1/s*K*P'
        residuals[0] = residual[0];
        residuals[1] = residual[1];

        if (jacobians != NULL)
        {
            if (jacobians[0] != NULL)
            {
                // jacobians是二级指针 *jacobians指向雅克比数据头
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J(*jacobians);

                double Xp = Pp[0], Yp = Pp[1], Zp = Pp[2];          // X' Y' Z'
                double Xp2 = Xp * Xp, Yp2 = Yp * Yp, Zp2 = Zp * Zp; // X'^2 Y'^2 Z'^2
                double fx = _K(0, 0), fy = _K(1, 1), cx = _K(0, 2), cy = _K(1, 2);

                J << (fx / Zp), (0), (-fx * Xp / Zp2), (-fx * Xp * Yp / Zp2), (fx + fx * Xp2 / Zp2), (-fx * Yp / Zp),
                    (0), (fy / Zp), (-fy * Yp / Zp2), (-fy - fy * Yp2 / Zp2), (fy * Xp * Yp / Zp2), (fy * Xp / Zp);
                J *= (-1);
            }
        }

        return true;
    }
};

