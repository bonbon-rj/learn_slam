#include <iostream>
#include "sophus/se3.hpp"
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp> //eigen 和opencv的eigen 顺序不能颠倒

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

        //parameters是二级指针 *parameters指向待优化的李代数数据头
        //residuals是一级指针 存放残差的数组
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> se3(*parameters); //se3
        Eigen::Vector3d Pp = Sophus::SE3d::exp(se3) * _points3d;        // P'= exp(xi)*P
        Eigen::Vector2d residual = _points2d - (_K * Pp).hnormalized(); // e = u-1/s*K*P'
        residuals[0] = residual[0];
        residuals[1] = residual[1];

        if (jacobians != NULL)
        {
            if (jacobians[0] != NULL)
            {
                //jacobians是二级指针 *jacobians指向雅克比数据头
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J(*jacobians);

                double Xp = Pp[0], Yp = Pp[1], Zp = Pp[2];          //X' Y' Z'
                double Xp2 = Xp * Xp, Yp2 = Yp * Yp, Zp2 = Zp * Zp; //X'^2 Y'^2 Z'^2
                double fx = _K(0, 0), fy = _K(1, 1), cx = _K(0, 2), cy = _K(1, 2);

                J << (fx / Zp), (0), (-fx * Xp / Zp2), (-fx * Xp * Yp / Zp2), (fx + fx * Xp2 / Zp2), (-fx * Yp / Zp),
                    (0), (fy / Zp), (-fy * Yp / Zp2), (-fy - fy * Yp2 / Zp2), (fy * Xp * Yp / Zp2), (fy * Xp / Zp);
                J *= (-1);
            }
        }

        return true;
    }
};

void feature_matches(std::vector<cv::Mat> &images,
                     std::vector<std::vector<cv::Point2d>> &points_uv)
{
    //特征点检测
    std::vector<std::vector<cv::KeyPoint>> keypoints(2);
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    detector->detect(images[0], keypoints[0]);
    detector->detect(images[1], keypoints[1]);

    //计算BRIEF描述子 特征点有n个则描述子为n*d d为描述子长度
    std::vector<cv::Mat> descriptor(2);
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
    extractor->compute(images[0], keypoints[0], descriptor[0]);
    extractor->compute(images[1], keypoints[1], descriptor[1]);

    //特征匹配
    std::vector<cv::DMatch> matches;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming"); //汉明距离
    matcher->match(descriptor[0], descriptor[1], matches);

    //找最小距离
    std::sort(matches.begin(), matches.end(), [](cv::DMatch a, cv::DMatch b) { return a.distance < b.distance; }); //升序
    double minDistance = matches[0].distance;

    std::vector<cv::DMatch> betterMatches;
    for (cv::DMatch d : matches)
    {
        //当距离大于两倍最小距离时认为有误，30是为了避免最小距离太小
        if (d.distance <= std::max(2 * minDistance, 30.0))
        {
            betterMatches.push_back(d);
        }
        else
        {
            break; //因为排序过 所以不满足可以直接跳出
        }
    }

    //把匹配点转换为vector<Point2d>的形式
    for (cv::DMatch d : betterMatches)
    {
        points_uv[0].push_back(keypoints[0][d.queryIdx].pt);
        points_uv[1].push_back(keypoints[1][d.trainIdx].pt);
    }
}

int main(void)
{
    //--思路--
    //3D点读取相机一对应的深度图一 通过内参变换(u1,v1) 得到 XYZ
    //2D点读取匹配相机一的相机二的坐标(u2,v2)
    //求得相机二到世界坐标系（相机一）的变换

    //读取图像
    std::vector<cv::Mat> images(2);
    images[0] = cv::imread("../1.png");
    images[1] = cv::imread("../2.png");

    //特征匹配
    std::vector<std::vector<cv::Point2d>> points_uv(2); //代表两幅图像匹配的像素点
    feature_matches(images, points_uv);

    //获得2D 3D点
    std::vector<cv::Point3d> points3d;
    std::vector<cv::Point2d> points2d;
    // cv::Mat depth_img = cv::imread("../1_depth.png", CV_LOAD_IMAGE_UNCHANGED);         //深度图
    cv::Mat depth_img = cv::imread("../1_depth.png", cv::IMREAD_UNCHANGED);         //深度图  opencv4
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); //内参
    for (int i = 0; i < points_uv[0].size(); i++)
    {
        ushort d = depth_img.ptr<ushort>(int(points_uv[0][i].y))[int(points_uv[0][i].x)]; //深度
        if (d == 0)
            continue;
        float dd = d / 5000.0;

        points3d.push_back(cv::Point3d(
            (points_uv[0][i].x - K.at<double>(0, 2)) / K.at<double>(0, 0) * dd,
            (points_uv[0][i].y - K.at<double>(1, 2)) / K.at<double>(1, 1) * dd,
            dd));
        points2d.push_back(points_uv[1][i]);
    }

    // Opencv pnp求解R t
    cv::Mat r, t; //求解r为旋转向量
    cv::solvePnP(points3d, points2d, K, cv::Mat(), r, t, false, 1);
    cv::Mat R;
    cv::Rodrigues(r, R); //旋转向量转矩阵

    //输出R t
    std::cout << "R=" << std::endl;
    std::cout << R << std::endl;
    std::cout << "t=" << std::endl;
    std::cout << t << std::endl;

    //类型转换 mat -> eigen
    Eigen::Matrix3d R_eigen;
    cv::cv2eigen(R, R_eigen);
    Eigen::Vector3d t_eigen;
    cv::cv2eigen(t, t_eigen);
    Eigen::Matrix3d K_eigen;
    cv::cv2eigen(K, K_eigen);
    std::vector<Eigen::Vector3d> points3d_eigen;
    std::vector<Eigen::Vector2d> points2d_eigen;
    for (int i = 0; i < points3d.size(); i++)
    {
        points3d_eigen.push_back(Eigen::Vector3d(points3d[i].x, points3d[i].y, points3d[i].z));
        points2d_eigen.push_back(Eigen::Vector2d(points2d[i].x, points2d[i].y));
    }

    //构造李群李代数
    t_eigen[0] = t_eigen[0]-1;
    Sophus::SE3<double> SE3(R_eigen, t_eigen); //李群
    Sophus::Vector6d se3 = SE3.log();          //李代数

    // Bundle Adjustment
    ceres::Problem problem;
    for (int i = 0; i < points2d_eigen.size(); i++)
    {
        problem.AddResidualBlock(new CostFunctor(points2d_eigen[i], points3d_eigen[i], K_eigen), NULL, se3.data());
    }

    //配置选项以及求解信息
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_QR; //求解方式
    options.minimizer_progress_to_stdout = true;  //输出到cout
    ceres::Solve(options, &problem, &summary);    //求解

    //输出结果
    std::cout << "求解信息：" << std::endl;
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "求解时间：" << std::endl;
    std::cout << summary.total_time_in_seconds << "s" << std::endl;
    std::cout << "求解结果：" << std::endl;
    std::cout << Sophus::SE3d::exp(se3).matrix() << std::endl;
    std::cout << "\n";

    return 0;
}