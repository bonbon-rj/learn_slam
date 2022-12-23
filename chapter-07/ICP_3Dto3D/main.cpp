#include <iostream>
#include "sophus/se3.hpp"
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp> //eigen 和opencv的eigen 顺序不能颠倒

// 定义残差怎么计算
class CostFunctor : public ceres::SizedCostFunction<3, 6>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    const std::vector<Eigen::Vector3d> _points3d;
    const Eigen::Matrix3d _K;
    CostFunctor(std::vector<Eigen::Vector3d> points3d, Eigen::Matrix3d K) : _points3d(points3d), _K(K) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        //对于e=P-exp(xi)P'  求解xi使得误差e最小化

        //parameters是二级指针 *parameters指向待优化的李代数数据头
        //residuals是一级指针 存放残差的数组
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> se3(*parameters); //se3
        Eigen::Vector3d Pp = Sophus::SE3d::exp(se3) * _points3d[1];     // P'= exp(xi)*P
        Eigen::Vector3d residual = _points3d[0] - Pp;                   // e = P-P'
        residuals[0] = residual[0];
        residuals[1] = residual[1];
        residuals[2] = residual[2];

        if (jacobians != NULL)
        {
            if (jacobians[0] != NULL)
            {
                //jacobians是二级指针 *jacobians指向雅克比数据头
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> J(*jacobians);

                double Xp = Pp[0], Yp = Pp[1], Zp = Pp[2]; //X' Y' Z'

                J << (-1), (0), (0), (0), (-Zp), (Yp),
                    (0), (-1), (0), (Zp), (0), (-Xp),
                    (0), (0), (-1), (-Yp), (Xp), (0);
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

void solve_ICP(
    std::vector<std::vector<cv::Mat>> &points3d,
    cv::Mat &R, cv::Mat &t)
{
    //求质心
    int N = points3d[0].size();
    cv::Mat P = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    cv::Mat Pp = P.clone();
    for (int i = 0; i < N; i++)
    {
        P += points3d[0][i];
        Pp += points3d[1][i];
    }
    P = P * 1 / N;
    Pp = Pp * 1 / N;

    // W = (pi-p)*(pi'-p')^T
    cv::Mat W = (cv::Mat_<double>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (int i = 0; i < N; i++)
    {
        W += (points3d[0][i] - P) * (points3d[1][i] - Pp).t();
    }

    // W = U*WW*V^T
    cv::Mat U, WW, VT;
    cv::SVD::compute(W, WW, U, VT, cv::SVD::FULL_UV);

    R = U * VT;
    t = P - R * Pp;
}

int main(void)
{
    //读取图像
    std::vector<cv::Mat> images(2);
    images[0] = cv::imread("../data/1.png");
    images[1] = cv::imread("../data/2.png");

    //特征匹配
    std::vector<std::vector<cv::Point2d>> points_uv(2); //代表两幅图像匹配的像素点
    feature_matches(images, points_uv);

    //深度图
    std::vector<cv::Mat> depth_imgs(2);
    // depth_imgs[0] = cv::imread("../data/1_depth.png", CV_LOAD_IMAGE_UNCHANGED);
    // depth_imgs[1] = cv::imread("../data/2_depth.png", CV_LOAD_IMAGE_UNCHANGED);
    depth_imgs[0] = cv::imread("../data/1_depth.png", cv::IMREAD_UNCHANGED); //opencv4
    depth_imgs[1] = cv::imread("../data/2_depth.png", cv::IMREAD_UNCHANGED); //opencv4

    //获得匹配3D点
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    std::vector<std::vector<cv::Mat>> points3d(2);
    for (int i = 0; i < points_uv[0].size(); i++)
    {
        ushort d1 = depth_imgs[0].ptr<ushort>(int(points_uv[0][i].y))[int(points_uv[0][i].x)]; //深度
        ushort d2 = depth_imgs[1].ptr<ushort>(int(points_uv[1][i].y))[int(points_uv[1][i].x)]; //深度
        if (d1 == 0 || d2 == 0)
            continue;
        float dd1 = d1 / 5000.0;
        float dd2 = d2 / 5000.0;

        points3d[0].push_back((
            cv::Mat_<double>(3, 1) << (points_uv[0][i].x - K.at<double>(0, 2)) / K.at<double>(0, 0) * dd1,
            (points_uv[0][i].y - K.at<double>(1, 2)) / K.at<double>(1, 1) * dd1,
            dd1));

        points3d[1].push_back((
            cv::Mat_<double>(3, 1) << (points_uv[1][i].x - K.at<double>(0, 2)) / K.at<double>(0, 0) * dd2,
            (points_uv[1][i].y - K.at<double>(1, 2)) / K.at<double>(1, 1) * dd2,
            dd2));
    }

    //ICP
    cv::Mat R, t;
    solve_ICP(points3d, R, t);

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
    std::vector<std::vector<Eigen::Vector3d>> points3d_eigen(2);
    for (int i = 0; i < points3d[0].size(); i++)
    {
        std::vector<Eigen::Vector3d> temp(2);
        cv::cv2eigen(points3d[0][i], temp[0]);
        cv::cv2eigen(points3d[1][i], temp[1]);
        points3d_eigen[0].push_back(temp[0]);
        points3d_eigen[1].push_back(temp[1]);
    }

    //构造李群李代数
    Sophus::SE3<double> SE3(R_eigen, t_eigen); //李群
    Sophus::Vector6d se3 = SE3.log();          //李代数

    // Bundle Adjustment
    ceres::Problem problem;
    for (int i = 0; i < points3d[0].size(); i++)
    {
        std::vector<Eigen::Vector3d> temp;
        temp.push_back(points3d_eigen[0][i]);
        temp.push_back(points3d_eigen[1][i]);
        problem.AddResidualBlock(new CostFunctor(temp, K_eigen), NULL, se3.data());
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
