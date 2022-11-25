#include <iostream>
#include <boost/format.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "sophus/se3.hpp"
#include <ceres/ceres.h>
void HorizontalMerge(std::vector<cv::Mat> &imgvec, cv::Mat &OutputImg)
{
    //图像宽和高
    int width = imgvec[0].cols;
    int height = imgvec[0].rows;

    //图像和矩形框暂时变量
    cv::Mat TempMat;
    cv::Rect TempRect;

    //横向拼接
    for (int i = 0; i < imgvec.size(); i++)
    {
        TempRect = cv::Rect(width * i, 0, width, height);
        TempMat = OutputImg(TempRect);
        imgvec[i].colRange(0, width).copyTo(TempMat);
    }
}

float getPixelValue(const cv::Mat *gray, float x, float y)
{
    int s = gray->step; //w*c

    // int 和 floor 都是向下取整
    // 对于图像来说就是获得x y左上角整数对应的像素
    int x1 = floor(x);
    int y1 = floor(y);

    //越界检查
    if (y1 * s + x1 > gray->rows * s)
        return 0.f;

    uchar *data = &(gray->data[y1 * s + x1]);

    //计算差异
    float dx = x - x1;
    float dy = y - y1;

    //双线性插值
    return float(
        (1 - dx) * (1 - dy) * data[0] + //左上
        dx * (1 - dy) * data[1] +       //右上
        (1 - dx) * dy * data[s] +       //左下
        dx * dy * data[s + 1]);         //右下
}

//定义残差怎么计算
class CostFunctor : public ceres::SizedCostFunction<1, 6>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    const cv::Mat _gray1, _gray2;
    const Eigen::Vector3d _P;
    const Eigen::Matrix3d _K;

    CostFunctor(cv::Mat gray1, cv::Mat gray2, Eigen::Vector3d P, Eigen::Matrix3d K)
        : _gray1(gray1), _gray2(gray2), _P(P), _K(K) {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        //* 对于e=I1(1/Z1*KP)-I2(1/Z2*Kexp(xi)P)  求解xi使得误差e最小化
        // I1
        Eigen::Vector3d P1(_P);
        Eigen::Vector2d uv1 = (_K * P1).hnormalized();
        float I1 = float(_gray1.ptr<uchar>(int(uv1[1]))[int(uv1[0])]);

        // I2
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> se3(*parameters); //se3
        Eigen::Vector3d P2 = Sophus::SE3d::exp(se3) * P1;
        Eigen::Vector2d uv2 = (_K * P2).hnormalized();
        float I2 = getPixelValue(&_gray2, uv2[0], uv2[1]); //双线性插值

        residuals[0] = I1 - I2;

        if (jacobians != NULL)
        {
            if (jacobians[0] != NULL)
            {
                //jacobians是二级指针 *jacobians指向雅克比数据头
                Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> J(*jacobians);

                //计算du  这里的P'也就是P2
                double Xp = P2[0], Yp = P2[1], Zp = P2[2];          //X' Y' Z'
                double Xp2 = Xp * Xp, Yp2 = Yp * Yp, Zp2 = Zp * Zp; //X'^2 Y'^2 Z'^2
                double fx = _K(0, 0), fy = _K(1, 1), cx = _K(0, 2), cy = _K(1, 2);
                Eigen::Matrix<double, 2, 6> du;
                du << (fx / Zp), (0), (-fx * Xp / Zp2), (-fx * Xp * Yp / Zp2), (fx + fx * Xp2 / Zp2), (-fx * Yp / Zp),
                    (0), (fy / Zp), (-fy * Yp / Zp2), (-fy - fy * Yp2 / Zp2), (fy * Xp * Yp / Zp2), (fy * Xp / Zp);

                //计算梯度
                Eigen::Matrix<double, 1, 2> gradient;
                float dx = (getPixelValue(&_gray2, uv2[0] + 1, uv2[1]) - getPixelValue(&_gray2, uv2[0] - 1, uv2[1])) / 2;
                float dy = (getPixelValue(&_gray2, uv2[0], uv2[1] + 1) - getPixelValue(&_gray2, uv2[0], uv2[1] - 1)) / 2;
                gradient << dx, dy;

                //计算雅克比
                J = -gradient * du;
            }
        }

        return true;
    }
};

int main(void)
{

    srand(time(NULL)); //以时间作为种子生成随机颜色

    // *先根据第一张depth和color提取3D点
    //读取图片
    cv::Mat color, gray, depth;
    color = cv::imread("../use_data/original_color.png");
    depth = cv::imread("../use_data/original_depth.png", -1);
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);

    //提取特征点
    std::vector<cv::KeyPoint> keypoints;
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
    detector->detect(color, keypoints);

    //内参
    Eigen::Matrix3d K;
    float cx = 325.5, cy = 253.5, fx = 518.0, fy = 519.0;
    float depth_scale = 1000.0;
    K << fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.0f;

    //根据特征点和深度获得3D点
    std::vector<Eigen::Vector3d> points3d;
    for (auto kp : keypoints)
    {
        //剔除边缘点
        if (kp.pt.x < 20 || kp.pt.y < 20 ||
            (kp.pt.x + 20) > color.cols ||
            (kp.pt.y + 20) > color.rows)
            continue;

        ushort d = depth.ptr<ushort>((int)(kp.pt.y))[(int)(kp.pt.x)];
        if (d == 0)
            continue;

        float z = float(d) / depth_scale;
        float x = z * (kp.pt.x - cx) / fx;
        float y = z * (kp.pt.y - cy) / fy;

        points3d.push_back(Eigen::Vector3d(x, y, z));
    }

    // *根据直接法算出后续对应特征点
    //初始化李代数
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t(0, 0, 0);
    Sophus::SE3<double> SE3(R, t);    //李群
    Sophus::Vector6d se3 = SE3.log(); //李代数

    //遍历图片
    cv::Mat now_color, now_gray;
    for (int i = 1; i < 9; i++)
    {
        // 读取图片
        boost::format fmt("../%s/%s/%d.%s");
        now_color = cv::imread((fmt % "use_data" % "rgb" % (i) % "png").str());
        if (now_color.empty())
            break;

        //转灰度
        cv::cvtColor(now_color, now_gray, cv::COLOR_BGR2GRAY);

        // Bundle Adjustment
        ceres::Problem problem;
        for (int i = 0; i < points3d.size(); i++)
        {
            problem.AddResidualBlock(new CostFunctor(gray, now_gray, points3d[i], K), NULL, se3.data());
        }

        //配置选项以及求解信息
        ceres::Solver::Options options;
        ceres::Solver::Summary summary;
        options.linear_solver_type = ceres::DENSE_QR; //求解方式
        options.minimizer_progress_to_stdout = true;  //输出到cout
        options.max_num_iterations = 100;             //要把迭代次数设大一点 要迭代比较多次才收敛
        ceres::Solve(options, &problem, &summary);    //求解

        //输出结果
        std::cout << "求解信息：" << std::endl;
        std::cout << summary.BriefReport() << std::endl;
        std::cout << "求解时间：" << std::endl;
        std::cout << summary.total_time_in_seconds << "s" << std::endl;
        std::cout << "求解结果：" << std::endl;
        std::cout << Sophus::SE3d::exp(se3).matrix() << std::endl;
        std::cout << "\n";

        //画图查看效果
        std::vector<cv::Mat> draw_img(2);
        draw_img[0] = color.clone();
        draw_img[1] = now_color.clone();
        for (int i = 0; i < points3d.size(); i++)
        {
            // 随机颜色
            float b = rand() % 256;
            float g = rand() % 256;
            float r = rand() % 256;

            // 根据3D点和优化后的李代数 求解两幅图上对应点坐标
            Eigen::Vector3d P1 = points3d[i];
            Eigen::Vector2d uv1 = (K * P1).hnormalized();
            Eigen::Vector3d P2 = Sophus::SE3d::exp(se3) * P1;
            Eigen::Vector2d uv2 = (K * P2).hnormalized();

            // 越界检查
            if (uv1[0] < 0 || uv1[0] > now_color.cols || uv1[1] < 0 || uv1[1] >= now_color.rows)
                continue;
            if (uv2[0] < 0 || uv2[0] > now_color.cols || uv2[1] < 0 || uv2[1] >= now_color.rows)
                continue;

            //画在图上
            cv::circle(draw_img[0], cv::Point2d(uv1[0], uv1[1]), 2, cv::Scalar(b, g, r), 2);
            cv::circle(draw_img[1], cv::Point2d(uv2[0], uv2[1]), 2, cv::Scalar(b, g, r), 2);
        }

        // 拼接展示
        cv::Mat displayImg = cv::Mat::zeros(draw_img[0].rows, draw_img[0].cols * 2, draw_img[0].type());
        HorizontalMerge(draw_img, displayImg);
        cv::imshow("result", displayImg);
        cv::waitKey(2000);
        cv::destroyWindow("result");
    }

    return 0;
}