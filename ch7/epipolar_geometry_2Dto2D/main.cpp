#include <iostream>
#include <opencv2/opencv.hpp>

void feature_matches(std::vector<cv::Mat> &images,
                     std::vector<std::vector<cv::Point2f>> &points_uv)
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

    //把匹配点转换为vector<Point2f>的形式
    for (cv::DMatch d : betterMatches)
    {
        points_uv[0].push_back(keypoints[0][d.queryIdx].pt);
        points_uv[1].push_back(keypoints[1][d.trainIdx].pt);
    }
}

void analyze_match_points(std::vector<std::vector<cv::Point2f>> &points_uv)
{
    //内参 fx 0 cx 0 fy cy 0 0 1
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 21.0, 249.7, 0, 0, 1);

    // 计算Fundamental矩阵 基础矩阵 F = K^(-T) t^R K^(-1)
    // cv::Mat F = cv::findFundamentalMat(points_uv[0], points_uv[1], CV_FM_8POINT);
    cv::Mat F = cv::findFundamentalMat(points_uv[0], points_uv[1], cv::FM_8POINT); //opencv4

    // 计算Essential矩阵 本质矩阵 E = t^R
    cv::Mat E = cv::findEssentialMat(points_uv[0], points_uv[1], K, cv::RANSAC);

    // 计算单应矩阵 H = K (R-tn^T/d) K^(-1)
    cv::Mat H = cv::findHomography(points_uv[0], points_uv[1], cv::RANSAC, 3);

    // 从Essential矩阵 获得R t
    cv::Mat R, t;
    cv::recoverPose(E, points_uv[0], points_uv[1], K, R, t); //获得模长为1的t t扩大任意倍数也可以满足

    // 验证E=t^*R*k
    cv::Point3d tp = cv::Point3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0));
    cv::Mat t_x = (cv::Mat_<double>(3, 3) << 0, -tp.z, tp.y, tp.z, 0, -tp.x, -tp.y, tp.x, 0);
    cv::Mat t_xR = t_x * R;
    double k = E.at<double>(0, 0) / t_xR.at<double>(0, 0); //粗略地以第一个元素为倍数
    std::cout << "验证E=t^*R*k，E-t^*R*k=" << std::endl
              << E - t_xR * k << std::endl; //3*3矩阵都接近0

    // 验证对极约束
    std::cout << "验证对极约束：" << std::endl;
    for (int i = 0; i < points_uv[0].size(); i++)
    {
        cv::Mat p1 = (cv::Mat_<double>(3, 1) << points_uv[0][i].x, points_uv[0][i].y, 1);
        cv::Mat p2 = (cv::Mat_<double>(3, 1) << points_uv[1][i].x, points_uv[1][i].y, 1);
        std::cout << p2.t() * (K.inv().t()) * t_x * R * K.inv() * p1 << std::endl; //接近0
    }

    // 初始化变换矩阵
    std::cout << "\n";
    cv::Mat T1 = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0); //相机1变换矩阵
    cv::Mat T2 = cv::Mat::zeros(3, 4, T1.type());                                //相机2变换矩阵
    R.colRange(0, 3).copyTo(T2(cv::Rect(0, 0, 3, 3)));
    t.colRange(0, 1).copyTo(T2(cv::Rect(3, 0, 1, 3)));

    //将像素坐标points_uv转为Z归一化坐标
    std::vector<std::vector<cv::Point2f>> points_xy(2); //K^(-1)*points_uv
    for (int i = 0; i < points_uv[0].size(); i++)
    {
        cv::Point2f x1 = cv::Point2f(
            (points_uv[0][i].x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (points_uv[0][i].y - K.at<double>(1, 2)) / K.at<double>(1, 1));
        cv::Point2f x2 = cv::Point2f(
            (points_uv[1][i].x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (points_uv[1][i].y - K.at<double>(1, 2)) / K.at<double>(1, 1));
        points_xy[0].push_back(x1);
        points_xy[1].push_back(x2);
    }

    //求解空间坐标XYZ 三角化
    cv::Mat Pmat;
    cv::triangulatePoints(T1, T2, points_xy[0], points_xy[1], Pmat);

    // 将四维P转为非齐次三维坐标
    std::vector<cv::Point3d> points_XYZ;
    for (int i = 0; i < Pmat.cols; i++)
    {
        cv::Mat P4d = Pmat.col(i);
        P4d /= P4d.at<float>(3, 0); // Z归一化
        points_XYZ.push_back(cv::Point3d(P4d.rowRange(0, 3)));
    }

    //验证重投影关系 x1,y1 =X/Z1,Y/Z1   x2,y2 =(RX+t)/Z,(RY+t)/Z
    for (int i = 0; i < points_xy[0].size(); i++)
    {
        std::cout << std::endl;
        std::cout << "x1,y1:" << points_xy[0][i] << std::endl;
        std::cout << "X/Z1,Y/Z1:" << points_XYZ[i].x / points_XYZ[i].z << "," << points_XYZ[i].y / points_XYZ[i].z << std::endl;
        std::cout << "x2,y2:" << points_xy[1][i] << std::endl;
        cv::Mat points_XYZ_mat = (cv::Mat_<double>(3, 1) << points_XYZ[i].x, points_XYZ[i].y, points_XYZ[i].z); //矩阵计算
        std::cout << "(RP+t)/Z:" << ((R * points_XYZ_mat + t) / points_XYZ[i].z).t() << std::endl;
        std::cout << "Z:" << points_XYZ[i].z << std::endl;
    }
}
int main(void)
{
    //读取图像
    std::vector<cv::Mat> images(2);
    images[0] = cv::imread("../1.png");
    images[1] = cv::imread("../2.png");

    //特征匹配
    std::vector<std::vector<cv::Point2f>> points_uv(2); //代表两幅图像匹配的像素点
    feature_matches(images, points_uv);

    //分析匹配点
    analyze_match_points(points_uv);

    return 0;
}