#include <iostream>
#include <vector>
#include <boost/format.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

int main(int argc, char **argv)
{
    const short n = 5; //图像数

    //读取图像
    std::vector<cv::Mat> colorImgs, depthImgs; //彩色图和深度图
    for (int i = 0; i < n; i++)
    {
        boost::format fmt("../data/%s/%d.%s");
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "pgm").str(), -1));
    }

    //读取位姿
    std::ifstream fin("../data/pose.txt");
    if (!fin)
        return -1;
    std::vector<Eigen::Isometry3d> poses;
    for (int i = 0; i < n; i++)
    {
        // 数据格式 px py pz qw qx qy qz
        double data[7] = {0};
        for (auto &d : data)
            fin >> d;

        // 四元数初始化是p0 p1 p2 p3 内置存储是p1 p2 p3 p0
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.rotate(Eigen::Quaterniond(data[6], data[3], data[4], data[5]));
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(T);
    }

    //相机内参
    double cx = 325.5, cy = 253.5;
    double fx = 518.0, fy = 519.0;
    double depthScale = 1000.0;

    //八叉树地图
    octomap::ColorOcTree tree(0.05); // 参数为分辨率
    std::cout << "Change images to octomap..." << std::endl;
    for (int i = 0; i < n; i++)
    {
        std::cout << "Process images " << i + 1 << "..." << std::endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];
        if (depth.empty())
            return -1;

        for (int v = 0; v < depth.rows; v++)
        {
            unsigned char *color_row_ptr = color.ptr<unsigned char>(v);
            unsigned short *depth_row_ptr = depth.ptr<unsigned short>(v);
            for (int u = 0; u < depth.cols; u++)
            {
                unsigned char *color_data_ptr = &color_row_ptr[u * color.channels()];
                unsigned int d = depth_row_ptr[u]; // 深度
                if (d == 0 || d >= 7000)           // 没有测量或深度太大不处理
                    continue;

                //相机坐标系xyz
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;

                //世界坐标xyz
                Eigen::Vector3d pointWorld = T * point;

                // 添加射线 则起点到观测点之间的点都未被占据
                tree.insertRay(
                    octomap::point3d(T(0, 3), T(1, 3), T(2, 3)),
                    octomap::point3d(pointWorld[0], pointWorld[1], pointWorld[2]));

                // 设置颜色
                tree.setNodeColor(
                    pointWorld[0], pointWorld[1], pointWorld[2],
                    color_data_ptr[0], color_data_ptr[1], color_data_ptr[2]);
            }
        }
    }

    // 更新中间节点的占据信息
    tree.updateInnerOccupancy();

    // 保存
    std::cout << "Save the octomap." << std::endl;
    tree.write("map.ot"); // 彩色地图要保存为ot

    return 0;
}
