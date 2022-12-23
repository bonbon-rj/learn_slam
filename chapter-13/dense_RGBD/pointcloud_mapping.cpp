#include <iostream>
#include <vector>
#include <fstream>
#include <boost/format.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

int main(int argc, char **argv)
{
    const short n = 5; //图像数

    // 读取图像
    std::vector<cv::Mat> colorImgs, depthImgs; //彩色图和深度图
    for (int i = 0; i < n; i++)
    {
        boost::format fmt("../data/%s/%d.%s");
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "pgm").str(), -1));
    }

    // 读取位姿
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

    // 已知像素位置(u v) 深度d 相机位资T 求空间XYZ并放入点云
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    PointCloud::Ptr allPointCloud(new PointCloud);
    std::cout << "Change images to pointcloud..." << std::endl;
    for (int i = 0; i < n; i++)
    {
        std::cout << "Process images " << i + 1 << "..." << std::endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];
        if (color.empty() | depth.empty())
            return -1;

        // 遍历图像
        PointCloud::Ptr pointCloud(new PointCloud);
        for (int v = 0; v < color.rows; v++)
        {
            unsigned char *color_row_ptr = color.ptr<unsigned char>(v);
            unsigned short *depth_row_ptr = depth.ptr<unsigned short>(v); //注意short类型
            for (int u = 0; u < color.cols; u++)
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

                //坐标放入电云
                PointT p;
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];

                //色彩放入点云
                p.b = color_data_ptr[0];
                p.g = color_data_ptr[1];
                p.r = color_data_ptr[2];

                pointCloud->points.push_back(p);
            }
        }

        // 离群点滤波
        pcl::StatisticalOutlierRemoval<PointT> outlier_filter; // 离群点滤波器
        outlier_filter.setMeanK(50);                           // 用于统计分析的某点的周围点的数量
        outlier_filter.setStddevMulThresh(1.0);                // 设定标准差乘数
        outlier_filter.setInputCloud(pointCloud);              // 设置待滤波的点云
        PointCloud::Ptr outlier_filter_result(new PointCloud); // 输出
        outlier_filter.filter(*outlier_filter_result);         // 滤波
        (*allPointCloud) += (*outlier_filter_result);
    }
    allPointCloud->is_dense = false; //指定点中的所有数据是否都是有限的 包含Inf/NaN值时设为false
    std::cout << "Now point cloud (after outlier filter) has " << allPointCloud->size() << " point." << std::endl;

    // 体素滤波
    pcl::VoxelGrid<PointT> voxel_filter;                 // 体素滤波器
    voxel_filter.setLeafSize(0.01, 0.01, 0.01);          // 设置体素网格大小 可以理解为分辨率
    voxel_filter.setInputCloud(allPointCloud);           // 设置待滤波的点云
    PointCloud::Ptr voxel_filter_result(new PointCloud); // 输出
    voxel_filter.filter(*voxel_filter_result);           // 滤波
    (*allPointCloud) = (*voxel_filter_result);
    std::cout << "After voxel filter, now point cloud has " << allPointCloud->size() << " point." << std::endl;

    // 保存
    std::cout << "Save the point cloud." << std::endl;
    pcl::io::savePCDFileBinary("map.pcd", *allPointCloud);

    return 0;
}
