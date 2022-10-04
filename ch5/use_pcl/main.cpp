#include <iostream>
#include <boost/format.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

int main(void)
{
    // // 读取文件夹所有图片可以这样子读取
    // // 但是是乱序的 可能左右相机对应不上 故没采用
    // std::string dir = "../color_img";
    // DIR *dir_ptr = opendir(dir.c_str());
    // if(dir_ptr == NULL) return -1;

    // std::vector<cv::Mat> colorImgs;
    // struct dirent *dirp;
    // int n=0;
    // while (((dirp=readdir(dir_ptr))!=NULL))
    // {
    //     if(dirp->d_name[0]=='.') continue;//剔除掉. 和 ..
    //     colorImgs.push_back(cv::imread(dir+"/"+dirp->d_name));
    // }

    const short n = 5; // 图像对数

    //读取图像
    std::vector<cv::Mat> colorImgs, depthImgs; //彩色图和深度图
    for (int i = 0; i < n; i++)
    {
        boost::format fmt("../%s/%d.%s");
        colorImgs.push_back(cv::imread((fmt % "color_img" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth_img" % (i + 1) % "pgm").str(), -1));
    }

    //读取位姿
    std::ifstream fin("../pose.txt");
    if (!fin)
        return -1;
    std::vector<Eigen::Isometry3d> poses;
    for (int i = 0; i < n; i++)
    {
        // 0,1,2对应xyz代表平移 3 4 5 6对应四元数p1 p2 p3 p0代表旋转
        double data[7] = {0};

        // c++11 引入的 for range-based loop写法
        // 对于for(auto &s:sp)，sp是一个序列，s是一个用于访问sp中基本元素的变量
        // 每次迭代都会用sp中的下一个元素来初始化s
        // 实现类似 for(int j=0;j<sizeof(data)/sizeof(data[0]);j++)fin >> data[j];
        // 对于for(auto s:sp) 不修改sp
        for (auto &d : data)
            fin >> d;

        // 这里要初始化 单独Eigen::Isometry3d T;最后没法实现预期效果
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

    // 已知u v d 以及相机位资T 求空间XYZ并放入点云
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    PointCloud::Ptr pointCloud(new PointCloud);
    for (int i = 0; i < n; i++)
    {
        std::cout << "转换图像中: " << i + 1 << std::endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];
        if (color.empty() | depth.empty())
            return -1;

        // 遍历图像
        for (int v = 0; v < color.rows; v++)
        {
            unsigned char *color_row_ptr = color.ptr<unsigned char>(v);
            unsigned short *depth_row_ptr = depth.ptr<unsigned short>(v); //注意short类型
            for (int u = 0; u < color.cols; u++)
            {
                unsigned char *color_data_ptr = &color_row_ptr[u * color.channels()];
                unsigned int d = depth_row_ptr[u]; //深度
                if (d == 0)
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
                //索引方式 color.data[v*color.step+u*color.channels()+c] step实际是col * channel
                p.b = color_data_ptr[0];
                p.g = color_data_ptr[1];
                p.r = color_data_ptr[2];

                pointCloud->points.push_back(p);
            }
        }
    }

    pointCloud->is_dense = false;                       //点云数据不有限
    pcl::io::savePCDFileBinary("./map.pcd", *pointCloud); //保存

    return 0;
}
