#include "main.h"

#include "yaml.h"
#include "camera.h"
#include "vo.h"

int main(void)
{
    // viz初始化
    cv::viz::Viz3d window("Visual Odometry");
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
    cv::Point3d cam_pos(0, 0, -1.0), cam_focal_point(0, 0, 0), cam_y_dir(0, 1, 0);
    cv::Affine3d cam_pose = cv::viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
    window.setViewerPose(cam_pose);
    window.showWidget("World", world_coor);
    window.showWidget("Camera", camera_coor);

    // 初始化类
    Yaml yaml("../config/default.yaml");
    Camera rgbd(yaml.getMat("K"), yaml.getArgs<double>("depth_scale"));
    VisualOdometry vo(&yaml, &rgbd);

    // 初始化vo 路径采用相对路径 所以在bin路径下才能运行
    Frame first_frame;
    first_frame.getColor("../data/rgb/0.png");
    first_frame.getDepth("../data/depth/0.png");
    if (!first_frame.checkSuccess())
        return false;
    first_frame.Tcw = Mat4d_to_SE3(Eigen::Matrix4d::Identity());
    vo.init();
    vo.init_map(first_frame);

    // vo循环
    int index = 1;
    while (1)
    {
        std::cout << "***************"<< "loop" << index << "***************" <<std::endl;
        //读取图像
        Frame frame;
        boost::format fmt("../data/%s/%d.%s");
        frame.getColor((fmt  % "rgb" % (index) % "png").str());
        frame.getDepth((fmt  % "depth" % (index) % "png").str());
        index++;
        if (!frame.checkSuccess())
            break;

        // vo计算
        if (!vo.calc(frame))
            continue;

        // 位姿展示
        Eigen::Matrix4d T = vo.current_frame.Tcw.inverse().matrix();
        cv::Affine3d M(
            cv::Matx33d(T(0, 0), T(0, 1), T(0, 2), T(1, 0), T(1, 1), T(1, 2), T(2, 0), T(2, 1), T(2, 2)),
            cv::Vec3d(T(0, 3), T(1, 3), T(2, 3)));
        window.setWidgetPose("Camera", M);
        window.spinOnce(1);

        // 路标点在图上展示
        cv::Mat img_show = vo.current_frame.color.clone();
        for (auto p : vo.map.mappoint_list)
        {
            Eigen::Vector2d uv = rgbd.world2pixel(p.position, vo.current_frame.Tcw);
            cv::circle(img_show, cv::Point2f(uv(0, 0), uv(1, 0)), 3, cv::Scalar(0, 255, 0));
        }
        cv::imshow("image_show", img_show);
        cv::waitKey(1);
    }
}
