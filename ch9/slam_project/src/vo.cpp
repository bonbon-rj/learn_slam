#include "vo.h"

void VisualOdometry::init()
{
    // 参数初始化
    map_point_erase_ratio = yaml->getArgs<double>("map_point_erase_ratio");
    number_of_features = yaml->getArgs<int>("number_of_features");
    scale_factor = yaml->getArgs<double>("scale_factor");
    level_pyramid = yaml->getArgs<double>("level_pyramid");
    min_inliers = yaml->getArgs<int>("min_inliers");
    keyframe_rotation = yaml->getArgs<double>("keyframe_rotation");
    keyframe_translation = yaml->getArgs<double>("keyframe_translation");

    // orb初始化 matcher初始化
    orb = cv::ORB::create(number_of_features, scale_factor, level_pyramid);
    matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
}

void VisualOdometry::init_map(Frame frame)
{
    // 提取特征点 计算描述子
    orb->detect(frame.color, keypoints);
    orb->compute(frame.color, keypoints, descriptor);

    // 用路标点和描述子初始化地图
    for (int i = 0; i < keypoints.size(); i++)
    {
        int x = cvRound(keypoints[i].pt.x);
        int y = cvRound(keypoints[i].pt.y);
        ushort d = frame.findDepth(x, y); //深度
        if (d == 0)
            continue;
        Eigen::Vector3d p_w = camera->pixel2world(Eigen::Vector2d(x, y), frame.Tcw, d);
        Eigen::Vector3d n = p_w - frame.getCamCenter(); // 第一帧光心为(0,0,0) 不减也可以
        n.normalize();

        MapPoint mp = MapPoint(p_w, descriptor.row(i).clone(), n);
        map.insertMapPoint(mp);
    }
    map.insertFrame(frame);

    //第一帧设为参考帧
    ref_frame = frame.clone();
}

bool VisualOdometry::calc(Frame frame)
{
    current_frame = frame.clone();

    // 提取特征点 计算描述子
    orb->detect(current_frame.color, keypoints);
    orb->compute(current_frame.color, keypoints, descriptor);

    // 取出地图中的描述子以及对应路标点 指向MapPoint的指针用于后续对MapPoint操作
    cv::Mat descriptor_map;
    std::vector<Eigen::Vector3d> candidate;
    std::vector<MapPoint *> mapPointPtr;
    for (auto &p : map.mappoint_list)
    {
        Eigen::Vector3d p_c = camera->world2camera(p.position, ref_frame.Tcw);
        if (p_c(2) < 0)
        {
            continue;
        }
        Eigen::Vector2d uv = camera->camera2pixel(p_c);

        if (ref_frame.isInFrame(uv(0), uv(1)))
        {
            p.visible_times++;
            candidate.push_back(p.position);
            mapPointPtr.push_back(&p);
            descriptor_map.push_back(p.descriptor);
        }
    }

    // 特征匹配
    matcher->match(descriptor_map, descriptor, matches);
    if (matches.size() == 0)
        return false;

    // 对匹配结果进行筛选
    std::vector<cv::DMatch> betterMatches;
    filter_match_result(matches, betterMatches);
    if (betterMatches.size() < 6)
        return false;

    // 将匹配结果对应取出
    std::vector<cv::Point3f> points3d;
    std::vector<cv::Point2f> points2d;
    for (cv::DMatch m : betterMatches)
    {
        // map
        Eigen::Vector3d map_match_point = candidate[m.queryIdx];
        mapPointPtr[m.queryIdx]->match_times++;
        points3d.push_back(cv::Point3f(map_match_point(0, 0), map_match_point(1, 0), map_match_point(2, 0)));

        // current frame
        points2d.push_back(cv::Point2f(keypoints[m.trainIdx].pt.x, keypoints[m.trainIdx].pt.y));
    }

    // 求解
    Sophus::SE3d Tcw;
    if (!solve_pose_2d3d(points2d, points3d, min_inliers, Tcw))
    {
        std::cout << "solve error" << std::endl;
        return false;
    }

    // 位姿检查
    Sophus::SE3<double> deltaT = ref_frame.Tcw * Tcw.inverse();
    Sophus::Vector6d d = deltaT.log();
    if (d.norm() > 0.25)
    {
        std::cout << "deltaT is too large: " << d.norm() << std::endl;
        return false;
    }

    // 上述没有返回说明位姿解可用
    current_frame.Tcw = Tcw;
    std ::cout << Tcw.matrix() << std::endl;

    // 剔除掉不需要的点
    erase_useless_mappoint();

    // 匹配结果太少 则补充
    if (betterMatches.size() < 100)
        add_mapoint(betterMatches, keypoints);

    // 地图太大则把剔除阈值加大
    if (map.mappoint_list.size() > 1000)
        map_point_erase_ratio += 0.05;
    else
        map_point_erase_ratio = map_point_erase_ratio;

    // 检查当前帧
    Eigen::Vector3d trans = d.block<3, 1>(0, 0);
    Eigen::Vector3d rot = d.block<3, 1>(3, 0);
    if (rot.norm() > keyframe_rotation || trans.norm() > keyframe_translation)
    {
        map.insertFrame(current_frame);
        ref_frame = current_frame.clone();
    }

    return true;
}

void VisualOdometry::add_mapoint(std::vector<cv::DMatch> &match, std::vector<cv::KeyPoint> &keypoint)
{
    // 剔除掉和地图匹配成功的点
    std::vector<bool> matched(keypoint.size(), false);
    for (cv::DMatch m : match)
    {
        matched[m.trainIdx] = true;
    }

    for (int i = 0; i < keypoint.size(); i++)
    {
        if (matched[i] == true)
        {
            continue;
        }

        int x = cvRound(keypoint[i].pt.x);
        int y = cvRound(keypoint[i].pt.y);
        ushort d = current_frame.findDepth(x, y); //深度
        if (d == 0)
            continue;
        Eigen::Vector3d p_w = camera->pixel2world(Eigen::Vector2d(x, y), current_frame.Tcw, d);
        Eigen::Vector3d n = p_w - current_frame.getCamCenter();
        n.normalize();

        MapPoint mp = MapPoint(p_w, descriptor.row(i).clone(), n);
        map.insertMapPoint(mp);
    }
}

void VisualOdometry::filter_match_result(std::vector<cv::DMatch> &match_result, std::vector<cv::DMatch> &filter_result)
{
    // 对结果按距离排序
    std::sort(match_result.begin(), match_result.end(), [](cv::DMatch a, cv::DMatch b)
              { return a.distance < b.distance; }); //升序
    double minDistance = matches[0].distance;

    // 筛选
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

    //拷贝
    filter_result.assign(betterMatches.begin(), betterMatches.end());
}

bool VisualOdometry::solve_pose_2d3d(std::vector<cv::Point2f> &p2d, std::vector<cv::Point3f> &p3d, int min_inlier_num, Sophus::SE3<double> &result)
{
    // 求解
    cv::Mat r, t, inliers;
    cv::solvePnPRansac(p3d, p2d, camera->K, cv::Mat(), r, t, false, 100, 4.0, 0.99, inliers, cv::SOLVEPNP_EPNP);

    // 类型转换
    cv::Mat R;
    cv::Rodrigues(r, R);
    Eigen::Matrix3d R_eigen, K_eigen;
    Eigen::Vector3d t_eigen;
    cv::cv2eigen(R, R_eigen);
    cv::cv2eigen(camera->K, K_eigen);
    cv::cv2eigen(t, t_eigen);
    std::vector<Eigen::Vector3d> points3d_eigen;
    std::vector<Eigen::Vector2d> points2d_eigen;
    for (int i = 0; i < p3d.size(); i++)
    {
        points3d_eigen.push_back(Eigen::Vector3d(p3d[i].x, p3d[i].y, p3d[i].z));
        points2d_eigen.push_back(Eigen::Vector2d(p2d[i].x, p2d[i].y));
    }

    // BA优化
    Sophus::SE3<double> SE3(R_eigen, t_eigen);
    Sophus::Vector6d se3 = SE3.log();
    ceres::Problem problem;
    for (int i = 0; i < points2d_eigen.size(); i++)
    {
        problem.AddResidualBlock(new CostFunctor(points2d_eigen[i], points3d_eigen[i], K_eigen), NULL, se3.data());
    }
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_QR; //求解方式
    options.minimizer_progress_to_stdout = false; //输出到cout
    ceres::Solve(options, &problem, &summary);    //求解

    result = Sophus::SE3d::exp(se3);

    if (inliers.rows < min_inlier_num)
        return false;
    return true;
}

void VisualOdometry::erase_useless_mappoint()
{
    for (auto iter = map.mappoint_list.begin(); iter != map.mappoint_list.end();)
    {
        Eigen::Vector3d p_cam = camera->world2camera(iter->position, current_frame.Tcw);
        if (p_cam(2, 0) < 0)
        {
            iter = map.mappoint_list.erase(iter);
            continue;
        }

        Eigen::Vector2d uv = camera->world2pixel(iter->position, current_frame.Tcw);
        if (!current_frame.isInFrame(uv(0, 0), uv(1, 0)))
        {
            iter = map.mappoint_list.erase(iter);
            continue;
        }

        // 剔除梯度太小的点
        int value = int(current_frame.getPixel(uv(0, 0), uv(1, 0)));
        int num = 3;
        int sum = 0;
        for (int i = num * (-1); i <= num; i++)
        {
            for (int j = num * (-1); j <= num; j++)
            {
                if (i == 0 && j == 0)
                {
                    continue;
                }
                sum = std::abs(value - int(current_frame.getPixel(uv(0, 0) + i, uv(1, 0) + j)));
            }
        }
        if (sum < 10)
        {
            iter = map.mappoint_list.erase(iter);
            continue;
        }

        float match_ratio = float(iter->match_times) / iter->visible_times;
        if (match_ratio < map_point_erase_ratio)
        {
            iter = map.mappoint_list.erase(iter);
            continue;
        }

        Eigen::Vector3d n = iter->position - current_frame.getCamCenter();
        n.normalize();
        double angle = acos(n.transpose() * iter->norm);
        if (angle > M_PI / 6.)
        {
            iter = map.mappoint_list.erase(iter);
            continue;
        }

        iter++;
    }
}

VisualOdometry::VisualOdometry(Yaml *yaml, Camera *camera)
{
    this->camera = camera;
    this->yaml = yaml;
}

VisualOdometry::~VisualOdometry()
{
}