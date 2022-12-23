#include "main.h"
#include "utils.h"

int main(int argc, char **argv)
{
    // 读取txt文件
    std::vector<std::string> images_paths;
    std::vector<Sophus::SE3d> T_w_c; // Twc
    std::string dataset_path = "../remode_test_data/test_data";
    std::ifstream fin(dataset_path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin)
        return -1;
    while (!fin.eof())
    {
        // 数据格式：图像文件名 tx, ty, tz, qx, qy, qz, qw  位姿是Twc
        std::string image_name;
        fin >> image_name;
        double data[7];
        for (double &d : data)
            fin >> d;

        images_paths.push_back(dataset_path + "/images/" + image_name);
        T_w_c.push_back(Sophus::SE3d(
            Eigen::Quaterniond(data[6], data[3], data[4], data[5]), // Eigen中四元数 构造是qw qx qy qz 内部存储是qx qy qz qw
            Eigen::Vector3d(data[0], data[1], data[2])));
        if (!fin.good())
            break;
    }

    // 第一张图
    cv::Mat image_0 = cv::imread(images_paths[0], 0);
    Sophus::SE3d T_w_c0 = T_w_c[0];
    int width = image_0.cols;
    int height = image_0.rows;

    // 初始化第一张图的深度图
    cv::Mat depth_image(height, width, CV_64F, 3.0);
    cv::Mat depth_cov2_image(height, width, CV_64F, 3.0);

    int images_num = 10; // images_paths.size() 可以调小 不用迭代所有图片
    // 迭代其他图 不断更新优化第一张图的深度图
    for (int i = 1; i < images_num; i++)
    {
        std::cout << "Update the depth from the image " << i << ", the final image is " << images_num - 1 << std::endl;

        cv::Mat image_i = cv::imread(images_paths[i], 0);
        if (image_i.empty())
            continue;

        Sophus::SE3d T_w_ci = T_w_c[i];
        Sophus::SE3d T_ci_c0 = T_w_ci.inverse() * T_w_c0;

        //迭代每一个像素
        int boarder = 20;      // 边缘宽度
        double min_cov2 = 0.1; // 最小方差
        double max_cov2 = 10;  // 最大方差
        for (int x = boarder; x < width - boarder; x++)
        {
            for (int y = boarder; y < height - boarder; y++)
            {
                // 方差太小或太大 该像素值不迭代
                double cov2 = depth_cov2_image.ptr<double>(y)[x];
                if (cov2 < min_cov2 || cov2 > max_cov2)
                    continue;

                double depth = depth_image.ptr<double>(y)[x]; // 参考帧的该点的深度值
                Eigen::Vector2d p_0 = Eigen::Vector2d(x, y);

                // ***********极线搜索 获得匹配点*********** //
                // 计算极线搜索参考坐标 最小坐标 最大坐标
                double cov = sqrt(cov2); // 标准差
                double d_min = depth - 3 * cov;
                double d_max = depth + 3 * cov;
                if (d_min < 0.1)
                    d_min = 0.1;
                Eigen::Vector2d p_i = cam2px(T_ci_c0 * (px2cam(p_0).normalized() * depth));     // 第i帧的该点的理论像素坐标
                Eigen::Vector2d p_i_min = cam2px(T_ci_c0 * (px2cam(p_0).normalized() * d_min)); // 第i帧的该点的理论最小像素坐标
                Eigen::Vector2d p_i_max = cam2px(T_ci_c0 * (px2cam(p_0).normalized() * d_max)); // 第i帧的该点的理论最大像素坐标

                // 计算极线方向和半长度
                Eigen::Vector2d epipolar_line = p_i_max - p_i_min;
                Eigen::Vector2d epipolar_direction = epipolar_line.normalized();

                double half_length = 0.5 * epipolar_line.norm();
                if (half_length > 100)
                    half_length = 100;

                // 以理论像素坐标为中心 左右各取半长度 计算零均值ncc 取出最高的作为匹配像素
                double best_ncc = -1.0;
                Eigen::Vector2d best_p_i;
                for (double l = -half_length; l <= half_length; l += 0.7) // l+=1
                {
                    Eigen::Vector2d p = p_i + l * epipolar_direction;
                    if (p(0) <= boarder || p(1) <= boarder || p(0) >= (width - boarder) || p(1) >= (height - boarder))
                        continue;

                    // 计算待匹配点与参考帧的 NCC
                    double ncc = calc_ncc(image_0, image_i, p_0, p);
                    if (ncc > best_ncc)
                    {
                        best_ncc = ncc;
                        best_p_i = p;
                    }
                }
                if (best_ncc < 0.85f) // 只相信 NCC 很高的匹配
                    continue;

                // ***********开始求解深度*********** //
                // 至此 image_0的p_0 已经与image_i的p_i匹配上了
                p_i = best_p_i;

                Sophus::SE3d T_c0_ci = T_ci_c0.inverse();
                Eigen::Vector3d f_0 = px2cam(p_0).normalized();
                Eigen::Vector3d f_i = px2cam(p_i).normalized();

                /*
                    d0f0 = diRfi + t =>
                    |(f0^T)f0 -(f0^T)Rfi| |d0| = |(f0^T)t|
                    |(fi^T)f0 -(fi^T)Rfi| |di|   |(fi^T)t|
                */
                Eigen::Matrix3d R = T_c0_ci.rotationMatrix();
                Eigen::Vector3d t = T_c0_ci.translation();

                // f0,fi均为列向量，有(f0^T)fi = (fi^T)f0 = f0 \cdot fi = fi \cdot f0
                double a0 = f_0.dot(f_0);
                double a1 = -f_0.dot(R * f_i);
                double a2 = f_i.dot(f_0);
                double a3 = -f_i.dot(R * f_i);
                double b0 = f_0.dot(t);
                double b1 = f_i.dot(t);

                // 二阶矩阵求逆直接用公式
                double determinant = a0 * a3 - a1 * a2;
                double d0 = 1.0 / determinant * (a3 * b0 - a1 * b1);
                double di = 1.0 / determinant * (-a2 * b0 + a0 * b1);

                // 取d0
                Eigen::Vector3d p0_tmp1 = d0 * f_0;     // d0计算p0
                Eigen::Vector3d p0_tmp2 = di * f_i + t; // d1计算p0
                Eigen::Vector3d p0_mean = (p0_tmp1 + p0_tmp2) / 2.0;
                double depth_e = p0_mean.norm(); //预估深度值

                // ***********高斯融合*********** //
                // 先计算cov_e
                Eigen::Vector3d p_0_e = f_0 * depth_e;
                Eigen::Vector3d a = p_0_e - t;
                double alpha = acos(p_0_e.dot(t) / (p_0_e.norm() * t.norm())); //也可以直接用单位向量 double alpha = acos(f_0.dot(t)/ t.norm());
                double beta = acos(a.dot(-t) / (a.norm() * t.norm()));
                double delta_beta = atan(1 / fx);
                double beta_p = beta - delta_beta; //这里如果改成加 conv_e计算要颠倒
                double p_norm = t.norm() * sin(beta - beta_p) / sin(M_PI - alpha - beta_p);
                double cov_e = p_0_e.norm() - p_norm;
                double cov2_e = cov_e * cov_e;

                // 融合(depth,cov2) (depth_e,cov2_e) ( (depth*cov2_e+depth_e*cov2)/(cov2+cov2_e), (cov2*cov2_e)/(cov2+cov2_e) )
                double depth_fuse = (depth * cov2_e + depth_e * cov2) / (cov2 + cov2_e);
                double cov2_fuse = (cov2 * cov2_e) / (cov2 + cov2_e);

                // 更新
                depth_image.ptr<double>(y)[x] = depth_fuse;
                depth_cov2_image.ptr<double>(y)[x] = cov2_fuse;
            }
            printf("\rCurrent progress:col(%d/%d)", x, width - boarder);
            fflush(stdout);
        }
        std::cout << std::endl;
    }

    // 展示
    std::cout << "Update all images completed, show the image and its depth image..." << std::endl;
    cv::Mat display;
    cv::normalize(depth_image,display,0,1,cv::NORM_MINMAX);
    cv::imshow("image", image_0);
    cv::imshow("depth", display);
    cv::waitKey(0);

    // 保存
    double maxValue = *std::max_element(depth_image.begin<double>(), depth_image.end<double>());
    int scale = floor(255.0 / maxValue);
    std::cout << "Save the depth image..." << std::endl;
    cv::imwrite("./scale" + std::to_string(scale) + "_depth.png", depth_image * scale);

    return 0;
}
