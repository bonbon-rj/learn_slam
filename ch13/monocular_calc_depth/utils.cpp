#include "utils.h"

// 像素到相机坐标系
Eigen::Vector3d px2cam(const Eigen::Vector2d px)
{
    return Eigen::Vector3d((px(0, 0) - cx) / fx, (px(1, 0) - cy) / fy, 1);
}

// 相机坐标系到像素
Eigen::Vector2d cam2px(const Eigen::Vector3d p_cam)
{
    return Eigen::Vector2d(p_cam(0, 0) * fx / p_cam(2, 0) + cx, p_cam(1, 0) * fy / p_cam(2, 0) + cy);
}

double getPixelValue(const cv::Mat *gray, Eigen::Vector2d p)
{
    double x = p(0);
    double y = p(1);

    int s = gray->step; // w*c

    // int 和 floor 都是向下取整
    // 对于图像来说就是获得x y左上角整数对应的像素
    int x1 = floor(x);
    int y1 = floor(y);

    //越界检查
    if (y1 * s + x1 > gray->rows * s)
        return 0.f;

    uchar *data = &(gray->data[y1 * s + x1]);

    //计算差异
    double dx = x - x1;
    double dy = y - y1;

    //双线性插值
    return double(
        (1 - dx) * (1 - dy) * data[0] + //左上
        dx * (1 - dy) * data[1] +       //右上
        (1 - dx) * dy * data[s] +       //左下
        dx * dy * data[s + 1]);         //右下
}

// 计算零均值ncc
double calc_ncc(const cv::Mat &ref, const cv::Mat &curr, const Eigen::Vector2d &pt_ref, const Eigen::Vector2d &pt_curr)
{
    // ncc参数
    const int window_size = 2;                                          // 半宽度
    const int ncc_area = (2 * window_size + 1) * (2 * window_size + 1); // 窗口面积

    // 保留参考帧和当前帧的像素值 以及计算它们的总和
    double mean_ref = 0, mean_curr = 0;
    std::vector<double> values_ref, values_curr;
    for (int x = -window_size; x <= window_size; x++)
    {
        for (int y = -window_size; y <= window_size; y++)
        {
            double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1)))[int(x + pt_ref(0))]) / 255.0;
            mean_ref += value_ref;

            double value_curr = getPixelValue(&curr, pt_curr + Eigen::Vector2d(x, y)) / 255.0;
            mean_curr += value_curr;

            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }
    }

    // 均值
    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    // 计算ncc
    double sum_AiBi = 0, sum_Ai2 = 0, sum_Bi2 = 0;
    for (int i = 0; i < values_ref.size(); i++)
    {
        // 去均值
        double Ai = values_ref[i] - mean_ref;
        double Bi = values_curr[i] - mean_curr;
        
        sum_AiBi += Ai * Bi;
        sum_Ai2 += Ai * Ai;
        sum_Bi2 += Bi * Bi;
    }

    return sum_AiBi / sqrt(sum_Ai2 * sum_Bi2 + 1e-10); // 防止分母出现零
}