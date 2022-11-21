#include <iostream>
#include <boost/format.hpp>
#include <opencv2/opencv.hpp>

int main(void)
{
    srand(time(NULL)); //以时间作为种子生成随机颜色

    std::vector<std::vector<int>> points_color;
    std::vector<cv::Point2f> now_keypoints, last_keypoints;
    cv::Mat now_color, last_color;
    for (int i = 0; i < 9; i++)
    {
        // 读取图片
        boost::format fmt("../%s/%d.%s");
        now_color = cv::imread((fmt % "simplify_name_rgb" % (i) % "png").str());
        if (now_color.empty())
            break;

        // 第一帧提取特征
        if (i == 0)
        {
            std::vector<cv::KeyPoint> original_keypoints;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect(now_color, original_keypoints);
            for (auto kp : original_keypoints)
            {
                now_keypoints.push_back(kp.pt);
                points_color.push_back(std::vector<int>{rand() % 256, rand() % 256, rand() % 256});
            }
        }
        else
        {
            // LK光流
            std::vector<uchar> status;
            std::vector<float> error;
            cv::calcOpticalFlowPyrLK(last_color, now_color, last_keypoints, now_keypoints, status, error);

            // 将status不为0的转移到temp后再拷贝给now_keypoints 避免erase导致下标对不齐 颜色同理
            std::vector<cv::Point2f> temp;
            std::vector<std::vector<int>> temp_color;
            for (int j = 0; j < now_keypoints.size(); j++)
            {
                if (status[j] != 0)
                {
                    temp.push_back(now_keypoints[j]);
                    temp_color.push_back(points_color[j]);
                }
            }
            points_color.assign(temp_color.begin(), temp_color.end());
            now_keypoints.assign(temp.begin(), temp.end());
            std::cout << "跟踪到的点数：" << now_keypoints.size() << std::endl;

            //画出keypoints
            if (now_keypoints.size() == 0)
                break;
            cv::Mat display_img = now_color.clone();
            for (int j = 0; j < now_keypoints.size(); j++)
            {
                cv::circle(display_img, now_keypoints[j], 2,
                           cv::Scalar(points_color[j][0], points_color[j][1], points_color[j][2]));
            }
            cv::imshow("circle", display_img);
            cv::waitKey(1000);
        }

        // 自迭代 now -> last
        last_keypoints.assign(now_keypoints.begin(), now_keypoints.end());
        last_color = now_color.clone();
    }
    return 0;
}