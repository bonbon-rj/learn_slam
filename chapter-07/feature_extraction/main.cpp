#include <iostream>
#include <opencv2/opencv.hpp>

//横向拼接图像
void HorizontalMerge(std::vector<cv::Mat> &imgvec, cv::Mat &OutputImg)
{
    //图像宽和高
    int width = imgvec[0].cols;
    int height = imgvec[0].rows;

    //图像和矩形框暂时变量
    cv::Mat TempMat;
    cv::Rect TempRect;
    for (int i = 0; i < imgvec.size(); i++)
    {
        TempRect = cv::Rect(width * i, 0, width, height);
        TempMat = OutputImg(TempRect);
        imgvec[i].colRange(0, width).copyTo(TempMat);
    }
}
int main(void)
{
    //读取图像
    std::vector<cv::Mat> images(2);
    images[0] = cv::imread("../data/1.png");
    images[1] = cv::imread("../data/2.png");

    //特征点检测
    std::vector<std::vector<cv::KeyPoint>> keypoints(2);           //关键点
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(2000); //2000个点
    detector->detect(images[0], keypoints[0]);
    detector->detect(images[1], keypoints[1]);

    //计算BRIEF描述子 特征点有n个则描述子为n*d d为描述子长度
    std::vector<cv::Mat> descriptor(2);
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
    extractor->compute(images[0], keypoints[0], descriptor[0]);
    extractor->compute(images[1], keypoints[1], descriptor[1]);

    //显示特征点
    std::vector<cv::Mat> draw_img(2);
    // Scalar::all(-1)表示随机颜色
    //DrawMatchesFlags::DEFAULT表示只画出坐标点，用DRAW_RICH_KEYPOINTS可以看方向
    cv::drawKeypoints(images[0], keypoints[0], draw_img[0], cv::Scalar::all(-1));
    cv::drawKeypoints(images[1], keypoints[1], draw_img[1], cv::Scalar::all(-1));
    cv::Mat displayImg = cv::Mat::zeros(draw_img[0].rows, draw_img[0].cols * 2, draw_img[0].type());
    HorizontalMerge(draw_img, displayImg); //拼接
    cv::imshow("特征点", displayImg);

    //特征匹配
    std::vector<cv::DMatch> matches;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming"); //汉明距离
    matcher->match(descriptor[0], descriptor[1], matches);

    //找最大最小距离
    std::sort(matches.begin(), matches.end(), [](cv::DMatch a, cv::DMatch b) { return a.distance < b.distance; }); //升序
    double minDistance = matches[0].distance;
    double maxDistance = matches[matches.size() - 1].distance;

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

    //画出匹配结果
    cv::Mat displayMatch, displayBetterMatch;
    cv::drawMatches(images[0], keypoints[0], images[1], keypoints[1], matches, displayMatch);
    cv::drawMatches(images[0], keypoints[0], images[1], keypoints[1], betterMatches, displayBetterMatch);
    cv::imshow("匹配结果", displayMatch);
    cv::imshow("过滤后的匹配结果", displayBetterMatch);
    cv::waitKey(0);

    return 0;
}
