#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>
#include <DBoW3/DBoW3.h>

int main(int argc, char **argv)
{
    // 获取数据集特征点描述子
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
    std::vector<cv::Mat> descriptors;
    boost::format fmt("../pic/%d.png");
    int index = 1;
    while (1)
    {
        cv::Mat img = cv::imread((fmt % (index++)).str());
        if (img.empty())
            break;

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        orb->detect(img, keypoints);
        orb->compute(img, keypoints, descriptor);
        descriptors.push_back(descriptor);
    }

    // 生成字典并保存
    DBoW3::Vocabulary vocabulary;
    vocabulary.create(descriptors);
    std::cout.setstate(std::ios_base::failbit); //保存时会把字典打印出来 所以暂时禁用掉cout
    vocabulary.save("./vocabulary.yml.gz");
    std::cout.clear(); //停止禁用

    return 0;
}