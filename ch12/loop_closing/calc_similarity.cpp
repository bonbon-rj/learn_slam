#include <iostream>
#include <DBoW3/DBoW3.h>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>

int main(int argc, char **argv)
{
    // 加载字典
    DBoW3::Vocabulary vocabulary("./vocabulary.yml.gz");
    if (vocabulary.empty())
        return -1;

    // 获取描述子
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

    // 通过图片计算相似度
    std::cout << "calc similarity from images..." << std::endl;
    for (int i = 0; i < descriptors.size(); i++)
    {
        // 图像描述子转为词袋向量
        DBoW3::BowVector v1;
        vocabulary.transform(descriptors[i], v1);

        // 计算每一张图像与其他图像之间的相似度
        for (int j = i; j < descriptors.size(); j++)
        {
            if (i == j)
                continue;

            DBoW3::BowVector v2;
            vocabulary.transform(descriptors[j], v2);

            double score = vocabulary.score(v1, v2);

            std::cout << "The similarity between images " << i << " and " << j << " is " << score << std::endl;
        }
    }
    std::cout << std::endl;

    // 通过database计算相似度
    std::cout << "calc similarity from database..." << std::endl;
    DBoW3::Database database(vocabulary, false, 0); // 第二个参数表示是否使用direct_index false表示不使用顺序索引
    for (int i = 0; i < descriptors.size(); i++)
        database.add(descriptors[i]);
    for (int i = 0; i < descriptors.size(); i++)
    {
        DBoW3::QueryResults ret;
        int max_result = 4; // max_result指定输出最相似的个数
        database.query(descriptors[i], ret, max_result);
        std::cout << "Serach " << max_result << " images most sililar to image " << i << ":" << std::endl;
        for (int j = 0; j < ret.size(); j++)
        {
            std::cout << "The similarity with image " << ret.at(j).Id
                      << " is " << ret.at(j).Score << std::endl;
        }
    }

    return 0;
}