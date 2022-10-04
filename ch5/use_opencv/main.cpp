#include <iostream>
#include <opencv2/opencv.hpp>

int main(void)
{

    cv::Mat img = cv::imread("../img.jpg");//注意相对路径按build中写
    if(img.empty()) return -1;
    cv::resize(img,img,cv::Size(256,256));


    // 遍历图像 将图像置黑 用指针效率高 不要用.at
    // 内循环 从左往右 外循环 从上到下
    // 宽 长 通道 -> 列 行 通道 -> cols rows channels()
    //图像拷贝要用img.clone()不能直接赋值 不然修改拷贝会导致原图像修改
    cv::Mat black = img.clone();
    for(int i=0;i<black.rows;i++)
    {
        unsigned char * row_ptr =black.ptr<unsigned char>(i);//行头指针
        for(int j=0;j<black.cols;j++)
        {
            unsigned char * data_ptr = &row_ptr[j*black.channels()]; //(x,y)处指针
            for(int c=0;c<black.channels();c++)
            {
                data_ptr[c] = 0;
            }
        }
    }
    
    cv::imshow("img",img);
    cv::imshow("black",black);
    cv::waitKey(0);

    cv::destroyAllWindows();
    return 0;
}