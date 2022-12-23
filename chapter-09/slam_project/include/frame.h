#pragma once
#include "main.h"

class Frame
{
public:

    cv::Mat color,gray,depth;
    Sophus::SE3d Tcw;

    bool getColorSuccess =false;
    bool getDepthSuccess = false;

    Frame();
    ~Frame();
    bool checkSuccess();
    bool isInFrame(double x,double y);
    void getColor(std::string path);
    void getDepth(std::string path); 
    Eigen::Vector3d getCamCenter();
    uchar getPixel(int u,int v);
    ushort findDepth(int u,int v);
    Frame clone();
};