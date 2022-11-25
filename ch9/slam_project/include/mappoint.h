#pragma once
#include "main.h"

class MapPoint
{
public:
    Eigen::Vector3d position;
    cv::Mat descriptor;
    Eigen::Vector3d norm; 

    int visible_times; 
    int match_times;

    MapPoint(Eigen::Vector3d position,cv::Mat descriptor, Eigen::Vector3d n);
    ~MapPoint();
};