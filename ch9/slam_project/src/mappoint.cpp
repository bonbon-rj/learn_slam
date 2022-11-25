#include "mappoint.h"

MapPoint::MapPoint(Eigen::Vector3d position,cv::Mat descriptor, Eigen::Vector3d n)
{
    this->position = position;
    this->descriptor = descriptor.clone();
    this->norm = n;

    visible_times = 0;
    match_times = 0;
}

MapPoint::~MapPoint()
{
    
}