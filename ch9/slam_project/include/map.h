#pragma once
#include "main.h"
#include "mappoint.h"
#include "frame.h"

class Map
{
public:
    std::list<Frame> frame_list;
    std::list<MapPoint> mappoint_list;

    void insertFrame(Frame frame);
    void insertMapPoint(MapPoint mappoint);
    
    Map();
    ~Map();
};