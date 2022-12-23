#include "map.h"

Map::Map()
{

}

Map::~Map()
{
    
}

void Map::insertFrame(Frame frame)
{
    frame_list.push_back(frame);
}

void Map::insertMapPoint(MapPoint mappoint)
{
    mappoint_list.push_back(mappoint);
}

