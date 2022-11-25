#include "frame.h"

Frame::Frame()
{

}
Frame::~Frame()
{
    
}

void Frame::getColor(std::string path)
{
    color = cv::imread(path);
    if (!color.empty())
    {
        getColorSuccess = true;
        cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        getColorSuccess = false;
    }
}

void Frame::getDepth(std::string path)
{
    depth = cv::imread(path, -1);
    if (!depth.empty())
    {
        getDepthSuccess = true;
    }
    else
    {
        getDepthSuccess = false;
    }
}

bool Frame::checkSuccess()
{
    if (getColorSuccess & getDepthSuccess)
        return 1;
    return 0;
}

uchar Frame::getPixel(int u, int v)
{
    return gray.ptr<uchar>(v)[u];
}

ushort Frame::findDepth(int u, int v)
{
    ushort d = depth.ptr<ushort>(v)[u];
    if ( d!=0 )
    {
        return d;
    }
    else 
    {
        // check the nearby points 
        int dx[4] = {-1,0,1,0};
        int dy[4] = {0,-1,0,1};
        for ( int i=0; i<4; i++ )
        {
            d = depth.ptr<ushort>( v+dy[i] )[u+dx[i]];
            if ( d!=0 )
            {
                return d;
            }
        }
    }
    return 0;
}

Frame Frame::clone()
{
    // Mat类是浅拷贝 要用clone
    Frame f;
    f.gray = gray.clone();
    f.color = color.clone();
    f.depth = depth.clone();
    f.Tcw = Tcw;
    return f;
}

bool Frame::isInFrame(double x,double y)
{
    return x > 0 && x < color.cols && y > 0 && y < color.rows;
}

Eigen::Vector3d Frame::getCamCenter()
{
    return Tcw.matrix().inverse().block<3, 1>(0, 3);
}
