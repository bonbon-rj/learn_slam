#pragma once
#include "main.h"

class Yaml
{
public:
    Yaml(std::string filename);
    ~Yaml();
    
    cv::Mat getMat(std::string key);

    template <typename T>
    T getArgs(std::string key);

private:
    cv::FileStorage f;
};