#include "yaml.h"

Yaml::Yaml(std::string filename)
{
    f = cv::FileStorage(filename.c_str(),cv::FileStorage::READ);
}

Yaml::~Yaml()
{
    f.release();
}

cv::Mat Yaml::getMat(std::string key)
{
    cv::Mat temp;
    f[key]>>temp;
    return temp;
}

template <typename T>
T Yaml::getArgs(std::string key)
{
    return T(f[key]);
}
//模板函数定义和声明分开放要显式表明支持的类型
template int Yaml::getArgs(std::string); 
template double Yaml::getArgs(std::string); 
template std::string Yaml::getArgs(std::string);
