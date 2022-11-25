#pragma once
#include "main.h"

// 内参
const double fx = 481.2f; 
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;

Eigen::Vector3d px2cam(const Eigen::Vector2d px);
Eigen::Vector2d cam2px(const Eigen::Vector3d p_cam);
double getPixelValue(const cv::Mat *gray, Eigen::Vector2d p);
double calc_ncc(const cv::Mat &ref, const cv::Mat &curr, const Eigen::Vector2d &pt_ref, const Eigen::Vector2d &pt_curr);