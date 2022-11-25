#pragma once
#include "main.h"
#include "utils.h"
#include "cost_function.h"

class BundleAdjustmentProblem
{
public:
    const int camera_param_num = 9; // 相机3+3+1+1+1=9个参数(-R t f k1 k2)
    const int point_param_num = 3;  //  点3个参数(x y z)

    // 扰动
    const double rotation_sigma = 0.0;
    const double translation_sigma = 0.0;
    const double point_sigma = 0.0;

    int cameras_num;
    int points_num;
    int observations_num;
    int paramters_num;

    int *camera_index;
    int *point_index;
    double *observations;
    double *parameters;

    void write_to_ply(std::string file_name);
    void normalize();
    void perturb();
    void build_problem(ceres::Problem *problem);
    void config_option(ceres::Solver::Options *options);
    ceres::Solver::Summary solve_problem(ceres::Solver::Options &options, ceres::Problem &problem);

    template <typename T>
    void my_fscanf(FILE *file, const char *format, T *value)
    {
        // 写这个函数只是为了编译器不警告
        int scan_num = fscanf(file, format, value);
        (void)scan_num;
    }

    BundleAdjustmentProblem(FILE *fptr);
    ~BundleAdjustmentProblem();
};