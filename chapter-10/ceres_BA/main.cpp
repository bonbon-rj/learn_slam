#include "main.h"
#include "bundle_adjustment_problem.h"

int main(int argc, char **argv)
{
    std::string file_name = "../data/problem-16-22106-pre.txt";

    FILE *fptr = fopen(file_name.c_str(), "r"); //只读
    if (fptr == NULL)
    {
        std::cerr << "Uable to open file:" << file_name << std::endl;
        return -1;
    }

    // 初始化
    BundleAdjustmentProblem *BA_problem = new BundleAdjustmentProblem(fptr);
    BA_problem->write_to_ply("./initialize.ply");

    // 正则化
    BA_problem->normalize();
    BA_problem->write_to_ply("./normalize.ply");

    // 添加扰动
    BA_problem->perturb();
    BA_problem->write_to_ply("./perturb.ply");

    // 求解
    ceres::Problem problem;
    ceres::Solver::Options options;
    BA_problem->build_problem(&problem);
    BA_problem->config_option(&options);
    BA_problem->solve_problem(options, problem);
    BA_problem->write_to_ply("./result.ply");

    return 0;
}
