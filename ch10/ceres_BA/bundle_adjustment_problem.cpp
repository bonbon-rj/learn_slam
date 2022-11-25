#include "bundle_adjustment_problem.h"

BundleAdjustmentProblem::BundleAdjustmentProblem(FILE *fptr)
{
	//读第一行 相机数量 点数数量 观测数量
	my_fscanf(fptr, "%d", &cameras_num);
	my_fscanf(fptr, "%d", &points_num);
	my_fscanf(fptr, "%d", &observations_num);
	std::cout << "File header:" << cameras_num << " " << points_num << " " << observations_num << std::endl;

	//读数据 每一行数据： 相机下标 点下标 x y
	camera_index = new int[observations_num];
	point_index = new int[observations_num];
	observations = new double[2 * observations_num];
	for (int i = 0; i < observations_num; i++)
	{
		my_fscanf(fptr, "%d", camera_index + i);
		my_fscanf(fptr, "%d", point_index + i);
		for (int j = 0; j < 2; j++)
		{
			my_fscanf(fptr, "%lf", observations + 2 * i + j);
		}
	}

	//读参数
	paramters_num = camera_param_num * cameras_num + point_param_num * points_num;
	parameters = new double[paramters_num];
	for (int i = 0; i < paramters_num; i++)
	{
		my_fscanf(fptr, "%lf", parameters + i);
	}

	fclose(fptr);
}

BundleAdjustmentProblem::~BundleAdjustmentProblem()
{
	delete[] camera_index;
	delete[] point_index;
	delete[] observations;
	delete[] parameters;
}

void BundleAdjustmentProblem::write_to_ply(std::string file_name)
{
	std::ofstream of(file_name.c_str());

	of << "ply" << '\n'
	   << "format ascii 1.0" << '\n'							// 编码格式
	   << "element vertex " << cameras_num + points_num << '\n' // 顶点
	   << "property float x" << '\n'							// 一行数据分别是x y z r g b
	   << "property float y" << '\n'
	   << "property float z" << '\n'
	   << "property uchar red" << '\n'
	   << "property uchar green" << '\n'
	   << "property uchar blue" << '\n'
	   << "end_header" << std::endl;

	// parameter前半段为相机参数 将相机中心表示为绿色
	for (int i = 0; i < cameras_num; i++)
	{
		//指针指向第i个相机参数头
		const double *camera = parameters + camera_param_num * i;

		//求光心 也就是齐次变换矩阵求逆后的平移向量部分 也就是-Rp
		/*
		T = |R  p|
			|0  1|
		T^(-1)  = |R^T  -Rp|
				  |0     1 |
		*/
		Eigen::Vector3d rotate_result;
		Eigen::Vector3d angle_axis(-(*(camera)), -(*(camera + 1)), -(*(camera + 2))); // 注意数据集给的是-R 所以需要取负
		Eigen::Vector3d point(*(camera + 3), *(camera + 4), *(camera + 5));
		angleAxis_rotate_point(angle_axis, point, rotate_result);
		Eigen::Vector3d camera_center = -rotate_result; // rotate_result是Rp 获得光心要取负

		//写入光心
		for (int j = 0; j < camera_center.size(); j++)
		{
			of << camera_center(j) << ' ';
		}

		//写入颜色
		of << "0 255 0\n";
	}

	// parameter后半段为点参数 将点表示为白色
	const double *points = parameters + camera_param_num * cameras_num; // 指针指向点参数头
	for (int i = 0; i < points_num; i++)
	{
		//指针指向第i个点参数头
		const double *point = points + i * point_param_num;

		//写入点
		for (int j = 0; j < point_param_num; j++)
		{
			of << *point << ' ';
			point++;
		}

		//写入颜色
		of << "255 255 255\n";
	}

	of.close();

	std::cout << "Write to " << file_name << " successfully" << std::endl;
}

void BundleAdjustmentProblem::normalize()
{
	//********************计算归一化因子********************//
	// 计算所有点的中位数
	double *points = parameters + camera_param_num * cameras_num;
	Eigen::Vector3d median;
	std::vector<std::vector<double>> xyz(3); // 二维数组内0 1 2 分别表示 x y z
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < points_num; j++)
		{
			xyz[i].push_back(*(points + 3 * j + i));
		}
		median(i) = get_vector_median(xyz[i]);
	}

	// 计算所有一范数 并取中位数作为中值绝对偏差
	std::vector<double> norm1(points_num); //一范数
	for (int i = 0; i < points_num; i++)
	{
		double *point = points + 3 * i;
		Eigen::Vector3d pt(*(point), *(point + 1), *(point + 2));
		norm1[i] = (pt - median).lpNorm<1>();
	}
	double median_absolute_deviation = get_vector_median(norm1); //中值绝对偏差
	double scale = 100.0 / median_absolute_deviation;			 //归一化因子

	//********************点归一化********************//
	for (int i = 0; i < points_num; i++)
	{
		//计算归一化后的向量
		double *point = points + 3 * i;
		Eigen::Vector3d pt(*(point), *(point + 1), *(point + 2));
		pt = scale * (pt - median);

		//更新到原指针处
		for (int j = 0; j < 3; j++)
		{
			(*(point + j)) = pt[j];
		}
	}

	//********************相机参数归一化********************//
	double *cameras = parameters; //指向相机参数头
	for (int i = 0; i < cameras_num; i++)
	{
		double *camera = cameras + camera_param_num * i; // 指向第i个相机参数头

		// R t 求光心
		Eigen::Vector3d rotate_result;
		Eigen::Vector3d angle_axis(-(*(camera)), -(*(camera + 1)), -(*(camera + 2))); // 注意数据集给的是-R 所以需要取负
		Eigen::Vector3d point(*(camera + 3), *(camera + 4), *(camera + 5));
		angleAxis_rotate_point(angle_axis, point, rotate_result);
		Eigen::Vector3d camera_center = -rotate_result; // rotate_result是Rp 获得光心要取负

		// 归一化
		camera_center = scale * (camera_center - median);

		// 光心返回更新t
		Eigen::Vector3d rotate_result1;
		Eigen::Vector3d angle_axis1(*(camera), *(camera + 1), *(camera + 2)); // 返回去 左乘逆矩阵 所以不用取负
		angleAxis_rotate_point(angle_axis1, camera_center, rotate_result1);
		Eigen::Vector3d result = -rotate_result1;

		// 更新到原指针处
		for (int j = 0; j < 3; j++)
		{
			*(camera + 3 + j) = result(j);
		}
	}

	std::cout << "Normalize successfully" << std::endl;
}

void BundleAdjustmentProblem::perturb()
{
	// 确定随机种子
	int random_seed = 38401;
	srand(random_seed);

	// 点加扰动
	double *points = parameters + camera_param_num * cameras_num;
	if (point_sigma > 0)
	{
		for (int i = 0; i < points_num; i++)
		{
			double *point = points + 3 * i;
			for (int j = 0; j < 3; j++)
			{
				(*(point + j)) = normal_rand(*(point + j), point_sigma);
			}
		}
	}

	//相机加扰动
	double *cameras = parameters;
	for (int i = 0; i < cameras_num; i++)
	{
		double *camera = cameras + camera_param_num * i;

		// R t 求光心
		Eigen::Vector3d rotate_result;
		Eigen::Vector3d angle_axis(-(*(camera)), -(*(camera + 1)), -(*(camera + 2))); // 注意数据集给的是-R 所以需要取负
		Eigen::Vector3d point(*(camera + 3), *(camera + 4), *(camera + 5));
		angleAxis_rotate_point(angle_axis, point, rotate_result);
		Eigen::Vector3d camera_center = -rotate_result; // rotate_result是Rp 获得光心要取负

		// R加扰动
		if (rotation_sigma > 0.0)
		{
			for (int j = 0; j < 3; j++)
			{
				(*(camera + j)) = normal_rand(*(camera + j), rotation_sigma);
			}
		}

		// 光心返回t
		Eigen::Vector3d rotate_result1;
		Eigen::Vector3d angle_axis1(*(camera), *(camera + 1), *(camera + 2)); // 返回去所以不用取负
		angleAxis_rotate_point(angle_axis1, camera_center, rotate_result1);
		Eigen::Vector3d result = -rotate_result;

		// t加扰动
		if (translation_sigma > 0.0)
		{
			for (int j = 0; j < 3; j++)
			{
				*(camera + 3 + j) = normal_rand(*(camera + 3 + j), translation_sigma);
			}
		}
	}

	std::cout << "Perturb successfully" << std::endl;
}

void BundleAdjustmentProblem::build_problem(ceres::Problem *problem)
{
	double *cameras = parameters;
	double *points = parameters + camera_param_num * cameras_num;

	for (int i = 0; i < observations_num; i++)
	{
		ceres::CostFunction *cost_function;
		cost_function = ReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);
		ceres::LossFunction *loss_function = NULL; // new ceres::HuberLoss(1.0); // CauchyLoss(1.0)
		double *camera = cameras + camera_param_num * camera_index[i];
		double *point = points + point_param_num * point_index[i];
		problem->AddResidualBlock(cost_function, loss_function, camera, point);
	}

	std::cout << "Build problem successfully" << std::endl;
}

void BundleAdjustmentProblem::config_option(ceres::Solver::Options *options)
{
	options->max_num_iterations = 10;
	options->minimizer_progress_to_stdout = true;
	options->num_threads = 1;
	options->trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;  // Options are: levenberg_marquardt, dogleg.
	options->linear_solver_type = ceres::DENSE_SCHUR;				   // Options are: sparse_schur, dense_schur, sparse_normal_cholesky.
	options->sparse_linear_algebra_library_type = ceres::SUITE_SPARSE; // Options are: suite_sparse and cx_sparse.
	options->dense_linear_algebra_library_type = ceres::EIGEN;		   // Options are: eigen and lapack.

	// Schur的消元顺序
	bool isAutomaticOrder = true;
	if (!isAutomaticOrder)
	{
		// 手动排序则把点排在相机前
		double *points = parameters + camera_param_num * cameras_num;
		double *cameras = parameters;
		ceres::ParameterBlockOrdering *ordering = new ceres::ParameterBlockOrdering;

		for (int i = 0; i < points_num; i++)
		{
			ordering->AddElementToGroup(points + point_param_num * i, 0);
		}
		for (int i = 0; i < cameras_num; i++)
		{
			ordering->AddElementToGroup(cameras + camera_param_num * i, 1);
		}

		options->linear_solver_ordering.reset(ordering);
	}

	options->gradient_tolerance = 1e-16;
	options->function_tolerance = 1e-16;

	std::cout << "Config options successfully" << std::endl;
}

ceres::Solver::Summary BundleAdjustmentProblem::solve_problem(ceres::Solver::Options &options, ceres::Problem &problem)
{
	ceres::Solver::Summary summary;
	std::cout << "Start solve problem..." << std::endl;
	ceres::Solve(options, &problem, &summary);

	std::cout << "Solve problem done" << std::endl;
	std::cout << summary.FullReport() << std::endl;

	return summary;
}
