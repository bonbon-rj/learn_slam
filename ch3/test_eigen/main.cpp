#include <iostream>
#include <Eigen/Dense>

int main(void)
{
    // //定义
    // Eigen::Matrix<float,2,3> M;//指定类型和行列
    // Eigen::Matrix3f R; //内置3*3的float矩阵
    // Eigen::Vector3d d; //内置的3*1的向量
    // Eigen::MatrixXd C;//动态矩阵

    // //初始化
    // for(int i =0;i<M.rows();i++) //循环
    // {
    //     for(int j=0;j<M.cols();j++)
    //     {
    //         M(i,j)= 1;
    //     }
    // }
    // R << 0,1,0,1,0,0,0,0,1; //流输入
    // d =  Eigen::Vector3d::Zero(); //内置函数初始化

    // //输出
    // std::cout<<M<<std::endl;
    // std::cout<<R<<std::endl;
    // std::cout<<d<<std::endl;
    // std::cout<<"\n";

    // //运算
    // std::cout<<R.cast<double>()*d<<std::endl; //不同类型运算要先转换
    // std::cout<<R.transpose()<<std::endl;//转置
    // std::cout<<R.sum()<<std::endl;//求和
    // std::cout<<R.trace()<<std::endl;//迹
    // std::cout<<R.inverse()<<std::endl;//逆
    // std::cout<<R.determinant()<<std::endl;//行列式
    // std::cout<<"\n";

    // //特征值分解
    // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(R);
    // std::cout<<solver.eigenvalues()<<std::endl; //特征值
    // std::cout<<solver.eigenvectors()<<std::endl; //特征向量
    // std::cout<<"\n";

    // //解方程 A*x = B 矩阵分解取代求逆 高效化
    // Eigen::Matrix3d A = Eigen::Matrix3d::Random();
    // Eigen::Matrix3d B = Eigen::Matrix3d::Random();
    // std::cout<<A.inverse()*B<<std::endl; //逆
    // std::cout<<A.colPivHouseholderQr().solve(B)<<std::endl;//求解方程
    // std::cout<<"\n";

    // *旋转矩阵用3*3矩阵表示即可 下列表示绕z轴旋转30°
    Eigen::Matrix3d R;
    // 这里犯了低级错误 记录一下 在c++里 1/2取整结果是0
    // R<< sqrt(3)/2,-1/2,0,
    //     1/2,sqrt(3)/2,0,
    //     0,0,1;
    R << cos(M_PI / 6), -sin(M_PI / 6), 0,
        sin(M_PI / 6), cos(M_PI / 6), 0,
        0, 0, 1;

    // *角轴 沿z轴转30° w.matrix()和w.toRotationMatrix()代表对应的旋转矩阵
    Eigen::AngleAxisd w(M_PI / 6, Eigen::Vector3d(0, 0, 1));

    // *四元数 Eigen::Quaternion
    Eigen::Quaterniond q = Eigen::Quaterniond(R);     //也可以Eigen::Quaterniond (w)
    std::cout << q.coeffs().transpose() << std::endl; //顺序是(p1,p2,p3,p0),p0为实部

    // *旋转向量
    Eigen::Vector3d v0(0.2, 0.3, 0.4), v;
    v = w * v0; //角轴旋转
    std::cout << v.transpose() << std::endl;
    v = R * v0; //旋转矩阵旋转
    std::cout << v.transpose() << std::endl;
    v = q * v0; // 四元数旋转
    std::cout << v.transpose() << std::endl;

    // *旋转矩阵转欧拉角
    Eigen::Vector3d euler = R.eulerAngles(2, 1, 0); // 012对应xyz 这里表示zyx顺序

    // *齐次变换矩阵 Eigen::Isometry
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity(); //单位阵
    T.rotate(R);                                         // 设置旋转部分 或者T.rotate(w);
    T.pretranslate(Eigen::Vector3d(1, 3, 4));            //设置平移部分
    std::cout << (T * v).transpose() << std::endl;       //齐次变换 不能用T.matrix() 因为是4*4的

    return 0;
}