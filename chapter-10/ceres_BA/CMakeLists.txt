cmake_minimum_required(VERSION 2.8)
project(ceres_BA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release") # 调试的时候用Debug 求解的时候用Release 用Debug求解时间会多出几十倍

# 使用eigen
include_directories("/usr/include/eigen3")

# 使用sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# 使用ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}
    main.cpp
    utils.cpp
    bundle_adjustment_problem.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${CERES_LIBRARIES}
    ${Sophus_LIBRARIES} 
    Sophus::Sophus
) 
