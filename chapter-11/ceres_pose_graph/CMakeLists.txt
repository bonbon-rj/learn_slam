cmake_minimum_required(VERSION 2.8)
project(ceres_pose_graph)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release") # Debug Release 

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
    pose.cpp
)


target_link_libraries(${PROJECT_NAME}
    ${CERES_LIBRARIES}
    ${Sophus_LIBRARIES} 
    Sophus::Sophus
) 