cmake_minimum_required(VERSION 2.8)
project(dense_RGBD)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release") # Debug Release 

# 使用eigen
include_directories("/usr/include/eigen3")

# 使用opencv
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

# 使用pcl
find_package( PCL REQUIRED )
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 使用octomap 
find_package( octomap REQUIRED )
include_directories(${OCTOMAP_INCLUDE_DIRS})

# 点云地图
add_executable(pointcloud_mapping
    pointcloud_mapping.cpp
)
target_link_libraries(pointcloud_mapping
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES}
) 

# 八叉树地图
add_executable(octomap_mapping
    octomap_mapping.cpp
)
target_link_libraries(octomap_mapping
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES}
    ${OCTOMAP_LIBRARIES}
) 