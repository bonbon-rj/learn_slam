cmake_minimum_required(VERSION 2.8)
project(direct_methods) 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")

# 使用eigen
include_directories("/usr/include/eigen3")

# 使用opencv
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

# 使用ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# 使用sophus
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

#稀疏
add_executable(direct_sparse
    direct_sparse.cpp
)
target_link_libraries(direct_sparse
    ${CERES_LIBRARIES}
    ${OpenCV_LIBS} 
    ${Sophus_LIBRARIES} 
    Sophus::Sophus
) 

# 半稠密
add_executable(direct_semidense
    direct_semidense.cpp
)
target_link_libraries(direct_semidense
    ${CERES_LIBRARIES}
    ${OpenCV_LIBS} 
    ${Sophus_LIBRARIES} 
    Sophus::Sophus
) 


