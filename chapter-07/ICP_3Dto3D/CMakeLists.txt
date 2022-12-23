cmake_minimum_required(VERSION 2.8)
project(ICP_3Dto2D) 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Debug")

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

add_executable(${PROJECT_NAME}
    main.cpp
)
target_link_libraries(${PROJECT_NAME}
    ${CERES_LIBRARIES}
    ${OpenCV_LIBS} 
    ${Sophus_LIBRARIES} 
) 

# 解决 fatal error: fmt/core.h: 没有那个文件或目录
target_link_libraries(${PROJECT_NAME} Sophus::Sophus) 
