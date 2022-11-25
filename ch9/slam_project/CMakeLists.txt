cmake_minimum_required(VERSION 2.8) 
project(slam_project) 
set(CMAKE_CXX_STANDARD 14) #设置版本
set(CMAKE_BUILD_TYPE "Debug") #设置模式
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin ) #设置输出可执行二进制路径
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib ) #设置编译库文件输出路径

# 使用eigen
include_directories("/usr/include/eigen3")

# 使用opencv
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

# 使用ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# 使用sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# 设置第三方库
set(THIRD_PARTY_LIBS
    ${CERES_LIBRARIES}
    ${OpenCV_LIBS} 
    ${Sophus_LIBRARIES} 
    Sophus::Sophus 
)

# 添加include目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 添加子目录
add_subdirectory(src)
add_subdirectory(test)