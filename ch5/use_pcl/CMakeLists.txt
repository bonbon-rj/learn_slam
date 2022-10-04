cmake_minimum_required(VERSION 2.8)
project(hello_pcl) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")

# 使用opencv
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

# 使用eigen
include_directories("/usr/include/eigen3")

# 使用pcl
find_package( PCL REQUIRED )
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(${PROJECT_NAME}
	main.cpp
)
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES}
) 