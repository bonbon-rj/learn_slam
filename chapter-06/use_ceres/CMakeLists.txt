cmake_minimum_required(VERSION 2.8)
project(hello_ceres) 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Debug")

# 使用ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}
	main.cpp
)

# 链接ceres
target_link_libraries(${PROJECT_NAME}
    ${CERES_LIBRARIES}
)
