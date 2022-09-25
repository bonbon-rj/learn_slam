cmake_minimum_required(VERSION 2.8)
project(hello_eigen) 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Debug")# Debug Release

# 使用eigen
include_directories("/usr/include/eigen3")

add_executable(${PROJECT_NAME}
	main.cpp
)
