cmake_minimum_required(VERSION 2.8)
project(hello_g2o) 
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")

# 使用G2O
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

#使用eigen
include_directories("/usr/include/eigen3")

add_executable(${PROJECT_NAME}
	main.cpp
)
# 链接G2O
target_link_libraries(${PROJECT_NAME}
    g2o_core g2o_stuff
)

