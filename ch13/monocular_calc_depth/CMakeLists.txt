cmake_minimum_required(VERSION 2.8)
project(monocular_calc_depth)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release") # Debug Release 

# 使用eigen
include_directories("/usr/include/eigen3")

# 使用sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# 使用opencv
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}
    main.cpp
    utils.cpp
    # dense_mapping.cpp
)
target_link_libraries(${PROJECT_NAME}
    ${Sophus_LIBRARIES} 
    Sophus::Sophus
    ${OpenCV_LIBS} 
) 

