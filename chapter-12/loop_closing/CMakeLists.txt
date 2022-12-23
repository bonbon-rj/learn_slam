cmake_minimum_required(VERSION 2.8)
project(loop_closing)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Debug") # Debug Release 

# 使用opencv
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

# 使用dbow
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.so" )

# get_vocabulary
add_executable(get_vocabulary
get_vocabulary.cpp
)
target_link_libraries(get_vocabulary
    ${OpenCV_LIBS} 
    ${DBoW3_LIBS}
) 

# calc_similarity 
add_executable(calc_similarity
    calc_similarity.cpp
)
target_link_libraries(calc_similarity
    ${OpenCV_LIBS} 
    ${DBoW3_LIBS}
) 