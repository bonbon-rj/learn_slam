#设置输出名
set(OUTPUT_NAME demo)

#编译可执行文件
add_executable(${OUTPUT_NAME} 
    main.cpp
)

#链接到自定义库
target_link_libraries(${OUTPUT_NAME} myslam)