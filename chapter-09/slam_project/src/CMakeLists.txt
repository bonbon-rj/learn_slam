# 添加自定义库名字 动态库
add_library( myslam SHARED
    yaml.cpp
    camera.cpp
    frame.cpp
    map.cpp
    mappoint.cpp
    utils.cpp
    vo.cpp
)

# 链接自定义库和设置的第三方库
target_link_libraries( myslam
    ${THIRD_PARTY_LIBS}
)
 

