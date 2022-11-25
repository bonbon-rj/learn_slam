# !usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(0)

    # 读取data 并根据文件名排序
    file_dir = sys.argv[1]
    try:
        rgb = os.listdir(file_dir)
        sort_time_rgb = sorted(rgb, key=lambda x: float(x[:-4]))
    except:
        print("请将data放在正确目录下")
        os.sys.exit(0)

    # 检查保存文件夹是否存在
    save_dir = sys.argv[2]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 拷贝
    count = 0
    for n in sort_time_rgb:
        save_name = str(count)+".png"
        cmd = "cp " + \
            os.path.join(file_dir, n) + " " + os.path.join(save_dir, save_name)
        print(cmd)
        os.system(cmd)
        count += 1
