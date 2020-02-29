# Gym-Carla Working Log

Copyright (c) 2020

created by Dongjie Yu (yudongjie.moon@foxmail.com)

* This work is modified from: <https://github.com/cjy1992/gym-carla>:
* Copyright (c) 2019:
* author: Jianyu Chen (jianyuchen@berkeley.edu)
* For a copy, see <https://opensource.org/licenses/MIT>.

---

## **Development log**
### **时间日志**
* 2020.02.27
    * 已开发
        1. 搭建出了整体框架，即确定了除了路径点之外的功能
        2. **指令方案初步定义为：纯指令，加上自车状态输入**
    * 待开发
        1. 先确定一些环境初始化参数
        2. 开发出创建环境与reset方法，确定terminal
        3. 能够在手动输入的情况下结束、重置
        4. resize后的render功能
* 2020.02.28
    * 已开发：
        1. 开始编写```__init__()```函数，目前写到自车生成部分
        2. **重要：**
           找到了起点与终点的数据，初步确定为前5个[0~4]，可用于多进程训练；任务可以精简为【直行】、【一次转弯】、【一次转弯带动态物体】，视时间决定
    * 待开发：
        1. 完成```__init__()```方法，并确认route teller方案
        2. 尝试运行__init__()与reset()测试路线有无交叉或重合，录入Town01\02剩余数据
        3. 编写terminal有关到达终点的条件，使用autopilot验证(?)
* 2020.02.29
    * 已开发：
        1. 成功实现了自车生成、传感器attach，并通过cv2显示图像
    * 待开发：
        1. 实现车辆的运动，实现一个简单的直行10s，判断图像是否实时改变
        2. 根据简单直行，调试step()、terminal()
        3. 学习route teller与驾驶执行的实现方式
---

## **Experiment Log**


---