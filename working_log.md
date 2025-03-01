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
* 2020.03.01
    * 已开发：
        1. 实现了自车的油门踩到底，图片会实时改变；设定终点并实现```terminal()```符合要求，即done之后```reset()```
    * 待开发：
        1. 完善```step()```、```terminal()```
        2. 学习route teller与驾驶执行的实现方式
        3. 接下来1-2天会暂停开发，学习[https://github.com/carla-rl-gym/carla-rl](https://github.com/carla-rl-gym/carla-rl)的结构与route指令的实现方式

* 2020.03.09（时隔多天终于开发了是吗）
    * 已开发：
        1. 完成了```step()```与```terminal()```的初步版本，并加入了planner，看起来直行的时候并没有问题
    * 待开发：
        1. 完成reward函数的设计与实现（考虑元素：距离目标远近……）
        2. 测试gym-carla框架
        3. 构思需要修改良哥代码的哪些部分

* 2020.03.11
    * 已开发：
        1. 完成了reward函数设计与实现；```state_info```的设计与实现
    * 待开发：
        1. 测试reward、state_info
* 2020.03.12(植树节快乐！)
    * 已开发：
        1. 测试了整个框架的运行，直行目前没有问题
        2. 路径规划部分在city_track注释了对A*算法的修正，变成了朴素的版本；目前没有出现**莫名其妙的右转**
    * 待开发：
        1. 构思DSAC代码需要修改的部分，明确需求
        2. 设计神经网络结构
        3. 确定需要的自车信息，将其也加入state_info

* 2020.03.29(三月就快结束了吗)
    * 中期前的工作：
        1. 完成了框架，精简了一下网络之后能运行50万次迭代，但是不知道训练了个什么东西
        2. 目前的reward除了吸收态之外包含对速度的奖励和对方向的惩罚(只适用直行)
        3. 改了一下相机的位置和图片的分辨率，准备再训练一次，网络大了非常多
    * 接下来一周的工作：
        1. 争取训练出成功的直行网络，并且在没有驾驶指令的情况下做一个对比
        2. 思考转弯的环境怎么设计：生成、奖励等
---

## **Experiment Log**


---