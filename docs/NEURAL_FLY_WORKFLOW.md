# Neural-Fly 完整工作流程指南

本文档详细说明如何在XTDrone仿真环境和实机上实现Neural-Fly的数据采集、模型训练与部署和测评的完整流程。

## 目录

- [概述](#概述)
- [环境准备](#环境准备)
- [第一部分：XTDrone仿真环境](#第一部分xtdrone仿真环境)
  - [1. 仿真数据采集](#1-仿真数据采集)
  - [2. 模型训练](#2-模型训练)
  - [3. 仿真部署与测评](#3-仿真部署与测评)
- [第二部分：实机环境](#第二部分实机环境)
  - [1. 实机数据采集](#1-实机数据采集)
  - [2. 模型训练](#2-模型训练-1)
  - [3. 实机部署与测评](#3-实机部署与测评)
- [配置参数说明](#配置参数说明)
- [常见问题](#常见问题)

---

## 概述

Neural-Fly复现工作分为三个主要阶段：
1. **数据采集**：使用SE3控制器飞行随机轨迹，采集状态数据用于训练
2. **模型训练**：使用采集的数据训练神经网络扰动观测器
3. **部署与测评**：部署训练好的模型，使用Figure8轨迹进行性能测评

### 控制器说明
- **SE3Controller**: 用于数据采集阶段的基准控制器
- **Neural_Fly_Control**: 带神经网络扰动补偿的自适应控制器

---

## 环境准备

### 软件依赖
```bash
# 1. ROS Noetic安装
sudo apt update
sudo apt install ros-noetic-desktop-full

# 2. MAVROS安装
sudo apt install ros-noetic-mavros
cd /opt/ros/noetic/lib/mavros
sudo ./install_geographiclib_datasets.sh

# 3. LibTorch安装 (用于神经网络推理)
# 下载地址: https://pytorch.org/get-started/locally/
# 解压到 /usr/local/libtorch

# 4. 编译工作空间
cd ~/catkin_ws  # 或您的工作空间路径
catkin_make
source devel/setup.bash
```

---

## 第一部分：XTDrone仿真环境

### 1. 仿真数据采集

#### 步骤1.1：启动XTDrone仿真环境

**终端1** - 启动Gazebo仿真
```bash
cd ~/PX4-Autopilot
source Tools/setup_gazebo.bash $(pwd) $(pwd)/build/px4_sitl_default
roslaunch px4 mavros_posix_sitl.launch
```

**终端2** - 启动通信节点
```bash
cd ~/XTDrone/communication
python multirotor_communication.py iris 0
```

#### 步骤1.2：配置数据采集参数

编辑配置文件 `src/realflight_modules/px4ctrl/config/ctrl_param_xtdrone.yaml`:

```yaml
# 数据采集关键参数
csv_filename: "/home/YOUR_USER/data/collect/data.csv"  # 数据保存路径
prefix: "/iris_0"  # 仿真无人机话题前缀

# 扰动观测器设置（数据采集时关闭）
disturbance_obs:
    use: false  # 数据采集时设为false
```

#### 步骤1.3：启动控制器（SE3模式）

> **重要**: 数据采集使用SE3Controller。需要修改代码中的控制器类型：

编辑 `src/realflight_modules/px4ctrl/src/PX4CtrlFSM.cpp` 第16行:
```cpp
// 数据采集时使用SE3Controller
controller = new SE3Controller(param_);

// 部署测评时使用Neural_Fly_Control
// controller = new Neural_Fly_Control(param_);
```

重新编译后启动控制器：
```bash
catkin_make
roslaunch px4ctrl run_ctrl_sim_vision_odom.launch
```

#### 步骤1.4：启动随机轨迹生成器

```bash
roslaunch planner random.launch
```

#### 步骤1.5：起飞并开始数据采集

**终端新开** - 起飞
```bash
rostopic pub -1 /px4ctrl/takeoff_land quadrotor_msgs/TakeoffLand "takeoff_land_cmd: 1"
```

**终端新开** - 启动数据采集服务
```bash
rosservice call /px4ctrl/collect_trigger "{}"
```

#### 步骤1.6：录制ROS Bag（可选）

```bash
rosbag record --tcpnodelay \
  /iris_0/mavros/rc/out \
  /iris_0/mavros/setpoint_raw/target_attitude \
  /iris_0/mavros/imu/data \
  /iris_0/mavros/vision_pose/odom \
  /iris_0/mavros/battery \
  /position_cmd
```

#### 步骤1.7：降落并停止采集

```bash
rostopic pub -1 /px4ctrl/takeoff_land quadrotor_msgs/TakeoffLand "takeoff_land_cmd: 2"
```

### 2. 模型训练

#### 步骤2.1：数据预处理

将采集的CSV数据转换为训练格式：
```python
import pandas as pd
import numpy as np

# 读取采集数据
data = pd.read_csv('/home/YOUR_USER/data/collect/data.csv')

# 提取特征: [velocity(3), quaternion(4), pwm(4)] = 11维
# 提取标签: 扰动力 f
```

#### 步骤2.2：训练神经网络

使用Neural-Fly论文中的网络结构训练模型：
```python
import torch
import torch.nn as nn

class PhiNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 输出3维特征向量
        )
    
    def forward(self, x):
        return self.net(x)

# 训练代码...
# 保存为TorchScript格式
model = PhiNetwork()
# ... 训练 ...
scripted_model = torch.jit.script(model)
scripted_model.save('/home/YOUR_USER/models/net.pt')
```

### 3. 仿真部署与测评

#### 步骤3.1：配置Neural-Fly控制器

编辑 `ctrl_param_xtdrone.yaml`:
```yaml
# 启用扰动观测器
disturbance_obs:
    constant: false
    use: true      # 开启Neural-Fly
    R: 40
    Q: 2.0
    P: 1.0
    lamda: 0.02

# 神经网络模型路径
model_path: "/home/YOUR_USER/models/net.pt"
```

#### 步骤3.2：修改代码使用Neural_Fly_Control

编辑 `PX4CtrlFSM.cpp`:
```cpp
controller = new Neural_Fly_Control(param_);
```

重新编译：
```bash
catkin_make
```

#### 步骤3.3：启动Figure8轨迹测评

**终端1** - 启动控制器
```bash
roslaunch px4ctrl run_ctrl_sim_vision_odom.launch
```

**终端2** - 起飞
```bash
rostopic pub -1 /px4ctrl/takeoff_land quadrotor_msgs/TakeoffLand "takeoff_land_cmd: 1"
```

**终端3** - 启动Figure8轨迹
```bash
roslaunch planner loop.launch
```

#### 步骤3.4：可视化与数据记录

**启动RViz**
```bash
roslaunch planner rviz_wind.launch
```

**记录测评数据**
```bash
sh shfiles/record_data.sh
```

---

## 第二部分：实机环境

### 1. 实机数据采集

#### 步骤1.1：硬件连接检查

```bash
# 检查飞控连接
ls /dev/ttyACM0

# 赋予串口权限
sudo chmod 777 /dev/ttyACM0
```

#### 步骤1.2：启动传感器和定位系统

```bash
sh shfiles/rspx4.sh
```

这将依次启动：
- RealSense相机: `roslaunch realsense2_camera rs_camera.launch`
- MAVROS: `roslaunch mavros px4.launch`
- VINS-Fusion: `roslaunch vins fast_drone_250.launch`

#### 步骤1.3：配置实机参数

编辑 `src/realflight_modules/px4ctrl/config/ctrl_param_fpv.yaml`:

```yaml
mass: 1.4  # 实际飞机重量(kg)
hover_percentage: 0.32  # 悬停油门百分比

# 数据保存路径
csv_filename: "/home/YOUR_USER/data/real_collect/data.csv"

# 数据采集时关闭扰动观测
disturbance_obs:
    use: false
```

#### 步骤1.4：启动SE3控制器

确保代码使用SE3Controller后：
```bash
roslaunch px4ctrl run_ctrl.launch
```

#### 步骤1.5：遥控器准备

- 5通道拨到内侧（进入OFFBOARD准备）
- 6通道拨到下侧（进入命令控制模式）
- 油门打到中位

#### 步骤1.6：自动起飞

```bash
sh shfiles/takeoff.sh
# 或
rostopic pub -1 /px4ctrl/takeoff_land quadrotor_msgs/TakeoffLand "takeoff_land_cmd: 1"
```

#### 步骤1.7：启动随机轨迹并采集数据

**终端新开** - 随机轨迹
```bash
roslaunch planner random.launch
```

**调用数据采集服务**
```bash
rosservice call /px4ctrl/collect_trigger "{}"
```

#### 步骤1.8：降落

```bash
sh shfiles/land.sh
# 或
rostopic pub -1 /px4ctrl/takeoff_land quadrotor_msgs/TakeoffLand "takeoff_land_cmd: 2"
```

### 2. 模型训练

与仿真环境相同，参考[仿真模型训练](#2-模型训练)部分。

### 3. 实机部署与测评

#### 步骤3.1：配置Neural-Fly控制器

编辑 `ctrl_param_fpv.yaml`:
```yaml
disturbance_obs:
    constant: false
    use: true  # 启用Neural-Fly
    R: 40.0
    Q: 1.0
    P: 2.0
    lamda: 0.05

model_path: "/home/YOUR_USER/models/real_net.pt"
```

#### 步骤3.2：修改代码并重新编译

```cpp
// PX4CtrlFSM.cpp
controller = new Neural_Fly_Control(param_);
```

```bash
catkin_make
```

#### 步骤3.3：启动系统

```bash
# 终端1: 传感器
sh shfiles/rspx4.sh

# 终端2: 控制器
roslaunch px4ctrl run_ctrl.launch

# 终端3: 起飞
sh shfiles/takeoff.sh
```

#### 步骤3.4：Figure8轨迹测评

```bash
roslaunch planner loop.launch
```

#### 步骤3.5：记录测评数据

```bash
sh shfiles/record.sh
```

---

## 配置参数说明

### 控制器参数 (ctrl_param_*.yaml)

| 参数 | 说明 | 数据采集 | 部署测评 |
|------|------|----------|----------|
| `disturbance_obs.use` | 是否启用扰动补偿 | `false` | `true` |
| `disturbance_obs.constant` | 网络输出是否设为常数 | - | `false` |
| `disturbance_obs.R` | 卡尔曼滤波测量噪声 | - | 40.0 |
| `disturbance_obs.Q` | 卡尔曼滤波过程噪声 | - | 1.0~2.0 |
| `disturbance_obs.lamda` | 遗忘因子 | - | 0.02~0.05 |
| `model_path` | 神经网络模型路径 | - | 实际路径 |

### 轨迹参数 (loop.launch)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `trajectory_type` | 0=圆形, 1=Figure8 | 1 |
| `a` | Figure8 X轴振幅(m) | 0.75 |
| `b` | Figure8 Z轴振幅(m) | 0.4 |
| `t` | 轨迹周期(s) | 6.28 |
| `r` | 圆形轨迹半径(m) | 1.0 |

---

## 常见问题

### Q1: 数据采集时飞机不稳定怎么办？
A: 检查SE3控制器的PID参数，适当降低Kp和Kv增益。

### Q2: 神经网络模型加载失败？
A: 确保LibTorch版本与训练时的PyTorch版本兼容，模型必须是TorchScript格式。

### Q3: 实机起飞后飞机超调严重？
A: 检查`hover_percentage`参数，通过实际悬停测试确定准确值。

### Q4: 如何判断Neural-Fly是否生效？
A: 观察控制器输出的扰动估计值`f`，应随飞行状态动态变化。

### Q5: 仿真和实机参数能否通用？
A: 不能。仿真和实机的动力学特性不同，需要分别采集数据和训练模型。

---

## 完整指令速查表

### XTDrone仿真

```bash
# === 数据采集 ===
# 1. 启动仿真
roslaunch px4 mavros_posix_sitl.launch
# 2. 启动通信
python multirotor_communication.py iris 0
# 3. 启动控制器 (SE3)
roslaunch px4ctrl run_ctrl_sim_vision_odom.launch
# 4. 启动随机轨迹
roslaunch planner random.launch
# 5. 起飞
rostopic pub -1 /px4ctrl/takeoff_land quadrotor_msgs/TakeoffLand "takeoff_land_cmd: 1"
# 6. 开始采集
rosservice call /px4ctrl/collect_trigger "{}"
# 7. 降落
rostopic pub -1 /px4ctrl/takeoff_land quadrotor_msgs/TakeoffLand "takeoff_land_cmd: 2"

# === 部署测评 ===
# 1-2同上
# 3. 启动控制器 (Neural-Fly)
roslaunch px4ctrl run_ctrl_sim_vision_odom.launch
# 4. 起飞
rostopic pub -1 /px4ctrl/takeoff_land quadrotor_msgs/TakeoffLand "takeoff_land_cmd: 1"
# 5. Figure8轨迹
roslaunch planner loop.launch
```

### 实机

```bash
# === 数据采集 ===
# 1. 启动传感器
sh shfiles/rspx4.sh
# 2. 启动控制器 (SE3)
roslaunch px4ctrl run_ctrl.launch
# 3. 起飞
sh shfiles/takeoff.sh
# 4. 启动随机轨迹
roslaunch planner random.launch
# 5. 开始采集
rosservice call /px4ctrl/collect_trigger "{}"
# 6. 降落
sh shfiles/land.sh

# === 部署测评 ===
# 1. 启动传感器
sh shfiles/rspx4.sh
# 2. 启动控制器 (Neural-Fly)
roslaunch px4ctrl run_ctrl.launch
# 3. 起飞
sh shfiles/takeoff.sh
# 4. Figure8轨迹
roslaunch planner loop.launch
# 5. 记录数据
sh shfiles/record.sh
```

---

## 版本信息

- 基于 Neural-Fly 论文: "Neural-Fly enables rapid learning for agile flight in strong winds"
- 适用于 ROS Noetic + PX4 v1.11.0
- 测试环境: Ubuntu 20.04

