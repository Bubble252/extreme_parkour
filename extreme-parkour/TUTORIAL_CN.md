# Extreme Parkour 项目完整教程

## 目录
1. [项目概述](#1-项目概述)
2. [项目架构](#2-项目架构)
3. [环境搭建](#3-环境搭建)
4. [仿真环境构建](#4-仿真环境构建)
5. [训练流程](#5-训练流程)
6. [适配新机器人](#6-适配新机器人)
7. [常见问题](#7-常见问题)

---

## 1. 项目概述

### 1.1 项目简介
Extreme Parkour 是一个基于强化学习的四足机器人极限跑酷项目，使用 NVIDIA Isaac Gym 进行物理仿真，采用 PPO（Proximal Policy Optimization）算法训练机器人在复杂地形上运动。

### 1.2 核心特性
- **两阶段训练**：基础策略 → 视觉蒸馏策略
- **特权学习**：训练时使用特权信息（地形高度图等），部署时使用视觉输入
- **复杂地形生成**：自动生成包含台阶、斜坡、踏脚石等多种障碍
- **历史编码器**：通过历史状态估计特权信息
- **动作延迟模拟**：模拟真实硬件的执行延迟

### 1.3 支持的机器人
- Unitree A1
- Unitree Go1
- ANYmal B
- ANYmal C
- Cassie

---

## 2. 项目架构

### 2.1 目录结构
```
extreme-parkour/
├── legged_gym/              # 仿真环境包
│   ├── envs/                # 环境定义
│   │   ├── base/           # 基础环境类
│   │   │   ├── legged_robot.py          # 主环境类
│   │   │   ├── legged_robot_config.py   # 环境配置
│   │   │   └── base_task.py             # Isaac Gym 基础任务
│   │   ├── a1/             # A1 机器人配置
│   │   ├── go1/            # Go1 机器人配置
│   │   ├── anymal_b/       # ANYmal-B 配置
│   │   ├── anymal_c/       # ANYmal-C 配置
│   │   └── cassie/         # Cassie 配置
│   ├── scripts/            # 训练和评估脚本
│   │   ├── train.py        # 训练脚本
│   │   ├── play.py         # 测试脚本
│   │   ├── save_jit.py     # 模型导出
│   │   └── evaluate.py     # 评估脚本
│   ├── utils/              # 工具函数
│   │   ├── terrain.py      # 地形生成
│   │   ├── task_registry.py # 任务注册器
│   │   └── helpers.py      # 辅助函数
│   └── resources/          # 机器人模型资源
│       └── robots/         # URDF 文件
│
├── rsl_rl/                  # 强化学习算法包
│   ├── algorithms/          # RL 算法
│   │   └── ppo.py          # PPO 实现
│   ├── modules/            # 神经网络模块
│   │   ├── actor_critic.py # Actor-Critic 网络
│   │   ├── estimator.py    # 状态估计器
│   │   └── depth_backbone.py # 深度视觉编码器
│   ├── runners/            # 训练运行器
│   │   └── on_policy_runner.py # PPO 训练循环
│   └── storage/            # 经验存储
│       └── rollout_storage.py # Rollout buffer
│
└── isaacgym/               # Isaac Gym (需单独下载)
```

### 2.2 核心类关系

```
TaskRegistry
    └── 注册和创建环境
         ├── LeggedRobot (继承 BaseTask)
         │   ├── 物理仿真步进
         │   ├── 观测计算
         │   ├── 奖励计算
         │   └── 终止条件检查
         │
         └── OnPolicyRunner
             ├── PPO 算法
             │   ├── ActorCriticRMA (神经网络)
             │   ├── Estimator (特权信息估计器)
             │   └── DepthEncoder (视觉编码器)
             └── 训练循环
```

### 2.3 数据流

```
环境观测 → Actor网络 → 动作 → 仿真环境 → 奖励 + 下一状态
    ↓                                        ↓
观测历史 → History Encoder → 估计特权信息    Rollout Buffer
                                            ↓
                                        PPO更新
```

### 2.4 核心组件说明

#### 2.4.1 环境 (LeggedRobot)
- **位置**: `legged_gym/legged_gym/envs/base/legged_robot.py`
- **功能**:
  - 与 Isaac Gym 交互
  - 计算观测向量（本体感知、扫描点、历史状态等）
  - 计算奖励函数（多项奖励的加权和）
  - 检测终止条件（倾翻、超时等）
  - 管理深度相机（可选）

#### 2.4.2 配置 (Config)
- **位置**: `legged_gym/legged_gym/envs/base/legged_robot_config.py`
- **功能**:
  - 定义环境参数（并行环境数、观测维度等）
  - 定义控制参数（PD控制器增益、动作缩放等）
  - 定义地形参数（类型、难度、尺寸等）
  - 定义奖励权重
  - 定义域随机化参数

#### 2.4.3 PPO 算法
- **位置**: `rsl_rl/rsl_rl/algorithms/ppo.py`
- **功能**:
  - 实现 PPO 损失函数
  - 价值函数学习
  - 特权信息估计器训练
  - 深度编码器训练（蒸馏阶段）

#### 2.4.4 地形生成
- **位置**: `legged_gym/legged_gym/utils/terrain.py`
- **功能**:
  - 生成多种地形类型（斜坡、台阶、踏脚石等）
  - 课程学习（难度递增）
  - 高度图转三角网格
  - 目标点生成

---

## 3. 环境搭建

### 3.1 系统要求
- **操作系统**: Ubuntu 18.04/20.04/22.04
- **GPU**: NVIDIA GPU with CUDA support (推荐 RTX 3090 或更好)
- **CUDA**: 11.3 或更高
- **Python**: 3.6-3.8

### 3.2 创建 Conda 虚拟环境

```bash
# 创建虚拟环境
conda create -n parkour python=3.8
conda activate parkour

# 安装 PyTorch (CUDA 11.3)
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 如果使用 CUDA 11.7
# pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

### 3.3 下载和安装 Isaac Gym

```bash
# 1. 访问 NVIDIA Isaac Gym 官网下载
# https://developer.nvidia.com/isaac-gym
# 需要注册 NVIDIA 账号

# 2. 下载后解压到项目目录
cd ~/桌面/extreme_parkour/extreme-parkour
tar -xvf ~/Downloads/IsaacGym_Preview_4_Package.tar.gz

# 3. 安装 Isaac Gym
cd isaacgym/python
pip install -e .

# 4. 测试安装
cd examples
python 1080_balls_of_solitude.py
# 如果看到物理仿真窗口，说明安装成功
```

### 3.4 安装项目依赖

```bash
cd ~/桌面/extreme_parkour/extreme-parkour

# 安装 rsl_rl (强化学习库)
cd rsl_rl
pip install -e .

# 安装 legged_gym (仿真环境)
cd ../legged_gym
pip install -e .

# 安装其他依赖
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask
pip install pymeshlab scipy scikit-learn matplotlib
```

### 3.5 验证安装

```bash
# 进入脚本目录
cd ~/桌面/extreme_parkour/extreme-parkour/legged_gym/legged_gym/scripts

# 运行测试（不训练，只是验证环境）
# 注意：如果系统配置较低，建议从4个环境开始测试
python train.py --exptid test-01 --device cuda:0 --num_envs 4 --max_iterations 1 --headless --no_wandb

# 如果4个环境运行正常，可以逐步增加：
# python train.py --exptid test-01 --device cuda:0 --num_envs 16 --max_iterations 1 --headless --no_wandb
# python train.py --exptid test-01 --device cuda:0 --num_envs 64 --max_iterations 1 --headless --no_wandb
```

**常见问题排查：**
- 如果系统卡死：环境数量太多，减少到2或4个
- 如果显存不足：使用 `--num_envs 2`
- 如果报错"index out of bounds"：**不要手动设置 --rows 和 --cols 参数**（默认值10×20是安全的）
- 地形参数建议：
  - ✅ 推荐：不指定（使用默认10×20）
  - ✅ 可用：`--rows 5 --cols 5` 或更大
  - ❌ 错误：`--rows 3 --cols 3`（太小会导致索引越界）

如果没有报错，说明环境搭建成功！

---

## 4. 仿真环境构建

### 4.1 环境初始化流程

```python
# 1. 创建环境 (train.py)
env, env_cfg = task_registry.make_env(name='a1', args=args)

# 2. TaskRegistry 查找注册的环境类
task_class = LeggedRobot  # 从 envs/__init__.py 注册

# 3. 创建 Isaac Gym 仿真
sim = gym.create_sim(...)

# 4. 创建地形
terrain = Terrain(cfg.terrain, num_robots)

# 5. 创建并行环境
for i in range(num_envs):
    env_handle = gym.create_env(sim, ...)
    actor_handle = gym.create_actor(env_handle, robot_asset, ...)

# 6. 初始化观测和状态缓冲区
obs_buf = torch.zeros(num_envs, num_obs)
```

### 4.2 地形生成详解

#### 4.2.1 地形类型
```python
# 在 legged_robot_config.py 中定义
terrain_dict = {
    "parkour_flat": 0.05,       # 平坦地形
    "parkour_slope": 0.1,        # 斜坡
    "parkour_stair": 0.15,       # 台阶
    "parkour_discrete": 0.1,     # 离散障碍
    "parkour_wave": 0.1,         # 波浪地形
    "parkour_step": 0.2,         # 踏脚石
    "demo": 0.15,                # 演示地形
    "parkour_hurdle": 0.15       # 栏架
}
```

#### 4.2.2 地形网格布局
```
地形网格 (例如 10 行 × 20 列):
┌─────────────────────────────────┐
│ 简单 → → → → → → → → → → 困难  │
│ ↓ [0,0] [0,1] ... [0,19]       │
│ ↓ [1,0] [1,1] ... [1,19]       │
│ ↓   ...   ...   ...   ...       │
│ ↓ [9,0] [9,1] ... [9,19]       │
└─────────────────────────────────┘
  每一列是一种地形类型
  每一行难度递增
```

#### 4.2.3 课程学习
```python
# 在 terrain.py 中
def curiculum(self):
    for i in range(num_rows):    # 难度维度
        for j in range(num_cols):  # 地形类型维度
            difficulty = i / (num_rows - 1)  # 0.0 → 1.0
            choice = j / num_cols             # 选择地形类型
            terrain = self.make_terrain(choice, difficulty)
```

### 4.3 观测空间构成

```python
# 观测向量组成 (legged_robot_config.py)
n_proprio = 3 + 2 + 3 + 4 + 36 + 5  # 本体感知
    # 3: 线速度 (base frame)
    # 2: 角速度 (roll, pitch rate)
    # 3: 重力投影
    # 4: 命令 (vx, vy, vyaw, height)
    # 36: 关节状态 (12pos + 12vel + 12target)
    # 5: 足端接触 + 动作历史

n_scan = 132  # 扫描点（地形高度采样）

n_priv_latent = 4 + 1 + 12 + 12  # 估计的特权信息
    # 4: 摩擦系数等
    # 1: 载荷
    # 12: 关节摩擦
    # 12: 电机强度

n_priv = 3 + 3 + 3  # 真实特权信息（仅训练时）
    # 3: 外部力
    # 3: 外部力矩
    # 3: 环境参数

history_len = 10  # 历史步数

# 总观测维度
num_obs = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv
```

### 4.4 动作空间

```python
# 动作: 12维向量 (四足机器人)
# 每条腿3个关节: [Hip, Thigh, Calf] × 4
action = [
    FL_hip, FL_thigh, FL_calf,   # 左前腿
    FR_hip, FR_thigh, FR_calf,   # 右前腿
    RL_hip, RL_thigh, RL_calf,   # 左后腿
    RR_hip, RR_thigh, RR_calf    # 右后腿
]

# 动作经过处理:
# 1. 裁剪: [-clip, +clip]
# 2. 缩放: × action_scale (通常 0.25)
# 3. 加到默认位置: target_pos = default_pos + action
# 4. PD控制器计算力矩: τ = Kp*(target - current) - Kd*velocity
```

### 4.5 奖励函数

```python
# 奖励组成 (在 legged_robot.py 中定义)
rewards = {
    "tracking_lin_vel": 1.5,      # 跟踪线速度
    "tracking_ang_vel": 0.5,      # 跟踪角速度
    "lin_vel_z": -2.0,            # 惩罚z方向速度
    "ang_vel_xy": -0.05,          # 惩罚roll/pitch角速度
    "orientation": -1.0,          # 惩罚姿态偏离
    "dof_acc": -2.5e-7,           # 惩罚关节加速度
    "action_rate": -0.01,         # 惩罚动作变化率
    "torques": -1e-5,             # 惩罚力矩
    "dof_pos_limits": -10.0,      # 惩罚关节超限
    "collision": -1.0,            # 惩罚碰撞
    "feet_air_time": 1.0,         # 奖励足端腾空时间
    "stumble": -0.0,              # 惩罚绊倒
    # ... 更多奖励项
}

# 总奖励 = Σ(weight_i × reward_i)
```

---

## 5. 训练流程

### 5.1 两阶段训练策略

#### 阶段一：基础策略训练 (Base Policy)
- **目标**: 学习使用特权信息在复杂地形上运动
- **输入**: 本体感知 + 地形扫描 + 特权信息
- **输出**: 关节动作
- **训练时间**: 10-15k 迭代 (8-10小时 on RTX 3090)

#### 阶段二：视觉蒸馏策略 (Distillation Policy)
- **目标**: 使用深度相机替代特权信息
- **输入**: 本体感知 + 深度图像
- **输出**: 关节动作
- **训练时间**: 5-10k 迭代 (5-10小时 on RTX 3090)

### 5.2 训练命令详解

#### 5.2.1 训练基础策略

```bash
cd ~/桌面/extreme_parkour/extreme-parkour/legged_gym/legged_gym/scripts

# 基础训练命令
python train.py \
    --exptid 001-01-baseline \     # 实验ID (xxx-xx-描述)
    --device cuda:0 \               # GPU设备
    --task a1 \                     # 机器人类型
    --num_envs 4096 \              # 并行环境数
    --max_iterations 15000 \        # 最大迭代数
    --headless                      # 无头模式（无GUI）

# 主要参数说明:
# --exptid: 实验标识符，格式建议 xxx-xx-描述
#           xxx-xx 用于自动匹配加载模型
# --device: GPU 设备 (cuda:0, cuda:1, cpu)
# --num_envs: 并行环境数量，越多越快但显存占用越大
#             推荐: 3090-24GB → 4096-6144
#                   2080Ti-11GB → 2048-4096
# --headless: 无GUI模式，训练更快
# --seed: 随机种子，用于复现
```

#### 5.2.2 训练蒸馏策略

```bash
# 从基础策略继续训练，添加视觉输入
python train.py \
    --exptid 001-02-vision \        # 新的实验ID
    --device cuda:0 \
    --resume \                      # 从检查点恢复
    --resumeid 001-01 \            # 基础策略ID (前6字符)
    --delay \                       # 添加动作延迟
    --use_camera \                  # 使用深度相机
    --max_iterations 10000

# 蒸馏阶段参数:
# --resume: 从之前的检查点恢复训练
# --resumeid: 要加载的基础策略ID
# --use_camera: 启用深度相机，生成深度图像
# --delay: 模拟真实硬件的执行延迟
# --checkpoint: 指定加载特定的检查点编号
```

### 5.3 训练过程监控

#### 5.3.1 使用 WandB (推荐)
```bash
# 训练时会自动上传到 wandb
# 访问: https://wandb.ai/parkour/项目名

# 如果不想使用 wandb
python train.py --exptid xxx-xx --no_wandb
```

#### 5.3.2 查看训练日志
```bash
# 日志保存位置
~/桌面/extreme_parkour/extreme-parkour/legged_gym/logs/parkour_new/001-01-baseline/

# 目录结构:
logs/parkour_new/001-01-baseline/
├── config.txt              # 训练配置
├── model_100.pt           # 检查点 (每100迭代)
├── model_200.pt
├── ...
└── model_15000.pt         # 最终模型
```

#### 5.3.3 关键指标
- **mean_reward**: 平均奖励（越高越好）
- **mean_episode_length**: 平均episode长度
- **lin_vel_tracking**: 线速度跟踪误差
- **policy_loss**: 策略损失
- **value_loss**: 价值函数损失
- **estimator_loss**: 估计器损失

### 5.4 视觉信息详解

#### 5.4.1 视觉系统概述

Extreme Parkour使用**深度相机**作为视觉输入，在第二阶段训练中替代特权信息：

```
阶段一（特权信息）:
输入 = 本体感知 + 地形扫描 + 特权信息（地形高度、摩擦系数等）
     ↓
  策略网络
     ↓
   关节动作

阶段二（视觉蒸馏）:
输入 = 本体感知 + 深度图像
     ↓
  深度编码器 → 特权信息估计
     ↓
  策略网络
     ↓
   关节动作
```

#### 5.4.2 深度相机配置

```python
# 在 legged_robot_config.py 中配置
class depth:
    use_camera = True           # 启用深度相机
    
    # 相机位置和角度
    position = [0.27, 0, 0.03]  # [x, y, z] 相对于躯干 (前置相机)
    angle = [-5, 5]             # [pitch, roll] 度数（正值向下倾斜）
    
    # 图像分辨率
    original = (106, 60)        # 原始分辨率 (宽, 高)
    resized = (87, 58)          # 处理后分辨率 (宽, 高)
    horizontal_fov = 87         # 水平视场角（度）
    
    # 深度范围
    near_clip = 0.0             # 最近可见距离 [m]
    far_clip = 2.0              # 最远可见距离 [m]
    
    # 更新频率
    update_interval = 5         # 每5个仿真步更新一次图像
    
    # 历史缓冲
    buffer_len = 2              # 保存最近2帧图像
    
    # 噪声模拟
    dis_noise = 0.0             # 距离测量噪声 [m]
```

#### 5.4.3 深度图像处理流程

```python
# 1. 获取原始深度图 (106 x 60 像素)
depth_raw = gym.get_camera_image_gpu_tensor(sim, env, camera, IMAGE_DEPTH)

# 2. 裁剪 (去除边缘噪声)
depth_cropped = depth_raw[:-2, 4:-4]  # 裁剪上下2像素，左右各4像素

# 3. 添加传感器噪声（模拟真实相机）
depth_noisy = depth_cropped + dis_noise * uniform(-1, 1)

# 4. 限制深度范围
depth_clipped = clip(depth_noisy, -far_clip, -near_clip)  # Isaac Gym深度值为负

# 5. 调整分辨率 (106x60 → 87x58)
depth_resized = resize(depth_clipped, (58, 87), mode='bicubic')

# 6. 归一化到 [-0.5, 0.5]
depth_normalized = (depth * -1 - near_clip) / (far_clip - near_clip) - 0.5

# 最终输出: [58, 87] 的归一化深度图
```

**可视化示例：**
```
深度图表示（俯视图）:
深色（-0.5）= 近处物体（0m）
灰色（0）    = 中等距离（1m）  
浅色（+0.5） = 远处物体（2m）

    █████████  ← 近处障碍物
    ░░░░░░░░░  ← 地面（中距离）
    ▓▓▓▓▓▓▓▓▓  ← 远处障碍物
```

#### 5.4.4 深度编码器架构

```python
# DepthOnlyFCBackbone58x87 网络结构
输入: [batch, 1, 58, 87]  # 单通道深度图

Conv2D(32 filters, 5x5)
  ↓ [batch, 32, 54, 83]
MaxPool2D(2x2)
  ↓ [batch, 32, 27, 41]
ELU激活

Conv2D(64 filters, 3x3)
  ↓ [batch, 64, 25, 39]
ELU激活

Flatten
  ↓ [batch, 64*25*39 = 62400]
Linear(62400 → 128)
  ↓ [batch, 128]
ELU激活

Linear(128 → 32)
  ↓ [batch, 32]  # 32维深度特征

# 与本体感知融合
Concat(depth_features[32] + proprioception[n_proprio])
  ↓ [batch, 32 + n_proprio]
MLP(128 hidden)
  ↓ [batch, 32]  # 最终特权信息估计
```

#### 5.4.5 历史帧处理

为了捕捉运动信息，使用**历史帧缓冲**：

```python
# buffer_len = 2，保存最近2帧
depth_buffer = [
    frame_t,      # 当前帧
    frame_t-1     # 前一帧（5个仿真步之前）
]

# 在推理时可以：
# 1. 只使用最新帧: depth_buffer[:, -1]
# 2. 堆叠多帧输入CNN提取时序特征
```

#### 5.4.6 视觉蒸馏训练

```python
# 训练目标：让深度编码器的输出接近特权信息

# 损失函数
depth_encoder_loss = MSE(
    depth_encoder(depth_image, proprioception),  # 从图像估计
    privileged_info                               # 真实特权信息
)

# 同时训练策略网络
policy_loss = PPO_loss(
    policy(proprioception + depth_encoder_output),
    advantages
)

total_loss = policy_loss + depth_encoder_loss
```

#### 5.4.7 深度图像可视化

```python
# 在play.py中保存深度图像
if self.cfg.depth.use_camera:
    import cv2
    depth_img = self.extras["depth"][0].cpu().numpy()  # 取第一个环境
    depth_img = (depth_img + 0.5) * 255  # 归一化到0-255
    cv2.imwrite(f"depth_{step}.png", depth_img.astype(np.uint8))
```

#### 5.4.8 真机部署注意事项

在真实机器人上使用深度相机时需要注意：

```python
# 1. 相机标定
- 确保相机位置和仿真中一致（position = [0.27, 0, 0.03]）
- 校准相机内参（焦距、畸变系数）
- 匹配视场角（horizontal_fov = 87°）

# 2. 图像预处理
- 使用相同的裁剪、缩放、归一化参数
- 添加适当的噪声容忍度
- 考虑光照变化的鲁棒性

# 3. 计算性能
- 深度编码器推理时间: ~2ms (GPU) / ~10ms (CPU)
- 控制频率: 50Hz (每20ms)
- 相机更新: 10Hz (每100ms) - update_interval=5

# 4. 常见问题
- 深度图有噪声 → 增加dis_noise进行训练
- 图像模糊 → 检查镜头对焦
- 推理延迟 → 使用TensorRT加速
```

### 5.5 测试训练好的策略

#### 5.5.1 测试基础策略
```bash
python play.py \
    --exptid 001-01 \              # 实验ID (自动匹配)
    --checkpoint 15000             # 指定检查点 (可选)

# 查看器控制:
# ALT + 鼠标左键 + 拖动: 移动视角
# [ / ]: 切换机器人
# 空格: 暂停/继续
# F: 切换自由相机/跟随相机
```

#### 5.5.2 测试蒸馏策略
```bash
python play.py \
    --exptid 001-02 \
    --delay \                      # 需要添加延迟标志
    --use_camera                   # 需要添加相机标志
```

#### 5.5.3 无头模式测试 (远程服务器)
```bash
python play.py \
    --exptid 001-01 \
    --headless \
    --web                          # 启用web查看器

# 然后在浏览器打开: http://localhost:8080
# 或使用 VSCode 的 Live Preview 扩展
```

### 5.5 导出模型用于部署

```bash
python save_jit.py --exptid 001-02

# 导出的文件:
# logs/parkour_new/001-02/traced/
# ├── 001-02-15000-base_jit.pt        # JIT编译的策略网络
# └── 001-02-15000-vision_weight.pt   # 深度编码器权重
```

### 5.6 训练技巧

#### 5.6.1 课程学习
```python
# 在配置文件中设置
terrain.curriculum = True           # 启用课程学习
terrain.max_difficulty = 1.0        # 最大难度
```

#### 5.6.2 调整环境数量
```bash
# 显存不足时减少环境数
--num_envs 2048

# 训练更快时增加环境数
--num_envs 8192
```

#### 5.6.3 调试模式
```bash
# 少量环境，显示GUI，便于调试
python train.py \
    --exptid debug-01 \
    --debug \                  # 自动设置: num_envs=64, rows=10, cols=8
    --max_iterations 100

# 或手动设置
python train.py \
    --exptid debug-01 \
    --num_envs 64 \
    --rows 5 \
    --cols 4 \
    --max_iterations 100
```

#### 5.6.4 恢复训练
```bash
# 如果训练中断，继续训练
python train.py \
    --exptid 001-01-baseline \
    --resume \
    --resumeid 001-01 \
    --checkpoint 8000              # 从第8000次迭代继续
```

---

## 6. 适配新机器人

### 6.1 准备 URDF 文件

#### 6.1.1 URDF 要求
- 使用标准 URDF 格式
- 包含完整的关节定义和质量属性
- 包含碰撞和视觉几何体
- 推荐使用 STL 或 DAE 格式的网格文件

#### 6.1.2 Tita双轮足机器人结构
项目中已包含Tita双轮足机器人的URDF文件，结构如下：
```
Tita双轮足机器人 (8个自由度):
- 躯干 (base_link): 质量13.2kg
- 左腿 (4个关节):
  ├─ joint_left_leg_1: 髋关节-横滚 (±45°)
  ├─ joint_left_leg_2: 髋关节-俯仰 (-110° ~ 200°)  
  ├─ joint_left_leg_3: 膝关节 (-153° ~ -40°)
  └─ joint_left_leg_4: 末端轮子 (连续旋转)
- 右腿 (4个关节):
  ├─ joint_right_leg_1: 髋关节-横滚 (±45°)
  ├─ joint_right_leg_2: 髋关节-俯仰 (-110° ~ 200°)
  ├─ joint_right_leg_3: 膝关节 (-153° ~ -40°)
  └─ joint_right_leg_4: 末端轮子 (连续旋转)
  
URDF位置: legged_gym/resources/robots/tita/urdf/tita_description.urdf
```

### 6.2 Tita机器人文件已就绪

Tita机器人的URDF和网格文件已经包含在项目中：

```bash
# Tita机器人文件结构
legged_gym/resources/robots/tita/
├── urdf/
│   └── tita_description.urdf      # 主URDF文件（包含完整的关节、惯性、碰撞定义）
└── meshes/
    ├── base_link.STL              # 躯干网格
    ├── left_leg_1.STL             # 左腿各连杆网格
    ├── left_leg_2.STL
    ├── left_leg_3.STL
    ├── left_leg_4.STL             # 左轮网格
    ├── right_leg_1.STL            # 右腿各连杆网格
    ├── right_leg_2.STL
    ├── right_leg_3.STL
    └── right_leg_4.STL            # 右轮网格
```

**如果你要添加其他机器人**，可参考Tita的文件结构创建新的目录。

### 6.3 创建Tita机器人配置文件

```bash
# 创建配置目录
cd ~/桌面/extreme_parkour/extreme-parkour/legged_gym/legged_gym/envs
mkdir tita
cd tita
```

创建 `tita_config.py`（基于实际的Tita机器人结构）:

```python
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TitaCfg(LeggedRobotCfg):
    """
    Tita双轮足机器人配置 - 8个自由度（每条腿4个关节）
    """
    class env(LeggedRobotCfg.env):
        # 环境参数
        num_envs = 4096
        num_observations = 56  # 需根据实际调整
        num_actions = 8        # 8个关节（每腿：髋横滚+髋俯仰+膝+轮）
        
    class init_state(LeggedRobotCfg.init_state):
        # 初始状态 - 调整为适合双轮足的高度
        pos = [0.0, 0.0, 0.5]  # x, y, z [m]
        
        # 默认关节角度 [rad] - 根据URDF中的关节名和限位设置
        default_joint_angles = {
            # 左腿 4个关节
            'joint_left_leg_1': 0.0,      # 髋关节-横滚 (范围: ±0.785)
            'joint_left_leg_2': 0.5,      # 髋关节-俯仰 (范围: -1.92~3.49)
            'joint_left_leg_3': -1.2,     # 膝关节 (范围: -2.67~-0.698)
            'joint_left_leg_4': 0.0,      # 末端轮子 (连续旋转)
            
            # 右腿 4个关节
            'joint_right_leg_1': 0.0,     # 髋关节-横滚 (范围: ±0.785)
            'joint_right_leg_2': 0.5,     # 髋关节-俯仰 (范围: -1.92~3.49)
            'joint_right_leg_3': -1.2,    # 膝关节 (范围: -2.67~-0.698)
            'joint_right_leg_4': 0.0,     # 末端轮子 (连续旋转)
        }
    
    class control(LeggedRobotCfg.control):
        # PD控制器参数（根据Tita实际电机性能调整）
        control_type = 'P'
        stiffness = {
            'joint_.*_leg_[123]': 80.,  # 腿部关节刚度 [N*m/rad]
            'joint_.*_leg_4': 5.,       # 轮子关节刚度更低
        }
        damping = {
            'joint_.*_leg_[123]': 2.0,  # 腿部关节阻尼 [N*m*s/rad]
            'joint_.*_leg_4': 0.5,      # 轮子关节阻尼
        }
        action_scale = 0.25          # 动作缩放
        decimation = 4               # 控制频率 = 50Hz (仿真200Hz / 4)
    
    class asset(LeggedRobotCfg.asset):
        # Tita机器人模型
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/tita/urdf/tita_description.urdf'
        
        # 足端名称（用于接触检测） - 根据URDF中的link名
        foot_name = "leg_4"  # 轮子连杆作为足端
        
        # 碰撞检测
        penalize_contacts_on = ["leg_2", "leg_3"]  # 大腿、小腿碰撞会受惩罚
        terminate_after_contacts_on = ["base_link"]  # 躯干碰撞会终止episode
        
        # 自碰撞
        self_collisions = 1  # 1 禁用，0 启用
        
        # 关节属性覆盖（可选）
        override_inertia = True
        override_com = True
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        # 域随机化
        randomize_friction = True
        friction_range = [0.5, 1.5]
        
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
    
    class rewards(LeggedRobotCfg.rewards):
        # 奖励权重（针对双轮足机器人优化）
        class scales:
            tracking_lin_vel = 1.5      # 跟踪线速度
            tracking_ang_vel = 0.5      # 跟踪角速度
            lin_vel_z = -2.0            # 惩罚垂直速度
            ang_vel_xy = -0.05          # 惩罚横滚/俯仰角速度
            orientation = -1.5          # 双足更需要保持姿态
            torques = -0.00001          # 惩罚大力矩
            dof_vel = -0.0              # 惩罚关节速度
            dof_acc = -2.5e-7           # 惩罚关节加速度
            base_height = -0.5          # 鼓励维持合适高度
            feet_air_time = 1.5         # 鼓励足端腾空时间（跳跃/奔跑）
            collision = -1.0            # 惩罚碰撞
            feet_stumble = -0.2         # 惩罚足端拖地
            action_rate = -0.01         # 惩罚动作突变
            stand_still = -0.0          # 静止时的惩罚
            wheel_slip = -0.1           # 惩罚轮子打滑（可选）
            
        # 其他奖励参数
        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.5     # Tita目标高度（根据实际调整）
        max_contact_force = 150.     # 轮子可能承受更大接触力
    
    class terrain(LeggedRobotCfg.terrain):
        # 地形配置
        mesh_type = 'trimesh'
        measure_heights = True
        curriculum = True
        
        # 地形尺寸
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # 难度级别
        num_cols = 20  # 地形类型数量
        
        # 地形类型比例
        terrain_proportions = [0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1]
        
    class commands(LeggedRobotCfg.commands):
        # 命令范围
        class ranges:
            lin_vel_x = [-1.0, 1.5]   # [m/s]
            lin_vel_y = [-0.5, 0.5]   # [m/s]
            ang_vel_yaw = [-1.0, 1.0] # [rad/s]
            heading = [-3.14, 3.14]   # [rad]


class TitaCfgPPO(LeggedRobotCfgPPO):
    """
    Tita机器人的PPO训练配置
    """
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1e-3
        num_learning_epochs = 5
        num_mini_batches = 4
        
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'tita_parkour'  # 实验名称
        max_iterations = 15000
        save_interval = 100
        
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
```

### 6.4 注册Tita机器人环境

编辑 `legged_gym/legged_gym/envs/__init__.py`:

```python
# 在文件末尾添加Tita机器人的导入
from .tita.tita_config import TitaCfg, TitaCfgPPO
from .base.legged_robot import LeggedRobot

# 注册Tita环境
task_registry.register(
    "tita",                             # 环境名称
    LeggedRobot,                        # 环境类
    TitaCfg(),                          # Tita配置
    TitaCfgPPO()                        # Tita训练配置
)
```

### 6.5 调整Tita的观测和动作空间

Tita有**8个关节**（vs 四足的12个），需要调整观测和动作维度：

```python
class TitaCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        # 重新计算观测维度
        n_proprio = 3 + 2 + 3 + 4 + 24 + 3  # 39维
        # 3: 线速度 (vx, vy, vz)
        # 2: 角速度 (roll rate, pitch rate)
        # 3: 重力投影向量 (gravity vector in body frame)
        # 4: 命令 (cmd_vx, cmd_vy, cmd_vyaw, heading)
        # 24: 关节状态 (8pos + 8vel + 8target)
        # 3: 足端接触历史 (2脚 × 1步 + padding)
        
        n_scan = 132        # 扫描点云（保持不变）
        n_priv_latent = 17  # 特权信息隐变量（根据需要调整）
        n_priv = 9          # 根据需要调整
        history_len = 10
        
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv
        num_actions = 8     # Tita有8个关节
```

### 6.6 测试Tita机器人

```bash
# 1. 首先测试Tita环境是否能正常加载（使用少量环境+GUI）
cd ~/桌面/extreme_parkour/extreme-parkour/legged_gym/scripts
python play.py \
    --task tita \
    --exptid test-00 \
    --num_envs 4 \
    --checkpoint 0

# 2. 确认无误后，开始Tita的完整训练
python train.py \
    --task tita \
    --exptid 200-01-tita-baseline \
    --device cuda:0 \
    --num_envs 4096 \
    --max_iterations 15000 \
    --headless
    
# 3. 训练完成后查看效果
python play.py \
    --task tita \
    --exptid 200-01-tita-baseline \
    --checkpoint 15000
```

### 6.7 常见调整项

#### 6.7.1 如果Tita不稳定（频繁摔倒）
```python
# 1. 增加关节阻尼
class control:
    damping = {
        'joint_.*_leg_[123]': 5.0,  # 增加腿部关节阻尼
        'joint_.*_leg_4': 1.0,
    }

# 2. 减少动作缩放（降低动作幅度）
class control:
    action_scale = 0.15  # 从0.25减小到0.15

# 3. 增加姿态和平衡相关奖励权重
class rewards:
    class scales:
        orientation = -2.0      # 增加姿态惩罚权重
        base_height = -1.0      # 增加高度偏差惩罚
        ang_vel_xy = -0.1       # 增加横滚/俯仰角速度惩罚
        
# 4. 降低初始高度，降低重心
class init_state:
    pos = [0.0, 0.0, 0.4]  # 从0.5降低到0.4
```

#### 6.7.2 如果Tita不动（不学习运动）
```python
# 1. 检查初始姿态是否合理（能否站立）
class init_state:
    default_joint_angles = {
        'joint_left_leg_2': 0.8,   # 增大髋关节角度
        'joint_left_leg_3': -1.5,  # 增大膝关节弯曲
        # ... 调整到稳定站立姿态
    }
    
# 2. 增加速度跟踪奖励
class rewards:
    class scales:
        tracking_lin_vel = 2.0  # 增加到2.0
        
# 3. 检查命令范围是否合理
class commands:
    class ranges:
        lin_vel_x = [0.5, 1.5]  # 最小速度不要太低
```

#### 6.7.3 如果轮子不转动或打滑严重
```python
# 1. 调整轮子关节的控制参数
class control:
    stiffness = {
        'joint_.*_leg_4': 10.,  # 增加轮子刚度
    }
    
# 2. 添加轮子速度奖励（鼓励滚动）
class rewards:
    class scales:
        wheel_vel = 0.1  # 奖励轮子转速
        
# 3. 调整摩擦力
class domain_rand:
    friction_range = [0.8, 1.2]  # 增大摩擦力
```

#### 6.7.4 如果训练太慢
```python
# 减少地形复杂度
class terrain:
    num_rows = 5
    num_cols = 10

# 增加环境数量
class env:
    num_envs = 8192
```

---

## 7. 常见问题

### 7.1 系统卡死/性能问题

**Q: 运行训练时电脑完全卡死，无法操作**

这是**最常见**的问题，通常是资源不足导致：

```bash
# 解决方案1：大幅减少环境数量
python train.py --exptid test-01 --device cuda:0 --num_envs 4 --max_iterations 1 --headless --no_wandb

# 解决方案2：减少地形复杂度
python train.py --exptid test-01 --device cuda:0 --num_envs 4 --rows 3 --cols 3 --max_iterations 1 --headless --no_wandb

# 解决方案3：使用CPU模式（最稳定但最慢）
python train.py --exptid test-01 --device cpu --num_envs 4 --max_iterations 1 --headless --no_wandb
```

**环境数量建议：**
```bash
# 根据你的硬件配置选择：
低配置（8GB显存以下）:     --num_envs 4
中配置（8-16GB显存）:       --num_envs 16
高配置（RTX 3080/3090）:    --num_envs 64-128
专业级（A100/H100）:        --num_envs 256-4096
```

**检查硬件资源：**
```bash
# 在另一个终端监控GPU使用情况
watch -n 1 nvidia-smi

# 监控CPU和内存
htop

# 如果看到显存接近100%，说明环境数太多了
```

**Q: 程序运行很慢，但没卡死**

```bash
# 1. 确认使用了GPU
python -c "import torch; print(torch.cuda.is_available())"  # 应该输出 True

# 2. 减少地形复杂度
--rows 3 --cols 3  # 默认是10×20，改成3×3

# 3. 禁用图形渲染（确保使用headless）
--headless

# 4. 减少物理仿真精度（在配置文件中）
# sim.dt = 0.01  # 增大时间步长
```

### 7.2 安装问题

**Q: Isaac Gym 安装失败**
```bash
# 检查 CUDA 版本
nvidia-smi

# 确保 PyTorch 和 CUDA 版本匹配
python -c "import torch; print(torch.cuda.is_available())"

# 如果使用 conda，清理环境重新安装
conda remove -n parkour --all
conda create -n parkour python=3.8
```

**Q: 显存不足**
```bash
# 减少并行环境数量
--num_envs 1024

# 或使用更小的网络
class policy:
    actor_hidden_dims = [256, 128, 64]
```

### 7.2 训练问题

**Q: 奖励一直很低**
- 检查初始姿态是否合理
- 检查 PD 控制器参数
- 降低地形难度
- 检查奖励权重设置

**Q: 机器人原地不动**
- 增加速度跟踪奖励权重
- 检查命令范围设置
- 减小动作限制

**Q: 机器人疯狂抖动**
- 增加阻尼系数
- 减小动作缩放
- 增加平滑奖励

### 7.3 部署问题

**Q: 导出的模型太大**
```bash
# 使用量化
# 在 save_jit.py 中添加
traced_policy = torch.jit.optimize_for_inference(traced_policy)
```

**Q: 真机效果不好**
- 增加域随机化
- 添加动作延迟训练
- 增加传感器噪声

### 7.4 调试技巧

```bash
# 1. 使用小规模环境调试
python train.py --debug --max_iterations 10

# 2. 可视化观测
# 在 legged_robot.py 的 compute_observations() 中添加
print("obs:", self.obs_buf[0])

# 3. 查看奖励分解
# 在训练日志中会输出各项奖励

# 4. 保存训练视频
RECORD_FRAMES = True  # 在 play.py 中设置
```

---

## 8. 进阶主题

### 8.1 自定义奖励函数

在 `legged_robot.py` 中添加新的奖励项：

```python
def _reward_custom_reward(self):
    """自定义奖励"""
    # 例如：奖励足端交替腾空
    contact_change = (self.contact_filt != self.last_contacts).float()
    return torch.sum(contact_change, dim=1)

# 在配置中添加权重
class rewards:
    class scales:
        custom_reward = 0.5
```

### 8.2 自定义地形

在 `terrain.py` 中添加新的地形类型：

```python
def custom_terrain(terrain, difficulty):
    """自定义地形生成函数"""
    # 实现你的地形生成逻辑
    pass

# 在配置中注册
class terrain:
    terrain_dict = {
        "custom": 0.2,  # 20% 概率
        # ... 其他地形
    }
```

### 8.3 多任务学习

```python
class env:
    task_both = True  # 启用多任务
    
# 训练时使用
python train.py --task_both
```

---

## 9. 参考资源

### 9.1 相关论文
- **Extreme Parkour with Legged Robots**: https://arxiv.org/abs/2309.14341
- **Isaac Gym**: https://arxiv.org/abs/2108.10470
- **PPO**: https://arxiv.org/abs/1707.06347

### 9.2 相关链接
- **项目主页**: https://extreme-parkour.github.io
- **Isaac Gym**: https://developer.nvidia.com/isaac-gym
- **legged_gym**: https://github.com/leggedrobotics/legged_gym

### 9.3 社区支持
- **GitHub Issues**: https://github.com/chengxuxin/extreme-parkour/issues
- **论文作者**: Xuxin Cheng (chengxuxin@cmu.edu)

---

## 10. 总结

本教程涵盖了 Extreme Parkour 项目的：
1. ✅ 完整架构说明
2. ✅ 环境搭建步骤
3. ✅ 仿真环境构建
4. ✅ 训练流程详解
5. ✅ 新机器人适配指南（包括双轮足机器人）

### 快速开始清单

```bash
# 1. 环境搭建
□ 创建conda环境
□ 安装PyTorch
□ 安装Isaac Gym
□ 安装项目依赖

# 2. 验证安装
□ 运行测试脚本

# 3. 开始训练
□ 训练基础策略 (15k iterations)
□ 训练蒸馏策略 (10k iterations)
□ 测试和评估

# 4. 适配新机器人（可选）
□ 准备URDF文件
□ 创建配置文件
□ 注册环境
□ 开始训练
```

祝你训练顺利！🚀
