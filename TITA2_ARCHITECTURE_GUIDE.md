# Tita2 双轮足机器人架构指南

## 目录
1. [项目概述](#项目概述)
2. [双轮足 vs 四足配置区别](#双轮足-vs-四足配置区别)
3. [从tita_rl借鉴的内容](#从tita_rl借鉴的内容)
4. [训练环境搭建](#训练环境搭建)
5. [添加新地形的方法](#添加新地形的方法)
6. [项目代码架构](#项目代码架构)
7. [完整训练流程](#完整训练流程)

---

## 项目概述

Tita2是一个基于Isaac Gym的双轮足机器人强化学习训练框架，专为极限跑酷环境设计。项目融合了：
- **extreme_parkour** 的地形生成和训练框架
- **tita_rl** 的简洁双轮足设计理念
- **Isaac Gym** 的高性能并行仿真

```
    极限跑酷地形      双轮足机器人      高性能训练
   (extreme_parkour) + (tita_rl) = Isaac Gym Tita2
         ↓                ↓              ↓
   复杂障碍物 + 8DOF轮式机器人 = 鲁棒运动策略
```

---

## 双轮足 vs 四足配置区别

### 关键差异对比表

| 配置项 | 四足机器人 (ANYmal) | 双轮足机器人 (Tita2) | 差异说明 |
|--------|---------------------|---------------------|----------|
| **自由度 (DOF)** | 12 (每腿3个关节) | 8 (每腿4个关节) | Tita2多了轮子关节 |
| **足端数量** | 4个足 | 2个轮子 | 接触面从点变为线/面 |
| **接触模式** | 点接触 | 连续滚动接触 | 影响摩擦力和稳定性 |
| **观测维度** | n_proprio=53 | n_proprio=39 | 减少了2个足端的状态 |
| **动作维度** | num_actions=12 | num_actions=8 | 对应8个关节 |
| **边缘检测** | 单点检测 | 圆周采样检测 | 轮子需要特殊处理 |
| **关节索引** | 标准顺序 | reindex重排序 | 适配真实机器人 |

### 详细配置差异

#### 1. 自由度配置
```python
# 四足机器人 (ANYmal)
每腿: [髋关节_横滚, 髋关节_俯仰, 膝关节] = 3 DOF
总计: 4腿 × 3关节 = 12 DOF

# 双轮足机器人 (Tita2)
每腿: [髋_横滚, 髋_俯仰, 膝关节, 轮子] = 4 DOF
总计: 2腿 × 4关节 = 8 DOF
```

#### 2. 观测空间差异
```python
# 四足 n_proprio = 53 计算
ang_vel(3) + imu(2) + yaw(3) + cmd(3) + flags(2) +
dof_pos(12) + dof_vel(12) + actions(12) + contact(4) = 53

# Tita2 n_proprio = 39 计算  
ang_vel(3) + imu(2) + yaw(3) + cmd(3) + flags(2) +
dof_pos(8) + dof_vel(8) + actions(8) + contact(2) = 39
```

#### 3. 足端处理差异
```python
# 四足：4个独立足端
feet_names = ["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"]
contact_buf = (num_envs, contact_buf_len, 4)

# Tita2：2个轮子
feet_names = [body for body in body_names if "leg_4" in body]
contact_buf = (num_envs, contact_buf_len, 2)
```

---

## 从tita_rl借鉴的内容

### 核心借鉴理念

```
 tita_rl 设计哲学     →     Tita2 实现
┌─────────────────────┐   ┌──────────────────────┐
│ 1. 简洁的8DOF设计   │ → │ 直接使用8个关节      │
│ 2. 统一PD参数       │ → │ 所有关节相同刚度     │
│ 3. 关节重排序       │ → │ reindex()方法        │
│ 4. 轮式接触模型     │ → │ 边缘圆周采样算法     │
│ 5. 稳定站姿参考     │ → │ default_joint_angles │
└─────────────────────┘   └──────────────────────┘
```

### 具体借鉴内容

#### 1. **关节配置与默认姿态**
```python
# 来源：tita_rl/configs/legged_robot_config.py
default_joint_angles = {
    'joint_left_leg_1': 0.0,    # 髋-横滚：保持水平
    'joint_left_leg_2': 0.8,    # 髋-俯仰：前倾站姿
    'joint_left_leg_3': -1.5,   # 膝关节：弯曲支撑
    'joint_left_leg_4': 0.0,    # 轮子：自由转动
    # 右腿镜像设置...
}
```

#### 2. **简化的PD控制参数**
```python
# tita_rl风格：统一参数，简化调试
stiffness = {'joint': 40.0}    # 所有关节统一刚度
damping = {'joint': 1.0}       # 所有关节统一阻尼
```

#### 3. **关节重排序策略**
```python
def reindex(self, tensor):
    """
    URDF顺序: [L1,L2,L3,L4, R1,R2,R3,R4]
    真机期望: [R1,R2,R3,R4, L1,L2,L3,L4]
    """
    return tensor[:, [4,5,6,7, 0,1,2,3]]  # 交换左右腿
```

#### 4. **轮式边缘检测算法**
```python
# 创新点：从点检测升级为圆周采样
# 原理：轮子是圆柱体，接触面是弧线而非点
for i in range(wheel_edge_sample_points):
    # 在轮子圆周均匀采样8个点
    sample_pos = wheel_center + radius * [cos(θ), sin(θ)]
    # 统计边缘点比例，超过40%则认为危险
```

---

## 训练环境搭建

### 系统架构图

```
                    Tita2 训练环境架构
    
    ┌─────────────────────────────────────────────────────┐
    │                Isaac Gym 核心                        │
    │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
    │  │ 4096个并行环境│  │  GPU物理仿真  │  │ 张量计算    │ │
    │  └──────────────┘  └──────────────┘  └─────────────┘ │
    └─────────────────────────────────────────────────────┘
                            ↑
    ┌─────────────────────────────────────────────────────┐
    │              Tita2Robot 环境类                      │
    │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
    │  │   机器人加载  │  │   地形生成    │  │  奖励计算   │ │
    │  │  (URDF资产)  │  │ (Parkour系列) │  │(轮式专用)   │ │
    │  └──────────────┘  └──────────────┘  └─────────────┘ │
    └─────────────────────────────────────────────────────┘
                            ↑
    ┌─────────────────────────────────────────────────────┐
    │               PPO 训练算法                          │
    │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
    │  │ Actor网络     │  │ Critic网络    │  │ RMA估计器   │ │
    │  │ (策略输出)    │  │ (价值估计)    │  │(特权信息)   │ │
    │  └──────────────┘  └──────────────┘  └─────────────┘ │
    └─────────────────────────────────────────────────────┘
```

### 环境初始化流程

```
    环境创建流程 (Tita2Robot._create_envs)
    
    ┌─ 开始 ─┐
    │        │
    ▼        │
   加载URDF   │
    │        │
    ▼        │
   设置关节   │ ← PD参数、默认角度、扭矩限制
    │        │
    ▼        │
   创建4096   │ ← 并行环境，域随机化
   个环境     │
    │        │
    ▼        │
   初始化     │ ← 足端索引、碰撞检测、边缘参数
   专用参数   │
    │        │
    ▼        │
   ┌─ 完成 ─┘
```

### 关键组件详解

#### 1. **资产加载 (Asset Loading)**
```python
# 路径：legged_gym/resources/robots/tita/urdf/tita_description.urdf
asset_options = gymapi.AssetOptions()
asset_options.max_motor_effort = 33.5  # 真实电机扭矩限制
robot_asset = self.gym.load_asset(sim, asset_root, asset_file, asset_options)
```

#### 2. **地形生成 (Terrain Generation)**
```python
terrain_dict = {
    "parkour": 0.2,         # 综合跑酷地形
    "parkour_hurdle": 0.2,  # 跨栏障碍
    "parkour_flat": 0.2,    # 平坦练习
    "parkour_step": 0.2,    # 台阶跳跃
    "parkour_gap": 0.2,     # 间隙跨越
}
```

#### 3. **奖励系统 (Reward System)**
```python
# 主要奖励组件
tracking_goal_vel: 1.0    # 目标点导航
tracking_yaw: 1.0         # 朝向对准
feet_edge: -0.5           # 边缘安全（轮式专用）
dof_error: -0.03          # 姿态稳定
hip_pos: -0.5             # 髋关节约束
```

---

## 添加新地形的方法

### 地形系统架构

```
        地形生成系统流程
        
    ┌────────────────────────────────────────────┐
    │            1. 配置文件定义                  │
    │   terrain_dict = {"new_terrain": 0.1}      │
    └────────────────┬───────────────────────────┘
                     │
    ┌────────────────▼───────────────────────────┐
    │            2. 地形生成函数                  │
    │   def new_terrain_function(terrain, ...)    │
    └────────────────┬───────────────────────────┘
                     │
    ┌────────────────▼───────────────────────────┐
    │            3. 高度图生成                    │
    │   修改 height_field_raw 数组               │
    └────────────────┬───────────────────────────┘
                     │
    ┌────────────────▼───────────────────────────┐
    │            4. 三角网格转换                  │
    │   convert_heightfield_to_trimesh()         │
    └────────────────────────────────────────────┘
```

### 添加新地形的步骤

#### 步骤 1：在配置文件中定义地形

编辑 `tita2_config.py`:
```python
class terrain(LeggedRobotCfg.terrain):
    terrain_dict = {
        # ... 现有地形 ...
        "my_custom_terrain": 0.1,     # 新地形，10%概率
        "slalom_course": 0.15,        # 另一个新地形，15%概率
    }
    terrain_proportions = list(terrain_dict.values())
```

#### 步骤 2：实现地形生成函数

在 `legged_gym/utils/terrain.py` 中添加：
```python
def my_custom_terrain(terrain, variation, width, length, height, step_width, step_height, platform_size=1.):
    """
    自定义地形生成函数
    Args:
        terrain: 地形对象
        variation: 难度变化 [0-1]
        width, length: 地形尺寸
        height: 基础高度
        step_width, step_height: 步长参数
        platform_size: 平台大小
    """
    # 1. 初始化平坦地面
    terrain.height_field_raw[:, :] = 0
    
    # 2. 添加你的地形特征
    # 例如：创建蛇形路径
    center_y = width // 2
    for x in range(length):
        # 计算蛇形偏移
        snake_offset = int(width * 0.2 * np.sin(x * 0.1))
        path_y = center_y + snake_offset
        
        # 创建路径（降低高度）
        terrain.height_field_raw[path_y-2:path_y+3, x] = -step_height * variation
        
        # 添加侧边障碍
        if x % 20 == 0:  # 每20个单位放一个障碍
            terrain.height_field_raw[path_y-5:path_y-3, x:x+5] = step_height * variation
            terrain.height_field_raw[path_y+3:path_y+5, x:x+5] = step_height * variation

def slalom_course(terrain, variation, width, length, height, step_width, step_height, platform_size=1.):
    """
    障碍滑雪地形：机器人需要绕过一系列障碍物
    """
    terrain.height_field_raw[:, :] = 0
    
    obstacle_spacing = int(30 * platform_size)  # 障碍物间距
    obstacle_width = int(8 * platform_size)     # 障碍物宽度
    
    for i, x in enumerate(range(obstacle_spacing, length, obstacle_spacing)):
        if x + obstacle_width < length:
            # 交替放置左右障碍
            if i % 2 == 0:  # 左侧障碍
                y_start = width // 4
                y_end = y_start + obstacle_width
            else:  # 右侧障碍  
                y_end = 3 * width // 4
                y_start = y_end - obstacle_width
                
            # 创建障碍物
            obstacle_height = step_height * (0.5 + 0.5 * variation)
            terrain.height_field_raw[y_start:y_end, x:x+obstacle_width] = obstacle_height
```

#### 步骤 3：注册地形函数

在 `legged_gym/utils/terrain.py` 的 `make_terrain` 方法中注册：
```python
def make_terrain(self, choice, difficulty):
    terrain = SubTerrain("terrain", 
                        width=self.width_per_env_pixels, 
                        length=self.length_per_env_pixels, 
                        vertical_scale=self.vertical_scale, 
                        horizontal_scale=self.horizontal_scale)
    
    # 现有地形选择...
    elif choice == 20:  # my_custom_terrain (新索引)
        my_custom_terrain(terrain, difficulty, self.width_per_env_pixels, 
                         self.length_per_env_pixels, 0.005, 
                         step_width, step_height, platform_size)
    elif choice == 21:  # slalom_course
        slalom_course(terrain, difficulty, self.width_per_env_pixels,
                     self.length_per_env_pixels, 0.005,
                     step_width, step_height, platform_size)
    
    return terrain
```

#### 步骤 4：测试新地形

```bash
# 1. 修改配置强制使用新地形 (调试用)
terrain_dict = {
    "my_custom_terrain": 1.0,  # 100%使用新地形
    # 注释掉其他地形...
}

# 2. 运行测试
cd extreme-parkour/legged_gym
python legged_gym/scripts/play.py --task=tita2 --sim_device=cuda:0
```

### 地形设计最佳实践

#### 1. **难度渐进设计**
```python
# 使用 variation 参数 [0-1] 控制难度
easy_height = 0.1 * variation      # 简单：最大10cm
medium_height = 0.2 * variation    # 中等：最大20cm  
hard_height = 0.5 * variation      # 困难：最大50cm
```

#### 2. **机器人尺寸适配**
```python
# 考虑Tita2的物理参数
robot_width = 0.3      # 机器人宽度
robot_length = 0.5     # 机器人长度  
wheel_radius = 0.0925  # 轮子半径

# 确保通道宽度 > robot_width + margin
min_passage_width = robot_width + 0.2  # 20cm余量
```

#### 3. **性能优化**
```python
# 避免过于复杂的几何体
# 使用高度图而非复杂网格
# 预计算常用参数
cache_sin = np.sin(np.linspace(0, 2*np.pi, 100))
cache_cos = np.cos(np.linspace(0, 2*np.pi, 100))
```

---

## 项目代码架构

### 整体架构图

```
                    Tita2 项目代码架构
                    
    ┌─────────────────────────────────────────────────────┐
    │                  训练入口脚本                        │
    │         legged_gym/scripts/train.py                 │
    └─────────────────┬───────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────┐
    │                环境注册中心                          │
    │       legged_gym/envs/__init__.py                   │
    │   task_registry.register("tita2", Tita2Robot)      │
    └─────────────────┬───────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────┐
    │              Tita2 环境实现                         │
    │  ┌─────────────────────┬─────────────────────────┐   │
    │  │    tita2_robot.py   │    tita2_config.py     │   │
    │  │   (环境逻辑实现)     │    (超参数配置)         │   │
    │  └─────────────────────┴─────────────────────────┘   │
    └─────────────────┬───────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────┐
    │                父类基础框架                          │
    │  ┌─────────────────────┬─────────────────────────┐   │
    │  │  legged_robot.py    │  legged_robot_config.py│   │
    │  │  (基础环境类)        │  (基础配置类)           │   │
    │  └─────────────────────┴─────────────────────────┘   │
    └─────────────────┬───────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────┐
    │                支撑工具模块                          │
    │  ┌──────────┬──────────┬──────────┬──────────────┐  │
    │  │ terrain  │  math    │ helpers  │ task_registry│  │
    │  │ (地形)   │ (数学)   │ (辅助)   │ (注册管理)   │  │
    │  └──────────┴──────────┴──────────┴──────────────┘  │
    └─────────────────────────────────────────────────────┘
```

### 核心文件详解

#### 1. **环境入口：`legged_gym/envs/__init__.py`**
```python
# 环境注册机制
from legged_gym.envs.tita2.tita2_robot import Tita2Robot
from legged_gym.envs.tita2.tita2_config import Tita2Cfg, Tita2CfgPPO

# 注册到任务注册表
task_registry.register("tita2", Tita2Robot, Tita2Cfg, Tita2CfgPPO)
```

#### 2. **配置系统：`tita2_config.py`**

```
    配置文件继承关系
    
    LeggedRobotCfg (基类)
           │
           ├─ env: 环境参数
           ├─ terrain: 地形参数  
           ├─ asset: 资产参数
           ├─ rewards: 奖励参数
           └─ ... 其他模块
           
    Tita2Cfg (派生类) ← 覆盖专用参数
           │
           ├─ env.num_actions = 8        # 8DOF
           ├─ env.n_proprio = 39         # 双足观测
           ├─ asset.foot_name = "leg_4"  # 轮子作为足
           ├─ rewards.feet_edge = -0.5   # 轮式边缘检测
           └─ terrain.terrain_dict = {...} # parkour地形
```

#### 3. **环境实现：`tita2_robot.py`**

```python
class Tita2Robot(LeggedRobot):
    """
    关键设计决策：
    1. 继承 LeggedRobot 基类，复用通用逻辑
    2. 覆盖专用方法：_create_envs, _init_buffers  
    3. 添加专用方法：reindex, _reward_feet_edge
    4. 适配轮式特性：轮子参数、边缘检测算法
    """
    
    def _create_envs(self):
        # 创建8DOF双足环境，设置轮子参数
        pass
        
    def _init_buffers(self):
        # 初始化2足缓冲区，跳过力传感器
        pass
        
    def reindex(self, tensor):
        # 关节重排序：URDF顺序 → 真机顺序
        pass
        
    def _reward_feet_edge(self):
        # 轮式边缘检测：圆周采样算法
        pass
```

#### 4. **继承层次关系**

```
    类继承关系图
    
    BaseTask (rsl_rl基类)
           │
           ▼
    LeggedRobot (extreme_parkour基类)
           │ ← 提供：地形、观测、奖励、仿真循环
           ▼
    Tita2Robot (我们的实现)
           │ ← 特化：8DOF、2足、轮式检测
           ▼
    实例化对象 (4096个并行环境)
```

### 数据流图

```
    训练循环中的数据流
    
    Environment State                    Neural Network
    ┌─────────────────┐                 ┌──────────────┐
    │ 机器人状态:      │   observations   │              │
    │ - 关节位置/速度  │ ──────────────► │    Actor     │
    │ - 基座姿态      │                 │   (策略网络)   │
    │ - 接触信息      │                 │              │
    │ - 地形扫描      │                 └──────┬───────┘
    └─────────────────┘                        │
            ▲                                  │ actions
            │                                  ▼
    ┌─────────────────┐                 ┌──────────────┐
    │ 物理仿真:        │  ◄──────────────│   动作执行    │
    │ - Isaac Gym     │                 │ - PD控制器    │
    │ - 4096个环境    │                 │ - 关节扭矩    │
    │ - 并行计算      │                 │ - 重排序      │
    └─────────────────┘                 └──────────────┘
            ▲                                  ▲
            │ next_state                       │
    ┌─────────────────┐                 ┌──────────────┐
    │ 奖励计算:        │                 │   Critic     │
    │ - 目标导航      │ ◄─────────────── │  (价值网络)   │
    │ - 边缘检测      │   value_estimate │              │
    │ - 姿态稳定      │                 │              │
    └─────────────────┘                 └──────────────┘
```

---

## 完整训练流程

### 训练流程图

```
                     Tita2 训练完整流程
                     
    ┌─── 开始训练 ───┐
    │               │
    ▼               │
┌─────────────────┐  │
│  1. 环境初始化   │  │
│  ├ 加载URDF     │  │
│  ├ 创建4096环境  │  │
│  ├ 初始化地形   │  │
│  └ 设置奖励权重  │  │
└─────────┬───────┘  │
          │          │
          ▼          │
┌─────────────────┐  │
│  2. 数据收集     │  │  ← 24步/环境
│  ├ 执行动作     │  │
│  ├ 物理仿真     │  │
│  ├ 计算奖励     │  │
│  └ 记录轨迹     │  │
└─────────┬───────┘  │
          │          │
          ▼          │
┌─────────────────┐  │
│  3. 策略更新     │  │  ← PPO算法
│  ├ 计算优势     │  │
│  ├ 策略梯度     │  │
│  ├ 价值函数     │  │
│  └ 网络更新     │  │
└─────────┬───────┘  │
          │          │
          ▼          │
┌─────────────────┐  │
│  4. 日志记录     │  │
│  ├ 奖励统计     │  │
│  ├ 模型保存     │  │
│  ├ Wandb上传    │  │
│  └ 性能分析     │  │
└─────────┬───────┘  │
          │          │
          ▼          │
    达到最大迭代？    │
    │         │      │
    是        否─────┘
    │
    ▼
 ┌─── 训练完成 ───┐
```

### 详细训练步骤

#### 阶段 1：环境准备 (Environment Setup)

```python
# 1.1 任务注册与配置加载
env = task_registry.make_env("tita2", args=args)
cfg = Tita2Cfg()

# 1.2 并行环境创建
num_envs = 4096  # 4096个并行环境
env._create_envs()  # 创建Isaac Gym仿真环境

# 1.3 地形生成  
terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]  # 5种parkour地形
env.terrain = Terrain(cfg.terrain, num_envs)

# 1.4 观测空间配置
obs_dim = 39 + 187 + 9 + 21 + 390 = 646  # proprio + scan + priv + history
action_dim = 8  # 8个关节动作
```

#### 阶段 2：数据收集 (Data Collection)

```python
# 2.1 仿真循环 (每个训练迭代)
for iteration in range(15000):  # 最大15000次迭代
    
    # 2.2 轨迹收集 (每次24步)
    for step in range(24):
        # 获取观测
        obs = env.compute_observations()  # (4096, 646)
        
        # 策略推理
        actions = policy.act(obs)  # (4096, 8)
        
        # 关节重排序 (URDF → 真机)
        actions_reindexed = env.reindex(actions)
        
        # 执行动作
        env.step(actions_reindexed)
        
        # 奖励计算
        rewards = env.compute_rewards()
        
        # 数据存储
        rollout_storage.add(obs, actions, rewards, ...)
```

#### 阶段 3：策略学习 (Policy Learning)

```python
# 3.1 优势估计 (Advantage Estimation)
advantages = compute_gae(rewards, values, dones)  # GAE-λ算法

# 3.2 PPO更新 (5个epoch)
for epoch in range(5):
    # 3.3 小批次训练 (4个mini-batch)
    for mini_batch in range(4):
        
        # 策略损失
        policy_loss = compute_policy_loss(
            old_actions, new_actions, advantages, clip_param=0.2
        )
        
        # 价值损失  
        value_loss = F.mse_loss(predicted_values, target_values)
        
        # 熵正则化
        entropy_loss = -entropy_coef * action_entropy
        
        # 总损失
        total_loss = policy_loss + value_loss + entropy_loss
        
        # 梯度更新
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm=1.0)
        optimizer.step()
```

#### 阶段 4：监控与评估 (Monitoring)

```python
# 4.1 性能指标记录
episode_rewards = env.episode_rewards.mean()
episode_length = env.episode_lengths.mean()
success_rate = (env.reach_goal_timer > 0).float().mean()

# 4.2 Wandb日志
wandb.log({
    "train/episode_reward": episode_rewards,
    "train/episode_length": episode_length, 
    "train/success_rate": success_rate,
    "train/policy_loss": policy_loss.item(),
    "train/value_loss": value_loss.item(),
})

# 4.3 模型检查点保存 (每100次迭代)
if iteration % 100 == 0:
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }, f"checkpoints/tita2_{iteration}.pt")
```

### 训练命令与监控

#### 基础训练命令
```bash
# 训练启动
cd extreme-parkour/legged_gym
python legged_gym/scripts/train.py \
    --task=tita2 \
    --sim_device=cuda:0 \
    --rl_device=cuda:0 \
    --num_envs=4096 \
    --headless

# 训练监控
tensorboard --logdir logs/  # 查看训练曲线
wandb login                  # 登录Wandb (如果配置)
```

#### 模型评估命令
```bash
# 加载模型测试
python legged_gym/scripts/play.py \
    --task=tita2 \
    --sim_device=cuda:0 \
    --load_run=Nov27_10-00-00_tita2_parkour \
    --checkpoint=5000
```

### 训练性能指标

#### 关键性能指标 (KPIs)
- **Episode Reward**: 总奖励 > 5.0 (目标)
- **Success Rate**: 完成率 > 80% (目标)
- **Episode Length**: 平均长度 > 500步 (目标)
- **Training Speed**: ~2000 FPS (4096环境)

#### 收敛时间预期
```
阶段           迭代次数    时间预估     性能表现
──────────────────────────────────────────
初始探索       0-1000      2-3小时      随机行为
基础学习       1000-5000   10-12小时    简单前进
技能提升       5000-10000  20-25小时    基础跑酷
熟练掌握       10000-15000 30-35小时    复杂地形
```

---

## 总结

Tita2项目成功融合了extreme_parkour的地形挑战性和tita_rl的设计简洁性，通过以下关键创新实现了双轮足机器人的极限跑酷能力：

### 主要贡献
1. **轮式边缘检测算法**：从点检测升级为圆周采样，适配轮式接触特性
2. **8DOF配置适配**：完整的观测/动作空间重构，支持双轮足形态
3. **关节重排序机制**：seamless的URDF到真机映射
4. **Parkour地形集成**：5种专业跑酷地形，渐进难度训练

### 技术优势
- **高性能仿真**：4096并行环境，2000+ FPS训练速度
- **模块化设计**：清晰的继承关系，易于扩展
- **鲁棒性训练**：域随机化 + 特权信息学习
- **实用性导向**：面向真机部署的设计理念

### 应用前景
通过本框架训练的Tita2机器人策略可以：
- 在复杂地形中稳定导航
- 执行跨栏、跳跃、爬坡等动态机动
- 适应不同摩擦力和扰动条件
- 支持sim-to-real迁移部署

这套完整的训练框架为双轮足机器人在极限环境下的应用奠定了坚实基础。
