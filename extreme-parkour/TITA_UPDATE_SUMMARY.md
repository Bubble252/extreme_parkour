# Tita机器人教程更新摘要

## 更新时间
2024年

## 更新内容

### 1. 基于实际URDF的Tita机器人结构说明

从URDF文件 `legged_gym/resources/robots/tita/urdf/tita_description.urdf` 中分析得出：

#### Tita机器人完整结构

**基本信息：**
- 机器人类型：双轮足机器人（Bipedal Wheeled Robot）
- 总自由度：**8个电机关节**
- 躯干质量：13.2 kg

**关节详细信息：**

##### 左腿（4个关节）：
1. **joint_left_leg_1**: 髋关节-横滚 (Hip Roll)
   - 轴向：Z轴旋转
   - 范围：-0.785 ~ 0.785 rad (约 ±45°)
   - 最大力矩：60 N·m
   - 最大速度：25 rad/s

2. **joint_left_leg_2**: 髋关节-俯仰 (Hip Pitch)
   - 轴向：Z轴旋转
   - 范围：-1.92 ~ 3.49 rad (约 -110° ~ 200°)
   - 最大力矩：60 N·m
   - 最大速度：25 rad/s

3. **joint_left_leg_3**: 膝关节 (Knee)
   - 轴向：Z轴旋转
   - 范围：-2.67 ~ -0.698 rad (约 -153° ~ -40°)
   - 最大力矩：60 N·m
   - 最大速度：25 rad/s

4. **joint_left_leg_4**: 末端轮子 (Wheel)
   - 轴向：Z轴旋转
   - 范围：无限位（连续旋转）
   - 最大力矩：15 N·m
   - 最大速度：20 rad/s
   - 轮子半径：0.0925 m (9.25 cm)

##### 右腿（4个关节）：
1. **joint_right_leg_1**: 髋关节-横滚（与左腿对称）
2. **joint_right_leg_2**: 髋关节-俯仰（与左腿对称）
3. **joint_right_leg_3**: 膝关节（与左腿对称）
4. **joint_right_leg_4**: 末端轮子（与左腿对称）

**连杆质量分布：**
- base_link: 13.2 kg
- leg_1 (髋部): 2.064 kg
- leg_2 (大腿): 3.098 kg
- leg_3 (小腿): 0.572 kg
- leg_4 (轮子): 1.509 kg

---

## 2. 教程更新的具体章节

### 第6章：适配新机器人 - 已全面更新

#### 6.1.2 节 - Tita双轮足机器人结构
- ✅ 从假设的"6关节双足"更新为**实际的8关节双轮足结构**
- ✅ 添加了每个关节的详细名称和角度限位
- ✅ 标注了URDF文件的确切位置

#### 6.2 节 - Tita机器人文件已就绪
- ✅ 列出了实际的文件结构（urdf/和meshes/目录）
- ✅ 说明Tita文件已包含在项目中，无需额外添加
- ✅ 列出了所有网格文件：base_link.STL, left_leg_1~4.STL, right_leg_1~4.STL

#### 6.3 节 - 创建Tita机器人配置文件
- ✅ 类名从 `BipedalRobotCfg` 改为 `TitaCfg`
- ✅ `num_actions` 从 6 更新为 **8**
- ✅ `default_joint_angles` 使用真实关节名：
  - `joint_left_leg_1~4`
  - `joint_right_leg_1~4`
- ✅ 为腿部关节和轮子关节设置了不同的PD控制参数：
  - 腿部关节：stiffness=80, damping=2.0
  - 轮子关节：stiffness=5, damping=0.5
- ✅ URDF路径指向真实文件：`resources/robots/tita/urdf/tita_description.urdf`
- ✅ 足端名称改为 `leg_4`（轮子连杆）
- ✅ 碰撞检测设置：`leg_2`, `leg_3` 碰撞惩罚，`base_link` 碰撞终止
- ✅ 添加了针对双轮足的特殊奖励项：`wheel_slip`, `wheel_vel`

#### 6.4 节 - 注册Tita机器人环境
- ✅ 导入语句改为：`from .tita.tita_config import TitaCfg, TitaCfgPPO`
- ✅ 注册名称改为 `"tita"`
- ✅ 实验名称改为 `tita_parkour`

#### 6.5 节 - 调整Tita的观测和动作空间
- ✅ 观测维度从 33 更新为 **39**
  - 关节状态从 18 (6×3) 更新为 **24 (8×3)**
- ✅ 动作维度从 6 更新为 **8**
- ✅ 添加了详细的观测空间分解说明

#### 6.6 节 - 测试Tita机器人
- ✅ 所有命令中的 `--task` 参数从 `bipedal_robot` 改为 `tita`
- ✅ 实验ID示例改为 `200-01-tita-baseline`

#### 6.7 节 - 常见调整项
- ✅ 6.7.1: 针对Tita的稳定性调整（腿部关节+轮子分别设置）
- ✅ 6.7.2: 针对Tita的运动学习调整
- ✅ **新增 6.7.3**: 专门针对轮子不转动或打滑的问题
  - 轮子刚度调整
  - 轮子速度奖励
  - 摩擦力调整
- ✅ 6.7.4: 训练速度优化（原6.7.3）

---

## 3. 关键技术要点

### 混合运动模式
Tita是**双轮足机器人**，不是单纯的双足机器人：
- **腿部关节（6个）**：用于跳跃、平衡、姿态控制
- **轮子关节（2个）**：用于滚动、滑行、高速移动

### 控制策略差异
- **腿部关节**需要较高的刚度和阻尼来维持姿态
- **轮子关节**需要较低的刚度和阻尼以允许自由转动
- 两种关节类型需要使用**正则表达式**分别设置控制参数

### 观测空间特点
相比四足机器人（12自由度）：
- 观测维度略小（39 vs 48）
- 足端接触只有2个（vs 4个）
- 需要特别关注平衡和姿态信息

### 奖励函数设计
新增针对双轮足的特殊奖励：
- `wheel_vel`: 鼓励轮子转动
- `wheel_slip`: 惩罚轮子打滑
- `base_height`: 维持合适的躯干高度
- `feet_air_time`: 鼓励适当的跳跃/腾空

---

## 4. 快速开始指南

### 步骤1: 创建Tita配置文件
```bash
cd ~/桌面/extreme_parkour/extreme-parkour/legged_gym/legged_gym/envs
mkdir tita
cd tita
# 创建 tita_config.py（参考教程6.3节）
```

### 步骤2: 注册Tita环境
编辑 `legged_gym/legged_gym/envs/__init__.py`
```python
from .tita.tita_config import TitaCfg, TitaCfgPPO
task_registry.register("tita", LeggedRobot, TitaCfg(), TitaCfgPPO())
```

### 步骤3: 测试环境
```bash
cd ~/桌面/extreme_parkour/extreme-parkour/legged_gym/scripts
python play.py --task tita --exptid test-00 --num_envs 4 --checkpoint 0
```

### 步骤4: 开始训练
```bash
python train.py --task tita --exptid 200-01-tita-baseline --device cuda:0 --num_envs 4096 --max_iterations 15000 --headless
```

---

## 5. 文件清单

### 已存在的文件（无需修改）：
- ✅ `legged_gym/resources/robots/tita/urdf/tita_description.urdf`
- ✅ `legged_gym/resources/robots/tita/meshes/*.STL` (9个网格文件)

### 需要创建的文件：
- ⬜ `legged_gym/legged_gym/envs/tita/tita_config.py`
- ⬜ `legged_gym/legged_gym/envs/tita/__init__.py`

### 需要修改的文件：
- ⬜ `legged_gym/legged_gym/envs/__init__.py` (添加Tita注册代码)

---

## 6. 与四足机器人的主要差异对比

| 项目 | 四足机器人 (A1/Go1) | Tita双轮足 |
|------|---------------------|-----------|
| 关节数 | 12 (每腿3×4) | 8 (每腿4×2) |
| 足端数 | 4 | 2 (轮子) |
| 平衡难度 | 较低 | 较高 |
| 运动模式 | 纯腿部运动 | 腿部+轮子混合 |
| 观测维度 | 48 | 39 |
| 动作维度 | 12 | 8 |
| 关节类型 | 同质（全为关节） | 异质（关节+轮子） |
| 控制策略 | 统一PD参数 | 分层PD参数 |
| 特殊挑战 | 步态协调 | 平衡+轮地接触 |

---

## 7. 注意事项

1. **关节名称必须精确匹配URDF**：
   - ❌ 错误：`left_hip`, `left_knee`, `left_wheel`
   - ✅ 正确：`joint_left_leg_1`, `joint_left_leg_2`, `joint_left_leg_3`, `joint_left_leg_4`

2. **关节顺序很重要**：
   - 在 `default_joint_angles` 字典中，Isaac Gym会按URDF中的关节定义顺序读取
   - 建议按照 left_1, left_2, left_3, left_4, right_1, right_2, right_3, right_4 的顺序

3. **轮子关节的特殊性**：
   - 轮子关节无角度限制（`lower=-999999, upper=999999`）
   - 轮子关节需要较低的控制刚度
   - 轮子关节的力矩限制较低（15 N·m vs 60 N·m）

4. **初始姿态调试**：
   - 建议先在GUI模式下测试（`--num_envs 4`，不加 `--headless`）
   - 确保初始姿态下机器人能稳定站立
   - 可以使用 `play.py --checkpoint 0` 查看初始姿态

5. **奖励函数调优**：
   - 双轮足机器人的平衡更加困难，可能需要更高的 `orientation` 惩罚权重
   - 轮子的滚动和打滑需要仔细平衡

---

## 8. 下一步工作

教程已完整更新，接下来可以：

1. **创建配置文件**：按照教程6.3节创建 `tita_config.py`
2. **注册环境**：按照教程6.4节修改 `__init__.py`
3. **测试环境**：按照教程6.6节运行测试命令
4. **开始训练**：按照教程第5章的训练流程进行

如有任何问题，可参考：
- 教程第7章：常见问题FAQ
- 教程第8章：进阶话题
- 项目已有的机器人配置：`legged_gym/legged_gym/envs/a1/`, `go1/`, `cassie/`

---

**更新状态：✅ 完成**

所有关于Tita机器人的信息已更新为基于真实URDF文件的准确数据。
