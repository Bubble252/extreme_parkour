# Tita2 Tracking Yaw 奖励系统说明

## 🎯 核心原理

`tracking_yaw` 是一个**目标点导航奖励**，它鼓励机器人朝向 parkour 地形中预设的目标点前进。

### 奖励函数公式

```python
reward = exp(-|target_yaw - current_yaw|)
```

- **target_yaw**: 从机器人当前位置指向目标点的理想朝向角度
- **current_yaw**: 机器人当前的实际朝向角度
- **奖励值范围**: [0.043, 1.0]
  - 当机器人正对目标时：`exp(-0) = 1.0` (最大奖励)
  - 当机器人偏离90°时：`exp(-π/2) ≈ 0.208`
  - 当机器人背对目标时：`exp(-π) ≈ 0.043` (最小奖励)

**指数函数的优势**：
- 平滑可微，梯度良好
- 对小角度偏差容忍度高（曲线平缓）
- 对大角度偏差惩罚重（快速衰减）

---

## 🗺️ 目标点系统架构

### 1. 地形生成阶段

每个 parkour 地形在生成时会创建一系列**路径点（waypoints）**：

```
起点平台 → 踏脚石1 → 踏脚石2 → ... → 踏脚石8 → 终点平台
   ↓           ↓           ↓                  ↓          ↓
goals[0]   goals[1]    goals[2]          goals[8]   goals[9]
```

**示例**（`parkour_terrain` 函数）：
- `num_stones = 8`：8个踏脚石
- `goals.shape = (10, 2)`：10个目标点（起点+8石+终点）
- 每个目标点记录 (x, y) 世界坐标

### 2. 运行时动态更新

**每个训练步骤都会执行**（`_update_goals()` 函数）：

```python
# Step 1: 检测是否到达当前目标
distance = ||robot_position - current_goal_position||
if distance < next_goal_threshold (0.8米):
    reach_goal_timer += 1
    
# Step 2: 延迟切换（防止抖动）
if reach_goal_timer > reach_goal_delay (0.5秒):
    cur_goal_idx += 1  # 切换到下一个目标
    reach_goal_timer = 0

# Step 3: 计算目标朝向
direction_vector = current_goal - robot_position
target_yaw = atan2(direction_vector.y, direction_vector.x)

# Step 4: 计算朝向偏差（进入观测）
delta_yaw = target_yaw - current_yaw
delta_next_yaw = next_target_yaw - current_yaw
```

**关键机制**：
- **自动推进**：机器人到达目标后自动瞄准下一个
- **延迟保护**：避免在目标点附近来回震荡
- **前瞻信息**：提供下一个目标的方向，帮助提前规划

---

## 📊 观测空间集成

目标导航信息被编码进观测的 `yaw_related(3)` 部分：

```python
obs_buf = [
    base_ang_vel (3),           # 角速度
    imu_obs (2),                # roll, pitch
    0*delta_yaw (1),            # ⚠️ 占位（未使用）
    delta_yaw (1),              # ⭐ 当前目标朝向偏差
    delta_next_yaw (1),         # ⭐ 下一个目标朝向偏差
    commands (3),               # 速度命令
    env_flags (2),              # 地形类型标记
    dof_pos (8),                # 关节位置
    dof_vel (8),                # 关节速度
    actions (8),                # 历史动作
    contact (2),                # 足端接触
]  # 总计 39 维
```

**delta_yaw 的物理含义**：
- `delta_yaw > 0`: 目标在右边 → 需要右转
- `delta_yaw < 0`: 目标在左边 → 需要左转
- `delta_yaw ≈ 0`: 正对目标 → 直走！

**更新频率**：每 5 步更新一次（降低噪声）

---

## 🎓 训练效果

启用 `tracking_yaw` 后，机器人会学习：

### ✅ 空间导航能力
- **任务理解**：从"保持平衡移动"提升到"到达特定位置"
- **路径规划**：主动寻找踏脚石中心，避免边缘
- **目标意识**：理解 parkour 地形的"最优路径"

### ✅ 精准转向策略
- **在线调整**：根据目标位置动态调整朝向
- **提前准备**：利用 `delta_next_yaw` 前瞻下一个转向
- **窄空间导航**：在狭小踏脚石上完成精准转向

### ✅ 与速度奖励的协同
- **tracking_goal_vel** (权重 1.0)：鼓励匹配目标速度
- **tracking_yaw** (权重 1.0)：鼓励朝向正确方向
- **协同效果**：既快又准地到达目标

---

## 🔧 Tita2 配置详解

### 配置文件修改

**1. 环境配置** (`tita2_config.py - class env`)：

```python
reach_goal_delay = 0.5          # 到达延迟 [秒]
# 原理：机器人需要在目标点附近停留 0.5 秒才会切换
# 作用：防止高速经过目标点导致频繁切换

next_goal_threshold = 0.8       # 到达判定距离 [米]
# 原理：距离目标 0.8 米内视为"到达"
# 调优：太小 → 难以触发；太大 → 触发过早

num_future_goal_obs = 2         # 前瞻目标数量
# 原理：env_goals 存储 num_goals + 2 个目标点
# 作用：让机器人知道"当前 → 下一个 → 下下个"的路径
```

**2. 奖励配置** (`tita2_config.py - rewards.scales`)：

```python
tracking_yaw = 1.0
# 权重 1.0 表示与 tracking_goal_vel 同等重要
# 调优建议：
#   - 开始训练：1.0（快速学习导航）
#   - 精细调优：0.5-2.0（平衡速度和精度）
```

### 父类已实现的功能

✅ **变量初始化**（`legged_robot.py` Line 1034-1040）：
```python
self.env_goals             # (num_envs, num_goals+2, 3)
self.cur_goal_idx          # (num_envs,) 当前目标索引
self.reach_goal_timer      # (num_envs,) 到达计时器
self.cur_goals             # (num_envs, 3) 当前目标位置
self.next_goals            # (num_envs, 3) 下一个目标位置
```

✅ **每步更新**（`legged_robot.py` Line 254）：
```python
self._update_goals()       # 更新目标点和 target_yaw
```

✅ **观测编码**（`legged_robot.py` Line 389-397）：
```python
delta_yaw, delta_next_yaw  # 已包含在 n_proprio 中
```

✅ **奖励计算**（`legged_robot.py` Line 1233-1235）：
```python
def _reward_tracking_yaw(self):
    return torch.exp(-torch.abs(self.target_yaw - self.yaw))
```

**Tita2 无需额外实现任何代码！** 🎉

---

## 📏 维度一致性验证

### Tita2 的 n_proprio 计算

```
组成部分：
1. base_ang_vel: 3
2. imu_obs (roll, pitch): 2
3. yaw_related (0*delta_yaw + delta_yaw + delta_next_yaw): 3 ⭐
4. commands (0*cmd[:2] + cmd[0]): 3
5. env_flags: 2
6. dof_pos: 8 (Tita2 特有)
7. dof_vel: 8 (Tita2 特有)
8. actions: 8 (Tita2 特有)
9. contact: 2 (Tita2 特有)

总计 = 3+2+3+3+2+8+8+8+2 = 39 ✓
```

### 完整观测空间

```
obs_buf = [
    obs_buf (n_proprio=39),
    heights (n_scan=187),
    priv_explicit (n_priv=9),
    priv_latent (n_priv_latent=21),
    obs_history (history_len * n_proprio = 10*39=390)
]

总计 = 39 + 187 + 9 + 21 + 390 = 646 ✓
```

**与配置文件完全匹配！**

---

## 🚀 测试步骤

### 快速测试

```bash
cd /home/bubble/桌面/extreme_parkour/extreme-parkour
./test_tracking_yaw.sh
```

### 手动测试

```bash
cd /home/bubble/桌面/extreme_parkour/extreme-parkour/legged_gym/legged_gym/scripts

python train.py \
    --task tita2 \
    --exptid tracking-yaw-test \
    --num_envs 4 \
    --max_iterations 2 \
    --no_wandb \
    --headless
```

### 预期输出

```
[Tita2] Environment created: 4 envs, 8 DOFs, 2 feet, 4 hips
[DEBUG] self.cfg.env.n_proprio = 39
[Tita2 DEBUG] Observation dimensions:
  obs_buf shape: torch.Size([4, 646])
  Expected n_proprio: 39
  Inferred n_proprio: 39.0
  
Reward: tracking_yaw = 0.85 (平均朝向偏差约 25°)
✓ 训练正常进行...
```

### 可能的错误

❌ **AttributeError: 'Tita2Cfg.env' has no attribute 'reach_goal_delay'**
- 原因：配置未正确添加
- 解决：检查 `tita2_config.py` 中 `class env` 是否包含三个新参数

❌ **维度不匹配错误**
- 原因：n_proprio 计算错误
- 解决：应该已经正确（39维），如果出现请报告

---

## 📈 调优建议

### 奖励权重调整

```python
# 保守策略（先学平衡，再学导航）
tracking_yaw = 0.5

# 激进策略（快速学习导航）
tracking_yaw = 2.0

# 平衡策略（当前推荐）
tracking_yaw = 1.0
```

### 到达判定调整

```python
# 宽松判定（适合初期训练）
next_goal_threshold = 1.0

# 严格判定（适合精细控制）
next_goal_threshold = 0.5

# 折中方案（当前推荐）
next_goal_threshold = 0.8
```

### 延迟时间调整

```python
# 短延迟（适合快速移动）
reach_goal_delay = 0.3

# 长延迟（适合稳定停留）
reach_goal_delay = 0.8

# 折中方案（当前推荐）
reach_goal_delay = 0.5
```

---

## 🎯 训练目标对比

### 阶段 1：不使用 tracking_yaw

**学习目标**：
- ✅ 在 parkour 地形上保持平衡
- ✅ 跟随速度命令移动
- ✅ 避免碰撞和跌倒

**行为特征**：
- 可能在踏脚石上随机游走
- 缺乏明确的移动方向
- 容易从踏脚石边缘掉落

### 阶段 2：使用 tracking_yaw（当前）

**学习目标**：
- ✅ 阶段1的所有目标
- ✅ **主动寻找并到达目标点**
- ✅ **沿最优路径穿越地形**
- ✅ **在窄空间中精准导航**

**行为特征**：
- 有目的地朝向踏脚石中心
- 提前规划转向动作
- 完成"起点 → 终点"的完整任务

---

## 🔍 常见问题

### Q1: 为什么 n_proprio 中包含 3 个 yaw 相关维度？

**A**: 这是父类设计的观测编码方式：
- `0*delta_yaw (1)`: 占位符，用于保持观测格式兼容
- `delta_yaw (1)`: 当前目标的朝向偏差（主要信号）
- `delta_next_yaw (1)`: 下一个目标的朝向偏差（前瞻信号）

### Q2: tracking_yaw 和 tracking_goal_vel 有什么区别？

**A**: 
- `tracking_goal_vel`: 鼓励匹配**速度命令**（方向+大小）
- `tracking_yaw`: 鼓励朝向**目标点**（空间导航）
- 两者协同：既要速度对，也要方向对

### Q3: 如果机器人到达终点后怎么办？

**A**: `cur_goal_idx` 会继续递增，但 `_gather_cur_goals()` 会返回最后一个目标点，机器人会保持在终点附近。

### Q4: 地形目标点是如何生成的？

**A**: 在 `terrain.py` 的 `parkour_terrain()` 函数中：
- 踏脚石位置决定目标点位置
- 每个踏脚石的中心即为目标点
- 保证路径连续且可达

---

## 📚 参考资料

### 代码位置

- **配置文件**: `legged_gym/envs/tita2/tita2_config.py`
  - Line 38-42: 目标点导航配置
  - Line 148-151: tracking_yaw 奖励权重

- **父类实现**: `legged_gym/envs/base/legged_robot.py`
  - Line 207-224: `_update_goals()` 目标更新逻辑
  - Line 389-397: 观测编码（包含 delta_yaw）
  - Line 1233-1235: `_reward_tracking_yaw()` 奖励计算
  - Line 1034-1040: 目标点系统初始化

- **地形生成**: `legged_gym/utils/terrain.py`
  - Line 418-530: `parkour_terrain()` 目标点生成

### 论文参考

- **Extreme Parkour**: 使用目标点导航训练四足机器人穿越极限地形
- **RMA (Rapid Motor Adaptation)**: 特权信息估计框架

---

## ✅ 总结

### 修改内容

1. **配置文件** (`tita2_config.py`):
   - ✅ 添加 3 个目标点导航参数
   - ✅ 启用 tracking_yaw 奖励（权重 1.0）

2. **无需修改代码**:
   - ✅ 父类已完整实现所有功能
   - ✅ 观测空间已包含 delta_yaw
   - ✅ 维度计算完全正确

### 工作原理

```
地形生成 → 创建目标点序列
    ↓
每步更新 → 计算 target_yaw 和 delta_yaw
    ↓
观测编码 → delta_yaw 进入 obs_buf
    ↓
策略网络 → 学习"朝向目标"的动作
    ↓
奖励信号 → exp(-|delta_yaw|) 引导学习
    ↓
目标到达 → 自动切换到下一个目标
```

### 预期效果

- 🎯 机器人会主动朝向目标点
- 🏃 沿最优路径穿越地形
- 🎪 完成完整的导航任务

**一切就绪，可以开始训练了！** 🚀
