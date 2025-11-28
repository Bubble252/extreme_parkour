# Tita2 完整维度验证报告

## 📊 总览

```
配置状态：
✅ tracking_yaw = 1.0   (已启用)
✅ feet_edge = -0.5     (已启用 - 轮式算法)
✅ hip_pos = -0.5       (已启用)
✅ dof_error = -0.03    (已启用)
✅ delta_torques = -1e-7 (已启用)

观测空间：
✅ n_proprio = 39
✅ n_scan = 187
✅ n_priv = 9
✅ n_priv_latent = 21
✅ num_observations = 646
```

---

## 🔍 详细维度分析

### 1. n_proprio = 39 维

#### **组成部分详解**：

```python
观测部分                      维度    累计    说明
─────────────────────────────────────────────────────────
base_ang_vel                  3      3      角速度 (XYZ)
imu_obs (roll, pitch)         2      5      IMU 姿态角
0*delta_yaw                   1      6      占位符
delta_yaw                     1      7      ⭐ 当前目标朝向偏差
delta_next_yaw                1      8      ⭐ 下一个目标朝向偏差
0*commands[:2]                2      10     占位符
commands[0]                   1      11     X方向速度命令
(env_class != 17)             1      12     环境类型标记1
(env_class == 17)             1      13     环境类型标记2
dof_pos (Tita2: 8 DOF)        8      21     关节位置
dof_vel (Tita2: 8 DOF)        8      29     关节速度
actions (Tita2: 8 DOF)        8      37     历史动作
contact (Tita2: 2 feet)       2      39     ⭐ 足端接触
─────────────────────────────────────────────────────────
总计                          39            ✓ 正确
```

**对比父类 (Go1, 53维)**：
```
差异项：
- dof_pos:  12 → 8  (-4)
- dof_vel:  12 → 8  (-4)
- actions:  12 → 8  (-4)
- contact:   4 → 2  (-2)
总差异：53 - 14 = 39 ✓
```

### 2. 完整观测空间 = 646 维

#### **公式验证**：

```python
num_observations = n_proprio + n_scan + n_priv + n_priv_latent + history_len*n_proprio

代入数值：
= 39 + 187 + 9 + 21 + 10×39
= 39 + 187 + 9 + 21 + 390
= 646 ✓

分解：
├─ 当前观测 (39)          ← n_proprio
├─ 地形高度 (187)         ← n_scan (17×11 网格)
├─ 显式特权 (9)           ← n_priv (lin_vel + 填充)
├─ 隐式特权 (21)          ← n_priv_latent (mass+friction+motor)
└─ 历史观测 (390)         ← 10步 × 39维
```

### 3. 奖励函数依赖的张量维度

#### **tracking_yaw**:
```python
self.target_yaw:    (num_envs,)        # 目标朝向
self.yaw:           (num_envs,)        # 当前朝向
delta_yaw:          (num_envs,)        # 差值
reward:             (num_envs,)        ✓
```

#### **feet_edge (轮式算法)**:
```python
# 输入
self.rigid_body_states:     (N, num_bodies, 13)
self.feet_indices:          (2,)                    # Tita2: 2轮

# 中间张量
wheel_center_pos:           (N, 2, 2)              # XY坐标
offsets:                    (8, 2)                 # 圆周采样点
sample_pos (每次循环):      (N, 2, 2)              # 广播正确
sample_grid:                (N, 2, 2)              # 网格坐标
at_edge (每次循环):         (N, 2)                 # 布尔查询
wheel_edge_count:           (N, 2)                 # 累加计数
wheels_at_edge:             (N, 2)                 # 阈值判定

# 输出
self.feet_at_edge:          (N, 2)                 # & 接触过滤
reward:                     (N,)                   # sum(dim=-1)
                                                   ✓ 所有维度正确
```

#### **hip_pos**:
```python
self.hip_indices:           (4,)                   # 4个髋关节
self.default_dof_pos:       (1, 8)                 # ⭐ 已 unsqueeze
self.dof_pos:               (N, 8)                 # 当前位置

# 索引操作
dof_pos[:, hip_indices]:            (N, 4)        # 当前髋位置
default_dof_pos[:, hip_indices]:    (1, 4)        # 默认位置（广播）
square((N,4) - (1,4)):              (N, 4)        # 平方误差
sum(..., dim=1):                    (N,)          ✓
```

#### **dof_error**:
```python
self.default_dof_pos_all:   (N, 8)                # 每环境默认
self.dof_pos:               (N, 8)                # 当前位置
square((N,8) - (N,8)):      (N, 8)                # 误差
sum(..., dim=1):            (N,)                  ✓
```

#### **delta_torques**:
```python
self.torques:               (N, 8)                # 当前扭矩
self.last_torques:          (N, 8)                # 上一步扭矩
square((N,8) - (N,8)):      (N, 8)                # 差值
sum(..., dim=1):            (N,)                  ✓
```

---

## ✅ 关键修改的维度影响

### 修改 1: tracking_yaw 启用

**新增观测维度**：0（delta_yaw 已在 n_proprio 中）
**新增变量**：
- `self.target_yaw`: (N,) ✓
- `self.delta_yaw`: (N,) ✓
- `self.next_target_yaw`: (N,) ✓

**配置依赖**：
```python
reach_goal_delay = 0.5
next_goal_threshold = 0.8
num_future_goal_obs = 2
```
**维度影响**：✅ 无（父类已初始化）

### 修改 2: feet_edge 启用（轮式算法）

**新增观测维度**：0（不增加观测）
**新增变量**：
- `self.wheel_radius`: 标量（0.0925）✓
- `self.wheel_edge_sample_points`: 标量（8）✓
- `self.feet_at_edge`: (N, 2) ✓（覆盖父类）

**方法内临时张量**：
```python
angles:           (8,)          ✓
offsets:          (8, 2)        ✓
wheel_edge_count: (N, 2)        ✓
wheels_at_edge:   (N, 2)        ✓
```

**维度影响**：✅ 无（所有张量正确）

### 修改 3: hip_pos 启用

**新增观测维度**：0（不增加观测）
**新增变量**：
- `self.hip_indices`: (4,) ✓

**关键修正**：
```python
# 修改前
self.default_dof_pos: (8,)  ❌ 会导致索引错误

# 修改后
self.default_dof_pos: (1, 8)  ✓ 允许广播索引
```

**维度影响**：✅ 已修正

---

## 🎯 所有奖励函数维度汇总

| 奖励函数 | 权重 | 输入维度 | 输出维度 | 状态 |
|---------|------|---------|---------|------|
| tracking_goal_vel | 1.0 | (N,3), (N,3) | (N,) | ✅ |
| tracking_yaw | 1.0 | (N,), (N,) | (N,) | ✅ |
| lin_vel_z | -2.0 | (N,3) | (N,) | ✅ |
| ang_vel_xy | -0.05 | (N,3) | (N,) | ✅ |
| orientation | -1.0 | (N,3) | (N,) | ✅ |
| torques | -0.0001 | (N,8) | (N,) | ✅ |
| dof_acc | -2.5e-7 | (N,8), (N,8) | (N,) | ✅ |
| collision | -1.0 | (N,M,3) | (N,) | ✅ |
| action_rate | -0.01 | (N,8), (N,8) | (N,) | ✅ |
| delta_torques | -1e-7 | (N,8), (N,8) | (N,) | ✅ |
| dof_error | -0.03 | (N,8), (N,8) | (N,) | ✅ |
| hip_pos | -0.5 | (N,8), (1,8) | (N,) | ✅ |
| feet_edge | -0.5 | (N,2,2) | (N,) | ✅ |

**所有奖励输出维度统一为 (N,)** ✓

---

## 🧮 Estimator 网络维度验证

### 配置覆盖

```python
class estimator(LeggedRobotCfgPPO.estimator):
    num_prop = 39           # ✅ 覆盖父类的 53
    num_scan = 187          # ✅ 覆盖父类的 132
    priv_states_dim = 9     # ✅ 继承父类
```

### 网络输入维度

```python
# Estimator 输入
input = obs_buf[:, :num_prop + num_scan]
      = obs_buf[:, :39 + 187]
      = obs_buf[:, :226]
input.shape = (N, 226) ✓

# Estimator 输出
priv_latent_pred.shape = (N, n_priv_latent)
                        = (N, 21) ✓

# 与真实 priv_latent 比较（训练时）
priv_latent_true = torch.cat([
    mass_params_tensor,      # (N, 4)
    friction_coeffs_tensor,  # (N, 1)
    motor_strength[0] - 1,   # (N, 8)
    motor_strength[1] - 1    # (N, 8)
], dim=-1)
priv_latent_true.shape = (N, 4+1+8+8)
                        = (N, 21) ✓ 匹配
```

### Actor-Critic 网络维度

```python
# Actor 输入
actor_input = obs_buf[:, :num_observations]
            = obs_buf[:, :646]
actor_input.shape = (N, 646) ✓

# Actor 输出
actions.shape = (N, num_actions)
              = (N, 8) ✓

# Critic 输入（与 Actor 相同）
critic_input.shape = (N, 646) ✓

# Critic 输出
value.shape = (N, 1) ✓
```

---

## 📐 PyTorch 张量形状追踪

### 关键运算广播验证

#### **例子 1: hip_pos 奖励**
```python
dof_pos = torch.randn(4096, 8)              # (N, 8)
default_dof_pos = torch.randn(1, 8)         # (1, 8)
hip_indices = torch.tensor([0, 1, 2, 3])    # (4,)

# 索引操作
a = dof_pos[:, hip_indices]                 # (4096, 8)[:, (4,)]
print(a.shape)  # torch.Size([4096, 4]) ✓

b = default_dof_pos[:, hip_indices]         # (1, 8)[:, (4,)]
print(b.shape)  # torch.Size([1, 4]) ✓

# 广播减法
diff = a - b                                # (4096, 4) - (1, 4)
print(diff.shape)  # torch.Size([4096, 4]) ✓

# 平方和
error = torch.sum(torch.square(diff), dim=1)  # sum over 4 hips
print(error.shape)  # torch.Size([4096]) ✓
```

#### **例子 2: feet_edge 采样**
```python
wheel_pos = torch.randn(4096, 2, 2)         # (N, 2, 2)
offset = torch.randn(2)                     # (2,)

# unsqueeze 链式
offset_expanded = offset.unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
print(offset_expanded.shape)  # torch.Size([1, 1, 2]) ✓

# 广播加法
sample_pos = wheel_pos + offset_expanded    # (4096,2,2) + (1,1,2)
print(sample_pos.shape)  # torch.Size([4096, 2, 2]) ✓
```

---

## ✅ 最终检查清单

### 配置文件 (tita2_config.py)

```
[✓] n_scan = 187
[✓] n_proprio = 39
[✓] n_priv = 9
[✓] n_priv_latent = 21
[✓] num_observations = 646
[✓] num_actions = 8

[✓] reach_goal_delay = 0.5
[✓] next_goal_threshold = 0.8
[✓] num_future_goal_obs = 2

[✓] tracking_yaw = 1.0
[✓] feet_edge = -0.5
[✓] hip_pos = -0.5
[✓] dof_error = -0.03
[✓] delta_torques = -1e-7

[✓] estimator.num_prop = 39
[✓] estimator.num_scan = 187
[✓] estimator.priv_states_dim = 9
```

### 机器人类 (tita2_robot.py)

```
[✓] self.num_dof = 8
[✓] self.feet_indices: (2,)
[✓] self.hip_indices: (4,)
[✓] self.default_dof_pos: (1, 8)  ← 已 unsqueeze
[✓] self.default_dof_pos_all: (N, 8)
[✓] self.wheel_radius = 0.0925
[✓] self.wheel_edge_sample_points = 8
[✓] _reward_feet_edge() 方法已实现
```

### 观测空间

```
[✓] obs_buf: (N, 646)
[✓] obs_history_buf: (N, 10, 39)
[✓] n_proprio 包含 delta_yaw (3维 yaw_related)
```

### 奖励函数

```
[✓] tracking_yaw: 输入(N,) + (N,) → 输出(N,)
[✓] feet_edge: 输入(N,2,2) → 8次采样 → 输出(N,)
[✓] hip_pos: 输入(N,8) + (1,8) → 输出(N,)
[✓] dof_error: 输入(N,8) + (N,8) → 输出(N,)
[✓] delta_torques: 输入(N,8) + (N,8) → 输出(N,)
```

### Estimator 网络

```
[✓] 输入: (N, 226)  = (N, 39+187)
[✓] 输出: (N, 21)   = (N, n_priv_latent)
[✓] 训练目标维度匹配
```

### Actor-Critic 网络

```
[✓] Actor 输入: (N, 646)
[✓] Actor 输出: (N, 8)
[✓] Critic 输入: (N, 646)
[✓] Critic 输出: (N, 1)
```

---

## 🎉 验证结论

### 所有维度完全正确！

```
✅ 观测空间：646 维
✅ 动作空间：8 维
✅ 历史缓冲：10 × 39 维
✅ Estimator：226 → 21 维
✅ 所有奖励：输出 (N,) 维
✅ 所有张量广播：正确无误
```

### 修改总结

| 功能 | 文件 | 行数 | 维度影响 |
|------|------|------|---------|
| tracking_yaw | config | 5 | ✅ 无 |
| feet_edge | robot | 85 | ✅ 无 |
| feet_edge | config | 5 | ✅ 无 |
| hip_pos | robot | 10 | ✅ 已修正 |
| hip_pos | config | 1 | ✅ 无 |
| **总计** | | **106** | **✅ 全部正确** |

---

## 🚀 准备训练

**所有系统检查通过，可以安全开始训练！**

```bash
cd /home/bubble/桌面/extreme_parkour/extreme-parkour/legged_gym/legged_gym/scripts

# 快速测试
python train.py --task tita2 --exptid complete-test \
    --num_envs 4 --max_iterations 2 --no_wandb --headless

# 完整训练
python train.py --task tita2 --exptid tita2-parkour-v1 \
    --num_envs 4096 --max_iterations 15000
```

**预期新奖励**：
- ✅ tracking_yaw ≈ 0.8 (朝向目标)
- ✅ feet_edge ≈ -0.2 (边缘安全)
- ✅ hip_pos ≈ -0.3 (髋关节约束)
- ✅ dof_error ≈ -0.1 (姿态保持)
- ✅ delta_torques ≈ -0.001 (扭矩平滑)

🎊 **Tita2 现在拥有完整的安全导航和姿态约束系统！**
