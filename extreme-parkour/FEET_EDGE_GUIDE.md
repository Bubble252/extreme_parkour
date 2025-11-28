# feet_edge 奖励详解：地形边缘检测与足端约束

## 🎯 核心功能

`feet_edge` 是 extreme_parkour 中的**地形边缘检测奖励**，它惩罚机器人将足端放在地形陡峭边缘（垂直悬崖）上的行为，鼓励机器人踩在平坦稳定的区域。

---

## 📐 工作原理

### 1. 边缘掩码（Edge Mask）的生成

#### **生成时机**：地形创建时（Terrain 类初始化）

**流程**（`terrain.py` Line 85-91）：

```python
# Step 1: 将高度图转换为三角网格时生成边缘掩码
self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(
    self.height_field_raw,
    self.cfg.horizontal_scale,    # 0.05 米
    self.cfg.vertical_scale,      # 0.005 米
    self.cfg.slope_treshold       # 1.5 (陡度阈值)
)

# Step 2: 形态学膨胀，扩展边缘区域
half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
# edge_width_thresh = 0.05 米 → half_edge_width = 1 个网格
structure = np.ones((half_edge_width*2+1, 1))  # (3, 1) 竖直膨胀核
self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)
```

#### **边缘判定条件**（`convert_heightfield_to_trimesh` Line 916-919）：

```python
slope_threshold = 1.5 * horizontal_scale / vertical_scale
# = 1.5 * 0.05 / 0.005 = 15.0

# 检测 X 方向的陡峭边缘（前后方向）
move_x[:num_rows-1, :] += (hf[1:, :] - hf[:-1, :] > slope_threshold)
move_x[1:, :] -= (hf[:-1, :] - hf[1:, :] > slope_threshold)

# 返回边缘掩码
return vertices, triangles, move_x != 0
```

**物理含义**：
- 如果相邻网格高度差 > 15 个单位（0.075米），标记为边缘
- `move_x != 0` 表示该位置发生了网格移动（垂直表面修正）
- 这些位置通常是悬崖、台阶边缘、跳跃间隙等

**掩码形状**：
```python
x_edge_mask.shape = (tot_rows, tot_cols)
# 例如：(1500, 1500) 表示整个地形的网格掩码
# 值：True = 边缘区域（危险），False = 平坦区域（安全）
```

---

### 2. 运行时边缘检测

#### **每个训练步骤执行**（`_reward_feet_edge()` Line 1278-1286）：

```python
def _reward_feet_edge(self):
    # Step 1: 获取足端在世界坐标系中的 XY 位置
    # self.rigid_body_states: (num_envs, num_bodies, 13)
    # self.feet_indices: (4,) 四个足端的索引
    feet_world_pos = self.rigid_body_states[:, self.feet_indices, :2]  # (N, 4, 2)
    
    # Step 2: 转换为地形网格坐标
    feet_pos_xy = (
        (feet_world_pos + self.terrain.cfg.border_size) /  # 加上边界偏移
        self.cfg.terrain.horizontal_scale                   # 除以网格尺寸
    ).round().long()  # 四舍五入为整数索引
    
    # Step 3: 裁剪索引防止越界
    feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
    feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
    
    # Step 4: 查询边缘掩码
    feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
    # feet_at_edge: (N, 4) 布尔张量，True 表示该足端在边缘上
    
    # Step 5: 只惩罚接触地面且在边缘的足端
    self.feet_at_edge = self.contact_filt & feet_at_edge
    # contact_filt: (N, 4) 足端接触过滤（考虑滞后）
    
    # Step 6: 计算奖励（只在高难度地形启用）
    rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
    # terrain_levels > 3：只在难度等级 4+ 启用
    # torch.sum(dim=-1)：统计有几只脚在边缘上
    # rew: (N,) 值域 [0, 4]，0=安全，4=四脚都在边缘
    
    return rew
```

---

### 3. 奖励计算详解

#### **奖励函数**：

```python
reward = -1.0 * num_feet_at_edge * (terrain_level > 3)
```

**奖励矩阵**：

| 场景 | terrain_level | 足端状态 | num_feet_at_edge | 最终奖励 |
|------|---------------|----------|------------------|----------|
| 初级地形 | ≤ 3 | 任意 | 0-4 | **0.0** (不惩罚) |
| 高级地形-安全 | > 3 | 全在平地 | 0 | **0.0** ✓ |
| 高级地形-轻度 | > 3 | 1脚边缘 | 1 | **-1.0** |
| 高级地形-中度 | > 3 | 2脚边缘 | 2 | **-2.0** |
| 高级地形-严重 | > 3 | 3脚边缘 | 3 | **-3.0** |
| 高级地形-危险 | > 3 | 4脚边缘 | 4 | **-4.0** ✗ |

#### **为什么只在 terrain_level > 3 启用？**

```python
# 课程学习策略：
Level 0-3: 简单地形（平地、小台阶）
    → 边缘较少，让机器人先学会基本移动
    → 不启用 feet_edge 惩罚，避免过早约束

Level 4+: 困难地形（大跳跃、窄石头）
    → 边缘很多，需要精准落脚
    → 启用 feet_edge 惩罚，强制学习安全落脚
```

---

## 🎨 可视化

#### **调试可视化**（`_draw_feet()` Line 1128-1139）：

```python
def _draw_feet(self):
    if hasattr(self, 'feet_at_edge'):
        # 绿球 = 安全区域的足端
        non_edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, color=(0, 1, 0))
        
        # 红球 = 边缘区域的足端
        edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, color=(1, 0, 0))
        
        feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
        for i in range(4):
            if self.feet_at_edge[self.lookat_id, i]:
                gymutil.draw_lines(edge_geom, ...)  # 画红球
            else:
                gymutil.draw_lines(non_edge_geom, ...)  # 画绿球
```

**运行可视化**：
```bash
python train.py --task go1 --no_wandb  # 不加 --headless
# 在 Isaac Gym 窗口中会看到足端的颜色标记
```

---

## 🔧 配置参数

### 地形配置（`legged_robot_config.py`）

```python
class terrain:
    # 边缘检测相关
    slope_treshold = 1.5            # 陡度阈值（高度差/水平距离）
    edge_width_thresh = 0.05        # 边缘膨胀宽度 [米]
    horizontal_scale = 0.05         # 网格水平分辨率 [米]
    vertical_scale = 0.005          # 网格垂直分辨率 [米]
```

**参数计算**：
```python
# 陡度判定：
actual_slope_threshold = slope_treshold * horizontal_scale / vertical_scale
                       = 1.5 * 0.05 / 0.005 = 15.0

# 物理含义：相邻网格高度差 > 0.075米（15 × 0.005）时标记为边缘

# 边缘膨胀：
half_edge_width = int(edge_width_thresh / horizontal_scale)
                = int(0.05 / 0.05) = 1 个网格

# 膨胀核大小：(2*1+1, 1) = (3, 1)
# 作用：将边缘标记向前后各扩展 1 个网格（0.05米）
```

### 奖励配置（`legged_robot_config.py`）

```python
class rewards:
    class scales:
        feet_edge = -1.0  # 每只脚在边缘上扣 1 分
```

---

## 🎓 训练效果

### ✅ 启用 feet_edge 后的行为改进

#### **Before (feet_edge = 0.0)**:
```
机器人行为：
❌ 落脚点随机，经常踩在边缘
❌ 在窄石头上不稳定，容易滑落
❌ 跳跃落地时脚尖悬空
❌ 在台阶边缘失去平衡
```

#### **After (feet_edge = -1.0)**:
```
机器人行为：
✅ 主动寻找平坦区域落脚
✅ 足端尽量靠近踏脚石中心
✅ 跳跃时预判安全着陆区
✅ 避免在陡峭边缘附近移动

学习效果：
✅ 成功率提升（减少失足跌落）
✅ 动作更稳定（足端位置更优）
✅ 能通过更困难的地形
```

---

## 🔬 技术细节

### 1. 坐标系转换

```python
# 世界坐标 → 网格坐标
feet_world_pos = [x_world, y_world]  # 单位：米
feet_grid_pos = (feet_world_pos + border_size) / horizontal_scale
              = (feet_world_pos + 5.0) / 0.05

# 示例：
# 世界坐标 (2.5, 3.0) → 网格坐标 ((2.5+5)/0.05, (3.0+5)/0.05) = (150, 160)
```

### 2. 边缘膨胀的必要性

**原因**：
- 原始 `x_edge_mask` 只标记陡峭点本身
- 但足端有物理尺寸（约 0.02-0.05米半径）
- 如果只在精确边缘惩罚，足端可能"挂"在边缘旁边
- 膨胀后，边缘周围 0.05 米都被标记，更安全

**示例**：
```
原始边缘: ███
膨胀后:   █████  (向前后各扩展 1 格)
```

### 3. contact_filt 的作用

```python
self.feet_at_edge = self.contact_filt & feet_at_edge
```

**为什么要 AND 操作？**
- `feet_at_edge`: 足端位置在边缘上（几何检测）
- `contact_filt`: 足端接触地面（力传感器）
- 只惩罚**同时满足**两个条件的情况

**场景分析**：
```
场景 1: 足端在边缘，但在空中（跳跃中）
  feet_at_edge = True, contact_filt = False
  → 不惩罚（还没落地，位置可能改变）

场景 2: 足端在边缘，且接触地面
  feet_at_edge = True, contact_filt = True
  → 惩罚！（真正危险的情况）

场景 3: 足端在平地，且接触地面
  feet_at_edge = False, contact_filt = True
  → 不惩罚（安全状态）
```

---

## 🚀 为什么 Tita2 暂时禁用？

### Tita2 的特殊性

```python
# 四足机器人 (Go1)
num_feet = 4              # 四个独立的足端
foot_name = "foot"        # 点接触

# 双轮足机器人 (Tita2)
num_feet = 2              # 两个轮子
foot_name = "leg_4"       # 轮子（连续接触）
```

### 挑战分析

#### **挑战 1: 轮子的连续接触特性**

```
四足足端：
  ●  ●
  ●  ●  → 4 个离散接触点，容易判定"在边缘"还是"在平地"

双轮足端：
  ╭───╮
  │ ● │
  ╰───╯ × 2 → 轮子是圆柱体，接触地面是一条线或弧面
```

**问题**：
- `self.rigid_body_states[:, self.feet_indices, :2]` 返回轮子刚体中心
- 但轮子半径约 0.05-0.1 米，实际接触区域更大
- 轮子中心在平地，但轮子边缘可能已经悬空

#### **挑战 2: 边缘掩码的网格分辨率**

```python
horizontal_scale = 0.05  # 5厘米网格
轮子半径 ≈ 0.05-0.1 米   # 5-10厘米

# 轮子可能跨越多个网格：
Grid: [ ][ ][轮][子][ ][ ]
Edge:     ███      
         ↑ 可能只有部分网格被标记为边缘
```

#### **挑战 3: 动态稳定性考虑**

```
四足机器人：
  - 静态稳定（三足支撑）
  - 可以停在狭窄区域
  - 足端精度要求高

双轮足机器人：
  - 动态稳定（类似倒立摆）
  - 需要持续调整平衡
  - 轮子需要"滚动空间"
  - 边缘约束可能限制平衡调整
```

---

## 💡 Tita2 启用建议

### 方案 1: 修改边缘检测逻辑（需要实现）

```python
def _reward_feet_edge_wheeled(self):
    """针对轮式机器人的边缘检测"""
    # 获取轮子位置
    wheel_pos = self.rigid_body_states[:, self.feet_indices, :2]  # (N, 2, 2)
    
    # 考虑轮子半径（例如 0.08 米）
    wheel_radius = 0.08
    n_sample_points = 8  # 在轮子周围采样 8 个点
    
    # 生成圆周采样点
    angles = torch.linspace(0, 2*np.pi, n_sample_points, device=self.device)
    offsets = wheel_radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    
    # 检查每个采样点是否在边缘
    wheel_edge_count = torch.zeros(self.num_envs, 2, device=self.device)
    for i in range(n_sample_points):
        sample_pos = wheel_pos + offsets[i].unsqueeze(0).unsqueeze(0)
        sample_grid = ((sample_pos + self.terrain.cfg.border_size) / 
                       self.cfg.terrain.horizontal_scale).round().long()
        # ... 检查边缘 ...
        at_edge = self.x_edge_mask[sample_grid[..., 0], sample_grid[..., 1]]
        wheel_edge_count += at_edge.float()
    
    # 如果超过一半的采样点在边缘，惩罚
    threshold = n_sample_points * 0.5
    self.feet_at_edge = (wheel_edge_count > threshold) & self.contact_filt
    
    rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
    return rew
```

### 方案 2: 使用更保守的参数

```python
# tita2_config.py
class rewards:
    class scales:
        feet_edge = -0.5  # 减半权重（相比四足的 -1.0）
        
class terrain:
    edge_width_thresh = 0.10  # 加倍膨胀（从 0.05 改为 0.10）
    # 作用：更早警告轮子接近边缘
```

### 方案 3: 渐进启用

```python
# 阶段 1: 先禁用，学会基本移动
feet_edge = 0.0

# 阶段 2: 低权重启用，轻度约束
feet_edge = -0.2

# 阶段 3: 正常权重，完整约束
feet_edge = -0.5 或 -1.0
```

---

## 📊 性能影响

### 计算开销

```python
# 每步操作：
1. 坐标转换: O(num_envs × num_feet)
2. 网格查询: O(num_envs × num_feet) - 仅索引操作，很快
3. 布尔运算: O(num_envs × num_feet)
4. 求和: O(num_envs)

# 总开销：可忽略（< 0.1ms）
```

### 训练影响

```
启用 feet_edge 后：
✅ 收敛速度：可能稍慢（额外约束）
✅ 最终性能：显著提升（更安全的策略）
✅ 泛化能力：更好（学会识别危险区域）
```

---

## 📚 参考资料

### 代码位置

- **奖励实现**: `legged_gym/envs/base/legged_robot.py`
  - Line 1278-1286: `_reward_feet_edge()` 奖励计算
  - Line 1128-1139: `_draw_feet()` 可视化
  - Line 864: `x_edge_mask` 初始化

- **地形生成**: `legged_gym/utils/terrain.py`
  - Line 85-91: 边缘掩码生成和膨胀
  - Line 879-943: `convert_heightfield_to_trimesh()` 边缘检测

- **配置文件**: `legged_gym/envs/base/legged_robot_config.py`
  - Line 142: `edge_width_thresh = 0.05`
  - Line 144-145: `horizontal_scale`, `vertical_scale`
  - Line 182: `slope_treshold = 1.5`
  - Line 310: `feet_edge = -1`

### 论文参考

- **Extreme Parkour with Legged Robots** (2023)
  - 使用边缘检测避免机器人从平台掉落
  - 课程学习策略：逐步增加地形难度

---

## ✅ 总结

### feet_edge 的核心价值

```
地形生成 → 检测陡峭边缘（高度差 > 阈值）
    ↓
创建边缘掩码 → 标记危险区域（包括膨胀）
    ↓
运行时检测 → 查询足端是否在边缘上
    ↓
奖励惩罚 → 引导机器人避开边缘
    ↓
学习结果 → 更安全的落脚策略
```

### 对 Tita2 的启示

- **优势**: 可以约束轮子不在危险边缘
- **挑战**: 轮子连续接触，需要特殊处理
- **建议**: 阶段 1 禁用，阶段 2+ 渐进启用

### 设计哲学

```
feet_edge 体现了强化学习的"安全约束"思想：
✅ 不是告诉机器人"应该怎么做"
✅ 而是告诉机器人"不应该怎么做"
✅ 留给机器人探索安全解的自由度
```

**这就是为什么 extreme_parkour 能训练出如此鲁棒的跳跃策略！** 🎉
