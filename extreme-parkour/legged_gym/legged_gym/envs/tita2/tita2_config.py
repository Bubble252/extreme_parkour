# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Tita2Cfg(LeggedRobotCfg):
    """
    Tita 双轮足机器人配置 (8 DOF)
    基于 tita_rl 项目的简洁设计，适配 extreme_parkour 的地形和训练框架
    """
    
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        
        # 观测空间维度（根据实际测试调整）
        n_scan = 187                    # 地形扫描点 (17x11 measured_points)
        n_priv = 3+3+3                  # 显式特权信息：base_lin_vel(3) + 2个零填充(3+3)
        n_priv_latent = 4 + 1 + 16      # 隐式特权信息：mass(4) + friction(1) + motor(16=2×8)
        # n_proprio 计算（Tita 专用：8 DOF + 2 feet）:
        # ang_vel(3) + imu(2) + yaw_related(3) + cmd(3) + env_flags(2) +
        # dof_pos(8) + dof_vel(8) + actions(8) + contact(2) = 39
        # 注意：父类默认是 53 (12 DOF + 4 feet)，但 Tita 只有 8+2
        n_proprio = 39
        history_len = 10
        
        # 总观测 = proprio + heights + priv + priv_latent + history
        # 注意：父类公式包含 n_priv，我们也需要包含
        num_observations = n_proprio + n_scan + n_priv + n_priv_latent + history_len*n_proprio
        # = 39 + 187 + 9 + 21 + 390 = 646
        num_privileged_obs = None
        num_actions = 8                 # 8个关节 (每条腿4个关节)
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20
        
        # 目标点导航系统配置（用于 tracking_yaw 奖励）
        # 原理：机器人到达一个目标点后，自动切换到下一个目标点
        reach_goal_delay = 0.5          # 到达目标后延迟 0.5 秒再切换（避免抖动）
        next_goal_threshold = 0.8       # 距离目标 0.8 米内视为"到达" [m]
        num_future_goal_obs = 2         # 观测中包含未来 2 个目标的信息（用于前瞻规划）
        
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.35]  # x, y, z [m] - 双足站立高度
        rot = [0.0, 0.0, 0.0, 1.0]  # x, y, z, w [quat]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        
        # 默认关节角度 [rad] - 参考 tita_rl 的稳定站姿
        default_joint_angles = {
            'joint_left_leg_1': 0.0,      # 左髋-横滚
            'joint_right_leg_1': 0.0,     # 右髋-横滚
            
            'joint_left_leg_2': 0.8,      # 左髋-俯仰
            'joint_right_leg_2': 0.8,     # 右髋-俯仰
            
            'joint_left_leg_3': -1.5,     # 左膝
            'joint_right_leg_3': -1.5,    # 右膝
            
            'joint_left_leg_4': 0.0,      # 左轮
            'joint_right_leg_4': 0.0,     # 右轮
        }
        
    class control(LeggedRobotCfg.control):
        control_type = 'P'
        # 简洁的刚度/阻尼设置 (参考 tita_rl)
        stiffness = {'joint': 40.0}     # 所有关节统一刚度 [N*m/rad]
        damping = {'joint': 1.0}        # 所有关节统一阻尼 [N*m*s/rad]
        action_scale = 0.5              # 动作缩放
        decimation = 4                  # 控制频率 = 50Hz (200Hz / 4)
        hip_scale_reduction = 0.5       # 髋关节额外缩放
        use_filter = True               # 使用动作滤波
        
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/tita/urdf/tita_description.urdf'
        foot_name = "leg_4"             # 轮子作为足端
        name = "tita"
        penalize_contacts_on = ["leg_3"]       # 小腿接触惩罚
        terminate_after_contacts_on = ["base"] # 躯干接触终止
        self_collisions = 0             # 启用自碰撞检测
        flip_visual_attachments = False
        max_motor_effort = 33.5         # 最大电机扭矩 [N*m] (参考 tita_rl)
        
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 2.75]
        
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        
        randomize_base_com = True
        added_com_range = [-0.1, 0.1]
        
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        
        randomize_motor = True
        motor_strength_range = [0.8, 1.2]
        
        randomize_kpkd = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        
        randomize_lag_timesteps = True
        lag_timesteps = 3
        
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.0
        heading_command = True
        global_reference = False
        
        class ranges:
            lin_vel_x = [-1.0, 1.0]   # [m/s]
            lin_vel_y = [-1.0, 1.0]   # [m/s]
            ang_vel_yaw = [-1.0, 1.0] # [rad/s]
            heading = [-3.14, 3.14]   # [rad]
            
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.35       # Tita 目标站立高度
        
        class scales(LeggedRobotCfg.rewards.scales):
            # 启用的奖励
            tracking_goal_vel = 1.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            torques = -0.0001
            dof_acc = -2.5e-7
            collision = -1.0
            feet_stumble = -0.0
            action_rate = -0.01
            
            # 已启用的姿态约束奖励
            delta_torques = -1.0e-7   # ✅ 鼓励平滑扭矩输出
            dof_error = -0.03         # ✅ 保持默认关节姿态 (所有8个关节)
            hip_pos = -0.5            # ✅ 约束髋关节位置 (4个髋关节: leg_1和leg_2)
            
            # 目标点导航奖励（已启用）
            tracking_yaw = 1.0        # ✅ 鼓励机器人朝向当前目标点
            # 原理：exp(-|target_yaw - current_yaw|)，正对目标时奖励最大(1.0)
            # 作用：引导机器人沿着 parkour 地形的最优路径移动
            
            # 地形边缘检测奖励（已启用 - 轮式专用算法）
            feet_edge = -0.5          # ✅ 惩罚轮子在陡峭边缘上
            # 原理：圆周采样8个点，>50%在边缘则惩罚
            # 权重：-0.5（相比四足的-1.0减半，因为轮子需要滚动空间）
            # 作用：引导机器人将轮子放在平坦稳定区域，避免边缘失足
            
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'           # 使用三角网格地形
        measure_heights = True
        curriculum = True
        
        terrain_length = 12.0           # 折中方案：足够长以容纳 parkour 元素，但比 18.0 小
        terrain_width = 8.0             # 保持正方形比例
        num_rows = 10                   # 难度级别
        num_cols = 15                   # 进一步减少列数（从 20 再减到 15）
        
        max_init_terrain_level = 5
        border_size = 25
        
        # 地形类型比例 - 轮足机器人优化配置
        # ============================================================
        # 设计原则：
        # 1. 轮子擅长：坡道、平地、宽通道
        # 2. 轮子困难：跳跃、大高度差、窄缝隙
        # 3. 课程学习：先简单后困难，difficulty 从 0→1
        # ============================================================
        terrain_dict = {
            # === 基础地形 (30%) - 轮子最擅长 ===
            "smooth slope": 0.1,        # ✅ 平滑坡道 (difficulty控制坡度: 0→16°)
            "rough slope up": 0.1,      # ✅ 粗糙上坡 (有地面噪声)
            "rough slope down": 0.1,    # ✅ 粗糙下坡
            
            # === 中等难度 (45%) - 平衡和路径规划 ===
            "smooth flat": 0.10,        # ✅ 纯平地 (热身)
            "parkour_flat": 0.15,       # ✅ 平坦parkour (宽通道训练)
            "log_bridge": 0.15,         # ✅ 独木桥 (窄通道平衡) - 宽度0.5m→0.3m
            "stepping stones": 0.10,    # ⚠️ 踏石 (已调大石块、减小间距)
            
            # === 挑战地形 (15%) - 高难度 ===
            "parkour": 0.10,            # ⚠️ 倾斜石块 (需要精确平衡)
            "platform": 0.10,           # ⚠️ 平台 (低高度跳跃)
            
            # === 禁用地形 (0%) - 不适合轮式 ===
            "rough stairs up": 0.0,     # ❌ 楼梯 - 轮子无法爬
            "rough stairs down": 0.0,
            "discrete": 0.0,            # ❌ 离散障碍
            "gaps": 0.0,                # ❌ 缝隙
            "pit": 0.0,                 # ❌ 深坑
            "wall": 0.0,                # ❌ 墙壁
            "large stairs up": 0.0,     # ❌ 大楼梯
            "large stairs down": 0.0,
            "parkour_hurdle": 0.0,      # ❌ 跨栏 - 轮子无法跳跃
            "parkour_step": 0.0,        # ❌ 大阶梯 - 高度差太大
            "parkour_gap": 0.0,         # ❌ 沟壑 - 轮子会卡住
            "demo": 0.0,
        }
        terrain_proportions = list(terrain_dict.values())
        
        slope_treshold = 0.75
        difficulty_scale = 1.0
        x_init_range = 0.2
        y_init_range = 0.2
        yaw_init_range = 0.0
        x_init_offset = 0.0
        y_init_offset = 0.0
        
        teleport_robots = True
        teleport_thresh = 0.3
        
        # 高度扫描
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        selected = False
        
    class depth(LeggedRobotCfg.depth):
        use_camera = False              # 第一阶段不使用视觉
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20
        
        position = [0.27, 0, 0.03]      # 前置相机
        angle = [-5, 5]                 # 俯仰角
        
        update_interval = 1
        
        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2
        
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        
        scale = 1
        invert = True
        
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            
        clip_observations = 100.0
        clip_actions = 100.0


class Tita2CfgPPO(LeggedRobotCfgPPO):
    """
    Tita2 的 PPO 训练配置
    """
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'tita2_parkour'
        algorithm_class_name = 'PPO'
        policy_class_name = 'ActorCriticRMA'
        
        max_iterations = 15000
        num_steps_per_env = 24
        save_interval = 100
        
        # 日志
        plot_input_gradients = False
        plot_parameter_gradients = False
        
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        
        # RMA (特权信息估计)
        estimator_hidden_dims = [128, 64]
        priv_encoder_dims = [64, 20]
        
    class estimator(LeggedRobotCfgPPO.estimator):
        # 关键：覆盖父类的 num_prop，使用 Tita2 的 n_proprio (39, not 53)
        num_prop = 39  # Tita2Cfg.env.n_proprio
        num_scan = 187  # Tita2Cfg.env.n_scan (17x11 height points)
        priv_states_dim = 9  # 继承父类 n_priv (3+3+3)
