# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class TitaCfg(LeggedRobotCfg):
    """
    Configuration for Tita bipedal wheeled robot (8 DOF)
    - 2 legs, each with 4 joints: hip_roll, hip_pitch, knee, wheel
    """
    
    class env(LeggedRobotCfg.env):
        # 环境参数
        num_envs = 4096
        num_observations = 235  # 根据实际情况调整
        num_privileged_obs = None  # 如果为 None，则从 num_obs 中推断
        num_actions = 8  # 8个关节：每条腿4个关节
        env_spacing = 3.0  # 环境之间的间距 [m]
        send_timeouts = True  # 发送时间超时信号
        episode_length_s = 20  # episode 长度 [s]
        
    class init_state(LeggedRobotCfg.init_state):
        # 初始状态 - 调整为适合双轮足的高度
        pos = [0.0, 0.0, 0.55]  # x, y, z [m] - 双足需要更高的初始高度
        
        # 默认关节角度 [rad] - 根据 URDF 中的关节名和限位设置
        # 这些角度需要让机器人能够稳定站立
        default_joint_angles = {
            # 左腿 4个关节
            'joint_left_leg_1': 0.0,      # 髋关节-横滚 (范围: ±0.785 rad = ±45°)
            'joint_left_leg_2': 0.6,      # 髋关节-俯仰 (范围: -1.92~3.49 rad)
            'joint_left_leg_3': -1.3,     # 膝关节 (范围: -2.67~-0.698 rad)
            'joint_left_leg_4': 0.0,      # 末端轮子 (连续旋转)
            
            # 右腿 4个关节
            'joint_right_leg_1': 0.0,     # 髋关节-横滚
            'joint_right_leg_2': 0.6,     # 髋关节-俯仰
            'joint_right_leg_3': -1.3,    # 膝关节
            'joint_right_leg_4': 0.0,     # 末端轮子
        }
        
    class control(LeggedRobotCfg.control):
        # PD控制器参数（根据Tita实际电机性能调整）
        control_type = 'P'
        
        # 刚度参数 [N*m/rad]
        # 注意：使用部分字符串匹配，不是正则表达式
        stiffness = {
            'leg_1': 80.,   # 髋横滚关节 (匹配 joint_left_leg_1, joint_right_leg_1)
            'leg_2': 80.,   # 髋俯仰关节 (匹配 joint_left_leg_2, joint_right_leg_2)
            'leg_3': 80.,   # 膝关节 (匹配 joint_left_leg_3, joint_right_leg_3)
            'leg_4': 20.,   # 轮子驱动电机 (匹配 joint_left_leg_4, joint_right_leg_4)
        }
        
        # 阻尼参数 [N*m*s/rad]
        damping = {
            'leg_1': 2.0,
            'leg_2': 2.0,
            'leg_3': 2.0,
            'leg_4': 1.0,   # 轮子驱动电机（适当阻尼）
        }
        
        action_scale = 0.25      # 动作缩放因子
        decimation = 4           # 控制频率 = 50Hz (仿真200Hz / 4)
        
    class asset(LeggedRobotCfg.asset):
        # Tita机器人模型路径
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/tita/urdf/tita_description.urdf'
        
        # 轮子接触检测 - 根据URDF中的link名
        # 注意：leg_4 是轮子连杆，不是传统的"足端"，Tita 没有独立的足部力传感器
        foot_name = "leg_4"  # 轮子连杆（用于检测轮子接触地面）
        
        # 碰撞检测 - 如果这些部位碰撞会受到惩罚
        penalize_contacts_on = ["leg_2", "leg_3"]  # 大腿、小腿
        
        # 如果这些部位碰撞会终止episode
        terminate_after_contacts_on = ["base_link"]  # 躯干碰撞会终止
        
        # 自碰撞设置 (1 = 禁用, 0 = 启用)
        self_collisions = 1
        
        # 翻转修正（如果URDF方向有问题）
        flip_visual_attachments = True
        fix_base_link = False
        
        # 关节属性覆盖
        override_inertia = True
        override_com = True
        
    class domain_rand(LeggedRobotCfg.domain_rand):
        # 域随机化 - 提高真机迁移能力
        randomize_friction = True
        friction_range = [0.5, 1.5]
        
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        
        randomize_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]
        
    class rewards(LeggedRobotCfg.rewards):
        # 奖励权重 - 使用父类已有的奖励函数，针对双轮足机器人调整权重
        class scales:
            # 速度跟踪奖励（使用父类已有的）
            tracking_goal_vel = 1.5     # 跟踪速度命令（父类已有）
            tracking_yaw = 0.5          # 跟踪偏航角（父类已有）
            
            # 姿态控制（双足更重要）
            lin_vel_z = -2.0            # 惩罚垂直方向速度（父类已有）
            ang_vel_xy = -0.1           # 惩罚横滚/俯仰角速度（父类已有）
            orientation = -2.0          # 惩罚姿态偏离（父类已有）
            
            # 关节控制（父类已有）
            torques = -0.00001          # 惩罚大力矩
            dof_acc = -2.5e-7           # 惩罚关节加速度
            action_rate = -0.1          # 惩罚动作突变
            delta_torques = -1.0e-7     # 惩罚力矩变化
            
            # 足端控制（父类已有）
            collision = -10.0           # 惩罚碰撞
            feet_stumble = -1.0         # 惩罚足端拖地
            feet_edge = -1.0            # 惩罚足端边缘接触
            
            # 关节位置（父类已有）
            hip_pos = -0.5              # 惩罚髋关节位置
            dof_error = -0.04           # 惩罚关节位置误差
            
        # 奖励函数参数
        only_positive_rewards = False
        tracking_sigma = 0.25           # 速度跟踪的容忍度
        soft_dof_pos_limit = 0.9        # 关节位置软限制（接近硬限制时开始惩罚）
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        
        base_height_target = 0.5        # Tita目标高度 [m]（根据实际调整）
        max_contact_force = 200.0       # 最大接触力 [N]（轮子可能承受更大力）
        
    class terrain(LeggedRobotCfg.terrain):
        # 地形配置
        mesh_type = 'trimesh'
        curriculum = True
        
        # 地形尺寸
        terrain_length = 18.0
        terrain_width = 4.0
        num_rows = 10
        num_cols = 40
        
        # 只使用有 goals 属性的地形类型（parkour 系列和 demo）
        # 其他地形类型会导致 AttributeError: 'SubTerrain' object has no attribute 'goals'
        terrain_dict = {
            "smooth slope": 0., 
            "rough slope up": 0.0,
            "rough slope down": 0.0,
            "rough stairs up": 0., 
            "rough stairs down": 0., 
            "discrete": 0., 
            "stepping stones": 0.0,
            "gaps": 0., 
            "smooth flat": 0,
            "pit": 0.0,
            "wall": 0.0,
            "platform": 0.,
            "large stairs up": 0.,
            "large stairs down": 0.,
            "parkour": 0.2,           # ✓ 有 goals
            "parkour_hurdle": 0.2,    # ✓ 有 goals  
            "parkour_flat": 0.2,      # ✓ 有 goals
            "parkour_step": 0.2,      # ✓ 有 goals
            "parkour_gap": 0.2,       # ✓ 有 goals
            "demo": 0.0,              # ✓ 有 goals
        }
        terrain_proportions = list(terrain_dict.values())
        
    class commands(LeggedRobotCfg.commands):
        # 命令范围
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4                # [vx, vy, vyaw, heading]
        
        class ranges:
            lin_vel_x = [-1.0, 1.5]     # [m/s] - 前后速度
            lin_vel_y = [-0.5, 0.5]     # [m/s] - 左右速度
            ang_vel_yaw = [-1.0, 1.0]   # [rad/s] - 转向角速度
            heading = [-3.14, 3.14]     # [rad] - 朝向
            
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            
    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0  # 噪声等级的缩放因子
        
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
            
    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]
        
    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        
        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0      # [m]
            bounce_threshold_velocity = 0.5  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps


class TitaCfgPPO(LeggedRobotCfgPPO):
    """
    Tita机器人的PPO训练配置
    """
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        # PPO算法参数
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = 'adaptive'  # 'adaptive' 或 'fixed'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # 每个环境的步数
        max_iterations = 15000  # 最大迭代次数
        run_name = ''
        experiment_name = 'tita_parkour'
        save_interval = 100     # 每100次迭代保存一次
        
        # 日志相关
        empirical_normalization = False
        log_interval = 10
        
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # 'elu', 'relu', 'selu', 'crelu', 'lrelu', 'tanh', 'sigmoid'
        
        # 适配器设置（用于特权信息学习）
        adaptation_module_branch_hidden_dims = [256, 128]
        
