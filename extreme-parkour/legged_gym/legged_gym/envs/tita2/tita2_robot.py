# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tita2Robot: 专为 8-DOF 双轮足机器人设计的环境类
参考 tita_rl 项目的简洁实现，覆盖必要的父类方法
"""

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.tita2.tita2_config import Tita2Cfg
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import torch


class Tita2Robot(LeggedRobot):
    """
    Tita 双轮足机器人环境
    
    关键设计：
    - 8 DOF (不填充，直接使用)
    - 2 个足端 (左右轮子)
    - reindex 方法：调整关节顺序以匹配真实机器人
    """
    
    def __init__(self, cfg: Tita2Cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
    def _create_envs(self):
        """
        覆盖父类方法，创建 Tita 机器人环境
        关键：只创建 2 个力传感器 (左右轮)
        """
        # 加载 URDF
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        
        # 保存默认关节角度
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.gym.get_asset_dof_name(robot_asset, i)
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    dof_props_asset['stiffness'][i] = self.cfg.control.stiffness[dof_name]
                    dof_props_asset['damping'][i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.cfg.control.stiffness[name] = 0.
                self.cfg.control.damping[name] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
            dof_props_asset['effort'][i] = self.cfg.asset.max_motor_effort
        
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        
        # Tita 特定：2 个足端（轮子）
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        print(f"[Tita2] Detected feet: {feet_names}")
        
        # 注意：不需要创建力传感器！contact_forces 由 Isaac Gym 自动提供
        
        # 碰撞检测
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        
        # 初始状态
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + \
                               self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        # 初始化 mass_params_tensor（用于域随机化）
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        # 初始化摩擦系数存储
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs = torch.zeros((self.num_envs, 1), dtype=torch.float, device='cpu', requires_grad=False)
        
        # 创建环境
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, 
                                                 self.cfg.asset.name, i, 
                                                 self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            
            # 收集质量参数
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)
        
        # 摩擦系数张量
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)
        
        # 足端索引
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        
        # 碰撞索引
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])
        
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        
        print(f"[Tita2] Environment created: {self.num_envs} envs, {self.num_dof} DOFs, {len(feet_names)} feet")
    
    def _init_buffers(self):
        """
        完全覆盖父类方法，跳过 force_sensor_tensor 初始化
        
        原因：我们没有创建力传感器，只使用 contact_forces（Isaac Gym 自动计算）
        """
        print("[Tita2] Initializing buffers...")
        print(f"[DEBUG] self.cfg.env.n_proprio = {self.cfg.env.n_proprio}")
        print(f"[DEBUG] Expected n_proprio = 53 (config file value)")
        print(f"[DEBUG] Config object id: {id(self.cfg)}")
        
        # 获取 Isaac Gym GPU 状态张量
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # 创建张量包装器
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        
        # 注意：不初始化 force_sensor_tensor，只使用 contact_forces
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        
        # 初始化其他数据
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        
        if self.cfg.env.history_encoding:
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_dofs, device=self.device, dtype=torch.float)
        
        # 注意：contact_buf 形状为 (num_envs, contact_buf_len, 2) - 只有 2 个足端
        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 2, device=self.device, dtype=torch.float)
        
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self._resample_commands(torch.arange(self.num_envs, device=self.device, requires_grad=False))
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False)
        
        # 足端空中时间追踪（2 个足端）
        self.feet_air_time = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device, requires_grad=False)
        
        # 基础速度和方向（父类需要）
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # 高度测量
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        
        # PD gains（已在 _create_envs 中设置 default_dof_pos）
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_all[:] = self.default_dof_pos.unsqueeze(0)
        
        # 设置 PD gains
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
        
        print(f"[Tita2] Buffers initialized: contact_forces shape = {self.contact_forces.shape}, contact_buf shape = {self.contact_buf.shape}")
    
    def reindex(self, tensor):
        """
        Tita 专用关节重排序
        
        URDF 中的关节顺序可能是：
        [left_leg_1, left_leg_2, left_leg_3, left_leg_4, 
         right_leg_1, right_leg_2, right_leg_3, right_leg_4]
        
        真实机器人期望的顺序（参考 tita_rl）：
        [right_leg_1, right_leg_2, right_leg_3, right_leg_4,
         left_leg_1, left_leg_2, left_leg_3, left_leg_4]
        
        即：交换左右腿顺序
        """
        # tensor shape: [num_envs, 8]
        # 重排为：右腿4个 + 左腿4个
        return tensor[:, [4, 5, 6, 7, 0, 1, 2, 3]]
    
    def reindex_feet(self, tensor):
        """
        足端重排序：交换左右足
        tensor shape: [num_envs, 2]
        """
        return tensor[:, [1, 0]]
    
    def compute_observations(self):
        """
        临时覆盖以调试观测维度
        """
        # 调用父类方法
        super().compute_observations()
        
        # 打印实际观测维度（只打印一次）
        if not hasattr(self, '_obs_dim_printed'):
            print(f"\n[Tita2 DEBUG] Observation dimensions:")
            print(f"  obs_buf shape: {self.obs_buf.shape}")
            print(f"  Expected n_proprio: {self.cfg.env.n_proprio}")
            print(f"  obs_history_buf shape: {self.obs_history_buf.shape}")
            
            # 尝试推断实际的 n_proprio
            # obs = n_proprio + n_scan + history*n_proprio + n_priv_latent
            # 让我们从 obs_buf[0] 中减去已知的部分
            total_obs = self.obs_buf.shape[1]
            n_scan = self.cfg.env.n_scan if hasattr(self.cfg.env, 'n_scan') else 132
            n_priv_latent = self.cfg.env.n_priv_latent if hasattr(self.cfg.env, 'n_priv_latent') else 21
            history_len = self.cfg.env.history_len if hasattr(self.cfg.env, 'history_len') else 10
            
            # total = n_proprio + n_scan + history_len * n_proprio + n_priv_latent
            # total = n_proprio * (1 + history_len) + n_scan + n_priv_latent
            # n_proprio = (total - n_scan - n_priv_latent) / (1 + history_len)
            inferred_n_proprio = (total_obs - n_scan - n_priv_latent) / (1 + history_len)
            
            print(f"  Inferred n_proprio: {inferred_n_proprio}")
            print(f"  Formula check: {inferred_n_proprio} * (1 + {history_len}) + {n_scan} + {n_priv_latent} = {inferred_n_proprio * (1 + history_len) + n_scan + n_priv_latent}")
            print()
            
            self._obs_dim_printed = True


# 导入必要的模块
import os
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
