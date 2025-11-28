#!/usr/bin/env python3
"""快速测试 Tita 环境是否能正确加载"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    print("1. 导入 legged_gym...")
    from legged_gym import LEGGED_GYM_ROOT_DIR
    print(f"   ✓ LEGGED_GYM_ROOT_DIR: {LEGGED_GYM_ROOT_DIR}")
    
    print("\n2. 导入 task_registry...")
    from legged_gym.utils.task_registry import task_registry
    print(f"   ✓ 可用任务: {list(task_registry.task_classes.keys())}")
    
    print("\n3. 导入 Tita 配置...")
    from legged_gym.envs.tita.tita_config import TitaCfg, TitaCfgPPO
    print("   ✓ TitaCfg 导入成功")
    
    print("\n4. 检查配置...")
    cfg = TitaCfg()
    print(f"   - 关节数量: {cfg.env.num_actions}")
    print(f"   - 环境数量: {cfg.env.num_envs}")
    print(f"   - URDF 路径: {cfg.asset.file}")
    print(f"   - 初始高度: {cfg.init_state.pos}")
    print(f"   - 地形类型: {cfg.terrain.mesh_type}")
    print(f"   - 地形网格: {cfg.terrain.num_rows} x {cfg.terrain.num_cols}")
    
    print("\n5. 检查默认关节角度...")
    for joint_name, angle in cfg.init_state.default_joint_angles.items():
        print(f"   - {joint_name}: {angle:.3f} rad")
    
    print("\n✅ 所有基础检查通过！")
    print("\n提示：现在可以运行完整测试：")
    print("python train.py --task tita --exptid test-tita-00 --num_envs 4 --max_iterations 1 --no_wandb --headless")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
