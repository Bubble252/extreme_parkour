#!/usr/bin/env python3
"""
Tita2 地形配置验证脚本
运行方式: conda activate park && python verify_terrain_config.py

直接解析配置文件，不导入任何模块，避免 isaacgym/torch 顺序问题
"""

import sys
import os
import re
import ast

print("=" * 60)
print("Tita2 地形配置验证")
print("=" * 60)

try:
    # 直接读取文件内容并解析 terrain_dict
    config_path = os.path.join(os.path.dirname(__file__), 
        'extreme-parkour/legged_gym/legged_gym/envs/tita2/tita2_config.py')
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # 使用正则表达式提取 terrain_dict
    pattern = r'terrain_dict\s*=\s*\{([^}]+)\}'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        raise ValueError("未找到 terrain_dict 定义!")
    
    # 解析字典
    dict_content = '{' + match.group(1) + '}'
    # 清理注释
    dict_lines = []
    for line in dict_content.split('\n'):
        # 移除 # 后的注释
        if '#' in line:
            line = line[:line.index('#')]
        dict_lines.append(line)
    dict_str = '\n'.join(dict_lines)
    
    terrain_dict = ast.literal_eval(dict_str)
    
    print(f"\n[1] terrain_dict 内容 (共 {len(terrain_dict)} 种地形)")
    print("-" * 50)
    print(f"{'序号':<4} {'状态':<4} {'地形名称':<25} {'权重':<8}")
    print("-" * 50)
    
    active_terrains = []
    total_weight = 0
    for idx, (name, weight) in enumerate(terrain_dict.items()):
        status = "✓" if weight > 0 else " "
        print(f"{idx:<4} {status:<4} {name:<25} {weight:.2f}")
        total_weight += weight
        if weight > 0:
            active_terrains.append((name, weight))
    
    print("-" * 50)
    print(f"\n[2] 统计信息")
    print(f"   - 激活地形数量: {len(active_terrains)}")
    print(f"   - 权重总和: {total_weight:.4f} {'✓ 正确' if abs(total_weight - 1.0) < 0.001 else '✗ 错误 (应为1.0)'}")
    
    print(f"\n[3] 激活的地形列表")
    for name, weight in active_terrains:
        print(f"   - {name}: {weight*100:.1f}%")
    
    print(f"\n[4] 新添加地形检查")
    if 'log_bridge' in terrain_dict:
        print(f"   ✓ log_bridge 已添加，权重: {terrain_dict['log_bridge']}")
    else:
        print(f"   ✗ log_bridge 未找到!")
    
    if 'stepping stones' in terrain_dict:
        print(f"   ✓ stepping stones 权重: {terrain_dict['stepping stones']}")
    else:
        print(f"   ✗ stepping stones 未找到!")
    
    print(f"\n[5] proportions 数组验证")
    proportions = list(terrain_dict.values())
    print(f"   - 数组长度: {len(proportions)} (原始20 + log_bridge = 21)")
    
    # 手动计算累积概率
    total = sum(proportions)
    normalized = [p/total for p in proportions]
    cumulative = []
    cum_sum = 0
    for p in normalized:
        cum_sum += p
        cumulative.append(cum_sum)
    
    print(f"   - 累积概率数组长度: {len(cumulative)}")
    print(f"   - 最后累积值: {cumulative[-1]:.4f} (应为1.0)")
    
    # 验证关键索引
    terrain_names = list(terrain_dict.keys())
    print(f"\n[6] 关键索引验证 (make_terrain 中使用)")
    print(f"   - proportions[6]  = {terrain_names[6]}: {proportions[6]} (stepping stones)")
    print(f"   - proportions[19] = {terrain_names[19]}: {proportions[19]} (demo)")
    print(f"   - proportions[20] = {terrain_names[20]}: {proportions[20]} (log_bridge)")
    
    # 检查累积概率
    print(f"\n[7] 累积概率检查 (决定地形选择)")
    print(f"   - cumulative[19] = {cumulative[19]:.4f} (choice < 这个值 → demo)")
    print(f"   - cumulative[20] = {cumulative[20]:.4f} (choice < 这个值 → log_bridge)")
    
    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()
