#!/bin/bash
# 测试 Tita2 的 tracking_yaw 奖励功能

echo "========================================="
echo "测试 Tita2 tracking_yaw 奖励系统"
echo "========================================="
echo ""

cd /home/bubble/桌面/extreme_parkour/extreme-parkour/legged_gym/legged_gym/scripts

echo "启动训练（2次迭代，4个环境，无界面模式）..."
echo "检查点："
echo "  ✓ 环境创建成功"
echo "  ✓ 观测维度 = 646"
echo "  ✓ 目标点系统初始化"
echo "  ✓ tracking_yaw 奖励正常计算"
echo ""

python train.py \
    --task tita2 \
    --exptid tracking-yaw-test \
    --num_envs 4 \
    --max_iterations 2 \
    --no_wandb \
    --headless

echo ""
echo "========================================="
echo "测试完成！请检查上方输出："
echo "========================================="
echo ""
echo "预期输出应包含："
echo "  [Tita2] Environment created: 4 envs, 8 DOFs, 2 feet, 4 hips"
echo "  [DEBUG] self.cfg.env.n_proprio = 39"
echo "  [Tita2 DEBUG] Observation dimensions:"
echo "    obs_buf shape: torch.Size([4, 646])"
echo ""
echo "如果看到以上输出且没有维度错误，说明 tracking_yaw 启用成功！"
