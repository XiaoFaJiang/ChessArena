#!/bin/bash

# 批量处理Chess Modeling任务

# 定义模型列表
models=(
    "deepseek-v3"
    "gpt-4.1"
    "claude-3-7-sonnet-20250219-v1:0"
    "doubao-seed-1-6-thinking-250615"
    "qwen3-235b-a22b"
    "deepseek-v3.1"
    "o3-2025-04-16"
    "gemini-2.5-pro"
    "deepseek-r1"
)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 计数器
total_tasks=${#models[@]}
current_task=0
completed_tasks=0
failed_tasks=0

echo -e "${GREEN}开始批量处理Basic understanding任务...${NC}"
echo "模型数量: $total_tasks"
echo "模型列表: ${models[*]}"
echo "=========================="

# 遍历所有模型
for model in "${models[@]}"; do
    ((current_task++))
    
    echo -e "${BLUE}[$current_task/$total_tasks] 处理模型: $model${NC}"
    echo "----------------------------------------"
    
    # 执行chess modeling任务
    if python fine_grained_evaluation.py --task "basic_understanding" \
        --model_id "$model" \
        --model_name "$model" \
        --eval_nums 200 \
        --max_tokens 32768;then
        
        echo -e "${GREEN}✓ 完成: $model${NC}"
        ((completed_tasks++))
    else
        echo -e "${RED}✗ 失败: $model${NC}"
        ((failed_tasks++))
    fi
    
    echo "------------------------"
done

# 最终统计
echo -e "${GREEN}所有任务处理完成!${NC}"
echo "总任务数: $total_tasks"
echo -e "${GREEN}成功: $completed_tasks${NC}"
echo -e "${RED}失败: $failed_tasks${NC}"

if [ $total_tasks -gt 0 ]; then
    success_rate=$(( completed_tasks * 100 / total_tasks ))
    echo "成功率: ${success_rate}%"
else
    echo "成功率: 0%"
fi

# 如果有失败的任务，显示提醒
if [ $failed_tasks -gt 0 ]; then
    echo -e "${YELLOW}注意: 有 $failed_tasks 个任务失败，请检查日志${NC}"
fi