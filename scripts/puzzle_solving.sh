#!/bin/bash

# 批量处理多个模型和多个rating分数段

# 定义所有rating分数段
ratings=(200 600 1000 1400 1800 2200 2600)

# 定义所有模型
models=(
    "deepseek-v3"
    "gpt-4.1"
    "claude-3-7-sonnet-20250219-v1:0"
    "deepseek-r1"
    "o3-2025-04-16"
    "gemini-2.5-pro"
    "doubao-seed-1-6-thinking-250615"
    "qwen3-235b-a22b"
    "deepseek-v3.1"
)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}开始批量处理模型评估...${NC}"
echo "模型数量: ${#models[@]}"
echo "Rating分数段: ${ratings[*]}"
echo "总任务数: $((${#models[@]} * ${#ratings[@]}))"
echo "=========================="

# 计数器
total_tasks=0
completed_tasks=0
failed_tasks=0

# 嵌套循环：外层是模型，内层是rating; with legal move
for model in "${models[@]}"; do
    echo -e "${BLUE}处理模型: $model${NC}"
    echo "----------------------------------------"
    
    for rating in "${ratings[@]}"; do
        ((total_tasks++))
        
        echo -e "${YELLOW}[$total_tasks] 模型: $model | Rating: $rating${NC}"
        
        # 执行命令
        if python fine_grained_evaluation.py --task "puzzle_solving" \
            --model_id "$model" \
            --model_name "$model" \
            --max_tokens 16384 \
            --rating $rating \
            --with_legal_move; then
            
            echo -e "${GREEN}✓ 完成: $model @ $rating${NC}"
            ((completed_tasks++))
        else
            echo -e "${RED}✗ 失败: $model @ $rating${NC}"
            ((failed_tasks++))
        fi
        
        echo "------------------------"
    done
    
    echo -e "${BLUE}模型 $model 所有rating处理完成${NC}"
    echo "========================================"
done

# 嵌套循环：外层是模型，内层是rating; without legal move
for model in "${models[@]}"; do
    echo -e "${BLUE}处理模型: $model${NC}"
    echo "----------------------------------------"
    
    for rating in "${ratings[@]}"; do
        ((total_tasks++))
        
        echo -e "${YELLOW}[$total_tasks] 模型: $model | Rating: $rating${NC}"
        
        # 执行命令
        if python fine_grained_evaluation.py --task "puzzle_solving" \
            --model_id "$model" \
            --model_name "$model" \
            --max_tokens 16384 \
            --rating $rating; then
            
            echo -e "${GREEN}✓ 完成: $model @ $rating${NC}"
            ((completed_tasks++))
        else
            echo -e "${RED}✗ 失败: $model @ $rating${NC}"
            ((failed_tasks++))
        fi
        
        echo "------------------------"
    done
    
    echo -e "${BLUE}模型 $model 所有rating处理完成${NC}"
    echo "========================================"
done

# 最终统计
echo -e "${GREEN}所有任务处理完成!${NC}"
echo "总任务数: $total_tasks"
echo -e "${GREEN}成功: $completed_tasks${NC}"
echo -e "${RED}失败: $failed_tasks${NC}"
echo "成功率: $(( completed_tasks * 100 / total_tasks ))%"