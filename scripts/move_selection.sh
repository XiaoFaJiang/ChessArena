#!/bin/bash

# 批量处理多个模型和多个play mode的chess能力评测

# 定义思考模型（使用standard或blindfold模式）
reasoning_models=(
    "deepseek-r1"
    "gemini-2.5-pro"
    "doubao-seed-1-6-thinking-250615"
    "doubao-1-5-thinking-pro-250415"
    "o3-2025-04-16"
)

# 定义非思考模型（使用blitz、bullet或blindfold模式）
chat_models=(
    "deepseek-v3"
    "gpt-4.1"
    "claude-3-7-sonnet-20250219-v1:0"
    "qwen3-235b-a22b"
    "deepseek-v3.1"  
)

# 定义play modes
# "bullet" "blindfold"
reasoning_play_modes=("standard" "blindfold")
chat_play_modes=("blitz" "bullet" "blindfold" )
eval_nums=1000
# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# 计算总任务数
total_reasoning_tasks=$((${#reasoning_models[@]} * ${#reasoning_play_modes[@]}))
total_chat_tasks=$((${#chat_models[@]} * ${#chat_play_modes[@]}))
total_tasks=$(($total_reasoning_tasks + $total_chat_tasks))

echo -e "${GREEN}开始批量处理模型Chess能力评估...${NC}"
echo "思考模型数量: ${#reasoning_models[@]} (${reasoning_models[*]})"
echo "非思考模型数量: ${#chat_models[@]} (${chat_models[*]})"
echo "思考模型Play Modes: ${reasoning_play_modes[*]}"
echo "非思考模型Play Modes: ${chat_play_modes[*]}"
echo "总任务数: $total_tasks (思考模型: $total_reasoning_tasks + 非思考模型: $total_chat_tasks)"
echo "=========================="

# 计数器
current_task=0
completed_tasks=0
failed_tasks=0

# 执行评估的函数
run_evaluation() {
    local model=$1
    local play_mode=$2
    local model_type=$3
    local with_legal=$4
    
    ((current_task++))
    echo -e "${YELLOW}[$current_task/$total_tasks] 模型: $model | 模式: $play_mode | 类型: $model_type${NC}"
    
    # 构建命令参数
    cmd_args="--task move_selection --model_id $model --model_name $model --max_tokens 32768 --play_mode $play_mode --eval_nums $eval_nums"
    
    if [ "$with_legal" = "true" ]; then
        cmd_args="$cmd_args --with_legal_move"
        echo -e "${PURPLE}  └─ 包含合法走法提示${NC}"
    else
        echo -e "${PURPLE}  └─ 不包含合法走法提示${NC}"
    fi
    
    cmd_args="$cmd_args "
    # 执行命令
    if python fine_grained_evaluation.py $cmd_args; then
        echo -e "${GREEN}✓ 完成: $model ($model_type) - $play_mode${NC}"
        ((completed_tasks++))
    else
        echo -e "${RED}✗ 失败: $model ($model_type) - $play_mode${NC}"
        ((failed_tasks++))
    fi
    
    echo "------------------------"
}

# 处理非思考模型
echo -e "${BLUE}=== 处理非思考模型 ===${NC}"
for model in "${chat_models[@]}"; do
    echo -e "${BLUE}处理非思考模型: $model${NC}"
    
    for play_mode in "${chat_play_modes[@]}"; do
        # 带合法走法提示的评估
        run_evaluation "$model" "$play_mode" "chat" "true"
        
        # 不带合法走法提示的评估
        run_evaluation "$model" "$play_mode" "chat" "false"
    done
    
    echo -e "${BLUE}非思考模型 $model 处理完成${NC}"
    echo "========================================"
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


# 处理思考模型
echo -e "${BLUE}=== 处理思考模型 ===${NC}"
for model in "${reasoning_models[@]}"; do
    echo -e "${BLUE}处理思考模型: $model${NC}"
    
    for play_mode in "${reasoning_play_modes[@]}"; do
        # 带合法走法提示的评估
        run_evaluation "$model" "$play_mode" "reasoning" "true"
        
        # 不带合法走法提示的评估
        #run_evaluation "$model" "$play_mode" "reasoning" "false"
    done
    
    echo -e "${BLUE}思考模型 $model 处理完成${NC}"
    echo "========================================"
done


# 生成总结报告
echo ""
echo -e "${PURPLE}=== 评估总结 ===${NC}"
echo "思考模型评估: ${#reasoning_models[@]} 个模型 × ${#reasoning_play_modes[@]} 种模式 × 2 种设置 = $total_reasoning_tasks 个任务"
echo "非思考模型评估: ${#chat_models[@]} 个模型 × ${#chat_play_modes[@]} 种模式 × 2 种设置 = $total_chat_tasks 个任务"