# ChessArena
这是ChessArena的提交代码，包含：1. chess competition sampling（对战匹配算法）; 2. Glicko-1 rating system; 3. fine-grained evaluation; 4. chess training四个部分

## Evaluation
### Chess competition sampling
如论文编写，我们的chess competition sampling基于Glicko公式推导：匹配两个实力接近的玩家RD的收敛最快；

** 随机选手匹配 ** 
```
python competition_sampling.py --games 4
```

** 定向选手匹配 **
```
python competition_sampling.py \
    --target \
    --player1_id "gpt-4.1" \
    --player1_name "gpt-4.1_blitz_True"
```
其中player1_id为调用openai API格式的model id;
player1_name可以自己设置任意名称，但后续需要跟上模型的play mode以及是否提供legal moves;
需要保证player1_name在./simulation_record/ratings.json中
通过这两种方式，进行比赛匹配与对弈；

### 手动编写config.json
进行比赛匹配算法，我们会在./config中添加对应比赛的config,并根据此config开始对弈
我们提供了一些config的示例在./config文件夹中，您可以自行查阅；
并根据此config编写您自己的想要的模型对弈；
使用以下代码：
```
python run_simulation.py \
    --config ./config/blitz_vs_blitz/qwen3-235b-a22b_blitz_vs_doubao-1.5-lite_blitz_legal.json \
    --games 4
    --parallel 2
```
启动对弈比赛

### Prompt
您可以在utils.py中调整每个模式的对弈prompt（如果您认为需要）

### Glicko rating system
您可以使用
```
python elo_calculate.py
```
得到所有模型对弈结果的Glicko分数

### Fine-grained evaluation
在ChessArena中，我们有三个fine-grained evaluation tasks，分别为:basic understanding, move selection and puzzle sovling；
您可以使用fine_grained_evaluation.py启动这三个任务的评估
例如: basic understanding
```
python static.py \
 --task "basic_understanding" \
 --model_id "your model id" \
 --api_key "your api key" \
 --model_name "your model name" \
 --url "your api url" \
```
进行评估，您也可以传入 $temperature,top_p,max_tokens$等参数控制LLMs生成结果的质量；
在fine-grained evaluation.py中，也有board_reconstruction任务，它是blindfold的棋盘重建任务，如您感兴趣，也可以进行实验；

### Sciprts
我们在./sciprts中保存了示例脚本，供您查阅


## Chess training
我们所有的训练都在8卡H800完成，SFT训练耗时4H左右；RL训练耗时60H左右；
** 训练代码 ** 
我们在./chess_train文件夹中保存了我们所有的训练代码：SFT-stage1, SFT-stage2(LLamaFactory)以及RL(Verl)训练

** 数据 **
因为上传文件大小关系，我们只保留了一部分训练数据在代码仓库中，供参考

### SFT
** 训练脚本 **
所有的SFT训练代码都保存在./chess_train/LLaMA-Factory文件夹中
SFT-stage1的脚本:./chess_train/LLaMA-Factory/qwen3-chess-sft-stage1.yaml
SFT-stage2的脚本:./chess_train/LLaMA-Factory/qwen3-chess-sft-stage2.yaml

** 数据 **
SFT的训练数据在./chess_train/LLaMA-Factory/data文件夹中
数据信息保存在./chess_train/LLaMA-Factory/data/dataset_info.json中
LLaMAFactory的具体使用方法，请参阅官方文档与Github（与本文作者无关）：https://llamafactory.readthedocs.io/en/latest/和https://github.com/hiyouga/LLaMA-Factory

### RL
** 数据处理 **
在进行RL训练之前，需要先进行数据处理；
数据处理脚本在./chess_train/verl/examples/data_preprocess/chess.py中
首先运行
```
cd ./chess_train/verl/examples/data_preprocess/chess.py
python chess.py
```
进行数据预处理


** 训练脚本 **
RL训练脚本保存在./chess_train/verl/run_qwen3-8b-chess.sh中
运行
```
cd ./chess_train/verl/run_qwen3-8b-chess.sh
```
进行RL训练。记得正确设置您的reference model(SFT model)路径；
如有需要，您可以设置其他的训练算法，如DAPO，GSPO等

** Reward **
我们的reward函数在./chess_train/verl/verl/utils/reward_score/chess.py中
您可以自行修改reward_score并进行训练

对于Verl框架的具体使用，您可以查阅Verl官方文档与Github（与本文作者无关）:https://verl.readthedocs.io/en/latest/start/install.html和https://github.com/volcengine/verl