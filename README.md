# ChessArena

This is the submission code for ChessArena, containing four components: 1. chess competition sampling (match-making algorithm); 2. Glicko-1 rating system; 3. fine-grained evaluation; 4. chess training.

## Evaluation

### Chess Competition Sampling

As described in our paper, our chess competition sampling is based on Glicko formula derivation: matching two players with similar skill levels achieves the fastest RD convergence.

**Random Player Matching**
```bash
python competition_sampling.py --games 4
```

**Targeted Player Matching**
```bash
python competition_sampling.py \
    --target \
    --player1_id "gpt-4.1" \
    --player1_name "gpt-4.1_blitz_True"
```

Where `player1_id` is the model ID in OpenAI API format; `player1_name` can be set to any name, but must be followed by the model's play mode and whether legal moves are provided. Ensure that `player1_name` exists in `./simulation_record/ratings.json`.

Through these two methods, you can perform match-making and conduct games.

### Manual Config.json Setup

For competition matching algorithms, we add corresponding competition configs in the `./config` directory and start games based on these configs. We provide some config examples in the `./config` folder for your reference. You can write your own desired model matchups based on these configs.

Use the following code:
```bash
python run_simulation.py \
    --config ./config/blitz_vs_blitz/qwen3-235b-a22b_blitz_vs_doubao-1.5-lite_blitz_legal.json \
    --games 4 \
    --parallel 2
```
to start the competition.

### Prompt

You can adjust the game prompts for each mode in `utils.py` (if you think it's necessary).

### Glicko Rating System

You can use:
```bash
python elo_calculate.py
```
to obtain Glicko scores for all model competition results.

### Fine-grained Evaluation

In ChessArena, we have three fine-grained evaluation tasks: basic understanding, move selection, and puzzle solving. You can use `fine_grained_evaluation.py` to start evaluation for these three tasks.

For example, basic understanding:
```bash
python static.py \
    --task "basic_understanding" \
    --model_id "your model id" \
    --api_key "your api key" \
    --model_name "your model name" \
    --url "your api url"
```

You can also pass parameters like `temperature`, `top_p`, `max_tokens` etc. to control the quality of LLM generation results.

In `fine-grained_evaluation.py`, there is also a `board_reconstruction` task, which is a board reconstruction task for blindfold mode. If you're interested, you can also conduct experiments with it.

### Scripts

We have saved example scripts in `./scripts` for your reference.

## Chess Training

All our training was completed on 8 H800 GPUs. SFT training takes about 4 hours; RL training takes about 60 hours.

**Training Code**

We have saved all our training code in the `./chess_train` folder: SFT-stage1, SFT-stage2 (LLamaFactory), and RL (Verl) training.

### SFT

**Training Scripts**

All SFT training code is saved in the `./chess_train/LLaMA-Factory` folder.

- SFT-stage1 script: `./chess_train/LLaMA-Factory/qwen3-chess-sft-stage1.yaml`
- SFT-stage2 script: `./chess_train/LLaMA-Factory/qwen3-chess-sft-stage2.yaml`

**Data**

SFT training data is in the `./chess_train/LLaMA-Factory/data` folder.
Data information is saved in `./chess_train/LLaMA-Factory/data/dataset_info.json`.

For specific usage of LLaMAFactory, please refer to the official documentation and GitHub (unrelated to this paper's authors): https://llamafactory.readthedocs.io/en/latest/ and https://github.com/hiyouga/LLaMA-Factory

### RL

**Data Processing**

Before conducting RL training, data processing is required first.
The data processing script is in `./chess_train/verl/examples/data_preprocess/chess.py`.

First run:
```bash
cd ./chess_train/verl/examples/data_preprocess/
python chess.py
```
for data preprocessing.

**Training Scripts**

RL training scripts are saved in `./chess_train/verl/run_qwen3-8b-chess.sh`.

Run:
```bash
cd ./chess_train/verl/
bash run_qwen3-8b-chess.sh
```
for RL training. Remember to correctly set your reference model (SFT model) path.

If needed, you can set other training algorithms like DAPO, GSPO, etc.

**Reward**

Our reward function is in `./chess_train/verl/verl/utils/reward_score/chess.py`.
You can modify the reward_score yourself and conduct training.

For specific usage of the Verl framework, you can refer to the official Verl documentation and GitHub (unrelated to this paper's authors): https://verl.readthedocs.io/en/latest/start/install.html and https://github.com/volcengine/verl