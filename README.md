# ♟️ ChessArena: A Chess Testbed for Evaluating Strategic Reasoning Capabilities of Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2509.24239-b31b1b.svg)](https://arxiv.org/abs/2509.24239)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗-Dataset-yellow.svg)](https://huggingface.co/datasets/ljcnju/ChessArena_Training_Dataset)
[![Hugging Face Models](https://img.shields.io/badge/🤗-Models-yellow.svg)](https://huggingface.co/ljcnju/Qwen3-8B-Chess-SFT)
[![Hugging Face Models](https://img.shields.io/badge/🤗-Models-yellow.svg)](https://huggingface.co/ljcnju/Qwen3-8B-Chess)


> *Official repository for ChessArena: A Chess Testbed for Evaluating Strategic Reasoning Capabilities of Large Language Models*

## 📖 Overview

ChessArena is a comprehensive framework for evaluating and enhancing strategic reasoning capabilities in Large Language Models through chess. This repository contains four key components:

1. **♟️ Chess Competition Sampling** - Automated competition systems for faster convergence
2. **📊 Glicko-1 Ranking System** - Robust rating calculations  
3. **🎯 Fine-grained Evaluation Tasks** - Three specialized assessment dimensions: why model fails on chess?
4. **🏋️ Chess Training** - Full training pipeline for chess reasoning

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/NEU-CS/ChessArena.git
cd ChessArena
pip install -r requirements.txt
```

### Basic Usage

As described in our paper, our chess competition sampling is based on Glicko formula derivation: matching two players with similar skill levels achieves the fastest RD convergence.
#### Run a quick 4-game competition:
```bash
python competition_sampling.py --games 4
```

This randomly selects the first player and uses our competition sampling algorithm to select the second player.

#### Target specific model matchups:
```bash
python competition_sampling.py \
    --target \
    --player1_id "gpt-4.1" \
    --player1_name "gpt-4.1" \
    --player1_play_mode "blitz" \
    --player1_provide_legal_moves \
    --player1_api_key "<your api key>" \
    --player1_base_url "<your api url>" 
```

Where `player1_id` is the model ID in OpenAI API format; `player1_name` can be set to any name, but must be followed by the model's play mode and whether legal moves are provided. Ensure that `player1_name` exists in `./simulation_record/ratings.json`.

**Ensure** that player1_name exists in ./simulation_record/ratings.json. 

This gives the first player and uses our competition sampling algorithm to select the second player. Through these two methods, you can perform match-making and conduct games.

#### Manual competition Configuration
For competition matching algorithms, we add corresponding competition configs in the 
```
./config
``` 
directory and start games based on these configs. You can write your own desired model matchups based on these configs.

Use the following code to start you game:
```bash
python run_simulation.py \
    --config ./config/blitz_vs_blitz/qwen3-235b-a22b_blitz_vs_doubao-1.5-lite_blitz_legal.json \
    --games 4 \
    --parallel 2
```

#### Prompt
You can adjust the game prompts for each mode in utils.py (if you think it's necessary).

#### Glicko Rating System
Calculate comprehensive model ratings:
```bash
python elo_calculate.py
```
Obtain Glicko scores for all model competition results.

### 🎯 Fine-grained Evaluation Tasks
In ChessArena, we have three fine-grained evaluation tasks: basic understanding, move selection, and puzzle solving. You can use fine_grained_evaluation.py to start evaluation for these three tasks.

For example, basic understanding:
```bash
python fine_grained_evaluation.py \
    --task "basic_understanding" \
    --model_id "your model id" \
    --api_key "your api key" \
    --model_name "your model name" \
    --url "your api url"
```
You can also pass parameters like $temperature, top\_p, max\_tokens$ etc. to control the quality of LLM generation results.

In fine-grained_evaluation.py, there is also a board_reconstruction task, which is a board reconstruction task for blindfold mode. If you're interested, you can also conduct experiments with it.

#### Scripts
We have saved example scripts in 
```
./scripts
```
for your reference.

### Chess Training

All our training was completed on 8 H800 GPUs. SFT training takes about 4 hours; RL training takes about 60 hours.

**Training Dataset**
https://huggingface.co/datasets/ljcnju/ChessArena_Training_Dataset

**Our trained models**

SFT stage-2: https://huggingface.co/ljcnju/Qwen3-8B-Chess-SFT

RL: https://huggingface.co/ljcnju/Qwen3-8B-Chess

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

For specific usage of LLaMAFactory, please refer to the official documentation and GitHub: https://llamafactory.readthedocs.io/en/latest/ and https://github.com/hiyouga/LLaMA-Factory

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

For specific usage of the Verl framework, you can refer to the official Verl documentation and GitHub: https://verl.readthedocs.io/en/latest/start/install.html and https://github.com/volcengine/verl

# Citation
If you use ChessArena in your research, please cite our paper:
```
@article{liu2025chessarena,
  title={ChessArena: A Chess Testbed for Evaluating Strategic Reasoning Capabilities of Large Language Models},
  author={Liu, Jincheng and He, Sijun and Wu, Jingjing and Wang, Xiangsen and Chen, Yang and Kuang, Zhaoqi and Bao, Siqi and Yao, Yuan},
  journal={arXiv preprint arXiv:2509.24239},
  year={2025}
}
```
