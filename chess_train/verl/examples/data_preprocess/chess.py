# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re

import datasets
import random
from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/chess_blitz_grpo")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "chess_blitz_grpo"

    dataset = datasets.load_dataset("json", data_files={
        "train":"../../data/grpo_data_train.jsonl",
        "test": "../../data/grpo_data_test.jsonl"
    })

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            system = example.pop("system")
            fen = example.pop("FEN")
            prompt = example.pop("prompt")
            prompt_without_legal_moves = example.pop("prompt_without_legal_moves")
            provide_legal_moves = False
            if random.random() < 0.5:
                instruction = prompt
                provide_legal_moves = True
            else:
                instruction = prompt_without_legal_moves
            
            leagl_moves = example.pop("legal_moves")
            top_moves = example.pop("top_moves")
            data = {
                "data_source": data_source + f"_with_legal({provide_legal_moves})",
                "prompt": [
                    {
                        "role": "system",
                        "content": system,
                    }
                    ,
                    {
                        "role": "user",
                        "content": instruction + "What is the best move? Think it step by step.",
                    }
                ],
                "ability": "chess_reasoning",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "fen": fen,
                    "legal_moves": leagl_moves,
                    "top_moves": top_moves,
                    "provide_legal_moves": provide_legal_moves
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
