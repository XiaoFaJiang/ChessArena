import json
import os
import random
import math
import subprocess
import time
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from openai import OpenAI
import openai
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Player:
    name: str
    rating: float
    rd: float  # Rating Deviation


@dataclass
class MatchConfig:
    white_player: Dict[str, Any]
    black_player: Dict[str, Any]
    max_moves: int
    max_retries: int
    stockfish_path: str


class ChessMatchSampling:
    """
    基于Glicko-1评分系统的国际象棋匹配采样器
    
    该系统根据Glicko-1算法计算棋手间的期望得分，
    并基于信息增强度进行智能匹配，以提高评分系统的准确性。
    """
    
    # 常量定义
    GLICKO_Q = math.log(10) / 400  # Glicko-1 q常数
    MIN_EXPECTED_SCORE = 0.001
    MAX_EXPECTED_SCORE = 0.999
    DEFAULT_MAX_RETRIES = 3
    API_RETRY_DELAY = 20  # 秒
    
    def __init__(
        self,
        args,
        ratings_file: str = "./simulation_record/ratings.json",
        provide_move_history: bool = True,
        max_moves: int = 200,
        max_retries: int = 5,
        stockfish_path: str = "./stockfish-8-linux/Linux/stockfish_8_x64",
        api_key: str = os.environ["OPENAI_API_KEY"], #your sk-key
        base_url: str = os.environ["OPENAI_BASE_URL"], #your api address
    ):
        """
        初始化匹配采样器
        
        Args:
            ratings_file: 评分文件路径
            provide_move_history: 是否提供走法历史
            max_moves: 最大走法数
            max_retries: 最大重试次数
            stockfish_path: Stockfish引擎路径
            api_key: API密钥
            base_url: API基础URL
        """
        self.players: Dict[str, Player] = {}
        self.provide_move_history = provide_move_history
        self.max_moves = max_moves
        self.max_retries = max_retries
        self.stockfish_path = stockfish_path
        self.args = args
        self.player1_url = self.args.player1_base_url #base_url of player1
        self.player1_api_key = self.args.player1_api_key #api_key of player1
        self.player2_url = base_url #base_url of player2
        self.player2_api_key = api_key #api_key of player2
        # 模型映射表
        self.model_map = {
            'doubao-1.5-lite': 'doubao-lite-1.5-32k',
            'maia-1100': 'lc0',
            #create your model_id map for your api call here
            #key: model id in our chessArena competitions, value: model id in your api call
            #you can see a model name in ./simulation_record/ratings.json
            #for example: qwen3-235b-a22b_blitz_True in ratings.json
            #the model id in our chessArena competitions is qwen3-235b-a22b
        }
        
        self.type_map = {
            'blindfold': 'blindfold_multiTurn'
        }
        self.thinking_models = {'doubao-seed-1-6-thinking-250615','doubao-1-5-thinking-pro-250415','deepseek-r1',"gemini-2.5-pro-preview-06-05","gemini-2.5-pro","O3"}
        # 加载评分数据
        self._load_ratings(ratings_file)
        
        logger.info(f"成功初始化匹配采样器，加载了 {len(self.players)} 个棋手")
    
    def _load_ratings(self, filename: str) -> None:
        """
        从JSON文件加载棋手评分数据
        
        Args:
            filename: 评分文件路径
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: JSON格式无效
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"评分文件不存在: {filename}")
        
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"无效的JSON格式: {e}")
        except Exception as e:
            raise ValueError(f"读取文件时发生错误: {e}")
        
        if 'players' not in data:
            raise ValueError("JSON文件中缺少'players'字段")
        
        self.players = {}
        for player_name, info in data['players'].items():
            if 'rating' not in info or 'RD' not in info:
                logger.warning(f"棋手 {player_name} 缺少必要的评分信息，跳过")
                continue
                
            self.players[player_name] = Player(
                name=player_name,
                rating=float(info['rating']),
                rd=float(info['RD'])
            )
        
        logger.info(f"成功加载 {len(self.players)} 个棋手的评分数据")
    
    def _calculate_g_rd(self, rd: float) -> float:
        """
        计算Glicko系统的g(RD)函数
        
        Args:
            rd: 评分偏差
            
        Returns:
            g(RD)值
        """
        return 1 / math.sqrt(1 + (3 * self.GLICKO_Q**2 * rd**2) / (math.pi**2))
    
    def calculate_expected_score(self, player_a: str, player_b: str) -> float:
        """
        计算棋手A对棋手B的期望得分
        
        Args:
            player_a: 棋手A名称
            player_b: 棋手B名称
            
        Returns:
            期望得分 (0-1之间)
            
        Raises:
            KeyError: 棋手不存在
        """
        if player_a not in self.players or player_b not in self.players:
            raise KeyError(f"棋手不存在: {player_a} 或 {player_b}")
        
        rating_a = self.players[player_a].rating
        rating_b = self.players[player_b].rating
        rd_b = self.players[player_b].rd
        
        g_b = self._calculate_g_rd(rd_b)
        expected_score = 1 / (1 + 10 ** (-g_b * (rating_a - rating_b) / 400))
        
        # 防止数值不稳定
        expected_score = max(min(expected_score, self.MAX_EXPECTED_SCORE), self.MIN_EXPECTED_SCORE)
        
        return expected_score
    
    def _calculate_d_squared(self, player1: str, player2: str) -> float:
        """
        计算两棋手间的d²值，用于衡量信息增强度
        
        Args:
            player1: 棋手1名称
            player2: 棋手2名称
            
        Returns:
            d²值
        """
        expected_score = self.calculate_expected_score(player1, player2)
        g_b = self._calculate_g_rd(self.players[player2].rd)
        d_squared = 1 / (g_b**2 * expected_score * (1 - expected_score))
        return d_squared
    
    def calculate_information_enhancement(self, player: str) -> List[Tuple[str, float]]:
        """
        计算指定棋手与其他所有棋手的信息增强度
        
        Args:
            player: 棋手名称
            
        Returns:
            按d²值从小到大排序的[(棋手名称, d²值)]列表
        """
        if player not in self.players:
            raise KeyError(f"棋手不存在: {player}")
        
        info_enhancement = {}
        for other_player in self.players:
            if player != other_player:
                info_enhancement[other_player] = self._calculate_d_squared(player, other_player)
        
        return sorted(info_enhancement.items(), key=lambda x: x[1])
    
    def _test_model_availability(self, url: str, model: str, api_key: str) -> bool:
        """
        测试模型是否可用
        
        Args:
            url: API URL
            model: 模型名称
            api_key: API密钥
            
        Returns:
            是否可用
        """
        if "maia" in model or 'random' in model:
            return True
        
        def _call_openai_api(model: str, url: str, prompt: str, api_key: str) -> bool:
            """调用OpenAI API"""
            client = OpenAI(api_key=api_key, base_url=url)
            
            for retry in range(self.DEFAULT_MAX_RETRIES):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    content = response.choices[0].message.content
                    logger.info(f"模型 {model} 测试成功: {content[:50]}...")
                    return True
                    
                except openai.RateLimitError:
                    logger.warning(f"API限流，等待 {self.API_RETRY_DELAY} 秒后重试...")
                    time.sleep(self.API_RETRY_DELAY)
                    
                except Exception as e:
                    logger.error(f"API调用失败 (重试 {retry + 1}/{self.DEFAULT_MAX_RETRIES}): {e}")
                    if retry == self.DEFAULT_MAX_RETRIES - 1:
                        return False
            
            return False
        
        return _call_openai_api(model, url, "hello", api_key)
    
    def _create_player_config(self, player_name: str, player_id: str, url: str, api_key: str) -> Tuple[Dict[str, Any], str]:
        """
        创建棋手配置
        
        Args:
            player_name: 棋手名称
            
        Returns:
            (配置字典, API URL)
        """
        parts = player_name.split("_")
        player_type = parts[-2]
        provide_legal_moves = True if parts[-1].lower() == "true" else False
        model_id = self.model_map.get(player_id, player_id)
        
        config = {
            "name": player_name,
            "api_key": api_key,
            "base_url": url,
            "model": model_id,
            "max_tokens": 4096 if player_id not in self.thinking_models else 16384,
            "play_mode": self.type_map.get(player_type,player_type),
            "provide_legal_moves": provide_legal_moves,
            "provide_move_history": self.provide_move_history
        }
        
        return config
    
    def _execute_match(self, player1_id: str, player1_name: str, player2_id: str, player2_name:str, show_simulation_output: bool = True, save_log: bool = False) -> bool:
        """
        执行棋局匹配
        
        Args:
            player1: 棋手1名称
            player2: 棋手2名称
            show_simulation_output: 是否显示仿真输出日志
            save_log: 是否保存仿真日志到文件
            
        Returns:
            是否成功执行
        """
        logger.info(f"开始匹配: {player1_name} vs {player2_name}")
        
        # 解析棋手类型
        player1_type = player1_name.split("_")[-2]
        player2_type = player2_name.split("_")[-2]
        
        # 创建配置目录
        dir_path = f"./config/{player1_type}_vs_{player2_type}"
        os.makedirs(dir_path, exist_ok=True)
        
        # 创建棋手配置
        try:
            white_config = self._create_player_config(player1_name,player1_id,self.player1_url,self.player1_api_key)
            black_config = self._create_player_config(player2_name,player2_id,self.player2_url,self.player2_api_key)
            
            # 测试模型可用性
            if not self._test_model_availability(white_url, white_config["model"], self.api_key):
                raise ValueError(f"白方模型不可用: {white_config['model']}")
                
            if not self._test_model_availability(black_url, black_config["model"], self.api_key):
                raise ValueError(f"黑方模型不可用: {black_config['model']}")
            
        except Exception as e:
            logger.error(f"配置创建失败: {e}")
            return False
        
        # 创建匹配配置
        match_config = MatchConfig(
            white_player=white_config,
            black_player=black_config,
            max_moves=self.max_moves,
            max_retries=self.max_retries,
            stockfish_path=self.stockfish_path
        )
        
        # 保存配置文件
        config_filename = f"{player1_name}_vs_{player2_name}.json"
        config_path = os.path.join(dir_path, config_filename)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(match_config.__dict__, f, indent=4, ensure_ascii=False)
            
            logger.info(f"配置文件已保存: {config_path}")
            
        except Exception as e:
            logger.error(f"配置文件保存失败: {e}")
            return False
        
        # 执行匹配
        try:
            logger.info(f"开始执行匹配仿真: {player1_name} vs {player2_name}")
            logger.info(f"配置文件: {config_path}")
            logger.info("-" * 60)
            
            # 使用 Popen 实时获取输出
            process = subprocess.Popen(
                ['python', '-u', 'run_simulation.py', '--config', config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 实时读取并打印输出
            output_lines = []
            log_file_path = None
            
            if save_log:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                log_file_path = os.path.join(dir_path, f"match_log_{timestamp}.txt")
                
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    clean_output = output.strip()
                    output_lines.append(clean_output)
                    
                    if show_simulation_output:
                        # 使用特殊的前缀来标识来自 run_simulation.py 的输出
                        logger.info(f"[SIMULATION] {clean_output}")
                    
                    # 保存到文件
                    if save_log and log_file_path:
                        with open(log_file_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {clean_output}\n")
                            log_file.flush()  # 确保实时写入
            
            # 等待进程结束
            return_code = process.wait(timeout=3600)  # 1小时超时
            
            if return_code == 0:
                logger.info("-" * 60)
                logger.info(f"匹配成功完成: {player1_name} vs {player2_name}")
                logger.info(f"总共输出了 {len(output_lines)} 行日志")
                return True
            else:
                logger.error(f"匹配执行失败，返回码: {return_code}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"匹配超时: {player1_name} vs {player2_name}")
            if 'process' in locals():
                process.kill()
            return False
            
        except Exception as e:
            logger.error(f"匹配执行异常: {e}")
            return False
    
    def random_sampling(self) -> bool:
        """
        随机采样匹配
        
        随机选择一个棋手，然后从其信息增强度最高的前3个对手中随机选择一个进行匹配
        
        Returns:
            是否成功执行匹配
        """
        if len(self.players) < 2:
            logger.error("棋手数量不足，无法进行匹配")
            return False
        
        # 随机选择第一个棋手
        player1_name = random.choice(list(self.players.keys()))
        player1_id = "_".join(player1_name.split("_")[:-2])
        logger.info(f"随机选择棋手: {player1_name}")
        
        info_enhancement = self.calculate_information_enhancement(player1_name)
        
        if len(info_enhancement) < 1:
            logger.error(f"棋手 {player1_name} 没有可匹配的对手")
            return False
        
        # 从前3个候选对手中随机选择
        top_candidates = info_enhancement[:min(3, len(info_enhancement))]
        player2_name = random.choice(top_candidates)[0]
        player2_id = "_".join(player2_name.split("_")[:-2])
        logger.info(f"选择对手: {player2} (信息增强度: {info_enhancement[0][1]:.4f})")
        
        # 执行匹配直到成功
        max_attempts = 5
        for attempt in range(max_attempts):
            if self._execute_match(player1_id,player1_name, player2_id,player2_name, show_simulation_output=True):
                return True
            
            logger.warning(f"匹配失败，尝试 {attempt + 1}/{max_attempts}")
            
            # 如果还有其他候选对手，换一个试试
            if len(top_candidates) > 1:
                top_candidates = [c for c in top_candidates if c[0] != player2]
                if top_candidates:
                    player2 = random.choice(top_candidates)[0]
                    logger.info(f"更换对手: {player2}")
        
        logger.error("所有匹配尝试都失败了")
        return False
    
    def targeted_sampling(self, player_id: str, player_name: str) -> bool:
        """
        定向采样匹配
        
        Args:
            player: 指定的棋手名称
            
        Returns:
            是否成功执行匹配
        """
        if player_name not in self.players:
            logger.error(f"指定的棋手不存在: {player_name}")
            return False
        
        logger.info(f"定向匹配棋手: {player_name}")
        
        # 获取信息增强度排序
        info_enhancement = self.calculate_information_enhancement(player_name)
        
        if len(info_enhancement) < 1:
            logger.error(f"棋手 {player_name} 没有可匹配的对手")
            return False
        
        # 从前3个候选对手中随机选择
        top_candidates = info_enhancement[:min(3, len(info_enhancement))]
        player2_name = random.choice(top_candidates)[0]
        player2_id = "_".join(player2_name.split("_")[:-2])
        logger.info(f"选择对手: {player2_name} (信息增强度: {info_enhancement[0][1]:.4f})")
        
        # 执行匹配直到成功
        max_attempts = 5
        for attempt in range(max_attempts):
            if self._execute_match(player_id, player_name, player2_id, player2_name, show_simulation_output=True):
                return True
            
            logger.warning(f"匹配失败，尝试 {attempt + 1}/{max_attempts}")
        
        logger.error("所有匹配尝试都失败了")
        return False
    
    def get_player_stats(self) -> Dict[str, Any]:
        """
        获取棋手统计信息
        
        Returns:
            统计信息字典
        """
        if not self.players:
            return {}
        
        ratings = [p.rating for p in self.players.values()]
        rds = [p.rd for p in self.players.values()]
        
        stats = {
            "total_players": len(self.players),
            "rating_stats": {
                "min": min(ratings),
                "max": max(ratings),
                "mean": sum(ratings) / len(ratings),
                "median": sorted(ratings)[len(ratings) // 2]
            },
            "rd_stats": {
                "min": min(rds),
                "max": max(rds),
                "mean": sum(rds) / len(rds),
                "median": sorted(rds)[len(rds) // 2]
            }
        }
        
        return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Run chess simulations between LLMs")
    parser.add_argument("--target", default=False,action="store_true", help='if target sampling')
    parser.add_argument("--player1_id", type=str, default="gpt-4.1", help="first player model id(for openai client call)")
    parser.add_argument("--player1_name", type=str, default="gpt-4.1", help="first player model name(for our chessArena ratings record),\
        need follow play mode and whether provide legal moves; player1_name must in our ratings.json")
    parser.add_argument("--player1_play_mode",type=str,default="blitz")
    parser.add_argument("--player1_provide_legal_moves",action="store_true")
    parser.add_argument("--player1_api_key",type="str",default="dummy")
    parser.add_argument("--player1_base_url",type="str",default="http://0.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--games", type=int, default=2, help="Number of games to run") 
    args = parser.parse_args()
    args.player1_name = f"{args.player1_name}_{args.player1_play_mode}_{str(args.player1_provide_legal_moves)}"
    for _ in range(args.games//2):
        try:
            # 初始化匹配采样器
            sampler = ChessMatchSampling(args)
            # 显示统计信息
            stats = sampler.get_player_stats()
            logger.info(f"系统统计: {stats}")
                
            # 执行随机采样匹配
            if args.target:
                success = sampler.targeted_sampling(args.player1_id,args.player1_name)
            else:
                success = sampler.random_sampling()
            if success:
                logger.info("匹配执行成功")
            else:
                logger.error("匹配执行失败")
                
        except Exception as e:
            logger.error(f"程序执行出错: {e}")
            raise


if __name__ == "__main__":
    main()