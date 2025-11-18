import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import os
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import math
from math import sqrt
from typing import Any
from collections import Counter

PLAY_MODE_MAP = {"blitz":"blitz","standard":"standard","bullet":"bullet","blindfold_multiTurn":"blindfold","blindfold":"blindfold"}
class EloCalculator:
    def __init__(self, k_factor: int = 32, initial_rating: int = 1500, initial_RD: int = 350, min_RD: int = 50):
        """
        Initialize Glicko calculator
        
        Args:
            k_factor: K-factor, determines the magnitude of rating changes
            initial_rating: Initial rating for new players
            initial_RD: Initial rating deviation for new players
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.initial_RD = initial_RD
        self.min_RD = min_RD
        self.players: Dict[str, Dict[str, Any]] = {}  # Store player information
        self.games_history = []  # Store all game history
        self.q = math.log(10)/400  # Glicko-1 q constant
        self.termination_stats = Counter()
        
    def g_RD(self, RD: float) -> float:
        """Calculate g(RD) function for Glicko system"""
        return 1 / sqrt(1 + (3 * self.q**2 * RD**2) / (math.pi**2))
    
    def expected_score(self, player_a: str, player_b: str) -> float:
        """
        Expected score for player A against player B
        """
        rating_a = self.players[player_a]['rating']
        rating_b = self.players[player_b]['rating']
        RD_b = self.players[player_b]['RD']
        g_b = self.g_RD(RD_b)
        E = 1 / (1 + 10 ** (-g_b * (rating_a - rating_b) / 400)) #防止数值不稳定
        if E < 0.001 or E > 0.999:
            E = max(min(E, 0.999), 0.001)
        return E
    
    def update_rating(self, player_a: str, player_b: str, result: float, game_id) -> Tuple[float, float]:
        """
        Update Glicko ratings for two players after a match
        
        Args:
            player_a: Player A name
            player_b: Player B name
            result: Match result (1 = A wins, 0 = A loses, 0.5 = draw)
            
        Returns:
            New ratings for players A and B
        """
        # Initialize new players if needed
        if player_a not in self.players:
            self.players[player_a] = {
                'rating': self.initial_rating,
                'RD': self.initial_RD,
                'games': 0,
                'history': [],
            }
        
        if player_b not in self.players:
            self.players[player_b] = {
                'rating': self.initial_rating,
                'RD': self.initial_RD,
                'games': 0,
                'history': [],
            }
        
        # Save pre-match ratings
        rating_a_old = self.players[player_a]['rating']
        RD_a_old = self.players[player_a]['RD']
        rating_b_old = self.players[player_b]['rating']
        RD_b_old = self.players[player_b]['RD']
        
        # Calculate expected scores
        E_a = self.expected_score(player_a, player_b)
        E_b = self.expected_score(player_b, player_a)
        
        # Calculate d² for both players
        g_b = self.g_RD(RD_b_old)
        d_sq_a = 1 / (self.q**2 * g_b**2 * E_a * (1 - E_a))
        
        g_a = self.g_RD(RD_a_old)
        d_sq_b = 1 / (self.q**2 * g_a**2 * E_b * (1 - E_b))
        
        # Update ratings and RDs
        # For Player A
        k_a = self.k_factor  # Could add streaming factor here if needed
        new_rating_a = rating_a_old + (self.q / ((1 / RD_a_old**2) + (1 / d_sq_a))) * g_b * (result - E_a)
        new_RD_a = sqrt(1 / ((1 / RD_a_old**2) + (1 / d_sq_a)))
        if new_RD_a < self.min_RD:
            new_RD_a = self.min_RD
                    
        # For Player B
        k_b = self.k_factor  # Could add streaming factor here if needed
        new_rating_b = rating_b_old + (self.q / ((1 / RD_b_old**2) + (1 / d_sq_b))) * g_a * ((1 - result) - E_b)
        new_RD_b = sqrt(1 / ((1 / RD_b_old**2) + (1 / d_sq_b)))
        if new_RD_b < self.min_RD:
            new_RD_b = self.min_RD
        
        # Apply updates
            self.players[player_a]['rating'] = new_rating_a
        
        self.players[player_a]['RD'] = new_RD_a
        self.players[player_a]['games'] += 1
        self.players[player_a]['history'].append({
            'rating': new_rating_a,
            'RD': new_RD_a,
            'opponent': player_b,
            'result': result,
            'timestamp': game_id
        })
        
        self.players[player_b]['rating'] = new_rating_b
            
        self.players[player_b]['RD'] = new_RD_b
        self.players[player_b]['games'] += 1
        self.players[player_b]['history'].append({
            'rating': new_rating_b,
            'RD': new_RD_b,
            'opponent': player_a,
            'result': 1 - result,
            'timestamp': game_id
        })
        
        # Record game history
        game_record = {
            'date': game_id,
            'player_a': player_a,
            'player_b': player_b,
            'result': result,
            'rating_a_before': rating_a_old,
            'rating_b_before': rating_b_old,
            'RD_a_before': RD_a_old,
            'RD_b_before': RD_b_old,
            'rating_a_after': new_rating_a,
            'rating_b_after': new_rating_b,
            'RD_a_after': new_RD_a,
            'RD_b_after': new_RD_b
        }
        
        self.games_history.append(game_record)
        
        return new_rating_a, new_rating_b
    
    
    
    def _update_single_player_attempt_stats(self, player, attempt_stats,win):
        """
        Update attempt statistics for a single player
        """
        if player not in self.players:
            self.players[player] = {}
        
        
        #total_attempts = attempt_stats['total_attempts']
        parsing_errors = attempt_stats['parsing_errors']
        illegal_moves = attempt_stats['illegal_moves']
        forbidden_thinking = attempt_stats.get('forbidden_thinking', 0)
        successful_moves = attempt_stats['successful_moves']
        total_attempts = parsing_errors + illegal_moves + forbidden_thinking + successful_moves
        #assert parsing_errors + illegal_moves + forbidden_thinking + successful_moves == total_attempts
        total_moves = attempt_stats['total_moves']
        optimal_moves = attempt_stats['optimal_moves']
        if 'attempt_stats' in self.players[player]:
            # Accumulate statistics
            stats = self.players[player]['attempt_stats']
            stats['total_attempts'] += total_attempts
            stats['parsing_errors'] += parsing_errors
            stats['illegal_moves'] += illegal_moves
            stats['forbidden_thinking'] += forbidden_thinking
            stats['successful_moves'] += successful_moves
            stats['total_moves'] += total_moves
            stats['optimal_moves'] += optimal_moves
            if win == 1:
                stats["win_num"] += 1
                if stats["win_avg_move"] == 0:
                    stats["win_avg_move"] = total_moves
                else:
                    stats["win_avg_move"] = (stats["win_avg_move"] * (stats["win_num"] - 1) + total_moves ) / stats["win_num"]
            if win == 0:
                stats["lose_num"] += 1
                if stats["lose_avg_move"] == 0:
                    stats["lose_avg_move"] = total_moves
                else:
                    stats["lose_avg_move"] = (stats["lose_avg_move"] * (stats["lose_num"] - 1) + total_moves ) / stats["lose_num"]
            #parsing error: 指模型输出move不符合format的次数，除以总的attempt次数
            #illegal move: 指模型输出move不符合legal moves的
            if win == 0.5:
                stats["draw_num"] += 1
                if stats["draw_avg_move"] == 0:
                    stats["draw_avg_move"] = total_moves
                else:
                    stats["draw_avg_move"] = (stats["draw_avg_move"] * (stats["draw_num"] - 1) + total_moves ) / stats["draw_num"]
            
            # Recalculate percentages
            if stats['total_attempts'] > 0:
                stats['parsing_errors_pct'] = stats['parsing_errors'] / stats['total_attempts'] * 100
                stats['forbidden_thinking_pct'] = stats['forbidden_thinking'] / stats['total_attempts'] * 100  
                stats['successful_moves_pct'] = stats['successful_moves'] / stats['total_attempts'] * 100
                stats['illegal_moves_pct'] = stats['illegal_moves'] / stats['total_attempts'] * 100
                
            if stats['total_moves'] > 0:
                
                stats['optimal_moves_pct'] = stats['optimal_moves'] / stats['total_moves'] * 100
        else:
            # Initialize statistics
            parsing_errors_pct = parsing_errors / total_attempts * 100 if total_attempts > 0 else 0
            forbidden_thinking_pct = forbidden_thinking / total_attempts * 100 if total_attempts > 0 else 0
            successful_moves_pct = successful_moves / total_attempts * 100 if total_attempts > 0 else 0
            illegal_moves_pct = illegal_moves / total_attempts * 100 if total_attempts > 0 else 0
            optimal_moves_pct = optimal_moves / total_moves * 100 if total_moves > 0 else 0
            
            self.players[player]['attempt_stats'] = {
                "total_attempts": total_attempts,
                "parsing_errors": parsing_errors,
                "illegal_moves": illegal_moves,
                "forbidden_thinking": forbidden_thinking,
                "successful_moves": successful_moves,
                "total_moves": total_moves,
                "optimal_moves": optimal_moves,
                "parsing_errors_pct": parsing_errors_pct,
                "illegal_moves_pct": illegal_moves_pct,
                "forbidden_thinking_pct": forbidden_thinking_pct,
                "successful_moves_pct": successful_moves_pct,
                "optimal_moves_pct": optimal_moves_pct,
                "win_num":1 if win == 1 else 0,
                "win_avg_move":total_moves if win == 1 else 0,
                "lose_num":1 if not win else 0,
                "lose_avg_move":total_moves if win == 0 else 0,
                "draw_num":1 if win == 0.5 else 0,
                "draw_avg_move":total_moves if win == 0.5 else 0
            }
        
    
    def update_player_attempt_stats(self, game_json):
        """
        Update player attempt statistics from file
        """
        try:
            player1 = game_json['white_player']
            if "random" in player1:
                player1 = "random_player_blitz_True"
            
            player1_lst = player1.split("_")
            player1_name = '_'.join(player1_lst[:-min(1,len(player1_lst))])
            player1_prefix = "_".join(player1_lst[:-min(2,len(player1_lst))])
            player1_attempt_stats = None
            # Find statistics for player1
            if game_json['player_attempt_stats'].get(player1, None):
                player1_attempt_stats = game_json['player_attempt_stats'][player1]
            elif game_json['player_attempt_stats'].get(player1_name, None):
                player1_attempt_stats = game_json['player_attempt_stats'][player1_name]
            elif game_json['player_attempt_stats'].get(player1_prefix, None):
                player1_attempt_stats = game_json['player_attempt_stats'][player1_prefix]
            if not (player1.endswith("False") or player1.endswith("True")):
                player1 = f"{player1}_{game_json.get('white_player_provide_legal_moves',True)}"
            win = (eval(game_json["result"]) + 1) / 2
            if player1_attempt_stats:
                self._update_single_player_attempt_stats(player1, player1_attempt_stats,win)
                
            # Find statistics for player2
            player2 = game_json['black_player']
            if "random" in player2:
                player2 = "random_player_blitz_True"
            player2_lst = player2.split("_")
            player2_name = '_'.join(player2_lst[:-min(1,len(player2_lst))])
            player2_prefix = "_".join(player2_lst[:-min(2,len(player2_lst))])
            player2_attempt_stats = None
                
            if game_json['player_attempt_stats'].get(player2, None):
                player2_attempt_stats = game_json['player_attempt_stats'][player2]
            elif game_json['player_attempt_stats'].get(player2_name, None):
                player2_attempt_stats = game_json['player_attempt_stats'][player2_name]
            elif game_json['player_attempt_stats'].get(player2_prefix, None):
                player2_attempt_stats = game_json['player_attempt_stats'][player2_prefix]
            if not (player2.endswith("False") or player2.endswith("True")):
                player2 = f"{player2}_{game_json.get('black_player_provide_legal_moves',True)}"
            if player2_attempt_stats:
                self._update_single_player_attempt_stats(player2, player2_attempt_stats,1-win)
                
        except Exception as e:
            print(f"Error updating attempt stats: {e}")
            print(game_json["game_id"],game_json["white_player"],game_json["black_player"])
        
    def load_game_result(self, game_json: dict):
        """
        Load game result from JSON file
        """
        try:
            result = game_json['result']
            result = (eval(result) + 1) / 2
            
            player1 = game_json['white_player']
            player2 = game_json['black_player']
            game_id = game_json["game_id"]
            termination = game_json["termination"]
            self.termination_stats[termination] += 1
            
            if "random" in player1.lower():
                player1 = "random_player_blitz"
            if "random" in player2.lower():
                player2 = "random_player_blitz"
            player1_with_legal_moves = game_json.get("white_player_provide_legal_moves",True)
            player2_with_legal_moves = game_json.get("black_player_provide_legal_moves",True)
            if not (player1.endswith("False") or player1.endswith("True")):
                player1 = f'{player1}_{player1_with_legal_moves}'
            if not (player2.endswith("False") or player2.endswith("True")):
                player2 = f'{player2}_{player2_with_legal_moves}'
            self.update_rating(player1, player2, result, game_id)
            #print(result,termination)
            if result == 1:
                if "win_termination" not in self.players[player1]:
                    self.players[player1]["win_termination"] = Counter()
                self.players[player1]["win_termination"][termination] += 1
                
                if "loss_termination" not in self.players[player2]:
                    self.players[player2]["loss_termination"] = Counter()
                self.players[player2]["loss_termination"][termination] += 1
            elif result == 0:
                if "win_termination" not in self.players[player2]:
                    self.players[player2]["win_termination"] = Counter()
                self.players[player2]["win_termination"][termination] += 1
                
                if "loss_termination" not in self.players[player1]:
                    self.players[player1]["loss_termination"] = Counter()
                self.players[player1]["loss_termination"][termination] += 1
                    
            elif result == 0.5:
                if "draw_termination" not in self.players[player2]:
                    self.players[player2]["draw_termination"] = Counter()
                self.players[player2]["draw_termination"][termination] += 1
                    
                if "draw_termination" not in self.players[player1]:
                    self.players[player1]["draw_termination"] = Counter()
                self.players[player1]["draw_termination"][termination] += 1
                    
                    
        except FileNotFoundError:
            print(f"File {game_json} not found")
        except Exception as e:
            print(f"Error updating Elo rating: {e}")
    
    def load_all_games(self, folder: str):
        """
        Load all game results from specified directory
        """
        game_type = ['blitz', 'bullet', 'standard', 'blindfold']
        all_game_json = []
        if not os.path.exists(folder):
            raise Exception(f"Directory {folder} not found")
            return
            
        for onedir in os.listdir(folder):
            dir_path = os.path.join(folder, onedir)
            if not os.path.isdir(dir_path):
                continue
                
            for onetype in game_type:
                if onedir.startswith(onetype) or onedir.endswith(onetype):
                    for simulation in os.listdir(dir_path):
                        sim_path = os.path.join(dir_path, simulation)
                        if not os.path.isdir(sim_path):
                            continue
                            
                        for x in os.listdir(sim_path):
                            x_path = os.path.join(sim_path, x)
                            if not os.path.isdir(x_path):
                                continue
                                
                            files = os.listdir(x_path)
                            
                            for onefile in files:
                                inner_path = os.path.join(x_path,onefile)
                                if not os.path.isdir(inner_path):
                                    continue
                                for game_json_path in os.listdir(inner_path):
                                    if game_json_path.endswith('.json'):
                                        file_path = os.path.join(inner_path, game_json_path)
                                        with open(file_path,'r') as f:
                                            all_game_json.append(json.loads(f.read()))
                    break
        all_game_json.sort(key = lambda x:x['game_id'])
        for one_json in all_game_json:
            player1 = one_json['white_player']
            player2 = one_json['black_player']
            if player1 != player2:
                self.load_game_result(one_json)
                self.update_player_attempt_stats(one_json)
    
    def add_game(self, player_a: str, player_b: str, result: float):
        """
        Add a new game
        """
        old_rating_a = self.players.get(player_a, {}).get('rating', self.initial_rating)
        old_rating_b = self.players.get(player_b, {}).get('rating', self.initial_rating)
        
        new_rating_a, new_rating_b = self.update_rating(player_a, player_b, result)
        
        print(f"Game update finished:")
        print(f"{player_a}: {old_rating_a:.1f} -> {new_rating_a:.1f} ({new_rating_a-old_rating_a:+.1f})")
        print(f"{player_b}: {old_rating_b:.1f} -> {new_rating_b:.1f} ({new_rating_b-old_rating_b:+.1f})")
    
    def get_rankings(self) -> List[Tuple[str, float, int]]:
        """
        Get rankings list
        
        Returns:
            [(name, rating, games), ...]
        """
        
        rankings = [(name, info.get('rating',1500), info.get('RD',350), info.get('games',0)) 
                   for name, info in self.players.items()]
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def get_attempt_stats(self):
        """
        Get attempt statistics
        """
        attempt_stats = [(name, info['attempt_stats'], info.get('rating',1500)) 
                        for name, info in self.players.items() 
                        if info.get('attempt_stats', None)]
        return sorted(attempt_stats, key=lambda x: x[2], reverse=True)
    
    def save_ratings(self, filename: str):
        """
        Save current ratings to JSON file
        """
        data = {
            'initial_rating': self.initial_rating,
            'initial_RD': self.initial_RD,
            'min_RD': self.min_RD,
            'players': self.players,
            'games_history': self.games_history,
        }
        
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        print(f"Elo ratings saved to {filename}")
    
    def print_rankings(self):
        """
        Print rankings table
        """
        rankings = self.get_rankings()
        print(f"\n{'='*110}")
        print(f"{'Rank':<4} {'Model':<50} {'Type':<10} {'Legal Moves':<8} {'Rating':<8} {'RD':<6} {'Interval':<12} {'Games':<6}")
        print(f"{'='*110}")
        index = 1
        for i, (name, rating, RD, games) in enumerate(rankings, 1):
            if RD > 300: #RD高于80的将不会被展示
                continue
            name_lst = name.split("_")
            model_name = "_".join(name_lst[:-2])
            with_legal_moves = name_lst[-1]
            oneType = name_lst[-2]
            interval_left = int(rating - 1.96 * RD)
            interval_right = int(rating + 1.96 * RD)
            print(f"{index:<4} {model_name:<50} {oneType:<10} {with_legal_moves:<8} {rating:<8.1f} {int(RD):<6} ({interval_left:<4}, {interval_right:<4}) {games:<6}")
            index += 1
        print(f"{'='*110}")
    
    def print_player_terminations(self):
        """
        Print player terminations and save to CSV file
        """
        # Define all termination types
        termination_types = ['checkmate', 'forfeit', 'stalemate', 'move_limit', 'insufficient_material','fivefold_repetition']
        
        # Collect termination data for all players
        termination_data = []
        player_terminated = [(name,info.get('win_termination',{}),info.get('loss_termination',{}),info.get('draw_termination',{}), info.get('rating',1500)) 
                        for name, info in self.players.items()]
        player_terminated = sorted(player_terminated,key = lambda x:x[-1],reverse=True)
        
        for player, win_terminations, loss_terminations, draw_terminations,_ in player_terminated:            
            # Prepare row data for CSV
            player_data = {'player': player}
            
            # Add win, loss and draw terminations for each type
            for term_type in termination_types:
                player_data[f'win_{term_type}'] = win_terminations.get(term_type, 0)
                player_data[f'loss_{term_type}'] = loss_terminations.get(term_type, 0)
                player_data[f'draw_{term_type}'] = draw_terminations.get(term_type, 0)
            
            # Calculate totals
            player_data['total_wins'] = sum(win_terminations.values())
            player_data['total_losses'] = sum(loss_terminations.values())
            player_data['total_draws'] = sum(draw_terminations.values())
            
            termination_data.append(player_data)
        
        # Create DataFrame and save to CSV
        if termination_data:
            df = pd.DataFrame(termination_data)
            
            # Reorder columns to have player first, then totals, then terminations
            columns_order = ['player', 'total_wins', 'total_losses', 'total_draws']
            for term_type in termination_types:
                columns_order.extend([f'win_{term_type}', f'loss_{term_type}', f'draw_{term_type}'])
            
            df = df[columns_order]
            
            # Save to CSV
            csv_filename = 'player_terminations.csv'
            df.to_csv(csv_filename, index=False)
            print(f"Termination statistics saved to {csv_filename}")
            print(f"DataFrame shape: {df.shape}")
            print("\nPreview of the data:")
            print(df.head())
        else:
            print("No termination data available to save.")
        


    def add_player(self,player_name,type):
        if f"{player_name}_{type}" not in self.players:
            self.players[f"{player_name}_{type}"] = {'rating': self.initial_rating, 'RD': self.initial_RD, 'games': 0}
    
    def print_attempt_stats(self):
        """
        Print attempt statistics
        """
        attempt_stats = self.get_attempt_stats()
        print(f"\n{'='*170}")
        print(f"{'Rank':<4} {'Model':<50} {'Type':<10} {'Legal':<8} {'win':<6} {'win_move':<8} {'lose':<6} {'lose_move':<8} {'draw':<6} {'draw_move':<8}")
        print(f"{'='*170}")
        for i, (name, one_stats, rating) in enumerate(attempt_stats, 1):
            name_lst = name.split("_")
            model_name = "_".join(name_lst[:-2])
            with_legal_moves = name_lst[-1]
            oneType = name_lst[-2]
            print(f"{i:<4} {model_name:<50} {oneType:<10} {with_legal_moves:<8} {one_stats['win_num']:<6.0f} {one_stats['win_avg_move']:<8.0f}  "
                  f"{one_stats['lose_num']:<6.0f} {one_stats['lose_avg_move']:<8.0f} "
                  f"{one_stats['draw_num']:<6.0f} {one_stats['draw_avg_move']:<8.0f}")
        print(f"{'='*170}")

    # Updated display methods in English
# Usage example
def main():
    # Create Elo calculator
    elo_calc = EloCalculator(k_factor=32, initial_rating=1500)
    elo_calc.load_all_games("./simulation_record/")
    elo_calc.add_player('gemini-2.5-pro-preview-06-05','standard')
    #elo_calc.add_player('doubao-1-5-pro-32k-250115','blindfold')
    # Create comprehensive dashboard
    elo_calc.print_rankings()
    #elo_calc.print_attempt_stats()
    #print(elo_calc.termination_stats)
    elo_calc.save_ratings("./simulation_record/ratings.json")
    #elo_calc.print_player_terminations()


if __name__ == "__main__":
    main()