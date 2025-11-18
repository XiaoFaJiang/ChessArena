# run_simulation.py
import os
import time
import chess
import random
import json
import logging
import concurrent.futures
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import re
from datetime import datetime
from openai import OpenAI
import openai
import copy

# Import stockfish functions
from stockfish import calculate_win_rate, get_best_moves_and_evaluate\
# Import lc0 engine
from chess_engine import lc0_engine as LC_ENGINE
from utils import parse_uci_move,parse_san_move,san_to_uci,get_blitz_move_prompt,get_bullet_move_prompt,get_multi_turn_blindfold_move_prompt,get_blindfold_move_prompt,\
    judge_thinking

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chess_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("chess_simulation")

@dataclass
class PlayerConfig:
    """Configuration for a chess player."""
    name: str
    api_key: str
    base_url: str
    model: str
    play_mode: str
    temperature: float = 0.2
    frequency_penalty: float = 0.1
    top_p: float = 1.0
    max_tokens: int = 500
    provide_legal_moves: bool = False
    provide_move_history: bool = False
    
@dataclass
class GameConfig:
    """Configuration for chess games."""
    white_player: PlayerConfig
    black_player: PlayerConfig
    max_moves: int = 100
    max_retries: int = 3
    log_directory: str = "game_logs"
    stockfish_path: str = "./stockfish"  # Add stockfish path parameter

def setup_game_logging(game_id: str, log_dir: str) -> logging.Logger:
    """Setup logging for a specific game."""
    # Create directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create game-specific logger
    game_logger = logging.getLogger(f"game_{game_id}")
    game_logger.setLevel(logging.INFO)
    
    # File handler for this game
    log_file = Path(log_dir) / f"game_{game_id}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    game_logger.addHandler(file_handler)
    
    return game_logger



def get_llm_move(client, player_config, fen, is_white, board, game_logger, max_retries=3,move_history=None,chat_history=None,last_opponent_move=None,lc0_engine=None):
    """
    Get a move from the LLM.
    Returns the move if successful, None if all retries are exhausted.
    Also returns attempt statistics.
    """
    # Get list of legal moves if configured to provide them
    
    legal_moves = None
    if player_config.provide_legal_moves:
        legal_moves = list(board.legal_moves)
        
    if not player_config.provide_move_history: #provide_move_history is False
        move_history = None
    
    partial_move_history = None
    if move_history:
        partial_move_history = move_history[-10:] #partial move history is used to help reduce fivefold repetition.
    start_prompt = ""
    #print("enter")
    if player_config.play_mode == "blitz":
        verified_prompt = get_blitz_move_prompt(fen, is_white, legal_moves, partial_move_history)
    elif player_config.play_mode == "bullet":
        verified_prompt = get_bullet_move_prompt(fen, is_white, legal_moves, partial_move_history)
    elif player_config.play_mode == "standard":
        verified_prompt = get_blitz_move_prompt(fen, is_white, legal_moves, partial_move_history)
    elif player_config.play_mode == "blindfold_singleTurn":
        verified_prompt = get_blindfold_move_prompt(fen, is_white, legal_moves, partial_move_history)
    elif player_config.play_mode == "blindfold_multiTurn":
        verified_prompt = get_multi_turn_blindfold_move_prompt(is_white, legal_moves, chat_history, last_opponent_move)[1]
    else:
        raise ValueError(f"Unknown play mode: {player_config.play_mode}")
    
    player_name = "White" if is_white else "Black"
    
    # Log the input

    game_logger.info(f"Prompt to {player_name} ({player_config.name}):")
    
    for prompt in verified_prompt:
        game_logger.info(f"\n{prompt['content']}")
    # Track attempt statistics
    attempt_stats = {
        "total_attempts": 0,
        "parsing_errors": 0,
        "illegal_moves": 0,
        "forbidden_thinking":0,
        "successful_moves": 0
    }
    
    if player_config.model == "random_player":
        #generate random move
        move_uci = random.choice(legal_moves).uci()
        attempt_stats["total_attempts"] += 1
        attempt_stats["successful_moves"] += 1
        game_logger.info(f"Valid move found: {move_uci}")
        return move_uci,f"{move_uci}",0,attempt_stats,[]
    elif player_config.model == "lc0":
        move_uci = lc0_engine.predict_move(board)
        attempt_stats["total_attempts"] += 1
        attempt_stats["successful_moves"] += 1
        game_logger.info(f"Valid move found: {move_uci}")
        return move_uci,f"{move_uci}",0,attempt_stats,[]
    
    #verified_prompt is used to record the prompt and response
    #wish LLMs could learn from history and improve their responses
    start_prompt = verified_prompt[:]
    all_records_prompt_and_response = []
    attempt = 0
    while attempt < max_retries:
        attempt_stats["total_attempts"] += 1
        try:
            game_logger.info(f"Attempt {attempt + 1}/{max_retries}...")
            
            # Record start time
            start_time = time.time()
            response = client.chat.completions.create(
                model=player_config.model,
                messages=[{"role":prompt['role'],"content":prompt["content"]} for prompt in verified_prompt],
                temperature=player_config.temperature,
                frequency_penalty=player_config.frequency_penalty,
                top_p=player_config.top_p,
                max_tokens=player_config.max_tokens,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            # Calculate response time
            response_time = time.time() - start_time
            content = response.choices[0].message.content
            reasoning_content = ""
            if hasattr(response.choices[0].message,'reasoning_content'):
                reasoning_content = response.choices[0].message.reasoning_content
                if not reasoning_content:
                    reasoning_content = ""
            if not content:
                raise Exception("No content returned by API call")
            verified_prompt.append({"role":"assistant", "content":str(content),"reasoning_content":reasoning_content})
            # Log the complete response
            game_logger.info(f"Raw response from {player_name} ({player_config.name}) (took {response_time:.2f}s):\n{content}")
            
            # Parse the move
            move_uci = parse_uci_move(content)
            if not move_uci:
                move_san = parse_san_move(content) #parse san style move and convert it to uci move
                if move_san:
                    move_uci = san_to_uci(move_san,fen)
                if not move_uci:
                    feedback = "Parsing error: No valid UCI move found in response. Rethink the valid UCI move notation and example we give you. Your answer must follow our example(i.e., adding ``` in the beginning and ending). Don't add \"x\" or \"+\" in UCI of your answer."
                    game_logger.warning(feedback)
                    attempt_stats["parsing_errors"] += 1
                    verified_prompt[-1]["status"] = "parsing_error"
                    verified_prompt[-1]["feedback"] = feedback
                    verified_prompt.append({"role":"user", "content":feedback})
                    attempt += 1
                    continue  # Try again
            #first check if there is valid UCI move
            #then check if it's forbidden thinking
            if player_config.play_mode == "bullet":
                if judge_thinking(content):
                    feedback = "Forbidden thinking detected."
                    game_logger.warning(feedback)
                    attempt_stats["forbidden_thinking"] += 1
                    verified_prompt[-1]["status"] = "forbidden_thinking"
                    verified_prompt[-1]["feedback"] = feedback
                    all_records_prompt_and_response.append(verified_prompt) #single record
                    verified_prompt = start_prompt[:] #reset the prompt
                    attempt += 1
                    continue
            # Check if the move is legal
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    game_logger.info(f"Valid move found: {move_uci}")
                    attempt_stats["successful_moves"] += 1
                    verified_prompt[-1]["status"] = "successful move"
                    verified_prompt[-1]["feedback"] = ""
                    all_records_prompt_and_response.append(verified_prompt)
                    return move_uci, content, response_time, attempt_stats, all_records_prompt_and_response
                else:
                    x = " Note that we have give you all legal moves." if player_config.provide_legal_moves else ""
                    feedback = f"Illegal move: {move_uci} is not a legal move in the current position. Please choose another one.{x} Don't add \"x\" or \"+\" in UCI of your answer."
                    game_logger.warning(feedback)
                    attempt_stats["illegal_moves"] += 1
                    verified_prompt[-1]["status"] = "illegal move"
                    verified_prompt[-1]["feedback"] = feedback
                    verified_prompt.append({"role":"user", "content":feedback})
                    
                    attempt += 1
                    continue  # Try again
            except ValueError as e:
                feedback = f"Invalid UCI format: {move_uci} - Error: {e}. Don't add \"x\" or \"+\" in UCI of your answer."
                game_logger.warning(feedback)
                attempt_stats["parsing_errors"] += 1  # Counting this as a parsing error
                verified_prompt[-1]["status"] = "Invalid UCI format"
                verified_prompt[-1]["feedback"] = feedback
                verified_prompt.append({"role":"user", "content":feedback})
                attempt += 1
                continue  # Try again
        except openai.RateLimitError:
            game_logger.error("Rate limit exceeded. Waiting...")
            time.sleep(20)
        except openai.APIConnectionError:
            game_logger.error("API connection error. Waiting...")
            time.sleep(20)
        except openai.APIError as e:
            game_logger.error(e)
            time.sleep(20)
        except Exception as e:
            if 'NoneType' in str(e): # No response, call LLM again
                game_logger.error(e)
                time.sleep(20)
                continue
            feedback = f"Error getting move: {e}"
            game_logger.error(feedback)
            verified_prompt[-1]["status"] = "other error"
            verified_prompt[-1]["feedback"] = feedback
            verified_prompt.append({"role":"user", "content":feedback})
            attempt += 1
            continue  # Try again
        
    all_records_prompt_and_response.append(verified_prompt)
    # If we've exhausted all retries
    game_logger.error(f"Failed to get a valid move after {max_retries} attempts. Forfeiting the game.")
    return None, None, None, attempt_stats, all_records_prompt_and_response

def play_game(game_id, game_config):
    """Simulate a chess game between two LLMs."""
    # Setup game-specific logging
    game_logger = setup_game_logging(game_id, game_config.log_directory)
    game_logger.info(f"Starting game {game_id}: {game_config.white_player.name} vs {game_config.black_player.name}")
    
    # Initialize game state
    board = chess.Board()
    move_history = []
    move_details = []  # For storing detailed move information
    move_count = 0
    game_over = False
    
    # Track attempt statistics for each player
    player_stats = {
        game_config.white_player.name: {
            "total_attempts": 0,
            "forbidden_thinking":0,
            "parsing_errors": 0,
            "illegal_moves": 0,
            "successful_moves": 0,
            "total_moves": 0,
            "optimal_moves": 0  # Count of moves that were in top 3 recommended by Stockfish
        },
        game_config.black_player.name: {
            "total_attempts": 0,
            "forbidden_thinking":0,
            "parsing_errors": 0,
            "illegal_moves": 0,
            "successful_moves": 0,
            "total_moves": 0,
            "optimal_moves": 0
        }
    }
    
    # Initialize OpenAI clients
    white_client = OpenAI(
        api_key=game_config.white_player.api_key,
        base_url=game_config.white_player.base_url
    )
    black_client = OpenAI(
        api_key=game_config.black_player.api_key,
        base_url=game_config.black_player.base_url
    )
    
    # Log initial board state
    game_logger.info(f"Initial board:\n{board}")
    
    # Game loop
    move_uci = None
    chat_histories = {}
    if (game_config.white_player.play_mode == "blindfold_multiTurn" or 
        game_config.black_player.play_mode == "blindfold_multiTurn"):
        
        if game_config.white_player.play_mode == "blindfold_multiTurn":
            system_prompt, _ = get_multi_turn_blindfold_move_prompt(True, None, None, None)
            chat_histories[game_config.white_player.name] = [
                {"role": "system", "content": system_prompt}
            ]
            
        if game_config.black_player.play_mode == "blindfold_multiTurn":
            system_prompt, _ = get_multi_turn_blindfold_move_prompt(False, None, None, None)
            chat_histories[game_config.black_player.name] = [
                {"role": "system", "content": system_prompt}
            ]
    
    lc0_engine = None
    if game_config.white_player.model == "lc0" or game_config.black_player.model == "lc0":
        lc0_engine = LC_ENGINE()
    
    while not game_over and move_count < game_config.max_moves:
        # Get the current board position's win rate before move
        #print(chat_history[game_config.white_player.name])
        try:
            current_win_rate = calculate_win_rate(board, stockfish_path=game_config.stockfish_path)
            game_logger.info(f"Current position win rate for {'White' if board.turn else 'Black'}: {current_win_rate:.2f}%")
            
            # Get top 3 recommended moves (no candidate move yet)
            top_moves, _ = get_best_moves_and_evaluate(board, stockfish_path=game_config.stockfish_path, n=3)
            game_logger.info("Top 3 recommended moves by Stockfish:")
            for i, (move, win_rate) in enumerate(top_moves, 1):
                game_logger.info(f"  {i}. {board.san(move)} ({move.uci()}) - Win rate: {win_rate:.2f}%")
        except Exception as e:
            game_logger.error(f"Error getting Stockfish evaluation: {e}")
            top_moves = []
            current_win_rate = None
        
        # White's turn
        if board.turn:
            game_logger.info(f"{game_config.white_player.name}'s turn (White) - Move {move_count//2 + 1}")
            current_chat_history = None
            if game_config.white_player.play_mode == "blindfold_multiTurn":
                current_chat_history = chat_histories.get(game_config.white_player.name,None)
            move_uci, raw_response, response_time, attempt_stats, all_record_prompt_and_response = get_llm_move(
                white_client, 
                game_config.white_player, 
                board.fen(), 
                True, 
                board, 
                game_logger,
                max_retries=game_config.max_retries,
                move_history=move_history,
                chat_history=current_chat_history,
                last_opponent_move=move_uci,
                lc0_engine=lc0_engine
            )
            current_player = game_config.white_player.name
            opponent = game_config.black_player.name
            if game_config.white_player.play_mode == "blindfold_multiTurn":
                chat_histories[game_config.white_player.name] = [{"role":item['role'],"content":item['content']} for item in all_record_prompt_and_response[-1]]
            # Update player statistics
            for stat_key in ["total_attempts", "parsing_errors", "illegal_moves","forbidden_thinking", "successful_moves"]:
                player_stats[current_player][stat_key] += attempt_stats.get(stat_key, 0)
            
        # Black's turn
        else:
            game_logger.info(f"{game_config.black_player.name}'s turn (Black) - Move {move_count//2 + 1}")
            current_chat_history = None
            if game_config.black_player.play_mode == "blindfold_multiTurn":
                current_chat_history = chat_histories.get(game_config.black_player.name,None)
            move_uci, raw_response, response_time, attempt_stats, all_record_prompt_and_response = get_llm_move(
                black_client, 
                game_config.black_player, 
                board.fen(), 
                False, 
                board, 
                game_logger,
                max_retries=game_config.max_retries,
                move_history=move_history,
                chat_history=current_chat_history,
                last_opponent_move=move_uci,
                lc0_engine=lc0_engine
            )
            current_player = game_config.black_player.name
            opponent = game_config.white_player.name
            if game_config.black_player.play_mode == "blindfold_multiTurn":
                chat_histories[game_config.black_player.name] = [{"role":item['role'],"content":item['content']} for item in all_record_prompt_and_response[-1]]
            # Update player statistics
            for stat_key in ["total_attempts", "parsing_errors", "illegal_moves","forbidden_thinking", "successful_moves"]:
                player_stats[current_player][stat_key] += attempt_stats.get(stat_key, 0)

        if move_uci is None:
            game_logger.info(f"{current_player} has forfeited the game due to failure to generate a legal move.")
            game_logger.info(f"{opponent} wins by forfeit!")
            game_over = True
            result = "0-1" if board.turn else "1-0"
            termination = "forfeit"
            
            # Add forfeit move details
            move_details.append({
                "move_number": move_count + 1,
                "player": current_player,
                "color": "white" if board.turn else "black",
                "fen_before": board.fen(),
                "chat_history": all_record_prompt_and_response,
                "move": None,
                "parsing_success": False,
                "move_legal": False,
                "outcome": "forfeit",
                "response_time": None,
                "attempt_stats": attempt_stats,
                "position_win_rate_before": current_win_rate,
                "top_moves": [(m.uci(), wr) for m, wr in top_moves] if top_moves else None,
                "move_win_rate": None,
                "move_ranking": None
            })
        else:
            # Make the move
            move = chess.Move.from_uci(move_uci)
            fen_before = board.fen()  # Save FEN before the move
            
            # Get top moves and evaluate the candidate move in a single Stockfish call
            try:
                top_moves, candidate_result = get_best_moves_and_evaluate(
                    board, 
                    candidate_move=move, 
                    stockfish_path=game_config.stockfish_path, 
                    n=3
                )
                
                # Extract candidate move data
                if candidate_result:
                    _, move_win_rate, move_ranking = candidate_result
                    
                    # Log the move quality
                    if move_ranking:
                        game_logger.info(f"Move {move_uci} is Stockfish's recommendation #{move_ranking}")
                        player_stats[current_player]["optimal_moves"] += 1
                        is_optimal = True
                    else:
                        game_logger.info(f"Move {move_uci} is not among Stockfish's top 3 recommendations")
                        is_optimal = False
                        
                    # Log the win rate
                    game_logger.info(f"Win rate after move {move_uci}: {move_win_rate:.2f}%")
                    
                else:
                    # This should not happen unless there's an error
                    move_win_rate = None
                    move_ranking = None
                    game_logger.warning(f"Failed to evaluate candidate move {move_uci}")
                    is_optimal = False
                
            except Exception as e:
                game_logger.error(f"Error analyzing move with Stockfish: {e}")
                move_win_rate = None
                move_ranking = None
                is_optimal = False
            
            # Update player's total moves
            player_stats[current_player]["total_moves"] += 1
            
            # Make the move on the board
            board.push(move)
            
            # Log the move
            game_logger.info(f"Move: {move_uci}")
            game_logger.info(f"Board after move:\n{board}")
            
            # Record move details
            move_history.append(move_uci)
            move_details.append({
                "move_number": move_count + 1,
                "player": current_player,
                "color": "white" if not board.turn else "black",  # Color of the player who just moved
                "fen_before": fen_before,
                "fen_after": board.fen(),
                "chat_history": all_record_prompt_and_response,
                "move": move_uci,
                "parsing_success": True,
                "move_legal": True,
                "outcome": None,  # Will be updated if the game ends
                "response_time": response_time,
                "attempt_stats": attempt_stats,
                "position_win_rate_before": current_win_rate,
                "top_moves": [(m.uci(), wr) for m, wr in top_moves] if top_moves else None,
                "move_win_rate": move_win_rate,
                "move_ranking": move_ranking
            })
            
            # Check for game end conditions
            if board.is_checkmate():
                game_logger.info(f"Checkmate! {current_player} wins!")
                game_over = True
                result = "1-0" if not board.turn else "0-1"
                termination = "checkmate"
                move_details[-1]["outcome"] = "checkmate_win"
            elif board.is_stalemate():
                game_logger.info("Stalemate! Game is a draw.")
                game_over = True
                result = "1/2-1/2"
                termination = "stalemate"
                move_details[-1]["outcome"] = "stalemate"
            elif board.is_insufficient_material():
                game_logger.info("Insufficient material! Game is a draw.")
                game_over = True
                result = "1/2-1/2"
                termination = "insufficient_material"
                move_details[-1]["outcome"] = "insufficient_material"
            elif board.is_seventyfive_moves():
                game_logger.info("75-move rule! Game is a draw.")
                game_over = True
                result = "1/2-1/2"
                termination = "75_move_rule"
                move_details[-1]["outcome"] = "75_move_rule"
            elif board.is_fivefold_repetition():
                game_logger.info("Fivefold repetition! Game is a draw.")
                game_over = True
                result = "1/2-1/2"
                termination = "fivefold_repetition"
                move_details[-1]["outcome"] = "fivefold_repetition"
            
            move_count += 1
    
    if lc0_engine:
        lc0_engine.quit_engine()
        
    if move_count >= game_config.max_moves and not game_over:
        game_logger.info(f"Maximum moves ({game_config.max_moves}) reached. Game is a draw.")
        result = "1/2-1/2"
        termination = "move_limit"
        if move_details:
            move_details[-1]["outcome"] = "move_limit"
    
    # Calculate and log optimal move percentages
    game_logger.info("=== Player Move Quality Statistics ===")
    for player_name, stats in player_stats.items():
        if stats["total_moves"] > 0:
            optimal_move_pct = (stats["optimal_moves"] / stats["total_moves"]) * 100
            game_logger.info(f"{player_name}:")
            game_logger.info(f"  Total moves: {stats['total_moves']}")
            game_logger.info(f"  Optimal moves (in top 3): {stats['optimal_moves']} ({optimal_move_pct:.1f}%)")
    
    # Add player attempt statistics to results
    game_logger.info("=== Player Attempt Statistics ===")
    for player_name, stats in player_stats.items():
        game_logger.info(f"{player_name}:")
        game_logger.info(f"  Total attempts: {stats['total_attempts']}")
        game_logger.info(f"  Parsing errors: {stats['parsing_errors']} ({stats['parsing_errors']/stats['total_attempts']:.1%} of attempts)")
        game_logger.info(f"  Illegal moves: {stats['illegal_moves']} ({stats['illegal_moves']/stats['total_attempts']:.1%} of attempts)")
        game_logger.info(f"  forbidden thinking: {stats['forbidden_thinking']} ({stats['forbidden_thinking']/stats['total_attempts']:.1%} of attempts)")
        game_logger.info(f"  Successful moves: {stats['successful_moves']} ({stats['successful_moves']/stats['total_attempts']:.1%} of attempts)")
    
    # Prepare game results
    game_results = {
        "game_id": game_id,
        "white_player": game_config.white_player.name,
        "white_play_mode":game_config.white_player.play_mode,
        "white_player_provide_legal_moves":game_config.white_player.provide_legal_moves,
        "black_player": game_config.black_player.name,
        "black_play_mode":game_config.black_player.play_mode,
        "black_player_provide_legal_moves":game_config.black_player.provide_legal_moves,
        "result": result,
        "termination": termination,
        "moves": move_history,
        "move_details": move_details,
        "final_fen": board.fen(),
        "total_moves": move_count,
        "player_attempt_stats": player_stats
    }
    
    # Log game results
    game_logger.info(f"Game completed!")
    game_logger.info(f"Result: {result}")
    game_logger.info(f"Termination: {termination}")
    game_logger.info(f"Final FEN: {board.fen()}")
    game_logger.info(f"Move history: {', '.join(move_history)}")
    
    # Save game results to JSON
    results_file = Path(game_config.log_directory) / f"game_{game_id}_results.json"
    with open(results_file, 'w') as f:
        json.dump(game_results, f, indent=2)
        
    return game_results

def load_config(config_file):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    # Create player configs
    white_player = PlayerConfig(
        name=config_data["white_player"]["name"] + "_" + config_data["white_player"]["play_mode"] + "_" + str(config_data["white_player"].get("provide_legal_moves", True)),
        api_key=config_data["white_player"].get("api_key", "dummy"),
        base_url=config_data["white_player"]["base_url"],
        model=config_data["white_player"]["model"],
        temperature=config_data["white_player"].get("temperature", 0.2),
        top_p=config_data["white_player"].get("top_p", 1.0),
        frequency_penalty=config_data["white_player"].get("frequency_penalty",0),
        max_tokens=config_data["white_player"].get("max_tokens", 500),
        provide_legal_moves=config_data["white_player"].get("provide_legal_moves", True),
        provide_move_history=config_data["white_player"].get("provide_move_history", False),
        play_mode=config_data["white_player"].get("play_mode","blitz"),
    )
    
    black_player = PlayerConfig(
        name=config_data["black_player"]["name"] + "_" + config_data["black_player"]["play_mode"] + "_" + str(config_data["black_player"].get("provide_legal_moves", True)),
        api_key=config_data["black_player"].get("api_key", "dummy"),
        base_url=config_data["black_player"]["base_url"],
        model=config_data["black_player"]["model"],
        temperature=config_data["black_player"].get("temperature", 0.2),
        top_p=config_data["black_player"].get("top_p", 1.0),
        frequency_penalty=config_data["black_player"].get("frequency_penalty",0),
        max_tokens=config_data["black_player"].get("max_tokens", 500),
        provide_legal_moves=config_data["black_player"].get("provide_legal_moves", True),
        provide_move_history=config_data["black_player"].get("provide_move_history", False),
        play_mode=config_data["black_player"].get("play_mode","blitz"),
    )
    
    log_dir = os.path.join("simulation_record",f"{white_player.play_mode}-{black_player.play_mode}",f"{white_player.name}-vs-{black_player.name}")
    # Create game config
    game_config = GameConfig(
        white_player=white_player,
        black_player=black_player,
        max_moves=config_data.get("max_moves", 100),
        max_retries=config_data.get("max_retries", 3),
        log_directory=config_data.get("log_directory", log_dir),
        stockfish_path=config_data.get("stockfish_path", "./stockfish")  # Add stockfish path
    )
    
    return game_config

def run_simulation(config_file, num_games, parallel_games):
    """Run a simulation of multiple chess games."""
    # Load configuration
    original_config = load_config(config_file)
    
    # Ensure num_games is even
    if num_games % 2 != 0:
        logger.warning(f"Number of games must be even. Adjusting from {num_games} to {num_games + 1}.")
        num_games += 1
    
    # Ensure log directory exists
    Path(original_config.log_directory).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting simulation with {num_games} games, {parallel_games} games in parallel")
    logger.info(f"Player 1: {original_config.white_player.name} ({original_config.white_player.model})")
    logger.info(f"Player 2: {original_config.black_player.name} ({original_config.black_player.model})")
    logger.info(f"Using Stockfish at: {original_config.stockfish_path}")
    
    # Use timestamp to create a unique simulation ID
    simulation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare log directory for this simulation
    simulation_log_dir = os.path.join(original_config.log_directory, f"simulation_{simulation_id}")
    Path(simulation_log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create game configurations with swapped players for half the games
    game_configs = []
    player1_name = original_config.white_player.name
    player2_name = original_config.black_player.name
    
    # Create configurations for the first half (original configuration)
    for i in range(num_games // 2):
        config = GameConfig(
            white_player=original_config.white_player,
            black_player=original_config.black_player,
            max_moves=original_config.max_moves,
            max_retries=original_config.max_retries,
            log_directory=os.path.join(simulation_log_dir, f"game_{i+1}_{player1_name}_white"),
            stockfish_path=original_config.stockfish_path
        )
        game_configs.append((f"{simulation_id}_{i+1}", config))
    
    
    #play mode exchange
    #original_config.white_player.play_mode, original_config.black_player.play_mode = \
        #original_config.black_player.play_mode, original_config.white_player.play_mode
    # Create configurations for the second half (swapped players)
    for i in range(num_games // 2, num_games):
        # Swap white and black players
        config = GameConfig(
            white_player=original_config.black_player,
            black_player=original_config.white_player,
            max_moves=original_config.max_moves,
            max_retries=original_config.max_retries,
            log_directory=os.path.join(simulation_log_dir, f"game_{i+1}_{player2_name}_white"),
            stockfish_path=original_config.stockfish_path
        )
        game_configs.append((f"{simulation_id}_{i+1}", config))
    
    # Run games in parallel
    start_time = time.time()
    all_results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_games) as executor:
        # Submit all games
        futures = {executor.submit(play_game, game_id, config): i 
                  for i, (game_id, config) in enumerate(game_configs)}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            game_index = futures[future]
            #try:
            result = future.result()
            all_results.append(result)
            logger.info(f"Game {game_index} completed: {result['result']}")
           # except Exception as e:
                #logger.error(f"Game {game_index} failed with error: {e}")
    
    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"Simulation completed in {total_time:.2f} seconds")
    
    # Analyze results
    player1_wins = 0
    player2_wins = 0
    draws = 0
    
    # Aggregate player statistics
    player_stats = {
        player1_name: {
            "total_attempts": 0,
            "parsing_errors": 0,
            "illegal_moves": 0,
            "forbidden_thinking": 0,
            "successful_moves": 0,
            "total_moves": 0,
            "optimal_moves": 0
        },
        player2_name: {
            "total_attempts": 0,
            "parsing_errors": 0,
            "illegal_moves": 0,
            "forbidden_thinking": 0,
            "successful_moves": 0,
            "total_moves": 0,
            "optimal_moves": 0
        }
    }
    
    for result in all_results:
        # Get player statistics
        if "player_attempt_stats" in result:
            for player, stats in result["player_attempt_stats"].items():
                for stat_key, value in stats.items():
                    if stat_key in player_stats[player]:
                        player_stats[player][stat_key] += value
        
        # Count wins based on who played white/black
        if result["result"] == "1-0":  # White win
            if result["white_player"] == player1_name:
                player1_wins += 1
            else:
                player2_wins += 1
        elif result["result"] == "0-1":  # Black win
            if result["black_player"] == player1_name:
                player1_wins += 1
            else:
                player2_wins += 1
        else:  # Draw
            draws += 1
    
    # Generate attempt statistics percentages
    for player in player_stats:
        if player_stats[player]["total_attempts"] > 0:
            for stat in ["parsing_errors", "illegal_moves", "successful_moves"]:
                player_stats[player][f"{stat}_pct"] = (
                    player_stats[player][stat] / player_stats[player]["total_attempts"] * 100
                )
        
        # Add optimal move percentage
        if player_stats[player]["total_moves"] > 0:
            player_stats[player]["optimal_moves_pct"] = (
                player_stats[player]["optimal_moves"] / player_stats[player]["total_moves"] * 100
            )
        else:
            player_stats[player]["optimal_moves_pct"] = 0
    
    # Generate summary report
    summary = {
        "simulation_id": simulation_id,
        "timestamp": datetime.now().isoformat(),
        "config_file": config_file,
        "num_games": num_games,
        "parallel_games": parallel_games,
        "player1": player1_name,
        "player2": player2_name,
        "results": {
            "player1_wins": player1_wins,
            "player1_win_rate": player1_wins / num_games if num_games > 0 else 0,
            "player2_wins": player2_wins,
            "player2_win_rate": player2_wins / num_games if num_games > 0 else 0,
            "draws": draws,
            "draw_rate": draws / num_games if num_games > 0 else 0,
        },
        "player_attempt_stats": player_stats,
        "termination_types": {},
        "total_time": total_time,
        "average_time_per_game": total_time / num_games if num_games > 0 else 0
    }
    
    # Count termination types
    for result in all_results:
        term_type = result["termination"]
        if term_type not in summary["termination_types"]:
            summary["termination_types"][term_type] = 0
        summary["termination_types"][term_type] += 1
    
    # Log summary
    logger.info("=== Simulation Summary ===")
    logger.info(f"Games played: {num_games}")
    logger.info(f"{player1_name} wins: {player1_wins} ({player1_wins/num_games:.1%})")
    logger.info(f"{player2_name} wins: {player2_wins} ({player2_wins/num_games:.1%})")
    logger.info(f"Draws: {draws} ({draws/num_games:.1%})")
    logger.info(f"Termination types: {summary['termination_types']}")
    
    # Log attempt statistics
    logger.info("=== Player Attempt Statistics ===")
    for player, stats in player_stats.items():
        if stats["total_attempts"] > 0:
            logger.info(f"{player}:")
            logger.info(f"  Total attempts: {stats['total_attempts']}")
            logger.info(f"  Parsing errors: {stats['parsing_errors']} ({stats['parsing_errors']/stats['total_attempts']:.1%})")
            logger.info(f"  Illegal moves: {stats['illegal_moves']} ({stats['illegal_moves']/stats['total_attempts']:.1%})")
            logger.info(f"  forbidden thinking: {stats['forbidden_thinking']} ({stats['forbidden_thinking']/stats['total_attempts']:.1%} of attempts)")
            logger.info(f"  Successful moves: {stats['successful_moves']} ({stats['successful_moves']/stats['total_attempts']:.1%})")
            
    # Log move quality statistics
    logger.info("=== Player Move Quality Statistics ===")
    for player, stats in player_stats.items():
        if stats["total_moves"] > 0:
            logger.info(f"{player}:")
            logger.info(f"  Total moves: {stats['total_moves']}")
            logger.info(f"  Optimal moves (in top 3): {stats['optimal_moves']} ({stats['optimal_moves_pct']:.1f}%)")
    
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Average time per game: {total_time / num_games:.2f} seconds")
    
    # Save summary to file
    summary_file = os.path.join(simulation_log_dir, "simulation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    """Main function to run the chess simulation."""
    parser = argparse.ArgumentParser(description="Run chess simulations between LLMs")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--games", type=int, default=2, help="Number of games to play (must be even)")
    parser.add_argument("--parallel", type=int, default=2, help="Number of games to run in parallel")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.games % 2 != 0:
        logger.warning(f"Number of games must be even. Adjusting from {args.games} to {args.games + 1}.")
        args.games += 1
    
    if args.parallel > args.games:
        args.parallel = args.games
        logger.warning(f"Adjusted parallel games to {args.parallel} (max number of games)")
    
    if args.parallel < 1:
        args.parallel = 1
        logger.warning("Adjusted parallel games to 1 (minimum value)")
    
    # Run the simulation
    summary = run_simulation(args.config, args.games, args.parallel)
    
    # Print final summary
    print("\n=== Final Simulation Results ===")
    print(f"Games: {summary['num_games']}")
    print(f"{summary['player1']} wins: {summary['results']['player1_wins']} ({summary['results']['player1_win_rate']:.1%})")
    print(f"{summary['player2']} wins: {summary['results']['player2_wins']} ({summary['results']['player2_win_rate']:.1%})")
    print(f"Draws: {summary['results']['draws']} ({summary['results']['draw_rate']:.1%})")
    
    # Print attempt statistics
    print("\n=== Player Attempt Statistics ===")
    for player, stats in summary["player_attempt_stats"].items():
        if stats["total_attempts"] > 0:
            print(f"{player}:")
            print(f"  Total attempts: {stats['total_attempts']}")
            print(f"  Parsing errors: {stats['parsing_errors']} ({stats.get('parsing_errors_pct', 0):.1f}%)")
            print(f"  Illegal moves: {stats['illegal_moves']} ({stats.get('illegal_moves_pct', 0):.1f}%)")
            print(f"  Successful moves: {stats['successful_moves']} ({stats.get('successful_moves_pct', 0):.1f}%)")
    
    # Print move quality statistics
    print("\n=== Player Move Quality Statistics ===")
    for player, stats in summary["player_attempt_stats"].items():
        if "total_moves" in stats and stats["total_moves"] > 0:
            print(f"{player}:")
            print(f"  Total moves: {stats['total_moves']}")
            print(f"  Optimal moves (in top 3): {stats['optimal_moves']} ({stats.get('optimal_moves_pct', 0):.1f}%)")
    
    print(f"\nTotal time: {summary['total_time']:.2f} seconds")

if __name__ == "__main__":
    main()