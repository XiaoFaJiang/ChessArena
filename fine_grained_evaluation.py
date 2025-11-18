import os
import sys
import json
import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from utils import parse_uci_move,parse_san_move,san_to_uci,parse_json,evaluate_sets,connect_gpt,judge_thinking,generate_random_move,parse_fen_from_user_prompt,extract_fen
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chess_agent import Chess_Agent
import logging
from datetime import datetime
import random
from chess_engine import lc0_engine
import chess

def setup_logger(model_name, rating, log_level=logging.INFO):
    """设置主logger和文件日志"""
    # 创建主logger
    logger = logging.getLogger(f"PuzzleEval_{model_name}_{rating}")
    logger.setLevel(log_level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    log_dir = f"./ablation_evaluation/puzzles_by_rating/lite/evaluation_results/provide_legal_move{args.with_legal_move}/{model_name}/{rating}_{rating+400}/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/puzzle_eval_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别的日志
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    return logger

def collect_response_from_gpt(question_list,args):
    '''
    :param db_path: str
    :param question_list: []
    :return: dict of responses collected from openai
    '''
    count = 0
    def call_api_once(i,question):
        nonlocal count
        if args.task == "board_reconstruction":
            messages = question["messages"]
        else:
            if args.play_mode == "blindfold":
                if args.with_legal_move:
                    messages = question["messages_with_legal_moves"]
                else:
                    messages = question["messages_with_out_legal_moves"]
            else:
                prompt = question["prompt"]
                if args.task == "move_choosing":
                    if args.play_mode != "blindfold":
                        if args.with_legal_move and args.with_move_history:
                            prompt = question["prompt_pgn"]
                        elif args.with_legal_move and not args.with_move_history:
                            prompt = question["prompt_without_move_histroy"]
                        elif not args.with_legal_move and args.with_move_history:
                            prompt = question["prompt_without_legal_moves_pgn"]
                        else:
                            prompt = question["prompt_without_legal_moves_without_move_histroy"]
                    else:
                        if args.with_legal_move:
                            prompt = question["prompt"]
                        else:
                            prompt = question["prompt_without_legal_moves"]
                    
                messages = [
                        {'role':'system','content':question_list[i]["system"]},
                        {'role':'user','content':prompt}
                ]
        if "random" in args.model_name:
            uci_move = generate_random_move(question_list[i]["legal_moves"])
            plain_result = f"""
```
{uci_move}
```
"""         
        elif "maia" in args.model_name:
            board = chess.Board(fen = question_list[i]["fen"].strip())
            uci_move = engine.predict_move(board)
            plain_result = f"""
```
{uci_move}
```
"""
        else:
            plain_result = connect_gpt(model=args.model_id, url=args.url, messages=messages, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, api_key=args.api_key,enable_thinking=args.enable_thinking)
        #print("the response is: \n",plain_result,"\n")
        count += 1
        
        print(f"processing:{count}/{len(question_list)}")
        return plain_result,messages
    
    response_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        # 使用 tqdm 显示进度
        response_list = list(executor.map(call_api_once, range(len(question_list)), question_list))

    return response_list

def move_choose_metric(question_list,response_list,args):
    result = []
    legal_nums = 0
    top_nums = 0
    uci_nums = 0
    san_nums = 0
    all_mar = []
    for i,v in enumerate(question_list):
        r = {}
        r['response'] = response_list[i][0]
        converted_uci = None
        r['success'] = False
        r['fen'] = v['fen']
        now_move = ""
        if args.play_mode == "bullet" and judge_thinking(r['response']): #if bullet play mode and thinking is detected, this move will not be accpetd
            pass
        else:
            parsed_uci = parse_uci_move(r['response'],args.parse_any_move)
            if not parsed_uci:
                parsed_san = parse_san_move(r['response'],args.parse_any_move)
                if parsed_san:
                    converted_uci = san_to_uci(parsed_san,v["fen"].strip())
            if parsed_uci in v['legal_moves']:
                legal_nums += 1
                uci_nums += 1
                now_move = parsed_uci
                r['success'] = True
                if parsed_uci in v["top_moves"]:
                    top_nums += 1
                    
            elif converted_uci in v['legal_moves']:
                legal_nums += 1
                san_nums += 1
                now_move = converted_uci
                r['success'] = True
                if converted_uci in v["top_moves"]:
                    top_nums += 1
                    
        move_winrate = 0
        move_with_winrate = v["move_with_winrate"]
        for i,v2 in enumerate(move_with_winrate):
            if v2[0] == now_move:
                move_winrate = v2[1]
                break
        amp = sum([v2[1] for v2 in move_with_winrate]) / len(move_with_winrate)
        mar = (move_winrate - amp) / (amp + 0.00001)
        all_mar.append(mar)
        
        r['final_uci'] = now_move
        r['top_moves'] = v['top_moves']
        r['legal_moves'] = v['legal_moves']
        r['messages'] = response_list[i][1]
        result.append(r)
        final_metrics = {"legal_rate":legal_nums/len(question_list),"optimal_rate":top_nums/len(question_list),\
                "AMR":sum(all_mar)/len(all_mar),"total_nums":len(question_list),"uci_nums":uci_nums,"san_nums":san_nums}
    return result,final_metrics

def eval_move_choose(args):
    if not args.only_compute_metric:
        question_list = []
        with open(f"./ablation_evaluation/move_choose_evaluation/{args.play_mode}_legal_evaluation.jsonl","r") as f:
            for line in f.readlines():
                question_list.append(json.loads(line))
        question_list = question_list[:args.eval_nums]
        if os.path.exists(f"./ablation_evaluation/move_choose_evaluation/{args.play_mode}/{args.model_name}_final_metrics_{args.with_legal_move}_{args.eval_nums}.json"):
            print("already exists.")
            print(f"./ablation_evaluation/move_choose_evaluation/{args.play_mode}/{args.model_name}_final_metrics_{args.with_legal_move}_{args.eval_nums}.json")
            print("return")
            return 
        response_list = collect_response_from_gpt(question_list,args)
    else:
        question_list = []
        with open(f"./ablation_evaluation/move_choose_evaluation/{args.play_mode}_legal_evaluation.jsonl","r") as f:
            for line in f.readlines():
                question_list.append(json.loads(line))
        fen_index_map = {}
        for i,v in enumerate(question_list):
            fen_index_map[v["fen"]] = v
            
        response_list = []
        with open(f"./ablation_evaluation/move_choose_evaluation/{args.play_mode}/{args.model_name}_prediction_{args.with_legal_move}_{args.eval_nums}.jsonl") as f:
            for line in f.readlines():
                response_list.append(json.loads(line))
        #根据response_list中的fen读取到question_list中的fen，然后再做判断
        if args.play_mode == "blindfold":
            response_list = [[v["response"],v["messages"]] for v in response_list]
            question_list = question_list[:len(response_list)]
        else:
            real_question_list = []
            for i,v in enumerate(response_list):
                fen = parse_fen_from_user_prompt(v["prompt"])
                item = fen_index_map[fen]
                assert v["legal_moves"] == item["legal_moves"]
                real_question_list.append(item)
            
            question_list = real_question_list[:]
            response_list = [v["response"] for v in response_list]
            assert len(question_list) == len(response_list)
    
    
    result,final_metrics = move_choose_metric(question_list,response_list,args)
    
    print(final_metrics)
    with open(f"./ablation_evaluation/move_choose_evaluation/{args.play_mode}/{args.model_name}_final_metrics_{args.with_legal_move}_{args.eval_nums}.json","w") as f:
        json.dump(final_metrics,f,indent = 2)
        
    with open(f"./ablation_evaluation/move_choose_evaluation/{args.play_mode}/{args.model_name}_prediction2_{args.with_legal_move}_{args.eval_nums}.jsonl","w") as f:
        for line in result:
            f.write(json.dumps(line) + "\n")
    
    if engine:
        engine.quit_engine()
    
def chess_modeling_metric(question_list,response_list):
    piece_match_num = 0
    result = []
    for i,v in enumerate(question_list):
        r = {}
        r['response'] = response_list[i][0]
        r['ground_truth'] = v['ground_truth']
        r["messages"] = response_list[i][1]
        parsed_json = parse_json(r['response'])
        if parsed_json is None:
            continue
        pred_piece = parsed_json.get('piece',"unknown")
        pred_legal_moves = parsed_json.get("legal_moves","unknown")
        r['pred_piece'] = pred_piece.strip() if pred_piece else None
        r['gt_piece'] = v['ground_truth']['piece'].strip() if v['ground_truth']['piece'] else None
        r['pred_legal_moves'] = pred_legal_moves if pred_legal_moves else []
        r['gt_legal_moves'] = v['ground_truth']['legal_moves'] if pred_legal_moves else []
        if r['pred_legal_moves'] and r['pred_legal_moves'] != "unknown":
            for i,m in enumerate(r['pred_legal_moves']):
                r['pred_legal_moves'][i] = san_to_uci(m,v['fen'])
                
        if  r['pred_piece'] == r['gt_piece']:
            piece_match_num += 1
            set_compare = evaluate_sets(set(r['pred_legal_moves']),set(r['gt_legal_moves']))
        else:
            set_compare = {'f1_score':0,'precision':0,'recall':0}
        r['f1_score'] = set_compare['f1_score']
        r['precision'] = set_compare['precision']
        r['recall'] = set_compare['recall']
        result.append(r)
    

    final_metrics = {"piece_match_rate":piece_match_num/len(question_list),\
        "f1":sum(x['f1_score'] for x in result)/len(result),\
        "precision":sum(x['precision'] for x in result)/len(result),\
        "recall":sum(x['recall'] for x in result)/len(result),\
        "total_num":len(question_list)}

    return final_metrics,result 
            
def eval_chess_modeling(args):
    args.play_mode = 'bullet'
    if not args.only_compute_metric:
        question_list = []
        with open("./ablation_evaluation/chess_modeling_evaluation/chess_modeling_evaluation.jsonl","r") as f:
            for line in f.readlines():
                question_list.append(json.loads(line))
        question_list = question_list[:args.eval_nums]
        args.with_legal_move = True
        if os.path.exists(f"./ablation_evaluation/chess_modeling_evaluation/{args.model_name}_final_metrics_{args.eval_nums}.json"):
            return 
        response_list = collect_response_from_gpt(question_list,args)
    else:
        question_list = []
        with open("./ablation_evaluation/chess_modeling_evaluation/chess_modeling_evaluation.jsonl","r") as f:
            for line in f.readlines():
                question_list.append(json.loads(line))
        response_list = []
        with open(f"./ablation_evaluation/chess_modeling_evaluation/{args.model_name}_predictions_{args.eval_nums}.jsonl") as f:
            for line in f.readlines():
                response_list.append(json.loads(line)["response"])
        args.eval_nums = len(response_list)
        question_list = question_list[:args.eval_nums]
        
    final_metrics,result = chess_modeling_metric(question_list,response_list)
    print(final_metrics)
    with open(f"./ablation_evaluation/chess_modeling_evaluation/{args.model_name}_final_metrics_{args.eval_nums}.json","w") as f:
        json.dump(final_metrics,f,indent=2)
    with open(f"./ablation_evaluation/chess_modeling_evaluation/{args.model_name}_predictions_{args.eval_nums}.jsonl","w") as f:
        for line in result:
            f.write(json.dumps(line) + "\n")

def eval_single_puzzle(puzzle_data, args, main_logger):
    fen = puzzle_data["FEN"]
    agent = Chess_Agent(
        args.url, args.api_key, args.model_id,
        args.model_name, args.temperature, args.top_p,
        args.max_tokens, args.enable_thinking, args.is_san,
        args.max_retry, args.play_mode, args.with_legal_move
    )
    agent.set_up_board(fen)
    agent.push_move(puzzle_data["Moves"][0])
    if "stockfish" in args.model_id:
        agent.set_up_stockfish_depth(args.stockfish_depth)
    ground_truth_moves = puzzle_data["Moves"]
    puzzle_id = puzzle_data["PuzzleId"]
    puzzle_logger = logging.getLogger(f"Puzzle_{puzzle_id}")
    puzzle_logger.setLevel(main_logger.level)

    puzzle_logger.parent = main_logger
    puzzle_logger.propagate = True
    
    print(f"Evaluating Puzzle {puzzle_id}...")
    
    #puzzle main loop
    #excatly match the moves in the puzzle
    success = True
    for index in range(1, len(ground_truth_moves), 2):
        move = agent.step()
        if not move:
            puzzle_logger.error(f"Puzzle {puzzle_id} Failed: No move found in model's response.")
            success = False
            break
        
        if move == ground_truth_moves[index]:
            if index + 1 < len(ground_truth_moves):
                agent.push_opponent_move(ground_truth_moves[index+1])
        else:
            if agent.board.is_checkmate():
                puzzle_logger.info(f"Puzzle {puzzle_id} Checkmate")
                break
            puzzle_logger.error(f"Puzzle {puzzle_id} Failed: Move {move} wrong, ground truth is {ground_truth_moves[index]}")
            success = False
            break
        
    if success:
        puzzle_logger.info(f"Puzzle {puzzle_id} Solved")
    
    model_path = f"./ablation_evaluation/puzzles_by_rating/lite/evaluation_results/provide_legal_move{args.with_legal_move}/{args.model_name}/{args.rating}_{args.rating+400}/generation"
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    agent.record_messages_and_response(f"{model_path}/puzzles_{puzzle_id}_san({args.is_san}_sccuess({success})).json")
    agent.clear_messages()
    agent.quit_engine()
    return success

def eval_puzzle_accuracy(args):
    # Load data
    with open(f"./ablation_evaluation/puzzles_by_rating/lite/puzzles_{args.rating}_{args.rating+400}.jsonl") as f:
        data = [json.loads(line) for line in f.readlines()][:args.eval_nums]
    if os.path.exists(f"./ablation_evaluation/puzzles_by_rating/lite/evaluation_results/provide_legal_move{args.with_legal_move}/{args.model_name}/{args.rating}_{args.rating+400}/final_metrics_issan({args.is_san}).json") or \
        os.path.exists(f"./ablation_evaluation/puzzles_by_rating/lite/evaluation_results/provide_legal_move{args.with_legal_move}/{args.model_name}/{args.rating}_{args.rating+400}/final_metrics_issan({args.is_san}).json.json"):
            return 
    main_logger = setup_logger(args.model_name, args.rating)
    main_logger.info("Starting concurrent puzzle evaluation...")
    
    # find puzzles that have been evaluated
    model_path = f"./ablation_evaluation/puzzles_by_rating/lite/evaluation_results/provide_legal_move{args.with_legal_move}/{args.model_name}/{args.rating}_{args.rating+400}/generation"
    evaluated_puzzles = set()
    if os.path.exists(model_path):
        for file in os.listdir(model_path):
            if file.endswith(".json"):
                evaluated_puzzles.add(file.split("_")[1])
    
    data = [puzzle for puzzle in data if puzzle["PuzzleId"] not in evaluated_puzzles]
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(eval_single_puzzle, puzzle, args, main_logger)
            for puzzle in data
        ]
        right_count = sum(future.result() for future in futures)
    accuracy = right_count / len(data)
    with open(f"./ablation_evaluation/puzzles_by_rating/lite/evaluation_results/provide_legal_move{args.with_legal_move}/{args.model_name}/{args.rating}_{args.rating+400}/final_metrics_issan({args.is_san}).json","w") as f:
        json.dump({"accuracy":accuracy,"right_count":right_count},f,indent=4)
    main_logger.info(f"\nFinal Accuracy: {accuracy:.2%} ({right_count}/{len(data)})")
    return accuracy
        
def eval_board_reconstruction(args):
    with open(f"./ablation_evaluation/board_reconstruction/blindfold_board_reconstruction.jsonl") as f:
        question_list = [json.loads(line) for line in f.readlines()][:args.eval_nums]
    if os.path.exists(f"./ablation_evaluation/board_reconstruction/evaluation_results/{args.model_name}/final_metrics_issan({args.is_san}).json") or \
        os.path.exists(f"./ablation_evaluation/board_reconstruction/evaluation_results/{args.model_name}/final_metrics_issan({args.is_san}).json.json"):
            return 
    response_list = collect_response_from_gpt(question_list,args)
    successful_cnt = 0
    total_cnt = 0
    successful_turn = []
    failed_turn = []
    write_items = []
    for v1,v2 in list(zip(question_list,response_list)):
        item = {}
        item['fen'] = v1['fen']
        item['response'] = v2[0]
        item['model_fen'] = extract_fen(v2[0])
        item["success"] = False
        if v1["fen"] == extract_fen(v2[0]):
            item["success"] = True
            successful_cnt += 1
            successful_turn.append(len(v1["messages"]))
        else:
            failed_turn.append(len(v1["messages"]))
        total_cnt += 1
        write_items.append(item)
    accuracy = successful_cnt / total_cnt
    if not os.path.exists(f"./ablation_evaluation/board_reconstruction/evaluation_results/{args.model_name}"):
        os.mkdir(f"./ablation_evaluation/board_reconstruction/evaluation_results/{args.model_name}")
    print({"accuracy":accuracy,"successful_cnt":successful_cnt,"total_cnt":total_cnt,"successful_turn":successful_turn,"failed_turn":failed_turn})
    with open(f"./ablation_evaluation/board_reconstruction/evaluation_results/{args.model_name}/final_metrics_issan({args.is_san}).json","w") as f:
        json.dump({"accuracy":accuracy,"successful_cnt":successful_cnt,"total_cnt":total_cnt,"successful_turn":successful_turn,"failed_turn":failed_turn},f,indent =4 )
    
    with open(f"./ablation_evaluation/board_reconstruction/evaluation_results/{args.model_name}/generation_({args.is_san}).json","w") as f:
        for line in write_items:
            f.write(json.dumps(line) + "\n")


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--max_tokens', type=int,default=2048)
    args_parser.add_argument("--eval_nums", type=int, default=200)
    args_parser.add_argument('--temperature', type=float,default=0.2)
    args_parser.add_argument('--top_p',type=float,default=1.0)
    args_parser.add_argument('--api_key', type=str,default="")
    args_parser.add_argument('--model_id', type=str,default="deepseek-v3")
    args_parser.add_argument('--model_name', type=str,default="deepseek-v3")
    args_parser.add_argument('--url', type=str,default="")
    args_parser.add_argument('--enable_thinking',action="store_true",default=False)
    args_parser.add_argument('--concurrency',type=int,default=20)
    args_parser.add_argument('--with_legal_move',action="store_true",default=False)
    args_parser.add_argument('--with_move_history',action="store_true",default=False)
    args_parser.add_argument('--task',type=str,default="puzzle")
    args_parser.add_argument('--only_compute_metric',action="store_true",default=False)
    args_parser.add_argument('--parse_any_move',action="store_true",default=False)
    args_parser.add_argument('--rating',type=int,default=200)
    args_parser.add_argument('--is_san',action="store_true",default=False)
    args_parser.add_argument('--max_retry',type=int,default=5)
    args_parser.add_argument('--play_mode',type=str,default="blitz")
    args_parser.add_argument('--stockfish_depth',type=int,default=20)
    
    args = args_parser.parse_args()
    if "maia" in args.model_id:
        engine = lc0_engine()
    if args.task == "chess_modeling":
        eval_chess_modeling(args)
    elif args.task == "move_choosing":
        eval_move_choose(args)
    elif args.task == "puzzle":
        eval_puzzle_accuracy(args)
    elif args.task == "board_reconstruction":
        eval_board_reconstruction(args)
    else:
        print("task not supported")