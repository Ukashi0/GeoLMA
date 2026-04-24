import json
import os
import time

import torch
from func_timeout import FunctionTimedOut, func_timeout
from formalgeo.data import DatasetLoader
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent import Evaluator, Expander, Reflector
from environment import FormalGeoEnv
from mcts import MCTS


def load_test_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return sorted(set(json.load(f)))


def safe_save_json(data, filename):
    tmp = f"{filename}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, filename)


def init_result_files(config, problem_count):
    log_dir = os.path.join(config["path_logs"], "search")
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(log_dir, f"{config['dataset_name']}-log.json")
    data_filename = os.path.join(log_dir, f"{config['dataset_name']}-data.json")

    log = {
        "start_pid": config["start_pid"],
        "end_pid": min(config["end_pid"], problem_count),
        "solved_pid": [],
        "unsolved_pid": [],
        "timeout_pid": [],
        "error_pid": [],
    }
    data = {
        "solved": {},
        "unsolved": {},
        "timeout": {},
        "error": {},
    }

    if os.path.exists(log_filename):
        with open(log_filename, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        safe_save_json(log, log_filename)

    if os.path.exists(data_filename):
        with open(data_filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        safe_save_json(data, data_filename)

    return log_filename, data_filename


def load_model(model_path, lora_weights):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, lora_weights)
    model.eval()
    return model, tokenizer


def solve_single_problem(config, problem_id, model, tokenizer):
    env = FormalGeoEnv(
        dataset_name=config["dataset_name"],
        datasets_path=config["path_datasets"],
    )
    expander = Expander(model, tokenizer)
    evaluator = Evaluator(model, tokenizer)
    reflector = Reflector(model, tokenizer)
    mcts = MCTS(env, expander, evaluator, reflector)

    start_time = time.time()

    try:
        done, solution = func_timeout(
            config["timeout"],
            mcts.search,
            args=(problem_id, config["num_iters"]),
        )
        timing = time.time() - start_time

        if done:
            return "solved", solution, timing, len(solution), solution
        return "unsolved", "None", timing, config["num_iters"], "None"

    except FunctionTimedOut:
        timing = time.time() - start_time
        return "timeout", f"timeout@{config['timeout']}s", timing, config["num_iters"], "None"

    except Exception as exc:
        timing = time.time() - start_time
        return "error", repr(exc), timing, 0, "None"


def get_problem_ids(config, problem_count, log):
    if config.get("test_json"):
        problem_ids = load_test_ids(config["test_json"])
    else:
        problem_ids = list(
            range(config["start_pid"], min(config["end_pid"] + 1, problem_count + 1))
        )

    processed_ids = set(log.get("solved_pid", []))
    processed_ids.update(log.get("unsolved_pid", []))
    processed_ids.update(log.get("timeout_pid", []))
    processed_ids.update(log.get("error_pid", []))

    return sorted(pid for pid in problem_ids if pid not in processed_ids)


def batch_solve(config):
    model, tokenizer = load_model(config["model_path"], config["lora_weights"])
    dl = DatasetLoader(config["dataset_name"], config["path_datasets"])

    log_filename, data_filename = init_result_files(config, dl.info["problem_number"])

    with open(log_filename, "r", encoding="utf-8") as f:
        log = json.load(f)
    with open(data_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    problem_ids = get_problem_ids(config, dl.info["problem_number"], log)

    for problem_id in problem_ids:
        result, msg, timing, step_size, seq_path = solve_single_problem(
            config, problem_id, model, tokenizer
        )

        data[result][str(problem_id)] = {
            "msg": msg if isinstance(msg, str) else str(msg),
            "timing": timing,
            "step_size": step_size,
            "seq_path": seq_path,
        }
        log[f"{result}_pid"].append(problem_id)

        safe_save_json(log, log_filename)
        safe_save_json(data, data_filename)


if __name__ == "__main__":
    config = {
        "dataset_name": "",
        "path_datasets": "",
        "path_logs": "",
        "model_path": "",
        "lora_weights": "",
        "start_pid": 0,
        "end_pid": 0,
        "num_iters": 0,
        "timeout": 0,
        "test_json": None,
    }
    batch_solve(config)

