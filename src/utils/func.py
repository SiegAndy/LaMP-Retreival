import json
import os
import subprocess
from copy import copy
import string
from typing import Any, List, Union
from src.utils.param import config_path


def copy2clip(txt):
    content = txt
    if isinstance(txt, list):
        content = ", ".join(txt)
    cmd = "echo " + content + " | clip"
    return subprocess.check_call(cmd, shell=True)


def number_of_tokens(prompt: str):
    return len(prompt.split(" "))


def remove_enter_punctuation(prompt: str):
    new_prompt = copy(prompt)
    for punctuation in string.punctuation:
        new_prompt = new_prompt.replace(punctuation, "")
    new_prompt.replace("\n", " ")
    return new_prompt


def extract_prompt_tokens_stats(prompt_json_path, price_per_1k_tokens: float = 0.001):
    total_tokens = 0
    length_of_json = 0
    with open(prompt_json_path, "r", encoding="utf-8") as file:
        id_prompts = json.load(file)
        length_of_json = len(id_prompts)
        for prompts in id_prompts.values():
            for prompt in prompts:
                cur_num_tokens = number_of_tokens(remove_enter_punctuation(prompt))
                total_tokens += cur_num_tokens
    return (
        total_tokens,
        total_tokens / length_of_json,
        round(total_tokens * price_per_1k_tokens / 1000, 2),
    )


def extract_output_tokens_stats(output_json_path, price_per_1k_tokens: float = 0.002):
    total_tokens = 0
    with open(output_json_path, "r", encoding="utf-8") as file:
        outputs = json.load(file)
        for output in outputs["golds"]:
            cur_num_tokens = number_of_tokens(output["output"])
            total_tokens += cur_num_tokens
    return total_tokens, round(total_tokens * price_per_1k_tokens / 1000, 2)


def check_config(
    key: str, default: Any = None, output_type: Any = None
) -> Union[bool, Any]:
    with open(config_path, "r", encoding="utf-8") as config:
        config = config.readlines()

    result = default
    for line in config:
        tag, status = line.strip("\n").split("=")
        if tag != key:
            continue
        if status.lower() == "true":
            status = True
        elif status.lower() == "false":
            status = False
        result = status
        break
    if output_type is not None:
        try:
            curr_type = type(result)
            result = output_type(result)
        except Exception as e:
            print(f"Unable to convert {curr_type} to {output_type}. Detail: {e}")
    return result


def config_to_env(key: str, default: Any = None, output_type: Any = None):
    value = check_config(key, default, output_type)
    os.environ[key] = value
