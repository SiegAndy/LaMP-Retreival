import string
from enum import Enum
import json
import os
import subprocess

from json import JSONEncoder
from typing import List
from copy import copy


class dto:
    __attribute__: List[str]
    __attribute_type__: List[object]

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            try:
                attr_index = self.__attribute__.index(key)
            except ValueError:
                continue
            attr_type = self.__attribute_type__[attr_index]
            if not isinstance(value, list):
                if dto in attr_type.__bases__:
                    setattr(self, key, attr_type(**value))
                else:
                    setattr(self, key, attr_type(value))
                continue
            if dto in attr_type.__bases__:
                setattr(self, key, [attr_type(**elem) for elem in value])
            else:
                setattr(self, key, [attr_type(elem) for elem in value])

    def to_json(self):
        result = dict()
        for elem in self.__attribute__:
            if hasattr(self, elem):
                result[elem] = self.__getattribute__(elem)
        return result

    def __str__(self) -> str:
        return json.dumps(self.to_json(), indent=4, cls=DTOEncoder)


class DTOEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, dto) or hasattr(o, "to_json"):
            return o.to_json()
        return o.__dict__


class label(dto):
    id: str
    output: str
    __attribute__ = ["id", "output"]
    __attribute_type__ = [str, str]


class labels(dto):
    task: str
    golds: List[label]
    __attribute__ = ["task", "golds"]
    __attribute_type__ = [str, label]


class DatasetType(Enum):
    data = "questions"
    label = "outputs"

    def __str__(self):
        return self.value


class DatasetCategory(Enum):
    train = "train"
    validate = "dev"
    test = "test"

    def __str__(self):
        return self.value


default_text_rank_window_size = 2
default_top_k_keywords = 5
default_data_path = os.path.join("src", "data")
default_prompt_path = os.path.join(default_data_path, "prompt")
default_extract_path = os.path.join(default_data_path, "extracts")


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


def extract_prompt_tokens_stats(
    flatten_prompts: List[str], price_per_1k_tokens: float = 0.001
):
    total_tokens = 0
    for prompt in flatten_prompts:
        cur_num_tokens = number_of_tokens(remove_enter_punctuation(prompt))
        total_tokens += cur_num_tokens
    return (
        total_tokens,
        total_tokens / len(flatten_prompts),
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
