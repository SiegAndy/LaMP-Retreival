import copy
import json
import random
import time
from typing import Callable, Dict, List, Type

from src.utils import labels, label, check_config, task_2_categories
import requests

from openai import OpenAI

default_model_name = "gpt-3.5-turbo"
random.seed(0)


class LMModel:
    def conversation(self, message: str) -> str:
        raise NotImplementedError


class OpenAIModel(LMModel):
    model_name: str
    client: OpenAI

    def __init__(self, model_name: str = default_model_name) -> None:
        self.client = OpenAI()
        self.model_name = model_name

    def conversation(self, message: List[str]) -> str:
        if isinstance(message, list):
            message = "\n".join(message)
        completion = None
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": message},
                ],
            )
        except:
            return 0
        return completion.choices[0].message.content


class HuggingFaceModel(LMModel):
    model_name = "Hugging-Face-Place_holder"
    API_URL = "https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": "Bearer {Hugging_Face_Key}"}

    def __init__(self) -> None:
        self.API_URL = self.API_URL.format(model_name=self.model_name)
        self.headers["Authorization"] = self.headers["Authorization"].format(
            Hugging_Face_Key=check_config("HUGGING_FACE_KEY")
        )


class DistilBERTModel(HuggingFaceModel):
    # Source: https://huggingface.co/distilbert-base-uncased-distilled-squad
    model_name = "distilbert-base-uncased-distilled-squad"

    def conversation(self, message: List[str]) -> str:
        message_before_modify = copy.copy(message)
        message = message[1 : len(message) - 1]  # remove the first and last sentence
        message = "\n".join(message)
        response = requests.post(
            self.API_URL,
            headers=self.headers,
            json={
                "inputs": {
                    "question": "is reference 1 or 2 related? Just answer with one token.",
                    "context": message,
                },
            },
        )
        try:
            result_json = response.json()
        except json.JSONDecodeError:
            time.sleep(1)
            return self.conversation(message_before_modify)
        if "answer" not in result_json:
            if "error" in result_json and "Rate limit reached" in result_json["error"]:
                print(
                    "Rate limit reached. You reached free usage limit (reset hourly). Please subscribe to a plan at https://huggingface.co/pricing to use the API at this rate"
                )
                exit()
            time.sleep(1)
            return self.conversation(message_before_modify)
        return result_json["answer"]


class BERTSERINIModel(HuggingFaceModel):
    # Source: https://huggingface.co/rsvp-ai/bertserini-bert-base-squad
    model_name = "rsvp-ai/bertserini-bert-base-squad"

    def conversation(self, message: List[str]) -> str:
        message_before_modify = copy.copy(message)
        message = message[1 : len(message) - 1]  # remove the first and last sentence
        message = "\n".join(message)
        response = requests.post(
            self.API_URL,
            headers=self.headers,
            json={
                "inputs": {
                    "question": "is reference 1 or 2 related? Just answer with one token.",
                    "context": message,
                },
            },
        )
        try:
            result_json = response.json()
        except json.JSONDecodeError:
            print("BERTSERINIModel catched json error")
            time.sleep(1)
            return self.conversation(message_before_modify)
        if "answer" not in result_json:
            if "error" in result_json and "Rate limit reached" in result_json["error"]:
                print(
                    "Rate limit reached. You reached free usage limit (reset hourly). Please subscribe to a plan at https://huggingface.co/pricing to use the API at this rate"
                )
                exit()
            time.sleep(1)
            return self.conversation(message_before_modify)
        return result_json["answer"]


class MiniLM(HuggingFaceModel):
    # Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def conversation(self, message: List[str]) -> str:
        message_before_modify = copy.copy(message)
        prompt_with_refs = message[-2]
        message = message[1 : len(message) - 1]  # remove the first and last sentence
        message = "\n".join(message)

        _, opt1, _, opt2, *_ = prompt_with_refs.split('"')
        response = requests.post(
            self.API_URL,
            headers=self.headers,
            json={
                "inputs": {
                    "source_sentence": message,
                    "sentences": [
                        opt1,
                        opt2,
                    ],
                },
            },
        )
        result: List[float] = response.json()
        if not isinstance(result, List):
            time.sleep(1)
            return self.conversation(message_before_modify)
        maximum_score = 0.0
        for score in result:
            if score > maximum_score:
                maximum_score = score
        if maximum_score == 0.0:
            return 0
        maximum_score_index = result.index(maximum_score)
        return maximum_score_index + 1


def task_1_parse_response(response: str, prompts: List[str]) -> str:
    prompt_with_refs = prompts[-2]

    _, opt1, _, opt2, *_ = prompt_with_refs.split('"')
    if not isinstance(response, str):
        response = str(response)
    potential_answer = []
    if "1" in response or opt1 in response:
        potential_answer.append("[1]")
    if "2" in response or opt2 in response:
        potential_answer.append("[2]")
    if len(potential_answer) == 2:
        # undetermined
        return "[3]"
    elif len(potential_answer) == 1:
        return potential_answer[0]
    else:
        # no answer
        return "[0]"


def task_2_parse_response(response: str, prompts: List[str]) -> str:
    possible_category = []
    for category in task_2_categories:
        if category in response:
            possible_category.append(category)
    if len(possible_category) == 0:
        return ""
    return random.choice(possible_category)


def feed_prompts_to_lm(
    prompts: Dict[str, str],
    model: Type[LMModel],
    ret: labels = None,
    callback: Callable[[str, List[str]], str] = None,
) -> labels:
    if ret is None:
        ret = labels(golds=list())
    container = ret.golds
    print(f"{model.__class__.__name__} is processing the questions")
    for id, prompt in prompts.items():
        print(f"{model.__class__.__name__} is processing the question {id}")
        start_time = time.time()
        model_response = model.conversation(message=prompt)
        finished_time = time.time()
        if callback is not None:
            model_response = callback(model_response, prompt)
        container.append(label(id=id, output=model_response))
        print(
            f"{model.__class__.__name__} finished the question {id} took {finished_time - start_time}"
        )
        time.sleep(5)
    return ret


def feed_prompt_to_lm(
    id: str,
    prompt: str | List[str],
    model: Type[LMModel],
    callback: Callable[[str, List[str]], str] = None,
) -> label:
    print(f"{model.__class__.__name__} is processing the question {id}")
    start_time = time.time()
    model_response = model.conversation(message=prompt)
    finished_time = time.time()
    if callback is not None:
        model_response = callback(model_response, prompt)
    print(
        f"{model.__class__.__name__} finished the question {id} took {finished_time - start_time}"
    )
    return label(id=id, output=model_response)
