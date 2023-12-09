import copy
import json
from math import ceil, floor
import random
import sys
import time
from typing import Callable, Dict, List, Tuple, Type

from transformers import pipeline, Pipeline
from sys import exit

from src.utils import labels, label, check_config, task_2_categories
import requests

from openai import OpenAI

default_model_name = "gpt-3.5-turbo"
random.seed(0)


class LMModel:
    def conversation(self, message: str, *args, **kwargs) -> str:
        raise NotImplementedError


class OpenAIModel(LMModel):
    model_name: str
    client: OpenAI

    def __init__(self, model_name: str = default_model_name) -> None:
        self.client = OpenAI()
        self.model_name = model_name

    def conversation(self, message: List[str], api_key: str = None) -> str:
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
    pipe: Pipeline
    API_URL = "https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": "Bearer {Hugging_Face_Key}"}

    def __init__(self, *args, **kwargs) -> None:
        self.API_URL = self.API_URL.format(model_name=self.model_name)
        self.headers["Authorization"] = self.headers["Authorization"].format(
            Hugging_Face_Key=check_config("HUGGING_FACE_KEY")
        )
        self.pipe = None

    def is_rate_limit(self, result_json: Dict) -> bool:
        if "error" not in result_json:
            return False
        if "Rate limit reached" in result_json["error"]:
            print(
                "Rate limit reached. You reached free usage limit (reset hourly). Please subscribe to a plan at https://huggingface.co/pricing to use the API at this rate"
            )
            return True
        print(result_json["error"])
        return True

    def switch_api_key(self, new_api_key: str) -> None:
        if new_api_key is None or new_api_key == "":
            return
        self.headers["Authorization"] = f"Bearer {new_api_key}"

    def url_request(
        self, question: str, message: str, api_key: str = None
    ) -> Dict[str, str]:
        self.switch_api_key(api_key)

        response = requests.post(
            self.API_URL,
            headers=self.headers,
            json={
                "inputs": {
                    "question": question,
                    "context": message,
                },
            },
        )

        try:
            result_json = response.json()
            return result_json
        except json.JSONDecodeError:
            time.sleep(1)
            return self.url_request(question=question, message=message, api_key=api_key)

    def pipe_request(self, question: str, message: str) -> Dict[str, str]:
        if self.pipe is None:
            self.pipe = pipeline("question-answering", model=self.model_name)
        return self.pipe(question=question, message=message)

    def conversation(
        self, message: List[str], api_key: str = None, use_pipeline: bool = False
    ) -> str:
        raise NotImplementedError


class QAModel(HuggingFaceModel):
    def __init__(self, task_name: str) -> None:
        self.task_name = task_name
        super().__init__(self)

    def pipe_request(self, question: str, message: str) -> None:
        if self.pipe is None:
            self.pipe = pipeline("question-answering", model=self.model_name)
        try:
            return self.pipe(question=question, context=message)
        except:
            return {
                "error": str(sys.exc_info()[0].args[0]),
            }

    def task_2_processing(self, message: List[str]) -> Tuple[str, str]:
        question = (
            'Given the above information, which category does the "Target Article" relate to? Just answer with the category name without further explanation. '
            + message[-2]
            + ". "
        )
        message[-1] = message[-1].replace("Article", '"Target Article"')
        tmp = message[-1].split(" With keywords: ")

        front = tmp[0]
        if len(front) > 600:
            front = front.split(". ")
            front = front[: ceil(len(front) / 2)]
            front = ". ".join(front)

        if len(tmp) == 2:
            keywords = tmp[1]
            keywords = keywords[1:-1]
            keywords = keywords.split(", ")
            keywords = ", ".join([f'"{token[1:-1]}"' for token in keywords])
            message[-1] = f"{front} With keywords: [{keywords}]."
        else:
            message[-1] = f"{front}"

        question += message[-1]

        context_list = message[2:-3]

        context = message[1] + "\n"
        # context_list = random.choices(context_list, k=8)
        for line in context_list:
            if '"[' in line and ']"' in line:
                # print(line.split("\"["))
                head, content_with_bracket = line.split('"[')
                # print(content_with_bracket.split("\"]"))
                content, _ = content_with_bracket.split(']"')
                tokens = content.split(", ")
                tokens_string = ", ".join([f'"{token}"' for token in tokens])
                # print(f"{head}[{tokens_string}].")
                context += f"{head}[{tokens_string}].\n"
        else:
            context += line + "\n"
        return context, question

    def conversation(
        self, message: List[str], api_key: str = None, use_pipeline: bool = False
    ) -> str:
        message_before_modify = copy.copy(message)
        # if "Freeing yourself from the material things and the need to keep up the appearance of wealth is as liberating a feeling as I've ever felt before and has helped me feel like the richest woman in the world. I hope you'll soon be able to get rich too!" not in message[-1]:
        #     exit(0)
        if self.task_name == "LaMP_1":
            message = message[1:-1]  # remove the first and last sentence
            message = "\n".join(message)
            question = "is reference 1 or 2 related? Just answer with one token."
        elif self.task_name == "LaMP_1_alt":
            message = message[1:-1]  # remove the first and last sentence
            message = "\n".join(message)
            question = "is reference 1 or 2 related? Just answer with one token."
        elif self.task_name == "LaMP_2":
            message, question = self.task_2_processing(message)
            f = open(".tmp", "w", encoding="utf-8")
            f.write(question + "\n\n")
            f.write(message + "\n")
            f.close()
        else:
            raise NotImplementedError(
                f"Need to deal with other dataset: {self.task_name}"
            )

        if use_pipeline:
            result = self.pipe_request(question=question, message=message)
        else:
            result = self.url_request(
                question=question, message=message, api_key=api_key
            )

        # if "error" in result and "overloaded" in result["error"]:
        # exit(0)
        # return ""
        # if self.task_name == "LaMP_2":
        #     if len(message_before_modify) > 4:

        #         need_to_remove = (
        #             random.choice(list(range(len(message_before_modify) - 4))) + 1
        #         )
        #         message_before_modify.pop(need_to_remove)
        #         return self.conversation(message_before_modify)

        if self.is_rate_limit(result):
            exit(0)
        if "answer" not in result:
            time.sleep(1)
            return self.conversation(message_before_modify)
        return result["answer"]


class DistilBERTModel(QAModel):
    # Source: https://huggingface.co/distilbert-base-uncased-distilled-squad
    model_name = "distilbert-base-uncased-distilled-squad"


class BERTSERINIModel(QAModel):
    # Source: https://huggingface.co/rsvp-ai/bertserini-bert-base-squad
    model_name = "rsvp-ai/bertserini-bert-base-squad"


class MiniLM(QAModel):
    # Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def conversation(self, message: List[str], api_key: str = None) -> str:
        self.switch_api_key(api_key)

        message_before_modify = copy.copy(message)

        if self.task_name == "LaMP_1":
            prompt_with_refs = message[-2]
            message = message[1:-1]  # remove the first and last sentence
            message = "\n".join(message)

            _, opt1, _, opt2, *_ = prompt_with_refs.split('"')
            options = [
                opt1,
                opt2,
            ]

        elif self.task_name == "LaMP_2":
            message = message[2:-3]  # remove the first two and last three sentence
            message = "\n".join(message)
            options = task_2_categories
        else:
            raise NotImplementedError(
                f"Need to deal with other dataset: {self.task_name}"
            )

        response = requests.post(
            self.API_URL,
            headers=self.headers,
            json={
                "inputs": {"source_sentence": message, "sentences": options},
            },
        )
        result: List[float] = response.json()
        if self.is_rate_limit(result):
            exit(0)
        if not isinstance(result, List):
            time.sleep(1)
            return self.conversation(message_before_modify)
        maximum_score = 0.0
        for score in result:
            if score > maximum_score:
                maximum_score = score
        if maximum_score == 0.0:
            return 0

        if self.task_name == "LaMP_1":
            maximum_score_index = result.index(maximum_score)
            return maximum_score_index + 1
        elif self.task_name == "LaMP_2":
            maximum_score_index = result.index(maximum_score)
            return options[maximum_score_index]
        else:
            raise NotImplementedError(
                f"Need to deal with other dataset: {self.task_name}"
            )


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
    api_key: str = None,
    log_path: str = None,
    callback: Callable[[str, List[str]], str] = None,
) -> label:
    output_fd = open(log_path, "a", encoding="utf-8") if log_path is not None else None

    printout = f"{model.__class__.__name__} is processing the question {id}"
    if output_fd is not None:
        output_fd.write(printout + "\n")
    else:
        print(printout)
    start_time = time.time()
    model_response = model.conversation(message=prompt, api_key=api_key)
    finished_time = time.time()
    if model_response == "":
        return None
    if callback is not None:
        model_response = callback(model_response, prompt)

    printout = f"{model.__class__.__name__} finished the question {id} took {finished_time - start_time}"
    if output_fd is not None:
        output_fd.write(printout + "\n")
    else:
        print(printout)
    if log_path is not None:
        output_fd.close()
    return label(id=id, output=model_response)
