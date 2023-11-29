from typing import Dict, Type

from src.util import labels, label


from openai import OpenAI

default_model_name = "gpt-3.5-turbo"


class LMModel:
    def conversation(self, message: str) -> str:
        raise NotImplementedError


class OpenAIModel:
    model_name: str
    client: OpenAI

    def __init__(self, model_name: str = default_model_name) -> None:
        self.client = OpenAI()
        self.model_name = model_name

    def conversation(self, message: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message},
            ],
        )
        return completion.choices[0].message.content


def feed_prompt_to_lm(
    prompts: Dict[str, str], model: Type[LMModel], ret: labels = None
) -> labels:
    if ret is None:
        ret = labels(golds=list())

    container = ret.golds

    for id, prompt in prompts.items():
        joined_prompt = "\n".join(prompt)
        # print(joined_prompt)
        model_response = model.conversation(message=joined_prompt)
        # print(model_response)
        container.append(label(id=id, output=model_response))

    return ret
