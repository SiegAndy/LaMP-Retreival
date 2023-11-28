from typing import Dict

from src.util import labels, label


def feed_prompt_to_lm(prompts: Dict[str, str], ret: labels = None) -> labels:
    if ret is None:
        ret = labels(golds=list())

    container = ret.golds

    for id, prompt in prompts.items():
        model_response = ""
        container.append(label(id=id, output=model_response))

    return ret
