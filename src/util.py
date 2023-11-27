from enum import Enum
import subprocess


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

def copy2clip(txt):
    content = txt
    if isinstance(txt, list):
        content = ", ".join(txt)
    cmd='echo '+content+' | clip'
    return subprocess.check_call(cmd, shell=True)