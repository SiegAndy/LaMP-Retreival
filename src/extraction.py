# Two comparable tasks:
#   1): Direct keywords extraction with title (no ranking on associated article).
#   2): Keywords extraction with title with ranking where ranks the author's profile
#           with respect to abstract of profile with specified title. In the meantime,
#           mention such rank order in the prompt.


import json
import os
from typing import Any, Callable, Dict, List

from src.TextRank import text_rank

from src.util import DatasetType
from src.tokenization import lemma_tokenizer


def PPEP_LaMP_1(profile: Dict[str, str], tokenizer: Callable[[str], str]) -> List[str]:

    abstract = profile["abstract"]
    # tokens = word_tokenize(abstract)
    # stemmed_tokens = [stemmer()]
    tokens = tokenizer(abstract)
    keywords = text_rank(tokens)

    return keywords


def PPEP_All_LaMP_1(profiles: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Extract PPEP for all article in the profile

    Return dictionary with profile ID as key and result string as value
    """
    PPEP_All_results = dict()
    for profile in profiles:
        keywords = PPEP_LaMP_1(profile, lemma_tokenizer)
        curr_result = f"article has title '{profile['title']}' with keywords: [{', '.join(keywords)}]"
        PPEP_All_results[profile["id"]] = curr_result
    return PPEP_All_results


def extract_info_LaMP_1(questions: List[Dict[str, str | List]]) -> Dict[str, str]:

    for question in questions:
        _, author_title, _, title_opt1, _, title_opt2, *_ = question["input"].split('"')
        print(
            f"Author title: {author_title}\nOption 1 Title: {title_opt1}\nOption 2 Title: {title_opt2}"
        )
        PPEP = PPEP_All_LaMP_1(question["profile"])
        print(PPEP)


def extract_info_LaMP_2(questions: List[Dict[str, str | List]]) -> Dict[str, str]:
    pass


def extract_labels(outputs: Dict[str, str | List]) -> Dict[str, str]:
    pass


def read_dataset(filename: str, task_index: int) -> Any:
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Dataset File <{filename}> not Found.")
    with open(filename, "r", encoding="utf-8") as file:
        contents = json.load(file)
        if not str(DatasetType.data) in filename:
            return extract_labels(contents)
        if task_index == 1:
            return extract_info_LaMP_1(contents)
        else:
            return extract_info_LaMP_2(contents)
