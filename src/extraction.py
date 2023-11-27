# Two comparable tasks:
#   1): Direct keywords extraction with title (no ranking on associated article).
#   2): Keywords extraction with title with ranking where ranks the author's profile
#           with respect to abstract of profile with specified title. In the meantime,
#           mention such rank order in the prompt.


import json
import os
import subprocess
from typing import Any, Callable, Dict, List


from src.BM25 import BM25Okapi as BM25

from src.TextRank import text_rank

from src.util import DatasetType, copy2clip
from src.tokenization import lemma_tokenizer


class dto:
    __attribute__: List[str]
    def to_json(self):
        result = dict()
        for elem in self.__attribute__:
            result[elem] = self.__getattribute__(elem)
        return result

    def __str__(self) -> str:
        return json.dumps(self.to_json(), indent=4)

class label(dto):
    id: str
    output: str
    __attribute__ = ["id", "output"]


class labels(dto):
    task: str
    golds: List[label]
    __attribute__ = ["task", "golds"]


task_1_template = "Given above information, for an author who has written the paper with the title \"{author_title}\", which reference is related? Just answer with [1] or [2] without explanation. [1]: \"{title_opt1}\" [2]: \"{title_opt2}\""

task_2_category = "categories: [women, religion, politics, style & beauty, entertainment, culture & arts, sports, science & technology, travel, business, crime, education, healthy living, parents, food & drink]"

task_2_template = "Given above information, which category does the following article relate to? Just answer with the category name without further explanation. article: {article}"


# def PPEP_LaMP_1(abstract: str, tokenizer: Callable[[str], str]) -> List[str]:
#     # abstract = profile["abstract"]
#     # tokens = word_tokenize(abstract)
#     # stemmed_tokens = [stemmer()]
#     tokens = tokenizer(abstract)
#     keywords = text_rank(tokens)

#     return keywords


# def PPEP_All_LaMP_1(profiles: List[Dict[str, str]], tokenizer: Callable[[str], str]) -> Dict[str, str]:
#     """
#     Extract PPEP for all article in the profile

#     Return dictionary with profile ID as key and result string as value
#     """
#     PPEP_All_results = dict()
#     for profile in profiles:
#         keywords = PPEP_LaMP_1(profile, tokenizer)
#         curr_result = f"article has title '{profile['title']}' with keywords: [{', '.join(keywords)}]"
#         PPEP_All_results[profile["id"]] = curr_result
#     return PPEP_All_results

def collect_feedback(text: str, copy_to_clipboard: bool = True):
    if copy_to_clipboard:
        copy2clip(text)
    return input("Response:")

def extract_info_LaMP_1(questions: List[Dict[str, str | List]], tokenizer: Callable[[str], str] = lemma_tokenizer, bm25_top_k = 5, **params) -> Dict[str, str]:
    """
    Return a dictionary with question id as key and prompt as value
    """
    prompt_collection = dict()
    for question in questions:
        _, author_title, _, title_opt1, _, title_opt2, *_ = question["input"].split('"')
        curr_prompt = [f"Here are the documents ranked by relevance, with titles and keywords, from most relevant to least relevant, for the topic of '{author_title}':"]
        ranker = BM25(corpus=question["profile"])
        rel_sequence = ranker.get_top_n(tokenizer(author_title), n = bm25_top_k, sequence_only=True)
        for rel_doc_index in rel_sequence:
            # id, title, abstract
            curr_profile = ranker.corpus[rel_doc_index]
            tokens = tokenizer(curr_profile[2])
            keywords = text_rank(tokens)
            curr_prompt.append(f"title '{curr_profile[1]}' with keywords: [{', '.join(keywords)}]")
        curr_prompt.append(task_1_template.format(author_title=author_title, title_opt1=title_opt1, title_opt2=title_opt2))
        # command = 'echo ' + " ".join(curr_prompt).strip() + '| clip'
        # print(command)
        # os.system(command)
        collect_feedback(curr_prompt)

        prompt_collection[curr_profile[0]] = curr_prompt
    return prompt_collection


def extract_info_LaMP_2(questions: List[Dict[str, str | List]], tokenizer: Callable[[str], str] = lemma_tokenizer, bm25_top_k = -1, **params) -> Dict[str, str]:
    """
    Return a dictionary with question id as key and prompt as value
    """
    prompt_collection = dict()
    # for question in questions:
    #     *_, curr_article = question["input"].split('article: ')
    #     curr_prompt = f"Here are the documents labeled by category, with titles and keywords:\n" 
    #     ranker = BM25(corpus=question["profile"])
    #     rel_sequence = ranker.get_top_n(tokenizer(author_title), n = bm25_top_k, sequence_only=True)
    #     profile_prompt_list = []
    #     for rel_doc_index in rel_sequence:
    #         # id, title, abstract
    #         curr_profile = ranker.corpus[rel_doc_index]
    #         tokens = tokenizer(curr_profile[2])
    #         keywords = text_rank(tokens)
    #         profile_prompt_list.append(f"title '{curr_profile[1]}' with keywords: [{', '.join(keywords)}]")
    #     curr_prompt += "\n".join(profile_prompt_list)
    #     curr_prompt += "\n" + task_1_template.format(author_title=author_title, title_opt1=title_opt1, title_opt2=title_opt2)
    #     prompt_collection[curr_profile[0]] = curr_prompt
    return prompt_collection


def extract_labels(outputs: Dict[str, str | List], **params) -> Dict[str, str]:
    pass


def read_dataset(filename: str, task_index: int, **params) -> Any:
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Dataset File <{filename}> not Found.")
    with open(filename, "r", encoding="utf-8") as file:
        contents = json.load(file)
        if not str(DatasetType.data) in filename:
            return extract_labels(contents, **params)
        if task_index == 1:
            return extract_info_LaMP_1(contents, **params)
        else:
            return extract_info_LaMP_2(contents, **params)
