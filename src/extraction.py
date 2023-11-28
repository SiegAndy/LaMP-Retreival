# Two comparable tasks:
#   1): Direct keywords extraction with title (no ranking on associated article).
#   2): Keywords extraction with title with ranking where ranks the author's profile
#           with respect to abstract of profile with specified title. In the meantime,
#           mention such rank order in the prompt.


from collections import defaultdict
import json
import os
from typing import Callable, Dict, List, Set


from src.BM25 import BM25Okapi as BM25

from src.TextRank import text_rank

from src.util import DatasetType, copy2clip
from src.tokenization import lemma_tokenizer

manual_feed_collect = False

task_1_template = 'Given above information, for an author who has written the paper with the title "{author_title}", which reference is related? Just answer with [1] or [2] without explanation. [1]: "{title_opt1}" [2]: "{title_opt2}"'

task_2_category = 'categories: ["women", "religion", "politics", "style & beauty", "entertainment", "culture & arts", "sports", "science & technology", "travel", "business", "crime", "education", "healthy living", "parents", "food & drink"]'

task_2_template = f"Given above information, which category does the following article relate to? Just answer with the category name without further explanation. {task_2_category} article: {{article}}"


def collect_feedback(text: str, copy_to_clipboard: bool = True):
    # ignore manual feedback collection loop where take feedback from input()
    if not manual_feed_collect:
        return
    if copy_to_clipboard:
        copy2clip(text)
    return input("Response:")


def extract_info_LaMP_1(
    questions: List[Dict[str, str | List]],
    tokenizer: Callable[[str], str] = lemma_tokenizer,
    bm25_top_k=5,
    **params,
) -> Dict[str, str]:
    """
    Rank documents by using specified author title as query and abstracts of documents as text,
    Extract key term within each text,
    Prepare Prompt given relevance to title, curr document title, and current key terms

    Return a dictionary with question id as key and prompt as value
    """
    prompt_collection = dict()
    for question in questions:
        _, author_title, _, title_opt1, _, title_opt2, *_ = question["input"].split('"')
        curr_prompt = [
            f"Here are the documents ranked by relevance, with titles and keywords, from most relevant to least relevant, for the topic of '{author_title}':"
        ]
        ranker = BM25(corpus=question["profile"])
        rel_sequence = ranker.get_top_n(
            tokenizer(author_title), n=bm25_top_k, sequence_only=True
        )
        for rel_doc_index in rel_sequence:
            # id, title, abstract
            curr_profile = ranker.corpus[rel_doc_index]
            tokens = tokenizer(curr_profile[2])
            keywords = text_rank(tokens)
            curr_prompt.append(
                f"title: \"{curr_profile[1]}\" with keywords: [{', '.join(keywords)}]"
            )
        curr_prompt.append(
            task_1_template.format(
                author_title=author_title, title_opt1=title_opt1, title_opt2=title_opt2
            )
        )

        collect_feedback(curr_prompt)

        prompt_collection[question["id"]] = curr_prompt
    return prompt_collection


def extract_info_LaMP_2(
    questions: List[Dict[str, str | List]],
    tokenizer: Callable[[str], str] = lemma_tokenizer,
    **params,
) -> Dict[str, str]:
    """
    Extract key term for each text,
    Prepare Prompt given category, key terms, and title

    Return a dictionary with question id as key and prompt as value
    """
    prompt_collection = dict()
    for question in questions:
        *_, curr_article = question["input"].split("article: ")
        curr_prompt = [
            f"Here are the documents labeled by category, with titles and keywords:"
        ]
        # ranker = BM25(corpus=question["profile"])
        # rel_sequence = ranker.get_top_n(tokenizer(author_title), n = bm25_top_k, sequence_only=True)
        for curr_profile in question["profile"]:
            tokens = tokenizer(curr_profile["text"])
            keywords = text_rank(tokens)
            curr_prompt.append(
                f"title: \"{curr_profile['title']}\" with keywords: [{', '.join(keywords)}] and category: \"{curr_profile['category']}\""
            )
        curr_prompt.append(task_2_template.format(article=curr_article))

        collect_feedback(curr_prompt)

        prompt_collection[question["id"]] = curr_prompt
    return prompt_collection


def extract_PPEP_LaMP_2_alt(
    question: Dict[str, str | List], tokenizer: Callable[[str], str] = lemma_tokenizer
) -> Dict[str, Set[str]]:
    categories_txt: Dict[str, Set[str]] = defaultdict(set)
    for curr_profile in question["profile"]:
        curr_category = curr_profile["category"]
        curr_tokens = tokenizer(curr_profile["text"])
        curr_keywords = text_rank(curr_tokens)
        categories_txt[curr_category].update(curr_keywords)
    return categories_txt


def extract_info_LaMP_2_alt(
    questions: List[Dict[str, str | List]],
    tokenizer: Callable[[str], str] = lemma_tokenizer,
    **params,
) -> Dict[str, str]:
    """
    Group text with the same category,
    Extract key term within each text,
    Add those term together as additional category information,
    Prepare Prompt given category and key terms (discard title/id info)

    Return a dictionary with question id as key and prompt as value
    """
    prompt_collection = dict()
    for question in questions:
        *_, curr_article = question["input"].split("article: ")

        curr_prompt = [
            f"Here are the keywords associated with different category, with titles and keywords:"
        ]
        curr_category_keywords_map = extract_PPEP_LaMP_2_alt(
            question=question, tokenizer=tokenizer
        )
        for category, keywords in curr_category_keywords_map.items():
            curr_prompt.append(
                f"category: \"{category}\" has keywords: \"[{', '.join(keywords)}]\""
            )
        curr_prompt.append(task_2_template.format(article=curr_article))

        collect_feedback(curr_prompt)

        prompt_collection[question["id"]] = curr_prompt
    return prompt_collection


def extract_labels(outputs: Dict[str, str | List], **params) -> Dict[str, str]:
    pass


def parse_dataset_to_prompt(filename: str, task_name: str, **params) -> Dict[str, str]:
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Dataset File <{filename}> not Found.")
    with open(filename, "r", encoding="utf-8") as file:
        contents = json.load(file)
        if not str(DatasetType.data) in filename:
            return extract_labels(contents, **params)
        if task_name == "LaMP_1":
            return extract_info_LaMP_1(contents, **params)
        else:
            return extract_info_LaMP_2_alt(contents, **params)
