# Two comparable tasks:
#   1): Direct keywords extraction with title (no ranking on associated article).
#   2): Keywords extraction with title with ranking where ranks the author's profile
#           with respect to abstract of profile with specified title. In the meantime,
#           mention such rank order in the prompt.
from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from typing import Callable, Dict, List, Set


from src.models.BM25 import BM25Okapi as BM25

from src.models.TextRank import text_rank

from src.utils import (
    DatasetType,
    copy2clip,
    default_top_k_keywords,
    task_2_categories,
    task_2_categories_str,
)
from src.tokenization import lemma_tokenizer


manual_feed_collect = False

task_1_options = '1 is "{title_opt1}", 2 is "{title_opt2}"'
task_1_template = 'Given above information, for an author who has written the paper with the title "{author_title}", which reference is related? Just choose 1 or 2 without further explanation.'


task_2_template = "Given the above information, which category does the following article relate to? Just answer with the category name without further explanation."

task_2_category_options = task_2_categories_str

task_2_article_with_keyword = 'Article: "{article}". With keywords: {article_keywords}'

task_2_article_no_keyword = 'Article: "{article}".'


def collect_feedback(text: str, copy_to_clipboard: bool = True):
    # ignore manual feedback collection loop where take feedback from input()
    if not manual_feed_collect:
        return
    if copy_to_clipboard:
        copy2clip(text)
    return input("Response:")


def extract_instance_info_LaMP_1(
    question: Dict[str, str | List],
    tokenizer: Callable[[str], str] = lemma_tokenizer,
    keyword_extraction: bool = True,
    bm25_top_k: int = default_top_k_keywords,
    text_rank_top_k_keywords: int = default_top_k_keywords,
    **params,
):
    if not keyword_extraction:
        return extract_instance_info_LaMP_1_no_keyword(
            question=question,
            tokenizer=tokenizer,
            bm25_top_k=bm25_top_k,
        )
    # if q_index >= limit: break
    _, author_title, _, title_opt1, _, title_opt2, *_ = question["input"].split('"')
    curr_prompt = [
        f"Here are the documents ranked by relevance, with titles and keywords, from most relevant to least relevant, for the topic of '{author_title}':"
    ]
    ranker = BM25(corpus=question["profile"], task_name="LaMP_1")
    rel_sequence = ranker.get_top_n(
        tokenizer(author_title), n=bm25_top_k, sequence_only=True
    )
    for rel_doc_index in rel_sequence:
        # id, title, abstract
        curr_profile = ranker.corpus[rel_doc_index]
        title = curr_profile[1]
        abstract = curr_profile[2]
        tokens = tokenizer(abstract)
        if len(tokens) <= 2:
            continue
        keywords = text_rank(tokens, top_k=text_rank_top_k_keywords)
        curr_prompt.append(f"Title: \"{title}\" with keywords: [{', '.join(keywords)}]")

    curr_prompt.append(
        task_1_options.format(title_opt1=title_opt1, title_opt2=title_opt2)
    )
    curr_prompt.append(
        task_1_template.format(
            author_title=author_title,
        )
    )

    collect_feedback(curr_prompt)
    return curr_prompt


def extract_instance_info_LaMP_1_no_keyword(
    question: Dict[str, str | List],
    tokenizer: Callable[[str], str] = lemma_tokenizer,
    bm25_top_k: int = 1,
    **params,
) -> str:
    _, author_title, _, title_opt1, _, title_opt2, *_ = question["input"].split('"')

    curr_prompt = [
        f"Here are the documents ranked by relevance, with titles and keywords, from most relevant to least relevant, for the topic of '{author_title}':"
    ]
    ranker = BM25(corpus=question["profile"], task_name="LaMP_1")
    rel_sequence = ranker.get_top_n(
        tokenizer(author_title), n=bm25_top_k, sequence_only=True
    )
    for rel_doc_index in rel_sequence:
        # id, title, abstract
        _, title, abstract = ranker.corpus[rel_doc_index]
        curr_prompt.append(f'Title: "{title}" with abtract: "{abstract}"')

    curr_prompt.append(
        task_1_options.format(title_opt1=title_opt1, title_opt2=title_opt2)
    )
    curr_prompt.append(
        task_1_template.format(
            author_title=author_title,
        )
    )

    collect_feedback(curr_prompt)
    return curr_prompt


def extract_instance_info_LaMP_1_alt(
    question: Dict[str, str | List],
    tokenizer: Callable[[str], str] = lemma_tokenizer,
    keyword_extraction: bool = True,
    bm25_top_k: int = default_top_k_keywords,
    text_rank_top_k_keywords: int = default_top_k_keywords,
    context_top_k_keywords: int = default_top_k_keywords,
    **params,
):
    if not keyword_extraction:
        return extract_instance_info_LaMP_1_alt_no_keyword(
            question=question,
            tokenizer=tokenizer,
            bm25_top_k=bm25_top_k,
        )
    # if q_index >= limit: break
    _, author_title, _, title_opt1, _, title_opt2, *_ = question["input"].split('"')
    curr_prompt = [f"Here are the related keywords for the topic of '{author_title}':"]
    # add title to abstract as content of BM25
    ranker = BM25(corpus=question["profile"], task_name="LaMP_1_alt")
    rel_sequence = ranker.get_top_n(
        tokenizer(author_title), n=bm25_top_k, sequence_only=True
    )
    keywords_collection = Counter()
    for rel_doc_index in rel_sequence:
        # id, title + abstract
        _, document = ranker.corpus[rel_doc_index]
        tokens = tokenizer(document)
        if len(tokens) <= 2:
            continue
        keywords = text_rank(tokens, top_k=text_rank_top_k_keywords)
        # add related keywords together
        keywords_collection.update(keywords)

    keywords = [
        token for token, _ in keywords_collection.most_common(context_top_k_keywords)
    ]
    curr_prompt.append(f"[{', '.join(keywords)}]")
    curr_prompt.append(
        task_1_options.format(title_opt1=title_opt1, title_opt2=title_opt2)
    )
    curr_prompt.append(
        task_1_template.format(
            author_title=author_title,
        )
    )

    collect_feedback(curr_prompt)
    return curr_prompt


def extract_instance_info_LaMP_1_alt_no_keyword(
    question: Dict[str, str | List],
    tokenizer: Callable[[str], str] = lemma_tokenizer,
    bm25_top_k: int = 1,
    **params,
) -> str:
    _, author_title, _, title_opt1, _, title_opt2, *_ = question["input"].split('"')

    curr_prompt = [
        f"Here are the related document ranked by relevance, from most relevant to least relevant, for the topic of '{author_title}':"
    ]
    ranker = BM25(corpus=question["profile"], task_name="LaMP_1_alt")
    rel_sequence = ranker.get_top_n(
        tokenizer(author_title), n=bm25_top_k, sequence_only=True
    )
    for rel_doc_index in rel_sequence:
        # id, title, abstract
        _, document = ranker.corpus[rel_doc_index]
        curr_prompt.append(f'Document: "{document}"')

    curr_prompt.append(
        task_1_options.format(title_opt1=title_opt1, title_opt2=title_opt2)
    )
    curr_prompt.append(
        task_1_template.format(
            author_title=author_title,
        )
    )

    collect_feedback(curr_prompt)
    return curr_prompt


def extract_info_LaMP_1(
    questions: List[Dict[str, str | List]],
    tokenizer: Callable[[str], str] = lemma_tokenizer,
    keyword_extraction: bool = True,
    bm25_top_k: int = default_top_k_keywords,
    text_rank_top_k_keywords: int = default_top_k_keywords,
    article_top_k: int = 1,
    **params,
) -> Dict[str, str]:
    """
    Rank documents by using specified author title as query and abstracts of documents as text,
    Extract key term within each text,
    Prepare Prompt given relevance to title, curr document title, and current key terms

    Return a dictionary with question id as key and prompt as value
    """

    prompt_collection = dict()
    for q_index, question in enumerate(questions):
        prompt_collection[question["id"]] = extract_instance_info_LaMP_1(
            question=question,
            tokenizer=tokenizer,
            keyword_extraction=keyword_extraction,
            text_rank_top_k_keywords=text_rank_top_k_keywords,
            bm25_top_k=bm25_top_k,
            article_top_k=article_top_k,
        )

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
        curr_prompt.append(task_2_template.format(article=curr_article))

        collect_feedback(curr_prompt)

        prompt_collection[question["id"]] = curr_prompt
    return prompt_collection


def extract_PPEP_LaMP_2_alt(
    question: Dict[str, str | List],
    tokenizer: Callable[[str], str] = lemma_tokenizer,
    keyword_extraction: bool = True,
    text_rank_top_k_keywords: int = default_top_k_keywords,
) -> Dict[str, Counter[str]] | Dict[str, list[Dict]]:
    categories_txt: Dict[str, list[str]] = defaultdict(list)
    if keyword_extraction:
        categories_txt: Dict[str, Counter[str]] = defaultdict(Counter)

    for curr_profile in question["profile"]:
        curr_category = curr_profile["category"]
        if keyword_extraction:
            curr_tokens = tokenizer(curr_profile["text"])
            curr_keywords = text_rank(curr_tokens, top_k=text_rank_top_k_keywords)
            categories_txt[curr_category].update(curr_keywords)
        else:
            categories_txt[curr_category].append(curr_profile)

    return categories_txt


def extract_instance_info_LaMP_2_alt(
    question: Dict[str, str | List],
    tokenizer: Callable[[str], str] = lemma_tokenizer,
    keyword_extraction: bool = True,
    text_rank_top_k_keywords: int = default_top_k_keywords,
    category_top_k_keywords: int = default_top_k_keywords,
    article_top_k: int = 1,
    **params,
) -> str:
    if not keyword_extraction:
        return extract_instance_info_LaMP_2_alt_no_keyword(
            question=question, tokenizer=tokenizer, article_top_k=article_top_k
        )
    *_, curr_article = question["input"].split("article: ")

    curr_prompt = [
        "Answer following question:",
        "Keywords associated with different categories:",
    ]
    curr_category_keywords_map = extract_PPEP_LaMP_2_alt(
        question=question, tokenizer=tokenizer, keyword_extraction=True
    )
    for category, keywords in curr_category_keywords_map.items():
        keywords = [token for token, _ in keywords.most_common(category_top_k_keywords)]
        curr_prompt.append(
            f"Category: \"{category}\" has keywords: \"[{', '.join(keywords)}]\""
        )
    curr_article_keywords = text_rank(
        tokenizer(curr_article), top_k=text_rank_top_k_keywords
    )

    curr_prompt.append(task_2_template)
    curr_prompt.append(task_2_category_options)

    if curr_article.endswith(".") or curr_article.endswith(","):
        curr_article = curr_article[:-1]
    curr_prompt.append(
        task_2_article_with_keyword.format(
            article=curr_article, article_keywords=curr_article_keywords
        )
    )

    collect_feedback(curr_prompt)
    return curr_prompt


def extract_instance_info_LaMP_2_alt_no_keyword(
    question: Dict[str, str | List],
    tokenizer: Callable[[str], str] = lemma_tokenizer,
    article_top_k: int = 1,
    **params,
) -> str:
    *_, curr_article = question["input"].split("article: ")

    curr_prompt = [
        "Answer following question:",
        "Here are top article associated with different categories",
    ]
    curr_category_article_map = extract_PPEP_LaMP_2_alt(
        question=question, tokenizer=tokenizer, keyword_extraction=False
    )
    for category, profiles in curr_category_article_map.items():
        ranker = BM25(corpus=profiles, task_name="LaMP_2")
        rel_sequence = ranker.get_top_n(
            tokenizer(category), n=article_top_k, sequence_only=True
        )
        best_articles = []
        for article_index in rel_sequence:
            _, curr = ranker.corpus[article_index]
            best_articles.append(curr)
        if len(best_articles) == 1:
            curr_prompt.append(
                f'Category: "{category}" has article: "{best_articles[0]}"'
            )
        else:
            curr_line = f'Category: "{category}" has articles:'
            for article in best_articles:
                article: str
                if article.endswith(".") or article.endswith(","):
                    article = article[:-1]
                curr_line += f' "{article}",'
            curr_line = curr_line[:-1] + "."
            curr_prompt.append(curr_line)

    curr_prompt.append(task_2_template)
    curr_prompt.append(task_2_category_options)

    if curr_article.endswith(".") or curr_article.endswith(","):
        curr_article = curr_article[:-1]
    curr_prompt.append(task_2_article_no_keyword.format(article=curr_article))

    collect_feedback(curr_prompt)
    return curr_prompt


def extract_info_LaMP_2_alt(
    questions: List[Dict[str, str | List]],
    tokenizer: Callable[[str], str] = lemma_tokenizer,
    keyword_extraction: bool = True,
    text_rank_top_k_keywords: int = default_top_k_keywords,
    category_top_k_keywords: int = default_top_k_keywords,
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
        prompt_collection[question["id"]] = extract_instance_info_LaMP_2_alt(
            question=question,
            tokenizer=tokenizer,
            keyword_extraction=keyword_extraction,
            text_rank_top_k_keywords=text_rank_top_k_keywords,
            category_top_k_keywords=category_top_k_keywords,
        )
    return prompt_collection


def extract_labels(outputs: Dict[str, str | List], **params) -> Dict[str, str]:
    pass


def parse_dataset_to_prompt(
    filename: str,
    task_name: str,
    keyword_extraction: bool,
    **params,
) -> Dict[str, str]:
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Dataset File <{filename}> not Found.")
    with open(filename, "r", encoding="utf-8") as file:
        contents = json.load(file)
        if not str(DatasetType.data) in filename:
            return extract_labels(contents, **params)
        if task_name == "LaMP_1":
            return extract_info_LaMP_1(
                contents,
                keyword_extraction=keyword_extraction,
                **params,
            )
        else:
            return extract_info_LaMP_2_alt(
                contents,
                keyword_extraction=keyword_extraction,
                **params,
            )
