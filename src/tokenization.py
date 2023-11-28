from multiprocessing import Pool, cpu_count
import string
from typing import Callable, List

import nltk
from nltk.corpus import stopwords

stemmer = nltk.stem.PorterStemmer()
stemmer_pipeline: Callable[[str], str] = lambda x: stemmer.stem(x)
lemmatizer = nltk.stem.WordNetLemmatizer()
lemmatizer_pipeline: Callable[[str], str] = lambda x: lemmatizer.lemmatize(x)
stop_words = set(stopwords.words("english"))


def tokenize_corpus(corpus: List[str], tokenizer: Callable[[str], str]):
    tokenizer_pool = Pool(cpu_count())
    return tokenizer_pool.map(tokenizer, corpus)


def tokenize(text: str, method: Callable[[str], str]) -> list[str]:
    tokens = nltk.word_tokenize(text.lower())
    # remove punctuation and stop words and lemmatize/stem token one by one, also remove token with less than 3 characters
    return [
        method(token)
        for token in tokens
        if token not in string.punctuation
        and token not in stop_words
        and len(token) >= 3
    ]


stem_tokenizer: Callable[[str], str] = lambda x: tokenize(x, stemmer_pipeline)
lemma_tokenizer: Callable[[str], str] = lambda x: tokenize(x, lemmatizer_pipeline)
