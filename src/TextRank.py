from typing import List, Tuple

import numpy as np
from src.util import default_text_rank_window_size, default_top_k_keywords

np.random.seed(0)


def text_rank_init(
    tokens: List[str], window_size: int = default_text_rank_window_size
) -> Tuple[np.ndarray, List[str], int]:
    """
    Create adjacent matrix for terms in text

    Consider token in the window has bi-directional connection
    """
    vocab_tracker = dict()
    index_vocab_tracker = 0
    id_vocab = list()
    term_pairs = list()

    windowed_pairs = zip(*[tokens[start_index:] for start_index in range(window_size)])
    for pair in windowed_pairs:
        term_pairs.append(pair)
        for curr_term in pair:
            if curr_term not in vocab_tracker:
                vocab_tracker[curr_term] = index_vocab_tracker
                index_vocab_tracker += 1
                id_vocab.append(curr_term)
    t_mat = np.zeros((index_vocab_tracker, index_vocab_tracker), dtype=int)
    for term1, term2 in term_pairs:
        t_mat[vocab_tracker[term1]][vocab_tracker[term2]] = 1
        t_mat[vocab_tracker[term2]][vocab_tracker[term1]] = 1
    row_sums = np.sum(t_mat, axis=1, keepdims=True)
    norm_mat = t_mat / row_sums
    return norm_mat, id_vocab, index_vocab_tracker


def text_rank(
    tokens: List[str],
    top_k: int = default_top_k_keywords,
    alpha: float = 0.2,
    iteration: int = 1000,
    tolerance: float = 1e-32,
) -> List[str]:
    adjacency_matrix, id_vocab, len_vocab = text_rank_init(tokens=tokens)
    n = len_vocab
    p_curr = np.random.rand(n, 1)
    p_curr = p_curr / p_curr.sum()
    pre_cal_first_term = alpha / n
    for i in range(iteration):
        left_term = (pre_cal_first_term + (1 - alpha) * adjacency_matrix).T
        p_next = np.matmul(left_term, p_curr)
        if i != 0 and (np.abs(np.sum(p_next - p_curr)) / n) <= tolerance:
            break
        p_curr = p_next
    flat_p = np.squeeze(p_next)
    ordered_indice = np.argsort(flat_p)[::-1][:top_k]
    term_weights = [id_vocab[index] for index in ordered_indice]
    # term_weights = [(id_vocab[index], flat_p[index]) for index in ordered_indice]
    return term_weights
