import numpy as np


def top_x_tokens(tokens_list: list, token_words: list,
                 top_x: int = 10) -> list:
    temp_list = tokens_list.copy()
    top_x_result = []
    for i in range(top_x):
        token_index = np.argmax(temp_list)
        top_x_result.append(token_words[token_index])
        temp_list[token_index] = 0
    return top_x_result
