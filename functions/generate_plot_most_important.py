import numpy as np
from matplotlib import pyplot as plt


def generate_plot_most_important(words_list: list, bow_dict: dict, title: str):
    keys = words_list[::-1]
    values = []

    for i in words_list:
        values.append(bow_dict[i])

    values = values[::-1]

    y = np.arange(len(keys))

    fig, axes = plt.subplots()

    axes.barh(y, values, align='center')
    axes.set_yticks(y, labels=keys)
    axes.set_title(f'Most important tokens based on vectorizer: {title}')

    plt.show()
