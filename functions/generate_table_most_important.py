# import prettytable as PrettyTable
from prettytable import PrettyTable


def generate_table_most_important(words_list: list,
                                  bow_dict: dict, title: str):
    pretty_result = PrettyTable()
    keys = []
    values = []

    pretty_result.field_names = ['Term', 'Count']

    keys = words_list

    for i in words_list:
        values.append(bow_dict[i])

    pretty_result.title = f'Most important tokens based on vectorizer: {title}'

    for k, v in zip(keys, values):
        pretty_result.add_row([k, v])
    print(pretty_result)
