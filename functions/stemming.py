from nltk.stem import PorterStemmer


def stemming(source_text: list) -> list:
    output_list = []
    porter = PorterStemmer()
    for word in source_text:
        output_list.append(porter.stem(word))

    return output_list
