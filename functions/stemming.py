from nltk.stem import PorterStemmer


def stemming(source_text: str) -> list:
    output_list = []
    porter = PorterStemmer()
    source_text = source_text.split(' ')
    for word in source_text:
        output_list.append(porter.stem(word))

    return output_list
