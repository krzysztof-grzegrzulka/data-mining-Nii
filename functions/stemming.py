from nltk.stem import PorterStemmer


def stemming(source_text: str) -> list:
    output_list = []
    porter = PorterStemmer()
    for word in list(filter((2).__ne__, source_text.split(','))):
        output_list.append(porter.stem(word))

    return output_list
