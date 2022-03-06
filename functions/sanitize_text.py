import re


def sanitize_text(source_text: str):
    get_emoticons = re.findall(r'[:;]{1}-?[)(<>]{1}', source_text)

    output = re.sub(' +', ' ',
                    re.sub(r'[,.;]',
                           '', re.sub(r'<\w*\s*\s*/*\w*[=\"]*>',
                                      '', re.sub(r'\d', '',
                                                 re.sub(r'[:;]{1}-?[)(<>]{1}',
                                                        '', source_text.lower()
                                                        ))))) + ' '.join(
                                                        get_emoticons)

    # print(text_with_emoticons_added_back)
    # print(output)
    return output
