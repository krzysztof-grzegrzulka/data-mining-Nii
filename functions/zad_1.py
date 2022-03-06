import re


def zad_1():
    delete_numbers = 'Dzisiaj mamy 4 stopnie na plusie, 1 marca 2022 roku'
    delete_html_tags = '<div><h2>Header</h2> <p>article<b>strong text</b> ' \
                       '<a href="">link</a></p></div>'
    delete_punctuation = 'Lorem ipsum dolor sit amet, consectetur; ' \
                         'adipiscing elit. Sed eget mattis sem. ' \
                         'Mauris egestas erat quam, ut faucibus ' \
                         'eros congue et. In blandit, ' \
                         'mi eu porta; lobortis, tortor nisl ' \
                         'facilisis leo, at tristique augue risus eu risus.'

    numbers_deleted = re.sub(r'\d', '', delete_numbers)
    html_tags_deleted = re.sub(r'<\w*\s*\s*/*\w*[=\"]*>', '', delete_html_tags)
    punctuation_deleted = re.sub(r'[,.;]', '', delete_punctuation)

    print(numbers_deleted)
    print(html_tags_deleted)
    print(punctuation_deleted)
