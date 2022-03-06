import re


def zad_3():
    text_wtih_emoticons = 'Lorem ipsum dolor :) sit amet, consectetur; ' \
                          'adipiscing elit. Sed eget mattis sem. ;) ' \
                          'Mauris ;( egestas erat quam, :< ut faucibus ' \
                          'eros congue :> et. In blandit, mi eu porta; ' \
                          'lobortis, tortor :-) nisl facilisis leo, at ;< ' \
                          'tristique augue risus eu risus ;-).'

    get_emoticons = re.findall(r'[:;]{1}-?[)(<>]{1}', text_wtih_emoticons)

    print(get_emoticons)
