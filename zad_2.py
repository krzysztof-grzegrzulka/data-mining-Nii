import re


def zad_2():
    text_with_hashtags = 'Lorem ipsum dolor sit amet, consectetur adipiscing' \
                         ' elit. Sed #texting eget mattis sem. Mauris ' \
                         '#frasista egestas erat #tweetext quam, ut faucibus' \
                         ' eros #frasier congue et. In blandit, mi eu porta' \
                         'lobortis, tortor nisl facilisis leo, at tristique ' \
                         '#frasistas augue risus eu risus.'

    get_hashtags = re.findall(r'#', text_with_hashtags)

    print(get_hashtags)
