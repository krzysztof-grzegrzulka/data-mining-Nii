import ssl

import nltk

import functions.sanitize_text
import functions.stop_words

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

source_txt = 'Lorem ipsum dolor :) sit amet, 12 consectetur; ' \
             'adipiscing elit. Sed eget mattis sem. ;) 15' \
             'Mauris ;( <div><h2>egestas erat quam, :< ut faucibus ' \
             'eros congue :> et. </div></h2>In blandit, mi eu porta; ' \
             'lobortis, tortor :-) <a href="">nisl facilisis leo</a>, at ;< ' \
             'tristique augue risus eu risus ;-).'

print(functions.sanitize_text.sanitize_text(source_txt))

nltk.download('stopwords')
nltk.download('punkt')
print(functions.stop_words.stop_words(source_txt))
