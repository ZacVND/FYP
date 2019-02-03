"""
Created on Dec 2018

THIS FILE IS FOR TESTING PREPROCESSING FUNCTIONS

@author: Zac Luong
"""
from nltk.corpus import stopwords
import util
import nltk
import json
import os
import re

sw = stopwords.words('english') + ['non']
script_dir = os.path.dirname(os.path.abspath(__file__))
stopword_trie = util.Trie(strings=(stopwords.words('english') + ['non']))

abbrev_path = os.path.join(script_dir, os.pardir, "data", "abbreviations.json")
with open(abbrev_path, 'r') as file:
    abbrev_dict = json.load(file)


# lemmatiser instance
# lem = nltk.WordNetLemmatizer()

# genia tagger instance
# executable_path = os.path.join(".", "geniatagger-3.0.2", "geniatagger")
# tagger = GENIATagger(executable_path)

# Normalisation

def normalise_sentence(sentence):
    '''
    given string 'sentence' expands abbreviations and substitutes patterns
    with normalisation tags
    '''
    # sent = expand_abbreviations(sentence)
    sent = nltk.word_tokenize(sentence)
    # sent_no_stopword = [w for w in sent if not w in sw]

    return sent


# Abbreviation expansion

def abbreviations(text):
    """
    finds all the abbreviations in text and returns them as a dictionary
    with the abbreviations as keys and expansions as values
    """
    tokens = nltk.word_tokenize(text)
    sent_out = str(text)
    foo = {}
    obrack_pt = re.compile(r'[\(\[]')
    cbrack_pt = re.compile(r's?[\)\]]')

    for i, t in enumerate(tokens[:-2]):
        tests = (obrack_pt.search(t) and tokens[i + 1].isupper()
                 and cbrack_pt.search(tokens[i + 2])
                 and not '=' in tokens[i + 1])
        if tests:
            foo[tokens[i + 1]] = [w.title() for w in tokens[:i]]
            sent_out = re.sub(r'[\(\[]' + tokens[i + 1] + '[\)\]]' + ' ', '',
                              sent_out)

    for a in foo.keys():
        candidates = []
        for i, w in enumerate(reversed(foo[a])):
            if i > len(a) + 1: break
            condition = (i > (len(a) - 3) if len(
                [1 for l in a if l == a[0]]) > 1 else True)
            if condition and w.lower().startswith(a[0].lower()) and \
                    stopword_trie.check(w) is None:
                candidates.append(foo[a][-(i + 1):])
        # sort the keys in a by ascending order of length
        candidates.sort(key=lambda x: len(x))
        foo[a] = (candidates[0] if candidates else [])

    return [foo, sent_out]


def expand_abbreviations(sent):
    """
    returns a copy of 'sent' with abbreviations expanded
    """

    [abbrev_new, sent_new] = abbreviations(sent)
    abbrev_dict.update({k: v for (k, v) in abbrev_new.items()
                        if v})
    # sort the keys in abbrev_dict in descending order of length
    keys = sorted(abbrev_dict.keys(), key=len, reverse=True)

    for k in keys:
        neww = (' '.join(abbrev_dict[k]) if type(abbrev_dict[k]) is list
                else abbrev_dict[k])

        if re.search(k, sent_new):
            next_i = re.search(k, sent_new).span()[1]
            if sent[next_i] is not ' ':
                neww += ' '
            sent_new = (re.sub(k, neww, sent_new)
                        if neww else sent_new)

    return sent_new


# Normalisation

if __name__ == "__main__":
    text = "Diurnal curves of intraocular pressure (IOP) were performed on the " \
           "baseline day and after 0.5, 3, 6, and 12 months of treatment. " \
           "The IOP measurements were performed at 8:00 AM, 12:00 noon, " \
           "4:00 PM, and 8:00 PM."

    text_2 = "Latanoprost is a PGF2 alpha analogue which reduces the intraocular pressure (IOP) by increasing the uveoscleral outflow."

    sent_new = expand_abbreviations(text_2)

    print(sent_new)

    pass
    # data_path = os.path.join(cur_dir, 'data/annotation I/')
    # for fn in os.listdir(data_path):
    #     with open(os.path.join(data_path, fn)) as f:
    #         raw = f.read()
    #         soup = bs4.BeautifulSoup(raw, "html5lib")
    #
    #     title = take_title(soup.pmid.text)
    #     for word, base_form, pos_tag, chunk, named_entity in tagger.tag(
    #             title.text):
    #         print("{:20}{:20}{:10}{:10}{:10}".format(word, base_form, pos_tag,
    #                                                  chunk, named_entity))
    #
    #     break
