"""
Created on Dec 2018

@author: Zac Luong
"""
import nltk, re, bs4, json, pickle, os
from geniatagger import GENIATagger
from nltk.corpus import stopwords
from Authentication import *
from take_abstract import *

sw = stopwords.words('english') + ['non']
cur_dir = os.path.dirname(os.path.abspath(__file__))

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
    sent_no_stopword = [w for w in sent if not w in sw]

    return sent_no_stopword


def get_arms(title, objective):
    """
    Using the title and the objective/background sentence to get the
    intervention arms. By finding the common medical terms in both sentences, a1
    is always mentioned before a2

    :param title: The title sentence
    :param objective: The objective sentence
    :return: a1, a2
    """

    pass


def get_semantic_class(chunk, debug=False):
    """
    returns the semantic class of chunk, where chunk is in string format
    """
    pass


# Abbreviation expansion

def abbreviations(text):
    """
    finds all the abbreviations in text and returns them as a dictionary
    with the abbreviations as keys and expansions as values
    """
    pass


def expand_abbreviations(sent):
    """
    returns a copy of 'sent' with abbreviations expanded
    """
    pass


# Normalisation

if __name__ == "__main__":
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
