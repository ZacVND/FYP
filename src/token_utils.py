from geniatagger import GENIATagger
from nltk.corpus import stopwords
from take_abstract import *
import preprocessing as pp
from enum import Enum
import feature as ft
from os import path
import numpy as np
import util
import sys
import bs4

script_dir = path.dirname(path.abspath(__file__))

G_WORD = 0
G_BASE_FORM = 1
G_POS_TAG = 2
G_CHUNK = 3
G_NAMED_ENTITY = 4


class EvLabel(Enum):
    __order__ = 'A1 A2 R1 R2 OC P'
    A1 = 0
    A2 = 1
    R1 = 2
    R2 = 3
    OC = 4
    P = 5


class EvLabelData:
    def __init__(self, word):
        self.word = word


class Token:
    def __init__(self, word, g_tags=[]):
        self.word = word.lower()
        self.og_word = word
        self.g_tags = g_tags

        self.ev_label = None
        self.predicted_ev_label = None

    def set_ev_label(self, ev_label):
        self.ev_label = ev_label


# genia tagger instance
executable_path = path.join(script_dir, "..", "geniatagger-3.0.2",
                            "geniatagger")
tagger = GENIATagger(executable_path)

xml_tag_to_ev_label = {
    'a1': EvLabel.A1,
    'a2': EvLabel.A2,
    'r1': EvLabel.R1,
    'r2': EvLabel.R2,
    'oc': EvLabel.OC,
    'p': EvLabel.P,
}
stopword_trie = util.Trie(strings=(stopwords.words('english') + ['non']))


class TokenCollection:
    def __init__(self, bs_doc):
        self.bs_doc = bs_doc

        self.tokens = None
        self.ev_labels = {}

        self.feature_vectors = None
        self.labels = None

    def normalize(self):

        # Tokenizing
        tokens = []
        abstract = self.bs_doc.abstract
        for abs_text in abstract.children:
            if isinstance(abs_text, bs4.NavigableString):
                continue  # Skip blank new lines
            for child in abs_text.children:
                xml_tag = child.name
                label = xml_tag_to_ev_label.get(xml_tag)
                if label is not None:
                    # Dealing with specific tags
                    tags = list(tagger.tag(child.text))
                    token = Token(child.text, g_tags=tags)
                    token.set_ev_label(label)
                    tokens.append(token)

                    self.ev_labels[label] = EvLabelData(token.word)
                else:
                    # Dealing with text
                    if not isinstance(child, bs4.NavigableString):
                        continue  # We only want navigable strings
                    tags = tagger.tag(str(child))
                    for tag in tags:
                        token = Token(tag[0], g_tags=[tag])
                        tokens.append(token)

        # Removing stopwords
        # TODO: Replace token list with linked list for quick removal
        tokens = [x for x in tokens
                  if not stopword_trie.check(x.word)]

        # NOTE: Token count should not change after this point!!
        self.tokens = tokens

    def generate_feature_matrix(self):
        token_count = len(self.tokens)
        feature_count = len(ft.Feature.__members__.items())
        feature_vectors = np.zeros((token_count, feature_count + 1))

        # Set bias term to 1
        feature_vectors[:, feature_count] += 1

        title = take_title(self.bs_doc.pmid.text)
        if title is None:
            print('Cannot fetch the title, assuming no title.')
        else:
            title_words = pp.normalise_sentence(title)
            title_trie = util.Trie(strings=title_words)

        for token_i in range(len(self.tokens)):
            token = self.tokens[token_i]

            # extracting features from UMLS class
            feature_class_map = ft.get_feature_classes(token.word)
            for feature_i, val in feature_class_map.items():
                feature_vectors[token_i, int(feature_i)] = val
            # sys.stdout.write(
            #     'UMLS Progress {}/{}\r'.format(token_i + 1, token_count))
            # sys.stdout.flush()

            # TODO: Add code to assign values to more features
            # g_tag is a list of tuples
            # in title feature
            if title is not None:
                in_title = title_trie.check(token.word)
                if in_title:
                    feature_vectors[token_i, ft.Feature.IS_IN_TITLE.value] = 1

            # is beginning of noun phrase
            if token.g_tags[0][G_CHUNK] == 'B-NP':
                feature_vectors[token_i, ft.Feature.IS_BNP.value] = 1

            # is in patient
            for tag in token.g_tags:
                result = ft.patient_dict_trie.check(tag[G_BASE_FORM])
                if result:
                    feature_vectors[token_i, ft.Feature.IS_IN_PDICT.value] = 1

        util.umls_cache.save()
        self.feature_vectors = feature_vectors

        return feature_vectors

    def generate_train_labels(self):
        token_count = len(self.tokens)

        # Generate label vectors
        a_labels = -1 * np.ones((token_count,))
        r_labels = -1 * np.ones((token_count,))
        oc_labels = -1 * np.ones((token_count,))
        p_labels = -1 * np.ones((token_count,))

        for token_i in range(len(self.tokens)):
            token = self.tokens[token_i]
            ev_label = token.ev_label
            if ev_label in [EvLabel.A1, EvLabel.A2]:
                a_labels[token_i] = 1
            if ev_label in [EvLabel.R1, EvLabel.R2]:
                r_labels[token_i] = 1
            if ev_label == EvLabel.OC:
                oc_labels[token_i] = 1
            if ev_label == EvLabel.P:
                p_labels[token_i] = 1

        labels = {
            'a': a_labels,
            'r': r_labels,
            'oc': oc_labels,
            'p': p_labels,
        }

        self.labels = labels
        return labels
