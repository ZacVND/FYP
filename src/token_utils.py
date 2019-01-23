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


class Label(Enum):
    # TODO: Add more features
    A1 = 0
    A2 = 1
    R1 = 2
    R2 = 3
    OC = 4
    P = 5


class Token:
    def __init__(self, word, g_tags=[]):
        self.word = word.lower()
        self.og_word = word
        self.g_tags = g_tags

        self.label = None

    def set_label(self, label):
        self.label = label


# genia tagger instance
executable_path = path.join("..", "geniatagger-3.0.2", "geniatagger")
tagger = GENIATagger(executable_path)

xml_tag_to_label = {
    'a1': Label.A1,
    'a2': Label.A2,
    'r1': Label.R1,
    'r2': Label.R2,
    'oc': Label.OC,
    'p': Label.P,
}
stopword_trie = util.make_trie({999: stopwords.words('english') + ['non']})


class TokenCollection:
    def __init__(self, bs_doc):
        self.bs_doc = bs_doc

        self.tokens = None
        self.a_tokens = []
        self.r_tokens = []
        self.oc_token = None
        self.p_token = None

        self.feature_vectors = None
        self.a_labels = None
        self.r_labels = None
        self.oc_labels = None
        self.p_labels = None

    def normalize(self):

        # Tokenizing
        tokens = []
        abstract = self.bs_doc.abstract
        for abs_text in abstract.children:
            if isinstance(abs_text, bs4.NavigableString):
                continue  # Skip blank new lines
            for child in abs_text.children:
                xml_tag = child.name
                label = xml_tag_to_label.get(xml_tag)
                if label is not None:
                    # Dealing with specific tags
                    tags = list(tagger.tag(child.text))
                    token = Token(child.text, g_tags=tags)
                    token.set_label(label)
                    tokens.append(token)
                    if label in [Label.A1, Label.A2]:
                        self.a_tokens.append(token)
                    if label in [Label.R1, Label.R2]:
                        self.r_tokens.append(token)
                    if label == Label.OC:
                        self.oc_token = token
                    if label == Label.P:
                        self.p_token = token
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
                  if not util.check_trie(stopword_trie, x.word)]

        # NOTE: Token count should not change after this point!!
        self.tokens = tokens

    def generate_feature_vector(self):
        # TODO: Complete this
        token_count = len(self.tokens)
        feature_count = len(ft.Feature.__members__.items())
        feature_vectors = np.zeros((token_count, feature_count + 1))

        # Set bias term to 1
        feature_vectors[:, feature_count] += 1

        umls_cache_path = path.join(script_dir, 'umls_cache.json')
        umls_cache = util.load_dict(umls_cache_path)

        title = take_title(self.bs_doc.pmid.text).text
        title_words = pp.normalise_sentence(title)
        title_mapping = {999: title_words}
        title_trie = util.make_trie(title_mapping)

        for token_i in range(len(self.tokens)):
            token = self.tokens[token_i]

            # extracting features from UMLS class
            feature_class_map = ft.get_feature_classes(token.word, umls_cache)
            for feature_i, val in feature_class_map.items():
                feature_vectors[token_i, int(feature_i)] = val
            # sys.stdout.write(
            #     'UMLS Progress {}/{}\r'.format(token_i + 1, token_count))
            # sys.stdout.flush()

            # in title feature
            key = util.check_trie(title_trie, token.word)
            in_title = key is not None
            if in_title:
                feature_vectors[token_i, ft.Feature.IS_IN_TITLE.value] = 1

            # is beginning of noun phrase
            if token.g_tags[0][3] == 'B-NP':
                feature_vectors[token_i, ft.Feature.IS_BNP.value] = 1

        util.save_dict(umls_cache_path, umls_cache)

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
            if token.label in [Label.A1, Label.A2]:
                a_labels[token_i] = 1
            if token.label in [Label.R1, Label.R2]:
                r_labels[token_i] = 1
            if token.label == Label.OC:
                oc_labels[token_i] = 1
            if token.label == Label.P:
                p_labels[token_i] = 1

        self.a_labels = a_labels
        self.r_labels = r_labels
        self.oc_labels = oc_labels
        self.p_labels = p_labels

        return a_labels, r_labels, oc_labels, p_labels
