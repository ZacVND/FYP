from geniatagger import GENIATagger
from nltk.corpus import stopwords
from string import punctuation
from enum import Enum
import feature as ft
from os import path
import numpy as np
import unicodedata
import nltk
import math
import util
import re

script_dir = path.dirname(path.abspath(__file__))

# constants to extract info from genia tagger output.
# Example:
#     tags = list(tagger.tag(string))
#     first_word_tags = tags[0]
#     pos_tag_of_first_word = first_word_tags[G_POS_TAG]
# See: https://github.com/bornabesic/genia-tagger-py#example-usage
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


# holds information for the evidence table output
class EvLabelData:
    def __init__(self, word):
        # self.word is deprecated
        self.word = word
        self.token = None


class Token:
    def __init__(self, word, g_tags=[]):
        self.word = word.lower()
        self.og_word = word
        self.g_tags = g_tags
        self.chunk = None
        self.ev_label = None
        self.sent_pos = 1  # position in decile
        self.abs_pos = 1
        self.para_cat = None
        self.para_label = None

    def set_sent_pos(self, sent_pos):
        if sent_pos > self.sent_pos:
            self.sent_pos = sent_pos

    def set_abs_pos(self, abs_pos):
        if abs_pos > self.abs_pos:
            self.abs_pos = abs_pos

    def set_ev_label(self, ev_label):
        self.ev_label = ev_label

    def set_chunk(self, chunk):
        self.chunk = chunk

    def set_para_cat(self, para_cat):
        self.para_cat = para_cat

    def set_para_label(self, para_label):
        self.para_label = para_label


class Chunk:
    def __init__(self, tokens):
        self.tokens = tokens
        self.features = {}
        self.string = " ".join([tok.word for tok in tokens])

    def extract_features(self):
        tok_1st = self.tokens[0].g_tags[G_CHUNK]
        if tok_1st.endswith("NP"):
            self.features[ft.Feature.CHUNK_TYPE_NP] = 1
        elif tok_1st.endswith("VP"):
            self.features[ft.Feature.CHUNK_TYPE_VP] = 1
        elif tok_1st.endswith("PP"):
            self.features[ft.Feature.CHUNK_TYPE_PP] = 1
        # elif tok_1st.endswith("ADVP"):
        #     self.features[ft.Feature.CHUNK_TYPE_ADVP] = 1
        elif tok_1st.endswith("ADJP"):
            self.features[ft.Feature.CHUNK_TYPE_ADJP] = 1


# genia tagger instance
executable_path = path.join(script_dir, "..", "geniatagger-3.0.2",
                            "geniatagger")

# list(tagger.tag(text)) output (word, base, POStag, chunktag, NEtag)
# NOTE: text has to be text, preferrably a sentence. NOT list of 'word'
tagger = GENIATagger(executable_path)

xml_tag_to_ev_label = {
    'a1': EvLabel.A1,
    'a2': EvLabel.A2,
    'r1': EvLabel.R1,
    'r2': EvLabel.R2,
    'oc': EvLabel.OC,
    'p': EvLabel.P,
}

stopword_trie = util.Trie(strings=stopwords.words('english') + ['non'])


class TokenCollection:
    def __init__(self, bs_doc):
        self.bs_doc = bs_doc
        self.tokens = None
        self.ev_labels = {}
        self.feature_vectors = None
        self.labels = None
        self.chunks = None

    def build_tokens(self, umls_cache=False):

        # Tokenizing
        tokens = []
        abstract = self.bs_doc.abstract

        # for unstructured abstracts
        if len(abstract.abstracttext.attrs) == 0:
            abstract.abstracttext['label'] = 'None'
            abstract.abstracttext['nlmcategory'] = 'None'

        # Populate the tokens list
        for abs_text in abstract.findAll('abstracttext'):

            try:
                para_cat = abs_text['nlmcategory']
            except KeyError:
                para_cat = 'None'
            try:
                para_label = abs_text['label']
            except KeyError:
                para_label = 'None'

            if para_cat == "BACKGROUND" or para_cat == "CONCLUSIONS":
                # Background of abstract is irrelevant
                # Conclusions of abstract does not contribute any info
                continue

            sents_ = nltk.sent_tokenize(abs_text.text)
            # This hack below is necessary because of the way bs4 encode texts
            sents_ = [unicodedata.normalize("NFKD", s) for s in sents_]
            sents_ = [s.replace('&lt;', '<') for s in sents_]
            sents_ = [s.replace('&gt;', '>') for s in sents_]
            tags = []
            sent_lens = []
            for s in sents_:
                tags_list = list(tagger.tag(s))
                sent_lens.append(len(tags_list))
                tags.extend(tags_list)

            tag_i = 0
            sent_i = 0
            sent_len_i = 0

            for child in abs_text.children:
                xml_tag = child.name
                label = xml_tag_to_ev_label.get(xml_tag)
                if label is not None:
                    # Dealing with specific tags
                    # This hack below is necessary because of the way bs4 encode texts
                    child_text = unicodedata.normalize("NFKD", child.text)
                    child_text = child_text.replace('&lt;', '<')
                    child_text = child_text.replace('&gt;', '>')
                    word_tokens = nltk.word_tokenize(child_text)
                    tokens_of_label = []
                    for tok in word_tokens:
                        token = Token(tok, g_tags=tags[tag_i])
                        token.set_ev_label(label)
                        # set sentence position in decile
                        sent_pos = math.ceil(sent_i / (sent_lens[sent_len_i] /
                                                       10))
                        token.set_sent_pos(sent_pos)
                        token.set_para_cat(para_cat)
                        token.set_para_label(para_label)
                        # append to full tokens list
                        tokens.append(token)
                        tokens_of_label.append(EvLabelData(token.word))
                        sent_i += 1
                        tag_i += 1

                    # get only the first token of the label
                    if len(tokens_of_label) == 0:
                        print("The current paper doesn't have token"
                              " associate with label: ", label)
                    else:
                        self.ev_labels[label] = tokens_of_label[0]
                else:
                    # Dealing with text
                    # This hack below is necessary because of the way bs4 encode texts
                    child = unicodedata.normalize("NFKD", child)
                    child = child.replace('&lt;', '<')
                    child = child.replace('&gt;', '>')
                    word_tokens = nltk.word_tokenize(child)
                    for tok in word_tokens:
                        token = Token(tok, g_tags=tags[tag_i])
                        # set sentence position in decile
                        sent_pos = math.ceil(
                            sent_i / (sent_lens[sent_len_i] / 10))
                        token.set_sent_pos(sent_pos)
                        token.set_para_cat(para_cat)
                        token.set_para_label(para_label)
                        # append to full tokens list
                        tokens.append(token)
                        sent_i += 1
                        tag_i += 1
                        if tok == '.':
                            sent_i = 0
                            sent_len_i += 1

        if umls_cache:
            # remove stopwords
            tokens = [x for x in tokens if not stopword_trie.check(x.word)]
            # remove punctuations
            tokens = [x for x in tokens if x.word not in punctuation]
            # remove anything that is not alphabetic or of type num-chars
            pattern = re.compile(r'[\d\w]+(?:-\w+)+')
            tokens = [x for x in tokens if ((x.word.isalpha()) or \
                                            (re.match(pattern,
                                                      x.word) is not None))]
        else:
            # Chunking and assigning the correct chunk to each token
            # TODO: write code to check if a chunk has measurement in it
            chunk_tokens = None
            chunks = []
            for tok in tokens:
                chunk_tag = tok.g_tags[G_CHUNK]
                if not chunk_tag.startswith("I"):
                    if chunk_tokens is not None:
                        # save chunk
                        chunk = Chunk(chunk_tokens)
                        chunks.append(chunk)
                        for c_tok in chunk_tokens:
                            c_tok.set_chunk(chunk)
                        chunk_tokens = None

                    if chunk_tag.startswith("B"):
                        chunk_tokens = [tok]
                else:
                    if chunk_tokens is not None:
                        chunk_tokens.append(tok)
            if chunk_tokens is not None:
                # save chunk
                chunk = Chunk(chunk_tokens)
                chunks.append(chunk)
                for c_tok in chunk_tokens:
                    c_tok.set_chunk(chunk)

            self.chunks = chunks

        # NOTE: Token count should not change after this point!!
        self.tokens = tokens

    def generate_feature_matrix(self):
        token_count = len(self.tokens)
        feature_count = len(ft.Feature.__members__.items())
        feature_vectors = np.zeros((token_count, feature_count))
        pattern = re.compile(r'[\d\w]+(?:-\w+)+')

        title = self.bs_doc.title.text
        if title is None:
            print('Cannot fetch the title, assuming no title.')
        else:
            # title_words = pp.normalise_sentence(title)
            title_words = nltk.word_tokenize(title)
            title_trie = util.Trie(strings=title_words)

        for chunk in self.chunks:
            chunk.extract_features()

        for token_i in range(token_count):
            token = self.tokens[token_i]

            if (not stopword_trie.check(token.word) and
                    token.word not in punctuation and
                    (token.word.isalpha() or
                     re.match(pattern, token.word) is not None)):
                # extracting features from UMLS class
                feature_class_map = ft.get_feature_classes(token.word)
                for feature_i, val in feature_class_map.items():
                    feature_vectors[token_i, int(feature_i)] = val
            # sys.stdout.write(
            #     'UMLS Progress {}/{}\r'.format(token_i + 1, token_count))
            # sys.stdout.flush()

            # in title feature
            if title is not None:
                in_title = title_trie.check(token.word)
                if in_title:
                    feature_vectors[
                        token_i, ft.Feature.TOK_IS_IN_TITLE.value] = 1

            # is beginning of noun phrase
            if token.g_tags[G_CHUNK] == 'B-NP':
                feature_vectors[token_i, ft.Feature.TOK_IS_BNP.value] = 1

            # is in patient dictionary
            # for tag in token.g_tags:
            result_pdict = ft.patient_dict_trie.check(token.g_tags[G_BASE_FORM])
            if result_pdict:
                feature_vectors[token_i, ft.Feature.TOK_IS_IN_PDICT.value] = 1

            # is in outcome dictionary
            result_odict = ft.outcome_dict_trie.check(token.g_tags[G_BASE_FORM])
            if result_odict:
                feature_vectors[token_i, ft.Feature.TOK_IS_IN_ODICT.value] = 1

            # is number
            try:
                float(token.g_tags[G_WORD])
                feature_vectors[token_i, ft.Feature.TOK_IS_NUMBER.value] = 1
            except ValueError:
                feature_vectors[token_i, ft.Feature.TOK_IS_NUMBER.value] = 0

            # # is placebo
            # if token.g_tags[G_BASE_FORM] == "placebo":
            #     feature_vectors[token_i, ft.Feature.TOK_IS_PLACEBO.value] = 1

            # is Cardinal Digit
            if token.g_tags[G_POS_TAG] == "CD":
                feature_vectors[token_i, ft.Feature.TOK_IS_CD.value] = 1

            # extract chunk features
            curr_chunk = token.chunk
            if curr_chunk:
                features = curr_chunk.features
                for ft_enum, value in features.items():
                    feature_vectors[token_i, ft_enum.value] = value

            if token.para_cat == "OBJECTIVE":
                feature_vectors[token_i, ft.Feature.PARA_CAT_OBJECTIVE.value] \
                    = 1

            if token.para_cat == "METHODS":
                feature_vectors[token_i, ft.Feature.PARA_CAT_METHODS.value] = 1

            if token.para_cat == "RESULTS":
                feature_vectors[token_i, ft.Feature.PARA_CAT_RESULTS.value] = 1

            # if token.para_cat == "CONCLUSIONS":
            #     feature_vectors[
            #         token_i, ft.Feature.PARA_CAT_CONCLUSIONS.value] = 1

            O_and_M_para = ["OBJECTIVE", "METHODS"]
            # patients group only mentioned in either OBJECTIVE or METHODS
            if ((token.para_cat in O_and_M_para) and
                    feature_vectors[token_i, ft.Feature.TOK_IS_IN_PDICT.value]):
                feature_vectors[token_i, ft.Feature.TOK_IS_PATIENTS.value] = 1

            result_tdict = ft.treatment_dict_trie.check(
                token.g_tags[G_BASE_FORM])
            if result_tdict:
                feature_vectors[token_i, ft.Feature.TOK_IS_DRUG.value] = 1
                feature_vectors[token_i, ft.Feature.TOK_IS_PROCEDURE.value] = 1

            # A1 and A2 are only in OBJECTIVE or METHODS
            # A1 and A2 are either drug, placebo or procedure
            if ((token.para_cat in O_and_M_para) and
                    (feature_vectors[token_i, ft.Feature.TOK_IS_DRUG.value] or
                     # feature_vectors[
                     #     token_i, ft.Feature.TOK_IS_PLACEBO.value] or
                     feature_vectors[
                         token_i, ft.Feature.TOK_IS_PROCEDURE.value])):
                feature_vectors[token_i, ft.Feature.TOK_IS_ARM.value] = 1

            # if token.para_cat == "BACKGROUND":
            #     feature_vectors[token_i, ft.Feature.PARA_CAT_OBJECTIVE] = 1

            # sentence position in decile
            feature_vectors[token_i,
                            ft.Feature.SENT_POSITION.value] = token.sent_pos

            # abstract position in decile
            abs_pos = math.ceil(token_i / (token_count / 10))
            token.set_abs_pos(abs_pos)
            feature_vectors[token_i,
                            ft.Feature.ABSTRACT_POSITION.value] = token.abs_pos

        # expand the cache if encounters new words
        util.umls_cache.save()
        self.feature_vectors = feature_vectors

        return feature_vectors

    def generate_train_labels(self):
        token_count = len(self.tokens)
        labels_matrix = np.zeros(
            (token_count, len(EvLabel.__members__.items())))

        # Generate label vectors, -1 for all tokens
        a1_labels = -1 * np.ones((token_count,))
        a2_labels = -1 * np.ones((token_count,))
        r1_labels = -1 * np.ones((token_count,))
        r2_labels = -1 * np.ones((token_count,))
        oc_labels = -1 * np.ones((token_count,))
        p_labels = -1 * np.ones((token_count,))

        # set label of token to 1 based on train data
        for token_i in range(len(self.tokens)):
            token = self.tokens[token_i]
            ev_label = token.ev_label
            if ev_label is not None:
                labels_matrix[token_i, ev_label.value] = 1

            if ev_label == EvLabel.A1:
                a1_labels[token_i] = 1
            if ev_label == EvLabel.A2:
                a2_labels[token_i] = 1
            if ev_label == EvLabel.R1:
                r1_labels[token_i] = 1
            if ev_label == EvLabel.R2:
                r2_labels[token_i] = 1
            if ev_label == EvLabel.OC:
                oc_labels[token_i] = 1
            if ev_label == EvLabel.P:
                p_labels[token_i] = 1

        labels = {
            EvLabel.A1: a1_labels,
            EvLabel.A2: a2_labels,
            EvLabel.R1: r1_labels,
            EvLabel.R2: r2_labels,
            EvLabel.OC: oc_labels,
            EvLabel.P: p_labels,
        }

        self.labels = labels
        return labels_matrix


if __name__ == "__main__":
    import os

    xml_file = os.path.join(util.data_path, "30022618.xml")
    ps = util.parse_paper(xml_file)
    col = TokenCollection(ps)
    col.build_tokens()
    print("Done")

    # # testing out tagger
    # text = "Mean (+/-SD) preoperative and 1-year postoperative intraocular pressures in the 5-fluorouracil group were 26.9 (+/-9.5) and 15.3 (+/-5.8) mm Hg, respectively. In the control group these were 25.9 (+/-8.1) mm Hg, and 15.8 (+/-5.1) mm Hg, respectively."
    #
    # # text = "The objective of the study was to compare the long-term efficacy and safety of tafluprost 0.0015% with latanoprost 0.005% eye drops in patients with open-angle glaucoma or ocular hypertension"
    #
    # tags = list(tagger.tag(text))
    # for tag in list(tags):
    #     print(tag)

    """
    text = "The study enrolled 148 patients with inadequately controlled " \
           "open-angle or pseudoexfoliation glaucoma"
    
    ('The', 'The', 'DT', 'B-NP', 'O')
    ('study', 'study', 'NN', 'I-NP', 'O')
    ('enrolled', 'enrol', 'VBD', 'B-VP', 'O')
    ('148', '148', 'CD', 'B-NP', 'O')
    ('patients', 'patient', 'NNS', 'I-NP', 'O')
    ('with', 'with', 'IN', 'B-PP', 'O')
    ('inadequately', 'inadequately', 'RB', 'B-NP', 'O')
    ('controlled', 'control', 'VBN', 'I-NP', 'O')
    ('open-angle', 'open-angle', 'JJ', 'I-NP', 'O')
    ('or', 'or', 'CC', 'I-NP', 'O')
    ('pseudoexfoliation', 'pseudoexfoliation', 'NN', 'I-NP', 'O')
    ('glaucoma', 'glaucoma', 'NN', 'I-NP', 'O')
    """
