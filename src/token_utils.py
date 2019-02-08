from geniatagger import GENIATagger
from nltk.corpus import stopwords
from enum import Enum
import feature as ft
from os import path
import numpy as np
import util
import nltk
import bs4

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

    def set_ev_label(self, ev_label):
        self.ev_label = ev_label

    def set_chunk(self, chunk):
        self.chunk = chunk
        # TODO: make sure this method is used instead of token.chunk=something


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
        elif tok_1st.endswith("ADVP"):
            self.features[ft.Feature.CHUNK_TYPE_ADVP] = 1
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


class TokenCollection:
    def __init__(self, bs_doc):
        self.bs_doc = bs_doc
        self.tokens = None
        self.ev_labels = {}
        self.feature_vectors = None
        self.labels = None
        self.paragraphs = []
        self.chunks = None

    def normalise(self):

        # Tokenizing
        tokens = []
        abstract = self.bs_doc.abstract
        if len(abstract.abstracttext.attrs) == 0:
            abstract.abstracttext['label'] = 'None'
            abstract.abstracttext['nlmcategory'] = 'None'

        # Populate the tokens list
        for abs_text in abstract.children:
            if isinstance(abs_text, bs4.NavigableString):
                continue  # Skip blank new lines

            sents_ = nltk.sent_tokenize(abs_text.text)
            tags = []
            for s in sents_:
                tags.extend(list(tagger.tag(s)))
            tag_i = 0
            for child in abs_text.children:
                xml_tag = child.name
                label = xml_tag_to_ev_label.get(xml_tag)
                if label is not None:
                    # Dealing with specific tags
                    word_tokens = nltk.word_tokenize(child.text)
                    tokens_of_label = []
                    for tok in word_tokens:
                        token = Token(tok, g_tags=tags[tag_i])
                        token.set_ev_label(label)
                        tokens.append(token)
                        tokens_of_label.append(EvLabelData(token.word))
                        tag_i += 1

                    # get only the first token of the label
                    if len(tokens_of_label) == 0:
                        print("The current paper doesn't have token"
                              " associate with label: ", label)
                    else:
                        self.ev_labels[label] = tokens_of_label[0]
                else:
                    # Dealing with text
                    word_tokens = nltk.word_tokenize(child)
                    if not isinstance(child, bs4.NavigableString):
                        continue  # We only want navigable strings

                    for tok in word_tokens:
                        # NOTE: paper #21921953, #22458918 has list index out of range
                        # because genia tagger returns missing "<15 mm Hg" causing tag_i to mismatch
                        # for now we take some files out of dataset
                        token = Token(tok, g_tags=tags[tag_i])
                        tokens.append(token)
                        tag_i += 1

        # Chunking and assigning the correct chunk to each token
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
                        c_tok.chunk = chunk
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
                c_tok.chunk = chunk

        # NOTE: Token count should not change after this point!!
        self.tokens = tokens
        self.chunks = chunks

    def generate_feature_matrix(self):
        token_count = len(self.tokens)
        feature_count = len(ft.Feature.__members__.items())
        feature_vectors = np.zeros((token_count, feature_count))

        title = self.bs_doc.title.text
        if title is None:
            print('Cannot fetch the title, assuming no title.')
        else:
            # title_words = pp.normalise_sentence(title)
            title_words = nltk.word_tokenize(title)
            title_trie = util.Trie(strings=title_words)

        for chunk in self.chunks:
            chunk.extract_features()

        for token_i in range(len(self.tokens)):
            token = self.tokens[token_i]

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

            # is in patient
            # for tag in token.g_tags:
            result = ft.patient_dict_trie.check(token.g_tags[G_BASE_FORM])
            if result:
                feature_vectors[token_i, ft.Feature.TOK_IS_IN_PDICT.value] = 1

            # is number
            if token.g_tags[G_POS_TAG] == "CD":
                feature_vectors[token_i, ft.Feature.TOK_IS_NUMBER.value] = 1

            # extract chunk features
            curr_chunk = token.chunk
            if curr_chunk:
                features = curr_chunk.features
                for ft_enum, value in features.items():
                    feature_vectors[token_i, ft_enum.value] = value

        util.umls_cache.save()
        self.feature_vectors = feature_vectors

        return feature_vectors

    def generate_train_labels(self):
        token_count = len(self.tokens)
        labels_matrix = np.zeros((token_count, len(EvLabel.__members__.items())))

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
    # testing out tagger
    # text = "To compare the clinical success rates and quality of life impact " \
    #        "of brimonidine 0.2% with timolol 0.5% in newly diagnosed patients naive to glaucoma therapy."

    text = "The objective of the study was to compare the long-term efficacy and safety of tafluprost 0.0015% with latanoprost 0.005% eye drops in patients with open-angle glaucoma or ocular hypertension"

    tags = list(tagger.tag(text))
    for tag in list(tags):
        print(tag)

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
