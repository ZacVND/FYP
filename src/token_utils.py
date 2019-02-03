from geniatagger import GENIATagger
from nltk.corpus import stopwords
import preprocessing as pp
from enum import Enum
import feature as ft
from os import path
import numpy as np
import json
import util
import nltk
import sys
import bs4
import re

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
        self.chunk = None
        self.ev_label = None
        self.predicted_ev_label = None

    def set_ev_label(self, ev_label):
        self.ev_label = ev_label


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

# - list of patterns
patterns = {
    # population
    '_POP_': r'''(?x) (?<=\W)[Nn]\s*\=\s*\d+
                    | (?<=_POP_\,\s)_NUM_
                    | (?<=_POP_\sand\s)_NUM_
                    | (?<=_POP_\,\s_NUM_\sand\s)_NUM_
                    | (?<=_POP_\,\s_NUM_\,\sand\s)_NUM_''',
    # confidence intervals
    '_CONFINT_': r'''(?x)\-?\d+(?:\.\d+)?\%?\s*
                        (?:\+\/\-|\Â\±)\s*
                        \d+(?:\.\d+)?\%?
                      | CI\s*(?:[\>\<]|(?:\&lt\;|\&gt\;))\s*\d+(?:\.\d+)?
                      | \(?\[?([Cc][Ii]|[Cc]onfidence\s[Ii]nterval)
                          \,?\)?\]?\s*\=?\s*_RANGE_
                      | (_NUM_|_PERC_)\s*\(?(\+\/|\Â\±|\±)\s*
                          (_NUM_|_PERC_)\)?
                      | _CONFINT_\s\(_NUM_\)
                      | _PERC_\s*_CONFINT_''',
    # confidence intervals with a measure indicator
    '_CONFINTM_': r'''(?x) _CONFINT_\s*mmHg
                         | _NUM_\s?[\(\[]_NUM_[\)\]]\s?mmHg
                         | _MEAS_\s*\(?(\+\/|\Â\±|\±)\s*_NUM_\)?
                         | _NUM_\s*\(?(\+\/|\Â\±|\±)\s*_MEAS_\)?''',
    # ranges
    '_RANGE_': r'''(?x)[\+\-]?\d+\.?\d*\s*(?:\-|to|\/)\s*[\+\-]?
                         \d+\.?\d*(?:\s*\%)?''',
    # ranges with a measure indicator
    '_RANGEM_': r'''(?x) _RANGE_\s*mmHg''',
    # p-values
    '_PVAL_': r'''(?x)[Pp]\s*
                      (?:[\>\<\=]|(?:\&lt\;|\&gt\;)){,2}\s*[01]?(\.\d+|\d+\%)
                    | [Pp]\s*
                      (?:\<|\>|\&gt\;|\&lt\;)\s*(?:or)\s*\=\s*(?:to)?\s*
                      [01]?(\.\d+|\d+\%)''',
    # percentages
    '_PERC_': r'''(?x)(?:[\-\+]\s*)?\d+\.?\d*\s*\%
                     | _NUM_\s[Pp]erc(\.|ent)? 
                     | _NUM_\s_PERC_''',
    # time indications
    '_TIME_': r'''(?x)\d+\W?\d*\s*([AaPp]\.?\s?[Mm]\.?)
                | \d+\:\d{2}
                | hrs|[Hh]ours|hh''',
    # measurements
    '_MEAS_': r'''(?x)  (\>|\<|\&lt\;|\&gt\;|\=|\≤)(\s*or\s*\=)?
                        \s*\-?\s*\d+\.?\d*\%?
                      | (_NUM_|_MEAS_)\s*
                         \/?(mm\s?[Hh][Gg]|mm|mg\/m[Ll]|mg|m[Ll]
                         |dB(\/(y|year|month))?|DB(\/(y|year|month))?)
                      | _NUM_(?=\sand\s_MEAS_)
                      | _NUM_(?=(\,\s_NUM_)*\,?\sand\s_MEAS_)''',
    # years
    '_YEAR_': r'(?<=[^\d])([12][019]\d{2})(?=[^\d])',
    # numbers (real, integers and in words)
    '_NUM_': r'''(?x) (?:[\-\+]\s*)?\d+(?:\.\d+)?(?=[^\>])
                    | _NUM_\d
                    | (?<=\b)([Oo]ne|[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive
                         |[Ss]ix|[Ss]even|[Ee]ight|[Nn]ine
                         |[Tt]en|[Ee]leven|[Tt]welve|[A-Za-z]+teen
                         |[Tt]wenty|[Tt]hirty|[Ff]orty|[Ff]ifty
                         |[Ss]ixty|[Ss]eventy|[Ee]ighty|[Nn]inety)(?=\W)
                    | _NUM_\s?([Hh]undred|[Tt]housand)(\s(and\s)?_NUM_)?
                    | ([Tt]wenty|[Tt]hirty|[Ff]orty|[Ff]ifty
                         |[Ss]ixty|[Ss]eventy|[Ee]igthty|[Nn]inety)
                      [\s\-]?_NUM_
                    | _NUM_\-_NUM_''',
    # urls
    '_URL_': r'(?:http\:\/\/)?www\..+\.(?:com|org|gov|net|edu)',
    # dates
    '_DATE_': r'''(?x)
                    (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?
                        |Apr(?:il)?|May|June?|July?
                        |Aug(?:ust)?|Sep(?:tember)?
                        |Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
                    (\s*[0123]?\d[\,\s+])?
                    (?:\s*(?:19|20)\d{2})? 
                    (?=\W+)                     #ie: January 01, 2011
                  | [0123]?\d\s*
                    (?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?
                        |[Aa]pr(?:il)?|[Mm]ay|[Jj]une?|[Jj]uly?
                        |[Aa]ug(?:ust)?|[Ss]ep(?:tember)?
                        |[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)
                    (?:[\,\s+]\s*(?:19|20)\d{2})?       #ie: 12 Jan 2011
                  | [0123]?\d[\-\/]
                      [01]?\d[\-\/]
                      (?:19|20)?\d{2}           #ie: 12/01/2001
                  | [0123]?\d[\-\/]
                      (?:[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul
                        |[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec)[\-\/]
                      (?:19|20)?\d{2}           #ie: 12/jan/2001''',
    # periods of time
    '_POFT_': r'''(?x) \d{1,3}\-(?:[Mm]inutes?|[Hh]ours?|[Dd]ays?|[Ww]eeks?
                                 |[Mm]onths?|[Yy]ears?)(?=[\s\W])
                      | (?<=\W)_NUM_\-?(?=\,?\s(_NUM_\-?\,?\s)?(to|or)\s_POFT_)''',
    # arm one references
    '_GONE_': r'''(?x) (?:[Aa]rm|[Gg]roup)\s*([1IiAa]|[Oo]ne)(?=[\s\W])
                    | (?<=\W)(?:1st|[Ff]irst|[Ii]ntervention|[Oo]ne|[Ss]tudy)\s+
                        (?:[Aa]rm|[Gg]roup)(?=[\s\W])''',
    # arm two references
    '_GTWO_': r'''(?x) (?:[Aa]rm|[Gg]roup)\s*(?:[2Bb]|II|ii|[Tt]wo)(?=[\s\W])
                    | (?:2nd|[Ss]econd|[Cc]ontrol|[Pp]lacebo)\s+
						(?:[Aa]rm|[Gg]roup)(?=[\s\W])
                    | (?<=\_GONE\_\sand\s)([2Bb]|II|ii)(?=[\s\W])
                    | (?<=\_GONE\_\,\s)([2Bb]|II|ii)(?=[\s\W])''',
    # ratios
    '_RATIO_': r'''(?x) (\_NUM\_|\_RATIO\_)[\:\/]\_NUM\_
                   | _NUM_\sof\s_NUM_''',
    # other
    'mmHg': r'mm[\s\/]*[Hh][Gg]',
    ' ': r'(\-|\s+)'
}

pat_ordered = ['mmHg', '_CONFINT_', '_CONFINTM_', '_GONE_', '_GTWO_',
               '_DATE_', '_POFT_', '_URL_', '_POP_', '_RATIO_', '_RANGE_',
               '_RANGEM_', '_PVAL_', '_TIME_', '_MEAS_', '_PERC_',
               '_YEAR_', '_NUM_', ' ']


class TokenCollection:
    def __init__(self, bs_doc):
        self.bs_doc = bs_doc

        self.tokens = None
        self.ev_labels = {}

        self.feature_vectors = None
        self.labels = None

    def chunker(self, text):
        """
        Take in text, tag it and return a list of strings (chunks)
        :param text: the text to be tagged and shallow parsed by GENIA tagger
        :return chunks: a list of tuple of type (chunk, chunk type)
        chunk = (["open", "angle", "glaucoma"], "NP")
        chunks = [chunk, chunk, ...]
        """
        chunks = []
        tags = list(tagger.tag(text))
        sent_len = len(tags)
        sent_i = 0
        # tags[i] == (G_WORD, G_BASE_FORM, G_POS_TAG, G_CHUNK, G_NAMED_ENTITY)
        chunk = []
        prev_pos_tags = []
        prev_chunk_tag = None
        while sent_i < sent_len:
            token = tags[sent_i]
            if token[G_CHUNK] == 'O':
                chunks.append((token[G_WORD], token[G_CHUNK]))
                prev_pos_tags = []
                chunk = []
                prev_chunk_tag = None
            elif token[G_CHUNK][0] == "B":
                chunk.append(token[G_WORD])
                prev_pos_tags.append(token[G_POS_TAG])
                prev_chunk_tag = token[G_CHUNK][2:]
            elif token[G_CHUNK][0] == "I" and \
                    token[G_CHUNK][2:] == prev_chunk_tag:
                chunk.append(token[G_WORD])

            sent_i += 1

        return chunks


    def normalize(self):

        # Tokenizing
        tokens = []
        abstract = self.bs_doc.abstract
        if len(abstract.abstracttext.attrs) == 0:
            abstract.abstracttext['label'] = 'None'
            abstract.abstracttext['nlmcategory'] = 'None'
        for abs_text in abstract.children:
            if isinstance(abs_text, bs4.NavigableString):
                continue  # Skip blank new lines

            # TODO: EXPAND ABBREVIATIONS HERE

            for child in abs_text.children:
                xml_tag = child.name
                label = xml_tag_to_ev_label.get(xml_tag)
                if label is not None:
                    # Dealing with specific tags
                    tags = list(tagger.tag(child.text.lower()))
                    l = []
                    for tag in tags:
                        token = Token(tag[G_WORD], g_tags=[tag])
                        token.set_ev_label(label)
                        tokens.append(token)
                        l.append(EvLabelData(token.word))

                    # if len(tags) > 1:
                    #     token = Token(tags[0][0], g_tags=[tags[0]])
                    # else:
                    #     token = Token(tags[0][0], g_tags=tags)
                    # token.set_ev_label(label)
                    # tokens.append(token)

                    # get only the first token of the label
                    # TODO: change this to the chunk
                    self.ev_labels[label] = l[0]
                else:
                    # Dealing with text
                    if not isinstance(child, bs4.NavigableString):
                        continue  # We only want navigable strings
                    tags = list(tagger.tag(child.lower()))
                    for tag in tags:
                        token = Token(tag[0], g_tags=[tag])
                        tokens.append(token)

        # Removing stopwords
        # TODO: Replace token list with linked list for quick removal
        # tokens = [x for x in tokens
        #           if not stopword_trie.check(x.word)]

        # NOTE: Token count should not change after this point!!
        self.tokens = tokens

    def generate_feature_matrix(self):
        token_count = len(self.tokens)
        feature_count = len(ft.Feature.__members__.items())
        feature_vectors = np.zeros((token_count, feature_count + 1))

        # Set bias term to 1
        feature_vectors[:, feature_count] += 1

        title = self.bs_doc.title.text
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
        a1_labels = -1 * np.ones((token_count,))
        a2_labels = -1 * np.ones((token_count,))
        r1_labels = -1 * np.ones((token_count,))
        r2_labels = -1 * np.ones((token_count,))
        oc_labels = -1 * np.ones((token_count,))
        p_labels = -1 * np.ones((token_count,))

        for token_i in range(len(self.tokens)):
            token = self.tokens[token_i]
            ev_label = token.ev_label
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
        return labels


if __name__ == "__main__":
    # testing out tagger
    # text = "To compare the clinical success rates and quality of life impact " \
    #        "of brimonidine 0.2% with timolol 0.5% in newly diagnosed patients naive to glaucoma therapy."

    text = "To compare the efficacy of the Ahmed S2 Glaucoma Valve with the " \
           "Baerveldt 250-mm(2) Glaucoma Implant in the treatment of adult glaucoma."
    tags = list(tagger.tag(text))
    for tag in list(tags):
        print(tag)
