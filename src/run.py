from geniatagger import GENIATagger
from nltk.corpus import stopwords
from take_abstract import *
import preprocessing as pp
import feature as ft
from os import path,listdir
import numpy as np
import time
import util
import nltk
import bs4

logger = util.get_logger("run")

sw = stopwords.words('english') + ['non']
cur_dir = path.dirname(path.abspath(__file__))

if __name__ == "__main__":

    # genia tagger instance
    executable_path = path.join("..", "geniatagger-3.0.2", "geniatagger")
    tagger = GENIATagger(executable_path)

    umls_cache_path = path.join(cur_dir, 'umls_cache.json')
    umls_cache = util.load_dict(umls_cache_path)

    data_path = path.join(cur_dir, "..", 'data/annotation I/')
    for fn in listdir(data_path):
        start = time.time()
        with open(path.join(data_path, fn)) as f:
            raw = f.read()
            soup = bs4.BeautifulSoup(raw, "html5lib")

        # tokenising
        text = soup.abstract.text
        # text = 'timolol some bullshit'
        sents = nltk.sent_tokenize(text)
        sents_words = [None] * len(sents)
        word_count = 0

        for i in range(len(sents)):
            sent = sents[i]
            chunks = tagger.tag(sent)

            words = nltk.word_tokenize(sent)
            logger.debug(["with sw", len(words)])
            words = [w for w in words if not w in sw]
            logger.debug(["without sw", len(words)])
            # TODO: Extract features that requires capitalisation,
            # then lowercase
            sents_words[i] = list(map(lambda w: w.lower(), words))
            # TODO: Do all preprocessing that affects word_count

            word_count += len(words)

        # Defining feature vectors
        feature_count = len(ft.Feature.__members__.items())
        # zero means unknown quantity of the feature
        feature_vectors = np.zeros((word_count, feature_count))

        # extracting features from UMLS classes
        curr_word_i = 0
        for sent_i, sent in enumerate(sents_words):
            for word in sent:
                feature_class_map = ft.get_feature_classes(word, umls_cache)
                for feature_i, val in feature_class_map.items():
                    feature_vectors[curr_word_i, int(feature_i)] = val
                curr_word_i += 1
                print('\rUMLS Progress {}/{}'.format(curr_word_i, word_count))

        # in title feature
        title = take_title(soup.pmid.text).text
        title_words = pp.normalise_sentence(title)
        title_mapping = {999: title_words}
        title_trie = util.make_trie(title_mapping)
        curr_word_i = 0
        for sent_i, sent in enumerate(sents_words):
            for word_i, word in enumerate(sent):
                key = util.check_trie(title_trie, word)
                in_title = key is not None
                if in_title:
                    feature_vectors[
                        curr_word_i, ft.Feature.IS_IN_TITLE.value] = 1
                curr_word_i += 1

        # Generate "true" labels for As
        a1 = soup.find("a1").text
        a2 = soup.find("a2").text
        a_labels = -1 * np.ones((word_count,))
        curr_word_i = 0
        for sent_i, sent in enumerate(sents_words):
            for word_i, word in enumerate(sent):
                if word == a1 or word == a2:
                    a_labels[curr_word_i] = 1
                curr_word_i += 1

        # Learn the weights
        a_feature_weights = np.linalg.lstsq(
            feature_vectors,
            a_labels.reshape(word_count, 1),
            rcond=None
        )[0]

        # Test the weights and see how well they perform.
        predicted_labels = \
            feature_vectors @ a_feature_weights.reshape(feature_count, 1)

        print(predicted_labels)

        curr_word_i = 0
        print('Words that are likely to be As:')
        for sent_i, sent in enumerate(sents_words):
            for word_i, word in enumerate(sent):
                # ideally pick the top n values instead of anything > 0
                if predicted_labels[curr_word_i] > 0:
                    print(word)
                curr_word_i += 1

        end = time.time()
        print('Time elapsed on current doc: ', end - start)
        break

    util.save_dict(umls_cache_path, umls_cache)
