from nltk.corpus import stopwords
from os import path, listdir
import token_utils as tu
import feature as ft
import numpy as np
import time
import util
import bs4

logger = util.get_logger("run")

sw = stopwords.words('english') + ['non']

# TODO: write code to download all of the UMLS query data

if __name__ == "__main__":

    # Prepare papers' XML
    # paper_files = sorted(listdir(data_path))[:4]
    paper_paths = util.get_paper_paths()
    paper_count = len(paper_paths)
    paper_soups = [None] * paper_count
    for i in range(len(paper_paths)):
        paper_soups[i] = util.parse_paper(paper_paths[i])

    # Extract feature vectors from all papers
    token_cols = [None] * paper_count
    feature_count = len(ft.Feature.__members__.items())
    all_feat_vecs = np.ones((0, feature_count + 1))
    all_a_labs = np.zeros((0,))
    for i in range(paper_count):
        soup = paper_soups[i]
        start = time.time()

        col = tu.TokenCollection(soup)
        col.normalize()
        feature_vector = col.generate_feature_vector()
        (a_labels,
         r_labels,
         oc_labels,
         p_labels) = col.generate_train_labels()

        all_feat_vecs = np.vstack((all_feat_vecs, feature_vector))
        all_a_labs = np.concatenate((all_a_labs, a_labels))

        token_cols[i] = col

        end = time.time()
        print('Time elapsed on paper #{}: {}'.format(i + 1, end - start))

    # Train: Try to learn the weights for each feature
    a_feature_weights = np.linalg.lstsq(
        all_feat_vecs,
        all_a_labs.reshape(len(all_a_labs), 1),
        rcond=None
    )[0].reshape(feature_count + 1, 1)

    # Test how good our prediction is
    print('\n\nEvaluating predictions:\n')
    for i in range(paper_count):
        soup = paper_soups[i]
        print('---- Paper #{} [{}]'.format(i + 1, soup.pmid.text))

        col = token_cols[i]
        prdct_a_labels = (col.feature_vectors @ a_feature_weights).flatten()

        # for j in range(len(col.tokens)):
        #     print(
        #         col.tokens[j].word,
        #         ' ==> ',
        #         col.feature_vectors[j, :],
        #         ' ==> ',
        #         a_labels[j],
        #         ' ==> ',
        #         all_a_labs[j]
        #     )

        print('True As are:', [t.word for t in col.a_tokens])
        print('Our best As are:')

        best_val_cnt = 4
        best_indices = np.argpartition(
            prdct_a_labels.flatten(),
            -best_val_cnt
        )[-best_val_cnt:]
        for best_index in best_indices:
            token = col.tokens[best_index]
            print('- {} ({})'.format(token.word, prdct_a_labels[best_index]))

        print('\n')
