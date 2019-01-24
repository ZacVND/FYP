import token_utils as tu
import feature as ft
import numpy as np
import time
import util
import heapq


class Classifier:
    def __init__(self, init_weights=None):
        self.names = ['a', 'r', 'oc', 'p']
        self.weights = {
            'a': np.zeros((ft.feature_count,)),
            'r': np.zeros((ft.feature_count,)),
            'oc': np.zeros((ft.feature_count,)),
            'p': np.zeros((ft.feature_count,)),
        }

        if init_weights is not None:
            for key, value in init_weights.items():
                self.weights[key] = value

    def train(self, paper_paths):
        paper_soups = util.load_paper_xmls(paper_paths)
        paper_count = len(paper_soups)

        all_labels = {
            'a': np.zeros((0,)),
            'r': np.zeros((0,)),
            'oc': np.zeros((0,)),
            'p': np.zeros((0,)),
        }

        # Extract feature vectors from all papers
        token_cols = [None] * paper_count
        all_feat_vecs = np.ones((0, ft.feature_count + 1))
        for i in range(paper_count):
            soup = paper_soups[i]
            start = time.time()

            col = tu.TokenCollection(soup)
            col.normalize()
            feature_vector = col.generate_feature_matrix()

            # Structure of tuple below:
            labels = col.generate_train_labels()

            all_feat_vecs = np.vstack((all_feat_vecs, feature_vector))

            for name in self.names:
                all_labels[name] = np.hstack([all_labels[name], labels[name]])

            token_cols[i] = col

            end = time.time()
            print('Time elapsed on paper #{}: {}'.format(i + 1, end - start))

        # Train: Try to learn the weights for each feature
        weights = {}
        for name in self.names:
            curr_labels = all_labels[name]
            weights[name] = np.linalg.lstsq(
                all_feat_vecs,
                curr_labels.reshape(len(curr_labels), 1),
                rcond=None
            )[0].reshape(ft.feature_count + 1, 1)

        self.weights = weights

    def test(self, paper_paths):
        # Test how good our prediction is
        paper_soups = util.load_paper_xmls(paper_paths)
        paper_count = len(paper_soups)

        # Extract feature vectors from all papers
        all_feat_vecs = np.ones((0, ft.feature_count + 1))
        for i in range(paper_count):
            soup = paper_soups[i]
            print('---- Paper #{} [{}]'.format(i + 1, soup.pmid.text))

            col = tu.TokenCollection(soup)
            col.normalize()
            feature_matrix = col.generate_feature_matrix()

            predictions = {}
            for name in self.names:
                weights = self.weights[name].reshape(ft.feature_count + 1, 1)
                predictions[name] = (feature_matrix @ weights).flatten()

            ev_labels_data = self.assign_ev_labels(col, predictions)

            for ev_label in tu.EvLabel:
                true_ev_label_data = col.ev_labels[ev_label]
                ev_label_data = ev_labels_data[ev_label]
                print("Predicted: ", ev_label.name, ev_label_data.word,
                      " --- True Label: ", true_ev_label_data.word)

    def assign_ev_labels(self, token_collection, predictions):
        ev_labels = {}
        # label_data = {"word": "timolol"}
        # ev_labels[tu.EvLabel.A1] = label_data

        tokens = token_collection.tokens
        for key, value in predictions.items():
            if key == 'a':
                token_is = util.get_n_max_indices(value, 2)
                a1 = tokens[token_is[0]]
                a2 = tokens[token_is[1]]
            if key == 'r':
                token_is = util.get_n_max_indices(value, 2)
                r1 = tokens[token_is[0]]
                r2 = tokens[token_is[1]]
            if key == 'oc':
                token_is = util.get_n_max_indices(value, 2)
                oc = tokens[token_is[0]]
            if key == 'p':
                token_is = util.get_n_max_indices(value, 2)
                p = tokens[token_is[0]]

        ev_labels[tu.EvLabel.A1] = tu.EvLabelData(a1.word)
        ev_labels[tu.EvLabel.A2] = tu.EvLabelData(a2.word)
        ev_labels[tu.EvLabel.R1] = tu.EvLabelData(r1.word)
        ev_labels[tu.EvLabel.R2] = tu.EvLabelData(r2.word)
        ev_labels[tu.EvLabel.OC] = tu.EvLabelData(oc.word)
        ev_labels[tu.EvLabel.P] = tu.EvLabelData(p.word)

        return ev_labels

    def compute_loss(self, token_collection):
        pass
