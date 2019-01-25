import token_utils as tu
import feature as ft
import numpy as np
import time
import util


class Classifier:
    def __init__(self, init_weights=None):
        self.names = ['a', 'r', 'oc', 'p']
        self.weights = {
            'a': np.zeros((ft.feature_count,)),
            'r': np.zeros((ft.feature_count,)),
            'oc': np.zeros((ft.feature_count,)),
            'p': np.zeros((ft.feature_count,)),
        }
        self.last_total_loss = None
        self.last_test_results = None

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

        print('Training...')

        # Extract feature vectors from all papers
        token_cols = [None] * paper_count
        all_feat_vecs = np.ones((0, ft.feature_count + 1))
        for i in range(paper_count):
            soup = paper_soups[i]
            start = time.time()
            paper_id = soup.pmid.text

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
            print('Time elapsed on paper #{} ({}): {}'
                  .format(i + 1, paper_id, end - start))

        # Train: Try to learn the weights for each feature
        weights = {}
        for name in self.names:
            curr_labels = all_labels[name]
            weights[name] = np.linalg.lstsq(
                all_feat_vecs,
                curr_labels.reshape(len(curr_labels), 1),
                rcond=None
            )[0].reshape(ft.feature_count + 1, 1)

        print('Done training.')

        self.weights = weights
        self.last_train_paths = paper_paths

    def test(self, paper_paths):
        # Test how good our prediction is
        paper_soups = util.load_paper_xmls(paper_paths)
        paper_count = len(paper_soups)

        # Extract feature vectors from all papers
        test_results = [None] * paper_count
        losses = np.zeros((paper_count,))
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

            label_assignment = self.assign_ev_labels(col, predictions)
            loss = self.compute_loss(col, label_assignment)
            losses[i] = loss

            for ev_label in tu.EvLabel:
                true_ev_label_data = col.ev_labels[ev_label]
                ev_label_data = label_assignment[ev_label]
                print("Predicted: ", ev_label.name, ev_label_data.word,
                      " --- True Label: ", true_ev_label_data.word)

            print('loss for this paper is: ', loss)
            test_result = {
                "soup": soup,
                "paper_path": paper_paths[i],
                "token_collection": col,
                "true_label_assignment": col.ev_labels,
                "predicted_label_assignment": label_assignment,
                "feature_matrix": feature_matrix,
                "loss": loss
            }
            test_results[i] = test_result

        total_loss = np.sum(losses)
        print("\n\n\n---------------")
        print("total loss is: ", total_loss)
        self.last_total_loss = total_loss
        self.last_test_results = test_results
        return total_loss

    def assign_ev_labels(self, token_collection, predictions):
        label_assignment = {}
        # label_data = {"word": "timolol"}
        # ev_labels[tu.EvLabel.A1] = label_data

        # TODO: Write better label assignment code
        # Currently it is taking duplicate values
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

        label_assignment[tu.EvLabel.A1] = tu.EvLabelData(a1.word)
        label_assignment[tu.EvLabel.A2] = tu.EvLabelData(a2.word)
        label_assignment[tu.EvLabel.R1] = tu.EvLabelData(r1.word)
        label_assignment[tu.EvLabel.R2] = tu.EvLabelData(r2.word)
        label_assignment[tu.EvLabel.OC] = tu.EvLabelData(oc.word)
        label_assignment[tu.EvLabel.P] = tu.EvLabelData(p.word)

        return label_assignment

    def compute_loss(self, token_collection, label_assignment):
        """
        Loss small -> result is good
        Loss high -> result is bad
        :param token_collection:
        :param ev_label_data:
        :return: loss
        """
        loss = 0
        for label, label_data in label_assignment.items():
            true_label_data = token_collection.ev_labels[label]
            if label_data.word != true_label_data.word:
                loss += 1

        return loss

    def save_result(self, json_path):
        """
        TODO: tCalculate the error between the desired labels and the true
        labels(
        posterior)
        :param json_path:
        :return:
        """
        train_data = {}

        test_counts = len(self.last_test_results)
        test_data = [None] * test_counts
        for i in range(test_counts):
            result = self.last_test_results[i]
            tokens = result["token_collection"].tokens
            new_result = {
                "pmid": result["soup"].pmid.text,
                "paper_path": result["paper_path"],
                "tokens": [{"word": t.word} for t in tokens],
                "true_label_assignment": {},
                "predicted_label_assignment": {},
                "feature_matrix": result["feature_matrix"].tolist(),
                "loss": result["loss"]
            }
            test_data[i] = new_result

        data = {
            "train_data": train_data,
            "test_data": test_data,
            "feature_names": [f.name for f in ft.Feature],
        }

        util.save_dict(json_path, data)
