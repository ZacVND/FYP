from sklearn.metrics import log_loss
from sklearn import tree
import token_utils as tu
import feature as ft
import numpy as np
import heapq
import time
import util


class Classifier:
    def __init__(self, init_weights=None):
        self.weights = {
            tu.EvLabel.A1: np.zeros((ft.feature_count,)),
            tu.EvLabel.A2: np.zeros((ft.feature_count,)),
            tu.EvLabel.R1: np.zeros((ft.feature_count,)),
            tu.EvLabel.R2: np.zeros((ft.feature_count,)),
            tu.EvLabel.OC: np.zeros((ft.feature_count,)),
            tu.EvLabel.P: np.zeros((ft.feature_count,)),
        }
        self.last_total_loss = None
        self.last_test_results = None
        self.last_train_paths = None

        cls_weights = {-1: 1, }
        for ev_label in tu.EvLabel:
            cls_weights[ev_label.value] = 100

        self.DT_clf = tree.DecisionTreeClassifier(class_weight=cls_weights)

        if init_weights is not None:
            for key, value in init_weights.items():
                self.weights[key] = value

    def train(self, paper_paths):
        paper_soups = util.load_paper_xmls(paper_paths)
        paper_count = len(paper_soups)

        all_labels_vec = np.ones((0, 1))
        all_labels = {
            tu.EvLabel.A1: np.zeros((0,)),
            tu.EvLabel.A2: np.zeros((0,)),
            tu.EvLabel.R1: np.zeros((0,)),
            tu.EvLabel.R2: np.zeros((0,)),
            tu.EvLabel.OC: np.zeros((0,)),
            tu.EvLabel.P: np.zeros((0,)),
        }

        print('Training on {} paper(s)...'.format(paper_count))

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
            tokens_count = len(labels[tu.EvLabel.A1])
            labels_vec = np.ones((tokens_count, 1)) * -1
            for token_i in range(tokens_count):
                for ev_label in tu.EvLabel:
                    if labels[ev_label][token_i] > 0:
                        labels_vec[token_i] = ev_label.value

            all_feat_vecs = np.vstack((all_feat_vecs, feature_vector))
            all_labels_vec = np.vstack((all_labels_vec, labels_vec))

            for ev_label in tu.EvLabel:
                all_labels[ev_label] = np.hstack(
                    [all_labels[ev_label], labels[ev_label]])

            token_cols[i] = col

            end = time.time()
            print('Time elapsed on paper #{} ({}): {}'
                  .format(i + 1, paper_id, end - start))

        # Train: Try to learn the weights for each feature
        weights = {}
        self.DT_clf.fit(all_feat_vecs, all_labels_vec)
        # for name in self.names:
        #     curr_labels = all_labels[name]
        #     weights[name] = np.linalg.lstsq(
        #         all_feat_vecs,
        #         curr_labels.reshape(len(curr_labels), 1),
        #         rcond=None
        #     )[0].reshape(ft.feature_count + 1, 1)

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

            prob_matrix = self.DT_clf.predict_proba(feature_matrix)

            predictions = {}
            for ev_label in tu.EvLabel:
                # weights = self.weights[ev_label].reshape(ft.feature_count + 1, 1)
                predictions[ev_label] = prob_matrix[:, ev_label.value + 1]

            label_assignment = self.assign_ev_labels(col, predictions)
            # loss = self.compute_loss(col, label_assignment)
            loss = self.compute_loss(col, prob_matrix, label_assignment)
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

        best_words_count = 6
        tokens = token_collection.tokens

        def pick_best_unique_words(labels, ev_label):
            heap = [(-x, i) for i, x in enumerate(labels)]
            heapq.heapify(heap)
            results = []
            seen = {}
            for _ in range(best_words_count):
                min_x, i = heapq.heappop(heap)
                token = tokens[i]
                if seen.get(token.word):
                    continue

                tup = (-min_x, i, ev_label)
                results.append(tup)
                seen[token.word] = True

            return results

        all_tuples = []
        for ev_label in tu.EvLabel:
            labels = predictions[ev_label]
            best_words = pick_best_unique_words(labels, ev_label)
            all_tuples += best_words
        # sorted by overall likelihood, we don't care which class yet
        sorted_tuples = sorted(all_tuples, key=lambda x: x[0], reverse=True)

        token_labelled = {}
        for tup in sorted_tuples:
            likelihood, index, ev_label = tup
            if token_labelled.get(index):
                continue
            if label_assignment.get(ev_label) is not None:
                continue
            token = tokens[index]
            label_assignment[ev_label] = tu.EvLabelData(word=token.word)
            token_labelled[index] = True

        return label_assignment

    def compute_loss(self, token_collection, prob_matrix, label_assignment):
        """
        TODO: Calculate the error between the desired labels and the true
        cross entropy loss

        labels(posterior)
        Loss small -> result is good
        Loss high -> result is bad
        :param token_collection:
        :param ev_label_data:
        :return: loss
        """
        # loss = 0
        # for label, label_data in label_assignment.items():
        #     true_label_data = token_collection.ev_labels[label]
        #     if label_data.word != true_label_data.word:
        #         loss += 1
        word_count, label_count = prob_matrix.shape
        true_mat = np.zeros((word_count, label_count))
        for i in range(word_count):
            if token_collection.tokens[i].ev_label is not None:
                label = token_collection.tokens[i].ev_label.value
                true_mat[i, label+1] = 1

        return log_loss(true_mat[:,1:],prob_matrix[:,1:])

    def generate_true_matrix(self):

        pass

    def save_result(self, json_path):
        """
        :param json_path:
        :return:
        """
        train_data = {}

        test_counts = len(self.last_test_results)
        test_data = [None] * test_counts
        for i in range(test_counts):
            result = self.last_test_results[i]
            tokens = result["token_collection"].tokens
            assignment = {}
            for label in tu.EvLabel:
                # TODO: Fix this bug where true_l is a tuple
                true_l = result["true_label_assignment"][label],
                if type(true_l) is tuple:
                    true_l = true_l[0]
                predicted_l = result["predicted_label_assignment"][label]
                assignment[label.name] = {
                    "true": true_l.word,
                    "predicted": predicted_l.word
                }
            new_result = {
                "pmid": result["soup"].pmid.text,
                "paper_path": result["paper_path"],
                "tokens": [{"word": t.word} for t in tokens],
                "label_assignment": assignment,
                "feature_matrix": result["feature_matrix"].tolist(),
                "loss": result["loss"]
            }
            test_data[i] = new_result

        data = {
            "train_data": train_data,
            "test_data": test_data,
            "feature_names": [f.name for f in ft.Feature],
            "label_names": [l.name for l in tu.EvLabel]
        }

        util.save_dict(json_path, data)


if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paper_paths = util.get_paper_paths()[:1]
    paper_path = os.path.join(script_dir, os.pardir, "data", "annotation II",
                              "9034838.xml")
    classifier = Classifier()
    classifier.train([paper_path])