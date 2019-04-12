from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
from nltk import word_tokenize
from string import punctuation
from sklearn.svm import SVC
from os import path, pardir
import numpy as np
import joblib
import heapq
import time
import re

import ie_tools.src.token_util as tu
import ie_tools.src.feature as ft
from ie_tools.src import util

# f_max_d=25, f_n_est=200, f_min_l=50

class Classifier:
    TypeSVM = "svm"
    TypeRF = "random_forest"
    TypeDT = "decision_tree"

    def __init__(self, clf_type=None, f_max_d=10, f_n_est=200, f_min_l=50,
                 persist=False):
        if clf_type is None:
            clf_type = Classifier.TypeSVM
        self.last_precisions = None
        self.last_total_loss = None
        self.last_test_results = None
        self.last_train_paths = None
        self.persist = persist
        self.clf_type = clf_type

        # weight = 100 for actual classes,
        # the ones we are not interested in will have {-1:1}
        cls_weights = {-1: 1, }
        for ev_label in tu.EvLabel:
            cls_weights[ev_label.value] = 100

        # Classifier
        if clf_type == self.TypeDT:
            self.clf = DecisionTreeClassifier(class_weight=cls_weights)
        elif clf_type == self.TypeRF:
            self.clf = RandomForestClassifier(class_weight=cls_weights,
                                              n_jobs=-1,
                                              min_samples_leaf=f_min_l,
                                              max_depth=f_max_d,
                                              n_estimators=f_n_est)
        elif clf_type == self.TypeSVM:
            self.clf = SVC(kernel="poly", degree=3, gamma="auto",
                           probability=True, class_weight=cls_weights)
        else:
            raise ValueError(
                "Unrecognised classifier clf_type: {}".format(clf_type))

    def save_model(self, output_path):
        joblib.dump(self.clf, output_path)

    def load_model(self, pretrained_path):
        if path.exists(pretrained_path):
            self.clf = joblib.load(pretrained_path)

    def train(self, paper_paths):
        paper_soups = util.load_paper_xmls(paper_paths)
        paper_count = len(paper_soups)

        # initializing the label vectors
        # each label has an empty list []
        train_start = time.time()
        print("Training on {} paper(s)...".format(paper_count))

        # Extract feature vectors from all papers
        token_cols = [None] * paper_count
        cum_feat_matrix = np.zeros(
            (0, ft.feature_count + 1))  # +1 is bias
        cum_labels_vec = np.zeros((0, 1))
        cum_labels = np.zeros((0, len(tu.EvLabel.__members__.items())))

        for i in range(paper_count):
            # going through all papers
            soup = paper_soups[i]
            paper_id = soup.pmid.text
            # print("Processing papers {} out of {}\r".format(i + 1, paper_count))
            # print("Paper #", paper_id)

            # start = time.time()
            col = tu.TokenCollection(soup)
            col.build_tokens()
            feature_matrix = col.generate_feature_matrix()
            tokens_count, _ = feature_matrix.shape
            bias_vec = np.ones((tokens_count, 1))
            feature_matrix = np.hstack([feature_matrix, bias_vec])

            # converts a one-hot matrix (labels) into a vector of size
            # (tokens_count,1) where each value corresponds to the class ID
            # from Enum EvLabel or -1 for unclassified tokens
            labels = col.generate_train_labels()
            labels_vec = np.ones((tokens_count, 1)) * -1
            for token_i in range(tokens_count):
                for ev_label in tu.EvLabel:
                    if labels[token_i, ev_label.value] > 0:
                        labels_vec[token_i] = ev_label.value

            cum_labels = np.vstack((cum_labels, labels))

            # append current feature_matrix to cum_feat_matrix
            cum_feat_matrix = np.vstack(
                (cum_feat_matrix, feature_matrix))
            cum_labels_vec = np.vstack((cum_labels_vec, labels_vec))
            token_cols[i] = col

            # end = time.time()
            # print("Time elapsed on paper #{} ({}): {}"
            #       .format(i + 1, paper_id, np.round(end - start, 4)))

        self.clf.fit(cum_feat_matrix, cum_labels_vec.flatten())

        train_end = time.time()
        print("Done training. Time elapsed: ", train_end - train_start)
        self.last_train_paths = paper_paths

    def test(self, paper_paths):
        # Test how good our prediction is
        paper_soups = util.load_paper_xmls(paper_paths)
        paper_count = len(paper_soups)
        print("Testing on {} paper(s)...".format(paper_count))

        # Extract feature vectors from all papers
        test_results = [None] * paper_count
        losses = np.zeros((paper_count,))
        precisions = [0] * 6
        for paper_i in range(paper_count):
            soup = paper_soups[paper_i]
            print("---- Paper #{} [{}]".format(paper_i + 1, soup.pmid.text))

            col = tu.TokenCollection(soup)
            col.build_tokens()
            feature_matrix = col.generate_feature_matrix()
            tokens_count, _ = feature_matrix.shape
            bias_vec = np.ones((tokens_count, 1))
            feature_matrix = np.hstack([feature_matrix, bias_vec])

            # classify A1, A2, R1, R2, OC, P
            prob_matrix = self.clf.predict_proba(feature_matrix)

            final_prob_matrix = prob_matrix.copy()

            for tok_i in range(len(final_prob_matrix)):
                final_prob_matrix[tok_i, :] /= np.sum(final_prob_matrix[tok_i,
                                                      :])

            predictions = {}
            for ev_label in tu.EvLabel:
                predictions[ev_label] = final_prob_matrix[:, ev_label.value + 1]

            label_assignment = self.assign_ev_labels(col, predictions)
            loss = self.eval_loss(col, prob_matrix)
            losses[paper_i] = loss

            predicted_phrases = [None] * 6

            for ev_label in tu.EvLabel:
                true_ev_label_data = col.ev_labels.get(ev_label)
                if true_ev_label_data is None:
                    print("Label not found: ", ev_label)
                else:
                    ev_label_data = label_assignment[ev_label]
                    if ev_label_data.token.chunk is None:
                        predicted_phrase = ev_label_data.token.word
                    else:
                        c_i = col.chunks.index(ev_label_data.token.chunk)
                        predicted_phrase = ev_label_data.token.chunk.string

                        # Select tokens and their chunks to display
                        if len(ev_label_data.token.chunk.tokens) == 1:
                            next_tok_i = 1 + col.tokens.index(
                                ev_label_data.token.chunk.tokens[-1])
                            next_tok = col.tokens[next_tok_i].word

                            if next_tok in "%&()*+,-/:<=>[]{|}~":
                                predicted_phrase += " {}".format(next_tok)

                            predicted_phrase = predicted_phrase + " {}".format(
                                col.chunks[c_i + 1].string)
                            if col.chunks[c_i + 1].string == "with":
                                predicted_phrase = predicted_phrase + " {}".format(
                                    col.chunks[c_i + 2].string)

                            if next_tok == "(":
                                predicted_phrase += " )"

                        elif ev_label == tu.EvLabel.P:
                            c_i = col.chunks.index(ev_label_data.token.chunk)
                            predicted_phrase = predicted_phrase + " {} {}".format(
                                col.chunks[c_i + 1].string,
                                col.chunks[c_i + 2].string)

                    print("Predicted: ", ev_label.name, predicted_phrase,
                          " --- True Label: ", true_ev_label_data.word)

                    predicted_phrases[ev_label.value] = predicted_phrase
                    phrase_lowered = predicted_phrase.lower()
                    if true_ev_label_data.word in phrase_lowered or \
                            (true_ev_label_data.word == "iop" and
                             "pressure" in phrase_lowered) or \
                            (true_ev_label_data.word == "pressure" and
                             "iop" in phrase_lowered):
                        precisions[ev_label.value] += 1

            loss = np.round(loss, 4)
            print("loss for this paper is: ", loss)
            test_result = {
                "soup": soup,
                "paper_path": paper_paths[paper_i],
                "token_collection": col,
                "true_label_assignment": col.ev_labels,
                "predicted_label_assignment": label_assignment,
                "predicted_phrases": predicted_phrases,
                "feature_matrix": feature_matrix,
                "loss": loss
            }
            test_results[paper_i] = test_result

        total_loss = np.sum(losses)
        print("\n\n---------------")
        precisions = [np.round(p / paper_count, 4) for p in precisions]
        print("Average precisions for this run is: \nA1:{}\t A2:{}\t R1:"
              "{}\t R2:{}\t OC:{}\t P:{}".format(precisions[0],
                                                 precisions[1],
                                                 precisions[2],
                                                 precisions[3],
                                                 precisions[4],
                                                 precisions[5]))
        dist_loss = np.round(total_loss, 4)
        print("total loss is: ", np.round(dist_loss, 4))
        print("average loss is: ", np.round(dist_loss / paper_count, 4))
        self.last_total_loss = dist_loss
        self.last_precisions = precisions
        self.last_test_results = test_results
        return total_loss, precisions

    def assign_ev_labels(self, token_collection, predictions):
        label_assignment = {}
        best_words_count = 6
        tokens = token_collection.tokens

        def pick_best_unique_words(labels, ev_label):
            heap = [(-x, i) for i, x in enumerate(labels)]
            heapq.heapify(heap)
            results = []
            bad_results = []
            seen = {}
            best_word_index = 0
            while best_word_index < best_words_count:
                if len(heap) == 0:
                    break

                min_x, i = heapq.heappop(heap)
                token = tokens[i]
                if seen.get(token.word):
                    continue

                good = True

                # Results has to be a Cardinal Digit
                if "R" in ev_label.name:
                    if token.g_tags[tu.G_POS_TAG] != "CD":
                        good = False

                    # R do not occur before half of the report
                    elif token.abs_pos < 4:
                        good = False

                    else:
                        next_tok = tokens[i + 1]
                        pattern_r = util.get_pattern_r()
                        if not re.match(pattern_r, next_tok.word):
                            good = False

                # ignore all punctuations
                if token.word in punctuation:
                    good = False

                if token.word in util.pat_ordered:
                    good = False

                tup = (-min_x, i, ev_label)
                if good:
                    results.append(tup)
                    seen[token.word] = True
                    best_word_index += 1
                else:
                    bad_results.append(tup)

            if len(results) < best_words_count:
                results = (results + bad_results)[:best_words_count]

            return results

        all_tuples = []
        for ev_label in tu.EvLabel:
            labels = predictions[ev_label]
            best_words = pick_best_unique_words(labels, ev_label)
            all_tuples += best_words

        # sorted by overall likelihood, we don"t care which class yet
        sorted_tuples = sorted(all_tuples, key=lambda x: x[0], reverse=True)

        token_labelled = {}
        # Pick the tokens by the highest likelihood
        for tup in sorted_tuples:
            likelihood, index, ev_label = tup

            # skip if the token has been assigned
            if token_labelled.get(index):
                continue

            # skip if the ev_label has been assigned
            if label_assignment.get(ev_label) is not None:
                continue

            token = tokens[index]

            ev_label_data = tu.EvLabelData(word=token.word)
            ev_label_data.token = token
            label_assignment[ev_label] = ev_label_data
            token_labelled[index] = True

        # l_1 = [e for e in sorted_tuples if "R1" == e[2].name
        #         and tokens[e[1]].word not in punctuation]
        #
        # l_2 = [e for e in sorted_tuples if "R2" == e[2].name
        #         and tokens[e[1]].word not in punctuation]
        #
        # with open("logging.txt", "a+") as log_file:
        #     log_file.write("\n{}\n".format(token_collection.bs_doc.pmid.text))
        #
        #     for e in l_1:
        #         log_file.write("{}, {}, {}\n".format(e, tokens[e[1]].og_word,
        #                                              tokens[e[1]].chunk.string))
        #
        #     log_file.write("\n")
        #
        #     for e in l_2:
        #         log_file.write("{}, {}, {}\n".format(e, tokens[e[1]].og_word,
        #                                              tokens[e[1]].chunk.string))
        #
        #     log_file.write("-------\n\n")
        #
        # freq_l1 = {}
        # freq_l2 = {}
        #
        # # l_1 and l_2 may not have the same length
        # for e in l_1:
        #     for tok in word_tokenize(tokens[e[1]].chunk.string):
        #         if freq_l1.get(tok) is None:
        #             freq_l1[tok] = 1
        #         else:
        #             freq_l1[tok] += 1
        #
        # for e in l_2:
        #     for tok in word_tokenize(tokens[e[1]].chunk.string):
        #         if freq_l2.get(tok) is None:
        #             freq_l2[tok] = 1
        #         else:
        #             freq_l2[tok] += 1
        #
        # unit_1 = [k for k, v in freq_l1.items() if v == max(freq_l1.values())]
        # unit_2 = [k for k, v in freq_l2.items() if v == max(freq_l2.values())]
        #
        # l_1 = [e for e in l_1 if any(u in unit_1 for
        #                              u in word_tokenize(tokens[e[1]].chunk.string))]
        #
        # l_2 = [e for e in l_2 if any(u in unit_2 for
        #                              u in word_tokenize(tokens[e[1]].chunk.string))]
        #
        # # sorting based on prob*pos
        # # sorted(l_1, key=lambda e: e[0]*e[1], reverse=True)
        #
        # likelihood, index, ev_label = l_1[0]
        # token = tokens[index]
        #
        # ev_label_data = tu.EvLabelData(word=token.word)
        # ev_label_data.token = token
        # label_assignment[ev_label] = ev_label_data
        #
        # _, _, ev_label = l_2[0]
        # foo = [e[1] for e in l_2 if e[1] != index]
        # index = np.argmin([np.abs(n-index) for n in foo])
        # token = tokens[foo[index]]
        #
        # ev_label_data = tu.EvLabelData(word=token.word)
        # ev_label_data.token = token
        # label_assignment[ev_label] = ev_label_data

        return label_assignment

    def eval_loss(self, token_collection, prob_matrix):
        word_count, label_count = prob_matrix.shape
        true_mat = np.zeros((word_count, label_count))

        for i in range(word_count):
            if token_collection.tokens[i].ev_label is not None:
                label = token_collection.tokens[i].ev_label.value
                true_mat[i, label + 1] = 1

        return log_loss(true_mat[:, 1:], prob_matrix[:, 1:])

    def save_result(self, json_path):
        test_counts = len(self.last_test_results)
        test_data = [None] * test_counts

        for i in range(test_counts):
            result = self.last_test_results[i]
            tokens = result["token_collection"].tokens
            assignment = {}

            for label in tu.EvLabel:
                true_l = result["true_label_assignment"].get(label)

                if type(true_l) is tuple:
                    true_l = true_l[0]
                predicted_l = result["predicted_label_assignment"][label]
                assignment[label.name] = {
                    "true": true_l.word if true_l is not None else "undefined",
                    "predicted": predicted_l.word,
                    "predicted_phrase": result["predicted_phrases"][label.value]
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
            "test_data": test_data,
            "feature_names": [f.name for f in ft.Feature],
            "label_names": [l.name for l in tu.EvLabel],
            "total_loss": self.last_total_loss,
            "precisions": self.last_precisions
        }

        util.save_dict(json_path, data)


def main():
    script_dir = path.dirname(path.realpath(__file__))
    # paper_paths = util.get_paper_paths()[:1]
    paper_path = path.join(script_dir, pardir, "data", "abstracts_structured",
                           "9034838.xml")
    classifier = Classifier()
    classifier.train([paper_path])


if __name__ == "__main__":
    main()
