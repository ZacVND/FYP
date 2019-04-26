"""
@author: ZacVND

k-fold cross-validation using LOO
"""

from datetime import datetime
from os import path
import numpy as np
import random

from ie_tools.src.classifier import Classifier
from ie_tools.src import util


def run():
    # choose between TypeRF, TypeDT, TypeSVM
    classifier_type = Classifier.TypeRF
    run_count = 20
    fold_count = 8
    max_papers = 120
    paper_paths = util.get_paper_paths()[:max_papers]
    paper_count = len(paper_paths)
    step_size = int(paper_count / fold_count)

    new_data = False
    if new_data:
        new_data_dir = util.get_new_data_dir()
        new_pps = util.get_paper_paths(dir=new_data_dir)
        paper_paths.extend(new_pps)

    curr_i = 1
    running_sum = 0
    running_pre_sum = [0] * 6

    print("---- INFO: First paper will always take longer to run than the "
          "subsequent papers because we starts genia tagger.\n\n")

    for run in range(run_count):
        random.shuffle(paper_paths)
        for fold in range(fold_count):
            train_pps = paper_paths[:step_size * fold]
            train_pps += paper_paths[step_size * (fold + 1):]
            test_pps = paper_paths[step_size * fold:step_size * (fold + 1)]
            classifier = Classifier(clf_type=classifier_type)
            classifier.train(train_pps)
            current_total_loss, current_precisions = classifier.test(test_pps)

            running_avg = np.divide(running_sum, curr_i) + \
                          np.divide(current_total_loss, curr_i)

            running_sum = np.add(running_sum, current_total_loss)

            running_pre_avg = np.divide(running_pre_sum, curr_i) + \
                              np.divide(current_precisions, curr_i)

            running_pre_sum = np.add(running_pre_sum, current_precisions)

            running_pre_avg = [np.round(rpa, 4) for rpa in running_pre_avg]

            print("Running average: {}\n".format(np.round(running_avg, 4)))
            print("Running precisions average:\nA1:{}\t A2:{}\t R1:{}\t "
                  "R2:{}\t OC:{}\t P:{}\n\n".format(running_pre_avg[0],
                                                    running_pre_avg[1],
                                                    running_pre_avg[2],
                                                    running_pre_avg[3],
                                                    running_pre_avg[4],
                                                    running_pre_avg[5]))

            curr_i += 1

    avg_total_loss = np.round(np.divide(running_sum,
                                        (run_count * fold_count)), 4)
    print("average total loss for {}-fold cross validation".format(fold_count),
          avg_total_loss)

    avg_total_precisions = np.round(np.divide(running_pre_sum,
                                              (run_count * fold_count)), 4)
    print("average precision for {}-fold cross validation".format(fold_count),
          avg_total_precisions)

    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    out_file = path.join(util.get_result_dir(), "{}-results-{}.txt".format(
        classifier_type, date_str))
    with open(out_file, 'w+') as file:
        file.write("Average total loss across {} runs of {}-fold cross "
                   "validation:\n{}\n\n".format(run_count, fold_count,
                                                avg_total_loss))
        file.write("Average precisions across {} runs of {}-fold cross "
                   "validation:\nA1:{}\t A2:{}\t R1:{}\t R2:{}\t "
                   "OC:{}\t P:{}".format(run_count, fold_count,
                                         avg_total_precisions[0],
                                         avg_total_precisions[1],
                                         avg_total_precisions[2],
                                         avg_total_precisions[3],
                                         avg_total_precisions[4],
                                         avg_total_precisions[5]))


if __name__ == "__main__":
    run()
