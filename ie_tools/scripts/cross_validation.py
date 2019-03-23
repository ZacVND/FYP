from datetime import datetime
from os import path
import numpy as np
import webbrowser
import random

from ie_tools.src.classifier import Classifier
from ie_tools.src import util

if __name__ == "__main__":
    # choose between TypeRF, TypeDT, TypeSVM
    classifier_type = Classifier.TypeRF
    run_count = 10
    fold_count = 5
    max_papers = 120
    paper_paths = util.get_paper_paths()[:max_papers]
    paper_count = len(paper_paths)
    step_size = int(paper_count / fold_count)

    curr_i = 1
    running_sum = 0
    running_avg = 0

    print("---- INFO: First paper will always take longer to run than the "
          "subsequent papers because we starts genia tagger.\n\n")

    for run in range(run_count):
        random.shuffle(paper_paths)
        for fold in range(fold_count):
            train_pps = paper_paths[:step_size * fold]
            train_pps += paper_paths[step_size * (fold + 1):]
            test_pps = paper_paths[step_size * fold:step_size * (fold + 1)]
            classifier = Classifier(clf_type=classifier_type, f_max_d=25,
                                    f_min_l=12, f_n_est=70)
            # first paper will always take longer to run than the subsequent
            # papers because we starts genia tagger.
            classifier.train(train_pps)
            current_total_loss = classifier.test(test_pps)
            running_avg = (running_sum / curr_i) + (current_total_loss / curr_i)
            running_sum += current_total_loss
            curr_i += 1
            print("Running average: {}\n\n".format(np.round(running_avg, 4)))

    avg_total_loss = np.round(running_sum / (run_count * fold_count), 4)
    print("average total loss for 5-fold cross validation", avg_total_loss)

    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    out_file = path.join(util.get_result_dir(), "{}-results-{}.txt".format(
        classifier_type, date_str))
    with open(out_file, 'w+') as file:
        file.write("Average total loss across 10 runs of 5-fold cross "
                   "validation:\n{}".format(avg_total_loss))

    # template_path = path.join(script_dir, "src", "results.pug")
    # date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    # out_file = path.join(results_dir, "results-{}.html".format(date_str))
    # json_path = path.join(results_dir,
    #                       "results-{}.json".format(date_str))
    #
    # classifier.save_result(json_path)
    # util.render_pug(template_path, out_file=out_file, json_path=json_path)
    # webbrowser.open("file://" + out_file)
