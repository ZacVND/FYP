"""
@author: ZacVND

Hold-out run
"""

import sklearn.model_selection as ms
from datetime import datetime
from os import path, listdir
import webbrowser

from ie_tools.src.classifier import Classifier
import ie_tools.src.util as util


def run(demo=False):
    # choose between TypeRF, TypeDT, TypeSVM
    classifier_type = Classifier.TypeRF
    prefix = classifier_type

    random = True
    persist = False
    pretrain = False
    unstructured = False
    new_test_only = False

    max_papers = 120
    paper_paths = util.get_paper_paths()[:max_papers]

    if random:
        train_pps, test_pps = ms.train_test_split(paper_paths, test_size=0.2)
    else:
        train_pps, test_pps = ms.train_test_split(paper_paths, test_size=0.2,
                                                  random_state=1)

    # uncomment the block below if you want new data to be included
    # new_data_dir = util.get_new_data_dir()
    # new_pps = util.get_paper_paths(dir=new_data_dir)
    # if new_test_only:
    #     test_pps = new_pps
    #     train_pps = paper_paths
    #     prefix = prefix + '-newtest'
    # else:
    #     paper_paths.extend(new_pps)
    #     train_pps, test_pps = ms.train_test_split(paper_paths, test_size=0.2)

    if unstructured:
        unstructured_dir = util.get_unstructured_dir()
        unstructured_pps = util.get_paper_paths(dir=unstructured_dir)
        # train_pps.extend(unstructured_pps)
        test_pps.extend(unstructured_pps)
        prefix = prefix + '-unstructured'

    classifier = Classifier(clf_type=classifier_type, persist=persist)
    pretrained = path.join(util.get_pretrained_dir(), "{}.sav".format(
        classifier_type))

    print("---- INFO: First paper will always take longer to run than the "
          "subsequent papers because we start genia tagger.\n\n")

    if path.isfile(pretrained) and pretrain:
        print("Pretrained weights for {} exist. "
              "Using existing weights.".format(classifier_type))
        classifier.load_model(pretrained)
    else:
        print("Pretrained weights for {} not found. "
              "Training from scratch.".format(classifier_type))
        classifier.train(train_pps)
        classifier.save_model(pretrained)

    classifier.test(test_pps)

    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    results_dir = util.get_result_dir()

    if demo:
        template_path = util.get_demo_template_path()
        prefix = 'demo-' + prefix
    else:
        template_path = util.get_template_path()

    out_file = path.join(results_dir, "{}-results-{}.html".format(prefix,
                                                                  date_str))
    json_path = path.join(results_dir, "{}-results-{}.json".format(prefix,
                                                                   date_str))

    classifier.save_result(json_path)
    util.render_pug(template_path, out_file=out_file, json_path=json_path)
    webbrowser.open("file://" + out_file)


if __name__ == "__main__":
    run()
