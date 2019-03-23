import sklearn.model_selection as ms
from datetime import datetime
from os import path, listdir
import webbrowser

from ie_tools.src.classifier import Classifier
import ie_tools.src.util as util

if __name__ == "__main__":
    # choose between TypeRF, TypeDT, TypeSVM
    classifier_type = Classifier.TypeRF
    random = True
    persist = True
    unstructured = False
    max_papers = 120
    paper_paths = util.get_paper_paths()[:max_papers]

    if random:
        train_pps, test_pps = ms.train_test_split(paper_paths, test_size=0.2)
    else:
        i = len(paper_paths)
        train_pps = paper_paths[:int(i / 2)]
        test_pps = paper_paths[int(i / 2) + 1:]

    prefix = classifier_type
    unstructured_dir = util.get_unstructured_dir()
    if unstructured:
        prefix = prefix + '-unstructured'
        for file in listdir(unstructured_dir):
            # ignore .DS_Store files
            if file.startswith(".DS"):
                continue
            test_pps.append(path.join(unstructured_dir, file))

    classifier = Classifier(clf_type=classifier_type, persist=True, f_max_d=25,
                            f_min_l=12, f_n_est=70)
    pretrained = path.join(util.get_pretrained_dir(), "{}.sav".format(
        classifier_type))

    print("---- INFO: First paper will always take longer to run than the "
          "subsequent papers because we starts genia tagger.\n\n")

    if path.isfile(pretrained):
        print("Pretrained weights for {} exist. "
              "Using existing weights.".format(classifier_type))
        classifier.load_model(pretrained)
    else:
        print("Pretrained weights for {} not found. "
              "Training from scratch.".format(classifier_type))
        classifier.train(train_pps)
        classifier.save_model(pretrained)

    classifier.test(test_pps)

    template_path = util.get_template_path()
    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    results_dir = util.get_result_dir()
    out_file = path.join(results_dir, "{}-results-{}.html".format(prefix,
                                                                  date_str))
    json_path = path.join(results_dir,
                          "{}-results-{}.json".format(prefix, date_str))

    classifier.save_result(json_path)
    util.render_pug(template_path, out_file=out_file, json_path=json_path)
    webbrowser.open("file://" + out_file)
