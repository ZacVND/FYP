import sklearn.model_selection as ms
from datetime import datetime
from os import path, listdir
import webbrowser

import ie_tools.src.util as util
from ie_tools.src.classifier import Classifier


if __name__ == "__main__":
    random = True
    persist = True
    unstruct = False

    # choose between TypeRF, TypeDT, TypeSVM
    classifier_type = Classifier.TypeRF
    max_papers = 120
    paper_paths = util.get_paper_paths()[:max_papers]

    if random:
        train_pps, test_pps = ms.train_test_split(paper_paths, test_size=0.2)
    else:
        i = len(paper_paths)
        train_pps = paper_paths[:int(i / 2)]
        test_pps = paper_paths[int(i / 2) + 1:]

    prefix = ''
    unstruct_dir = util.get_unstruct_dir()
    if unstruct:
        prefix = 'unstruct-'
        for file in listdir(unstruct_dir):
            # ignore .DS_Store files
            if file.startswith(".DS"):
                continue
            test_pps.append(path.join(unstruct_dir, file))

    classifier = Classifier(type=classifier_type, persist=True)
    pretrained = path.join(util.get_pretrained_dir(), "{}.sav".format(
        classifier_type))

    print("---- INFO: First paper will always take longer to run than the "
          "subsequent papers because we starts genia tagger.\n\n")

    if path.isfile(pretrained):
        print("Pretrained weights exist.")
        classifier.load_model(pretrained)
    else:
        print("Pretrained weights not found. Training from scratch.")
        classifier.train(train_pps)
        classifier.save_model(pretrained)

    classifier.test(test_pps)

    template_path = util.get_template_path()
    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    results_dir = util.get_result_dir()
    out_file = path.join(results_dir, "{}results-{}.html".format(prefix,
                                                                 date_str))
    json_path = path.join(results_dir,
                          "{}results-{}.json".format(prefix, date_str))

    classifier.save_result(json_path)
    util.render_pug(template_path, out_file=out_file, json_path=json_path)
    webbrowser.open("file://" + out_file)
