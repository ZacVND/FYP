from src.classifier import Classifier
import sklearn.model_selection as ms
from datetime import datetime
import src.util as util
from os import path, listdir
import webbrowser

script_dir = path.dirname(path.abspath(__file__))
unstr_data_path = path.join(script_dir, 'data', 'annotation I excluded files',
                            'unstructured')
results_dir = path.join(script_dir, "results")

if __name__ == "__main__":
    random = True
    unstruct = True

    classifier_type = 'forest'  # choose between forest, svm, tree
    max_papers = 5
    paper_paths = util.get_paper_paths()[:max_papers]

    if random:
        train_pps, test_pps = ms.train_test_split(paper_paths, test_size=0.2)
    else:
        i = len(paper_paths)
        train_pps = paper_paths[:int(i / 2)]
        test_pps = paper_paths[int(i / 2) + 1:]

    prename = ''

    if unstruct:
        prename = 'unstruct_'
        for file in listdir(unstr_data_path):
            # ignore .DS_Store files
            if file.startswith(".DS"):
                continue
            test_pps.append(path.join(unstr_data_path, file))

    classifier = Classifier(clf_type=classifier_type)
    # first paper will always take longer to run than the subsequent
    # papers because we starts genia tagger.
    classifier.train(train_pps)
    classifier.test(test_pps)

    template_path = path.join(script_dir, "src", "results.pug")
    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    out_file = path.join(results_dir, "results-{}.html".format(prename,
                                                                  date_str))
    json_path = path.join(results_dir,
                          "results-{}.json".format(date_str))

    classifier.save_result(json_path)
    util.render_pug(template_path, out_file=out_file, json_path=json_path)
    webbrowser.open("file://" + out_file)
