from src.classifier import Classifier
import sklearn.model_selection as ms
from datetime import datetime
import src.util as util
from os import path
import webbrowser

script_dir = path.dirname(path.abspath(__file__))
results_dir = path.join(script_dir, "results")

if __name__ == "__main__":
    random = True

    max_papers = 120
    paper_paths = util.get_paper_paths()[:max_papers]

    if random:
        train_pps, test_pps = ms.train_test_split(paper_paths, test_size=0.2)
    else:
        i = len(paper_paths)
        train_pps = paper_paths[:int(i/2)]
        test_pps = paper_paths[int(i/2)+1:]

    classifier = Classifier(clf_type='svm')
    # first paper will always take longer to run than the subsequent
    # papers because we starts genia tagger.
    classifier.train(train_pps)
    classifier.test(test_pps)

    template_path = path.join(script_dir, "src", "results.pug")
    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    out_file = path.join(results_dir, "results-{}.html".format(date_str))
    json_path = path.join(results_dir,
                          "results-{}.json".format(date_str))

    classifier.save_result(json_path)
    util.render_pug(template_path, out_file=out_file, json_path=json_path)
    webbrowser.open("file://" + out_file)
