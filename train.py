from src.classifier import Classifier
import sklearn.model_selection as ms
from datetime import datetime
import src.util as util
from os import path
import webbrowser

script_dir = path.dirname(path.abspath(__file__))
results_dir = path.join(script_dir, "results")

if __name__ == "__main__":
    max_papers = 50
    paper_paths = util.get_paper_paths()[:max_papers]
    train_pps, test_pps = ms.train_test_split(paper_paths,
                                              test_size=0.1)
    classifier = Classifier()
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
