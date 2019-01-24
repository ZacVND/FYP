from src.classifier import Classifier
import sklearn.model_selection as ms
import src.util as util

if __name__ == "__main__":
    paper_paths = util.get_paper_paths()
    train_pps, test_pps = ms.train_test_split(paper_paths,
                                            test_size=0.1)
    classifier = Classifier()
    classifier.train(train_pps)
    classifier.test(test_pps)

