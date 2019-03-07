from nltk.corpus import stopwords
import token_utils as tu
import feature as ft
import numpy as np
import time
import util
import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))

max_papers = 100
paper_paths = util.get_paper_paths()[:max_papers]
paper_soups = util.load_paper_xmls(paper_paths)

label_dict = dict()

logger = util.get_logger("run")

sw = stopwords.words('english') + ['non']

if __name__ == "__main__":
    xml_file = os.path.join(util.data_path, "25847610.xml")
    ps = util.parse_paper(xml_file)
    col = tu.TokenCollection(ps)
    col.build_tokens()
    # extracting labels and nlmcategory
    # for ps in paper_soups:
    #     for tag in ps.findAll("abstracttext"):
    #         value = tag['nlmcategory']
    #         key = tag['label']
    #         label_dict[key] = value
    #
    # label_dict = sorted(label_dict.items(), key=lambda x: x[1])
    # with open('label_dict.json', 'w') as fp:
    #     json.dump(dict(label_dict), fp)

    # # Prepare papers' XML
    # # paper_files = sorted(listdir(data_path))[:4]
    # paper_paths = util.get_paper_paths()
    # paper_count = len(paper_paths)
    # paper_soups = [None] * paper_count
    # for i in range(len(paper_paths)):
    #     paper_soups[i] = util.parse_paper(paper_paths[i])
    #
    # # Extract feature vectors from all papers
    # token_cols = [None] * paper_count
    # feature_count = ft.feature_count
    # all_feat_vecs = np.ones((0, feature_count + 1))
    # all_a1_labs = np.zeros((0,))
    # all_a2_labs = np.zeros((0,))
    # all_r1_labs = np.zeros((0,))
    # all_r2_labs = np.zeros((0,))
    # all_oc_labs = np.zeros((0,))
    # all_p_labs = np.zeros((0,))
    # for i in range(paper_count):
    #     soup = paper_soups[i]
    #     start = time.time()
    #
    #     col = tu.TokenCollection(soup)
    #     col.build_tokens()
    #     feature_vector = col.generate_feature_matrix()
    #     (a1_labels,
    #      a2_labels,
    #      r1_labels,
    #      r2_labels,
    #      oc_labels,
    #      p_labels) = col.generate_train_labels()
    #
    #     all_feat_vecs = np.vstack((all_feat_vecs, feature_vector))
    #     all_a1_labs = np.concatenate((all_a1_labs, a1_labels))
    #     all_a2_labs = np.concatenate((all_a2_labs, a2_labels))
    #     all_r1_labs = np.concatenate((all_r1_labs, r1_labels))
    #     all_r2_labs = np.concatenate((all_r2_labs, r2_labels))
    #     all_oc_labs = np.concatenate((all_oc_labs, oc_labels))
    #     all_p_labs = np.concatenate((all_p_labs, p_labels))
    #
    #     token_cols[i] = col
    #
    #     end = time.time()
    #     print('Time elapsed on paper #{}: {}'.format(i + 1, end - start))
    #
    #
    #     print('\n')

    print("Done")
