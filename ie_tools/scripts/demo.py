from nltk.corpus import stopwords
import numpy as np
import time
import json
import os

import ie_tools.src.token_utils as tu
import ie_tools.src.feature as ft
from ie_tools.src import util

script_dir = os.path.dirname(os.path.abspath(__file__))


max_papers = 100
paper_paths = util.get_paper_paths()[:max_papers]
paper_soups = util.load_paper_xmls(paper_paths)

label_dict = dict()

logger = util.get_logger("run")

sw = stopwords.words('english') + ['non']

if __name__ == "__main__":

    print("Done")
