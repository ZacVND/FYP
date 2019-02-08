from Authentication import Authentication
from os import path, listdir
import requests
import logging
import shutil
import heapq
import json
import time
import bs4
import os

logging.basicConfig(format='[%(name)s|%(levelname)s] %(message)s',
                    level=logging.INFO)

api_key = "bea4b3d4-f1ef-439e-b68f-3564c8c7a231"
auth_client = Authentication(api_key)
tgt = auth_client.gettgt()

_end = '__end'

script_dir = path.dirname(path.abspath(__file__))
data_path = path.join(script_dir, "..", 'data/annotation I/')

last_time = time.time()


def log(string):
    global last_time
    diff = time.time() - last_time
    last_time = time.time()
    print("{}, (+{}s)".format(string, round(diff, 4)))


def get_logger(name):
    return logging.getLogger(name)


def load_dict(file_path):
    try:
        cache = json.load(open(file_path, 'r'))
    except (IOError, ValueError):
        cache = {}
    return cache


def save_dict(file_path, dict):
    json.dump(dict, open(file_path, 'w'))


def render_pug(template_path, out_path=None, out_file=None, json_path=None):
    if shutil.which("pug") is None:
        print("The command `pug` is not available! You will need to install it"
              " yourself - https://pugjs.org")
        return

    command = "pug"
    if out_file is not None:
        command += " < {}".format(template_path)
        command += " > {}".format(out_file)
        if json_path is not None:
            command += " -O {}".format(json_path)

    else:
        command += " {}".format(template_path)
        if out_path is not None:
            command += " -o {}".format(json_path)
            if json_path is not None:
                command += " -O {}".format(json_path)

    os.system(command)


def get_paper_paths():
    return [path.join(data_path, file) for file in sorted(listdir(data_path))]


def load_paper_xmls(paper_paths):
    paper_count = len(paper_paths)
    paper_soups = [None] * paper_count
    for i in range(len(paper_paths)):
        paper_soups[i] = parse_paper(paper_paths[i])

    return paper_soups


def parse_paper(paper_path):
    # NOTE: if error encounters then remove .DS_Store file in data folder
    with open(paper_path) as f:
        raw = f.read()
        soup = bs4.BeautifulSoup(raw, "html5lib")
    return soup


def get_n_max_indices(values, n):
    heap = [(-x, i) for i, x in enumerate(values)]
    heapq.heapify(heap)
    results = []
    for i in range(n):
        curr_item = heapq.heappop(heap)
        results.append(curr_item[1])

    return results


class Cache:
    def __init__(self, file_path):
        self.file_path = file_path
        self.cache = load_dict(file_path)

    def set(self, key, value):
        self.cache[key] = value

    def get(self, key):
        return self.cache.get(key)

    def save(self):
        save_dict(self.file_path, self.cache)


class Trie:
    def __init__(self, mapping=None, strings=None):
        if mapping is not None:
            self.trie = self.make_trie_from_mapping(mapping)
        else:
            self.trie = self.make_trie_from_strings(strings)

    def make_trie_from_mapping(self, mapping):
        root = dict()
        for key, str_arr in mapping.items():
            for string in str_arr:
                string = string.lower()
                current_dict = root
                for letter in string:
                    current_dict = current_dict.setdefault(letter, {})
                current_dict[_end] = key
        return root

    @staticmethod
    def make_trie_from_strings(strings):
        root = dict()
        for string in strings:
            string = string.lower()
            current_dict = root
            for letter in string:
                current_dict = current_dict.setdefault(letter, {})
            current_dict[_end] = True
        return root

    def check(self, string):
        curr_dict = self.trie
        string = string.lower()
        for letter in string:
            curr_dict = curr_dict.get(letter)
            if curr_dict is None:
                return None

        key = curr_dict.get(_end)
        if key is not None:
            return key

        return None


umls_cache_path = path.join(script_dir, 'umls_cache.json')
umls_cache = Cache(file_path=umls_cache_path)


def get_umls_classes(string):
    """
    Query the param with UMLS REST API and build up its class, only consider
    the first item in the results.
    Original source from NLM UMLS API examples.
    https://github.com/HHS/uts-rest-api/blob/master/samples/python/search-terms.py
    :param string:
    :return classes: The Semantic classes of the string
    """
    cached = umls_cache.get(string)
    if cached is not None:
        return cached

    uri = "https://uts-ws.nlm.nih.gov"
    version = "2018AB"
    content_endpoint = "/rest/search/" + version

    # REQUIRED: generate a ticket for each request
    ticket = auth_client.getst(tgt)

    query = {'string': string, 'ticket': ticket, 'pageNumber': 1,
             'pageSize': 1,
             'searchType': 'exact'}
    search_result = requests.get(uri + content_endpoint, params=query)
    search_result.encoding = 'utf-8'
    results = json.loads(search_result.text)["result"]["results"]

    uri_2 = 'https://uts-ws.nlm.nih.gov/rest'
    content_endpoint_2 = '/content/' + version + '/CUI/'

    classes = []

    for result in results:

        uid = result["ui"]
        if uid is None or uid == 'NONE':
            continue

        try:
            ticket = auth_client.getst(tgt)
            query_2 = {'ticket': ticket}
            r_2 = requests.get(uri_2 + content_endpoint_2 + uid,
                               params=query_2)

            if not 200 <= r_2.status_code < 300:
                continue

            r_2.encoding = 'utf-8'
            items = json.loads(r_2.text)
            json_sem = items['result']['semanticTypes']
            for result in json_sem:
                classes.append(result['name'])
        except json.JSONDecodeError:
            print('Got bad JSON!')

    umls_cache.set(string, classes)
    return classes


if __name__ == '__main__':
    classes = get_umls_classes("timolol")
    assert ('Pharmacologic Substance' in classes)
    class_to_feature_mapping = {
        13: ['Pharmacologic Substance', 'Antibiotic',
             'Organic Chemical', 'Biomedical or Dental Material']
    }
    pass
