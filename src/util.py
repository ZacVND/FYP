from Authentication import Authentication
from os import path, listdir
import requests
import logging
import json
import bs4

logging.basicConfig(format='[%(name)s|%(levelname)s] %(message)s',
                    level=logging.INFO)

api_key = "bea4b3d4-f1ef-439e-b68f-3564c8c7a231"
auth_client = Authentication(api_key)
tgt = auth_client.gettgt()

_end = '__end'

script_dir = path.dirname(path.abspath(__file__))
data_path = path.join(script_dir, "..", 'data/annotation I/')


def get_logger(name):
    return logging.getLogger(name)


def load_dict(filename):
    try:
        cache = json.load(open(filename, 'r'))
    except (IOError, ValueError):
        cache = {}
    return cache


def save_dict(filename, dict):
    json.dump(dict, open(filename, 'w'))


def make_trie(mapping):
    root = dict()
    for key, str_arr in mapping.items():
        for string in str_arr:
            string = string.lower()
            current_dict = root
            for letter in string:
                current_dict = current_dict.setdefault(letter, {})
            current_dict[_end] = key
    return root


def check_trie(trie, string):
    curr_dict = trie
    string = string.lower()
    for letter in string:
        curr_dict = curr_dict.get(letter)
        if curr_dict is None:
            return None

    key = curr_dict.get(_end)
    if key is not None:
        return key

    return None


def get_paper_paths():
    return [path.join(data_path, file) for file in sorted(listdir(data_path))]


def parse_paper(paper_path):
    with open(paper_path) as f:
        raw = f.read()
        soup = bs4.BeautifulSoup(raw, "html5lib")
    return soup


def get_umls_classes(string):
    """
    Query the param with UMLS REST API and build up its class, only consider
    the first item in the results.
    Original source from NLM UMLS API examples.
    https://github.com/HHS/uts-rest-api/blob/master/samples/python/search-terms.py
    :param string:
    :return class:
    """
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

    return classes


if __name__ == '__main__':
    # classes = get_umls_classes("timolol")
    # assert ('Pharmacologic Substance' in classes)
    class_to_feature_mapping = {
        13: ['Pharmacologic Substance', 'Antibiotic',
             'Organic Chemical', 'Biomedical or Dental Material']
    }
    trie = make_trie(class_to_feature_mapping)
    print(trie)
    print(check_trie(trie, 'Pharmacologic Substance'))
    print(check_trie(trie, 'Random bullshit'))
