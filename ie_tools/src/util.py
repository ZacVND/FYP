"""
@author: ZacVND

Based on Antonio Trenta's take_abstract.py which was written in Python 2
"""

from nltk.corpus import stopwords
from os import path, listdir
import requests
import logging
import shutil
import heapq
import json
import time
import bs4
import os
import re
import nltk

from ie_tools.libraries.Authentication import Authentication

# PLEASE CHANGE THE API KEY HERE!!!
# Refer to README on how to get one
# TODO: MOVE API key to a file
api_key = "bea4b3d4-f1ef-439e-b68f-3564c8c7a231"

auth_client = Authentication(api_key)
tgt = auth_client.gettgt()

_end = "__end"

logging.basicConfig(format="[%(name)s|%(levelname)s] %(message)s",
                    level=logging.INFO)

script_dir = path.dirname(path.abspath(__file__))

results_dir = path.normpath(path.join(script_dir, "..", "..", "results"))

structured_dir = path.normpath(path.join(script_dir, "..", "..",
                                   "data", "abstracts_structured"))

preprocessed_dir = path.normpath(path.join(script_dir, "..", "..",
                                           "data", "preprocessed"))

unstructured_dir = path.normpath(path.join(script_dir, "..", "..", "data",
                                           "abstracts_unstructured"))

testing_dir = path.normpath(path.join(script_dir, "..", "..", "data",
                                      "testing"))

new_data_dir = path.normpath(path.join(script_dir, "..", "..", "new_data"))

pretrained_dir = path.normpath(path.join(script_dir, "..", "..", "pretrained"))

template_path = path.normpath(path.join(script_dir, "results.pug"))

demo_template_path = path.normpath(path.join(script_dir, "demo.pug"))

genia_tagger_path = path.normpath(path.join(script_dir, "..", "..",
                                            "geniatagger-3.0.2", "geniatagger"))

umls_cache_path = path.normpath(path.join(script_dir, "..", "..", "data",
                                          "umls_cache.json"))

abbrev_dict_path = path.normpath(path.join(script_dir, "..", "..", "data",
                                           "abbreviations.json"))

# anything that is of type num-chars
pattern_num_char = re.compile(r"[\d\w]+(?:-\w+)+")

# pattern that matches Results tokens/phrases
pattern_r = re.compile(r"mm|mm\s*[Hh][Gg]|mg|percent|patients|months|vs|"
                       r"%|,|to|\(|\+\s*/\s*-?|±")

last_time = time.time()

with open(abbrev_dict_path, "r") as file:
    abbrev_dict = json.load(file)

sw = stopwords.words("english") + ["non"]


# Abbreviation expansion
def abbreviations(text):
    """
    finds all the abbreviations in text and returns them as a dictionary
    with the abbreviations as keys and expansions as values
    """

    tokens = nltk.word_tokenize(text)
    sent_out = str(text)
    foo = {}
    obrack_pt = re.compile(r"[\(\[]")
    cbrack_pt = re.compile(r"s?[\)\]]")

    for i, t in enumerate(tokens[:-2]):
        tests = (obrack_pt.search(t) and tokens[i + 1].isupper()
                 and cbrack_pt.search(tokens[i + 2])
                 and not "=" in tokens[i + 1])
        if tests:
            foo[tokens[i + 1]] = [w.title() for w in tokens[:i]]
            sent_out = re.sub(r"[\(\[]" + tokens[i + 1] + "[\)\]]", "",
                              sent_out)

    for a in foo.keys():
        candidates = []
        for i, w in enumerate(reversed(foo[a])):
            if i > len(a) + 1: break
            condition = (i > (len(a) - 3) if len(
                [1 for l in a if l == a[0]]) > 1 else True)
            if condition and w.lower().startswith(a[0].lower()) and not w in sw:
                candidates.append(foo[a][-(i + 1):])
        candidates.sort(key=lambda x: len(x))
        foo[a] = (candidates[0] if candidates else [])

    return [foo, sent_out]


def expand_abbreviations(sent):
    """
    returns a copy of "sent" with abbreviations expanded
    """

    [abbrev_new, sent_new] = abbreviations(sent)
    abbrev_dict.update({k: v for (k, v) in abbrev_new.items() if v})
    keys = sorted(abbrev_dict.keys(), key=lambda x: -len(x))

    for k in keys:
        neww = (" ".join(abbrev_dict[k]) if type(abbrev_dict[k]) is list
                else abbrev_dict[k])
        sent_new = (re.sub(k, neww, sent_new)
                    if neww else sent_new)

    return sent_new


# Normalisation

# - list of patterns
patterns = {
    "mmHg": "[Mm]+\s*[Hh][Gg]",
    # dates
    "_DATE_": r"""(?x)
                    (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?
                        |Apr(?:il)?|May|June?|July?
                        |Aug(?:ust)?|Sep(?:tember)?
                        |Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
                    (\s*[0123]?\d[\,\s+])?
                    (?:\s*(?:19|20)\d{2})? 
                    (?=\W+)                     #ie: January 01, 2011
                  | [0123]?\d\s*
                    (?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?
                        |[Aa]pr(?:il)?|[Mm]ay|[Jj]une?|[Jj]uly?
                        |[Aa]ug(?:ust)?|[Ss]ep(?:tember)?
                        |[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)
                    (?:[\,\s+]\s*(?:19|20)\d{2})?       #ie: 12 Jan 2011
                  | [0123]?\d[\-\/]
                      [01]?\d[\-\/]
                      (?:19|20)?\d{2}           #ie: 12/01/2001
                  | [0123]?\d[\-\/]
                      (?:[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul
                        |[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec)[\-\/]
                      (?:19|20)?\d{2}           #ie: 12/jan/2001""",
    # periods of time
    "_POFT_": r"""(?x) \d{1,3}(\.\d*)?\-(?:[Mm]inute(s?)?|[Hh]ours?|[Dd]ay(s?)?|[Ww]eek(s?)?
                                 |[Mm]onth(s?)?|[Yy]ear(s?)?)(?=[\s\W])
                      | (?<=\W)\d+(\.\d*)?-?(?=\,?\s(\d+-?\,?\s)?(to|or)\s_POFT_)
                      | \d+(\.\d*)?-?\s*(?:[Mm]inute(s?)?|[Hh]our(s?)?|[Dd]ay(s?)?
                      | [Ww]eek(s?)?|[Mm]onth(s?)?|[Yy]ear(s?)?)(?=[\s\W])
                      | ([Ww]eek|[Mm]onth|[Yy]ear)[Ss]?\s+\d
                      | (?<=_POFT_\,\s)+\s*\d+
                      | (?<=_POFT_\,\sand\s)\d+
                      | (?<=_POFT_\sand\s)\d+""",
    # urls
    "_URL_": r"(?:http\:\/\/)?www\..+\.(?:com|org|gov|net|edu)",
    # population
    "_POP_": r"""(?x) (?<=\W)[Nn]\s*\=\s*\d+
                    | _POP_\,\s*\d+
                    | _POP_\sand\s\d+
                    | _POP_\,\s\d+\sand\s\d+
                    | _POP_\,\s\d+\,\sand\s\d+""",
    # ratios
    "_RATIO_": r"""(?x) (\d+|\_RATIO\_)[\:\/]\d
                   | d+\sof\sd+""",
    # p-values
    "_PVAL_": r"""(?x)[Pp]\s*
                      (?:[\>\<\=]|(?:\&lt\;|\&gt\;)){,2}\s*[01]?(\.\d+|\d+\%)
                    | [Pp]\s*
                      (?:\<|\>|\&gt\;|\&lt\;)\s*(?:or)\s*\=\s*(?:to)?\s*
                      [01]?(\.\d+|\d+\%)
                    | \(_PVAL_\)""",
    # time indications
    "_TIME_": r"""(?x)\d+\W?\d*\s*([AaPp]\.?\s?[Mm]\.?)
                | \d+\:\d{2}
                | hrs|[Hh]ours|hh
                | [Nn]oon""",
    # measurements in brackets
    "_MEAS_": r"""(?x)\(\d+(\.\d*)?\s*mmHg\s*(;\s*_PVAL_\s*)?\)
                    | \(\-?\d+(\.\d*)?\%\s*(;\s*_PVAL_\s*)?\)""",
    # criteria
    "_CRI_": r"""([<=>]\s*or\s*)?[<=>]+\s*\-?\d+(\.\d*)?\s*(\%|mmHg)?""",
    # years
    "_YEAR_": r"(?<=[^\d])([12][019]\d{2})(?=[^\d])",
    # other
    " ": r"(\-|\s+)"
}

pat_ordered = ["mmHg", "_DATE_", "_POFT_", "_URL_", "_POP_", "_PVAL_",
               "_TIME_", "_RATIO_", "_CRI_", "_MEAS_", "_YEAR_", " "]


def normalise_sentence(sent):
    """
    given string "sent" expands abbreviations and substitutes patterns
    with normalisation tags
    """
    sent = re.sub("&lt;", "<", sent)
    sent = re.sub("&gt;", ">", sent)
    sent = re.sub("vs", "versus", sent)
    sent = re.sub("±", "+/-", sent)
    sent = re.sub("\s*\+/-\s*", " +/- ", sent)
    sent = expand_abbreviations(sent)
    for key in pat_ordered * 3:
        sent = re.sub(patterns[key], key, sent)
    return sent


def get_pattern_num_char():
    return pattern_num_char


def get_pattern_r():
    return pattern_r


def get_unstructured_dir():
    return unstructured_dir


def get_testing_dir():
    return testing_dir


def get_pretrained_dir():
    return pretrained_dir


def get_preprocessed_dir():
    return preprocessed_dir


def get_new_data_dir():
    return new_data_dir


def get_result_dir():
    return results_dir


def get_genia_tagger_path():
    return genia_tagger_path


def get_template_path():
    return template_path


def get_demo_template_path():
    return demo_template_path


def log(string):
    global last_time
    diff = time.time() - last_time
    last_time = time.time()
    print("{}, (+{}s)".format(string, round(diff, 4)))


def get_logger(name):
    return logging.getLogger(name)


def load_dict(file_path):
    try:
        cache = json.load(open(file_path, "r"))
    except (IOError, ValueError):
        cache = {}

    return cache


def save_dict(file_path, dict):
    json.dump(dict, open(file_path, "w"))


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


def get_paper_paths(dir=structured_dir):
    paper_paths = []
    for file in sorted(listdir(dir)):
        # ignore .DS_Store files
        if file.startswith(".DS"):
            continue
        paper_paths.append(path.join(dir, file))

    return paper_paths


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

    query = {"string": string, "ticket": ticket, "pageNumber": 1,
             "pageSize": 1,
             "searchType": "exact"}
    search_result = requests.get(uri + content_endpoint, params=query)
    search_result.encoding = "utf-8"
    results = json.loads(search_result.text)["result"]["results"]

    uri_2 = "https://uts-ws.nlm.nih.gov/rest"
    content_endpoint_2 = "/content/" + version + "/CUI/"

    classes = []

    for result in results:

        uid = result["ui"]
        if uid is None or uid == "NONE":
            continue

        try:
            ticket = auth_client.getst(tgt)
            query_2 = {"ticket": ticket}
            r_2 = requests.get(uri_2 + content_endpoint_2 + uid,
                               params=query_2)

            if not 200 <= r_2.status_code < 300:
                continue

            r_2.encoding = "utf-8"
            items = json.loads(r_2.text)
            json_sem = items["result"]["semanticTypes"]
            for result in json_sem:
                classes.append(result["name"])

        except json.JSONDecodeError:
            print("Got bad JSON!")

    umls_cache.set(string, classes)

    return classes


def analyse_data(dir):
    from unicodedata import normalize
    tok_count = 0
    sent_count = 0

    paper_paths = get_paper_paths(dir=dir)
    paper_soups = load_paper_xmls(paper_paths)
    paper_count = len(paper_soups)

    for i in range(paper_count):
        soup = paper_soups[i]
        abstract = soup.abstract

        for element in abstract.find_all(text=True):
            sent = normalize("NFKD", element)
            sent = re.sub("&lt;", "<", sent)
            sent = re.sub("&gt;", ">", sent)
            element.replace_with(sent)

        for abs_text in abstract.findAll('abstracttext'):
            sents_ = nltk.sent_tokenize(abs_text.text)
            sent_count += len(sents_)

            for sent_ in sents_:
                toks = nltk.word_tokenize(sent_)
                tok_count += len(toks)

    print("The data in {} has:\n{} sentences\n{} tokens\n\n".format(dir,
                                                                sent_count,
                                                                tok_count))


def main():

    dir_list = [structured_dir, new_data_dir, unstructured_dir]
    for dir in dir_list:
        analyse_data(dir)


if __name__ == "__main__":
    main()
