"""
Created on Dec 2018

@author: Zac Luong
"""
from __future__ import division
import nltk, re, bs4, json, pickle, os
from geniatagger import GENIATagger
from nltk.corpus import stopwords
from Authentication import *
from take_abstract import *

sw = stopwords.words('english') + ['non']
cur_dir = os.path.dirname(os.path.abspath(__file__))

# -  abbreviations dictionary
foo = open(r"./data/abbreviations2.dat", 'rb')
abbrev_dict = pickle.load(foo)
foo.close()

# - common words
common_words = []
f = open(r'./data/common_words.txt')
for line in f:
    common_words.append(line.strip())

f.close()

# lemmatiser instance
lem = nltk.WordNetLemmatizer()

# genia tagger instance
executable_path = os.path.join(".", "geniatagger-3.0.2", "geniatagger")
genia_tagger_instance = GENIATagger(executable_path)

api_key = "bea4b3d4-f1ef-439e-b68f-3564c8c7a231"

def umls_query(string, apikey=api_key, version='2018AB', max_pages=2,
               **extra_args):
    """
    Query the param with UMLS REST API and build up its class
    Original source from NLM UMLS API examples.
    https://github.com/HHS/uts-rest-api/blob/master/samples/python/search-terms.py
    :param word:
    :return class:
    """
    uri = "https://uts-ws.nlm.nih.gov"
    content_endpoint = "/rest/search/"+version

    auth_client = Authentication(apikey)
    tgt = auth_client.gettgt()

    # REQUIRED: generate a ticket for each request
    ticket = auth_client.getst(tgt)

    query = {'string':string, 'ticket':ticket, 'pageNumber':1, 'pageSize':1}
    r = requests.get(uri + content_endpoint, params=query)
    r.encoding = 'utf-8'
    items = json.loads(r.text)
    json_data = items["result"]

    uri_2 = 'https://uts-ws.nlm.nih.gov/rest'
    content_endpoint_2 = '/content/'+version+'/CUI/'

    classes = []

    for result in json_data["results"]:
        try:
            ticket = auth_client.getst(tgt)
            query_2 = {'ticket': ticket}
            r_2 = requests.get(uri_2 + content_endpoint_2 + result["ui"],
                               params=query_2)
            r_2.encoding = 'utf-8'
            items = json.loads(r_2.text)
            json_sem = items['result']['semanticTypes']
            for result in json_sem:
                classes.append(result['name'])
        except:
            NameError

        print("\n")
    print(classes)

def remove_paratag(paragraph):
    """
    removes paragraph tags from paragraph
    """
    pattern = r'\<abstracttext.*\"\>(.*)\<\/abstracttext\>'
    return re.findall(pattern,paragraph)

def take_paragraph_label(paragraph):
    """
    returns the paragraph label
    """
    pattern = r'\<abstracttext.*label\=\"(.*)\"\snlmcategory.*\>'
    return re.findall(pattern,paragraph)[0]

def take_paragraph_category(paragraph):
    """
    returns the paragraph category
    """
    pattern = r'\<abstracttext.*nlmcategory\=\"(.*)\"\>'
    return re.findall(pattern,paragraph)[0]

def normalise_sentence(sentence):
    '''
    given string 'sentence' expands abbreviations and substitutes patterns
    with normalisation tags
    '''
    pass

data_path = os.path.join(cur_dir, 'data/annotation I/')
for fn in os.listdir(data_path):
    with open(os.path.join(data_path, fn)) as f:
        raw = f.read()
        soup = bs4.BeautifulSoup(raw, "html5lib")

    title = take_title(soup.pmid.text).text.encode('utf-8')
    print(title)
    title = [w.lower() for w in nltk.word_tokenize(normalise_sentence(title))]

def get_semantic_class(chunk, debug=False):
    """
    returns the semantic class of chunk, where chunk is in string format
    """
    pass

# Abbreviation expansion

def abbreviations(text):
    """
    finds all the abbreviations in text and returns them as a dictionary
    with the abbreviations as keys and expansions as values
    """
    pass

def expand_abbreviations(sent):
    """
    returns a copy of 'sent' with abbreviations expanded
    """
    pass

# Normalisation