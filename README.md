# FYP_NLP
## Extraction of medical evidence from abstracts of randomised clinical trials

Systematic Reviews are regarded as the highest quality evidence in Evidence Based Medicine (EBM) practice. Authors of systematic reviews use randomised  clinical trials (RCT) as the main source of data. 

The PICO (Patients, Intervention, Comparison, Outcome) framework is used to assess whether a given RCT is relevant, all PICO elements are required by the CONSORT statement to be included in the abstract of RCT reports. Therefore, researchers do not need to read through the full reports, additionally abstracts are always available for free whereas full paper may require an access fee. 

Our system uses conditional random field to automatically read the abstracts and extracting these evidence: Intervention arm, Comparison arm, Patient group, Outcome measure, and results of the 2 arms respectively.

## Requirements ##
General
* Python 3.7 or higher
* sklearn
* scipy
* numpy
* nltk (\*)
* [GeniaTagger 3.0.2](http://www.nactem.ac.uk/GENIA/tagger/) or later
* [genia-tagger-py](https://github.com/bornabesic/genia-tagger-py) by bornabesic

(\*) The stopword corpus is needed. Instructions [here](http://www.nltk.org/data.html). 

Data collection
* requests 
* [beatifulsoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* html5lib
* lxml 
* json
* urllib3
* certifi

Results rendering
* [pugjs](https://pugjs.org/api/getting-started.html)

The project structure should look something like this:
```bash
.
├── README.md
├── data
│   ├── abstracts_structured
│   │   └── ...
│   ├── abstracts_unstructured
│   │   └── ...
│   └── umls_cache.json
├── geniatagger-3.0.2
│   └── ...
├── ie_tools
│   ├── __init__.py
│   ├── generated
│   ├── libraries
│   │   ├── Authentication.py
│   │   ├── __init__.py
│   │   └── geniatagger.py
│   ├── scripts
│   │   ├── __init__.py
│   │   ├── cross_validation.py
│   │   ├── data_collection.py
│   │   ├── demo.py
│   │   ├── hold_out.py
│   │   └── unit_test.py
│   └── src
│       ├── classifier.py
│       ├── demo.pug
│       ├── feature.py
│       ├── results.html
│       ├── results.pug
│       ├── take_abstract.py
│       ├── token_utils.py
│       └── util.py
├── pretrained
│   ├── decision_tree.sav
│   └── random_forest.sav
└── results
    └── ...
```

Due to absolute import python structure, please run the code as follows:

### Demo:
```python -m ie_tools.scripts.demo```

### Hold out:
```python -m ie_tools.scripts.hold_out```

### Cross validation:
```python -m ie_tools.scripts.cross_validation```


Genia Tagger wrapper gonna be based on this: https://github.com/bornabesic/genia-tagger-py

Genia Tagger output help: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
