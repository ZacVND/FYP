# FYP_NLP
## Extraction of medical evidence from abstracts of randomised clinical trials

Systematic Reviews are regarded as the highest quality evidence in Evidence Based Medicine (EBM) practice. Authors of systematic reviews use randomised  clinical trials (RCT) as the main source of data. 

The PICO (Patients, Intervention, Comparison, Outcome) framework is used to assess whether a given RCT is relevant, all PICO elements are required by the CONSORT statement to be included in the abstract of RCT reports. Therefore, researchers do not need to read through the full reports, additionally abstracts are always available for free whereas full paper may require an access fee. 

Our system uses conditional random field to automatically read the abstracts and extracting these evidence: Intervention arm, Comparison arm, Patient group, Outcome measure, and results of the 2 arms respectively.

We compare the performance between Decision Tree, Random Forest and SVM classifiers for this task:

Cross entropy loss shows how close the predicted probability distribution (of tokens) is to the true probability distribution.

|                    | Random Forest |  SVM  | Decision Tree |
|--------------------|:-------------:|:-----:|:-------------:|
| **Cross Entropy Loss** |     0.2063    | 0.236 |     1.5371    |

Below is the comparison between the classifiers' precisions. Precision is the a simple metric showing how well the system selected the phrases that contain the key information. The more phrases that contain the true token, the higher the precision.

|               | Intervention (A1) | Comparison (A2) | Intervention Result (R1) | Comparison Result (A2) | Outcome Measure (OC) | Patient Group (P) |
|---------------|:-----------------:|:---------------:|:------------------------:|:----------------------:|:--------------------:|:-----------------:|
| **Random Forest** |       0.619       |      0.5105     |           0.398          |          0.304         |         0.719        |       0.863       |
| **SVM**           |       0.534       |      0.4795     |          0.3375          |         0.2795         |        0.6855        |       0.823       |
| **Decision Tree** |       0.4185      |      0.4805     |          0.3335          |          0.304         |        0.6135        |       0.767       |

## Sample Output ##

| **Intervention Arm**       |           phacoemulsification ( phaco )           |
| **Comparison Arm**         | combined phacoemulsification and viscogonioplasty |
| **Patient Group**          |               82 patients with pacg               |
| **Outcome Measure**        |                    the mean iop                   |
| **Result of Intervention** |           22.3+/-6.3 to 14.0+/-3.7 mm hg          |
| **Result of Comparison**   |           23.3+/-7.3 to 14.5+/-2.5 mm hg          |

## Requirements ##
**General**
* Python 3.7 or higher
* sklearn
* scipy
* numpy
* nltk (\*)
* GeniaTagger 3.0.2 or later
* [genia-tagger-py](https://github.com/bornabesic/genia-tagger-py) by bornabesic

(\*) The stopword corpus is needed. Instructions [here](http://www.nltk.org/data.html). 

**Data collection**
* Sign up and request your **[UMLS API key](https://uts.nlm.nih.gov/home.html)**
* requests 
* [beatifulsoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* html5lib
* lxml 
* json
* urllib3
* certifi

**Results rendering**
* [pugjs](https://pugjs.org/api/getting-started.html)

## Installing ##
**Step 1:** Fork this repository, install all requirements

**Step 2:** Install the [GENIA Tagger](http://www.nactem.ac.uk/GENIA/tagger/) in the project root repository

**Step 3:** Get [genia-tagger-py](https://github.com/bornabesic/genia-tagger-py) and put `geniatagger.py` in *./ie_tools/libraries/*

**Step 4:** Insert your UMLS API key
```python
# ./ie_tools/src/util.py 

...
api_key = "<Your API key here>"
...
```

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
├── new_data
│   └── ...
├── pretrained
│   └── ...
└── results
    └── ...
```

***take_abstract.py***: 
* `take_abstract.take(pmid)` retrieves the abstract of a RCT report (identified by `pmid`) from the PubMed website and saves it in *./new_data/* folder.
* `take_title(pmid)` retrieves the title of a RCT report from its `pmid` and write it to the corresponding abstract file.

***feature.py***: 
* `feature.get_feature_classes(word)` retrieves the corresponding semantic classes of a given `word`
* Defines the features which the classifier uses in class `Feature`

***util.py***: 
* Defines all paths to necessary directories and files, **if you would like to change the structure please change this file accordingly**
* Defines utilities functions such as rendering results into HTML, query UMLS Metathesaurus, building knowledge base, loading and parsing abstracts, etc...

***token_utils.py***: 
* Define functions to tokenise the abstract, do preprocessing tasks (tokenizing, chunking, shallow parsing), generate feature matrix for classifiers

***classifier.py***: 
* Define classifier functions, training and test, calculating precision and loss, pattern matching and selecting best tokens/phrases.
* Saves results in `.json` format

**Step 5:** Run *data_collection.py* to build the knowledge base for our model, output file is `umls_cache.json`:

```python -m ie_tools.scripts.data_collection```

**NOTE:** Due to absolute import python structure, we have to use `python -m <module>` to run from command line. Otherwise running from PyCharm (2018) works by default.

**Step 6:** Run the code.

Specify the classifier type with: `classifier_type = Classifier.TypeRF`. Choose between TypeRF, TypeSVM or TypeDT.

### Hold out:
***hold_out.py***:
Train the classifier on 80% of the abstract and test the classifier on the remaining 20%, if no pretrained classifier exist in *./pretrained/* it will train a new classifier.

```python -m ie_tools.scripts.hold_out```

### Demo:
***demo.py***:
Runs hold_out script but with different final output being rendered

```python -m ie_tools.scripts.demo```

### Cross validation:
***cross_validation.py***:
Train a new classifier at every run. There are 20 runs of 10-fold cross validation, these values can be modified in the file.

```python -m ie_tools.scripts.cross_validation```

Results from running the code are saved in *./results/* as `.json` and `.html` files

