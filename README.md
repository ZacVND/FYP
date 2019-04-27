# Final Year Project
## Extraction of medical evidence from abstracts of randomised clinical trials

Systematic Reviews are regarded as the highest quality evidence in Evidence Based Medicine (EBM) practice. Authors of systematic reviews use randomised  clinical trials (RCT) as the main source of data. 

The PICO (Patients, Intervention, Comparison, Outcome) framework is used to assess whether a given RCT is relevant, all PICO elements are required by the CONSORT statement to be included in the abstract of RCT reports. Therefore, researchers do not need to read through the full reports, additionally abstracts are always available for free whereas full paper may require an access fee. 

Our system uses conditional random field to automatically read the abstracts and extracting these evidence: Intervention arm, Comparison arm, Patient group, Outcome measure, and results of the 2 arms respectively.

We compare the performance between Decision Tree, Random Forest and SVM 
classifiers using **the average of 20 runs of 8-fold cross validation**:

Cross entropy loss shows how close the predicted probability distribution (of tokens) is to the true probability distribution. Lower is better.

|                    | Random Forest |  SVM  | Decision Tree |
|--------------------|:-------------:|:-----:|:-------------:|
| **Cross Entropy Loss** |     0.0223    | 0.0260 |     0.1709    |

Below is the comparison between the classifiers' precisions. Precision is the a simple metric showing how well the system selected the phrases that contain the key information. The more phrases that contain the true token, the higher the precision. Higher is better.

|               | Intervention (A1) | Comparison (A2) | Intervention Result (R1) | Comparison Result (R2) | Outcome Measure (OC) | Patient Group (P) |
|---------------|:-----------------:|:---------------:|:------------------------:|:----------------------:|:--------------------:|:-----------------:|
| **Random Forest** |       0.7484       |      0.549     |           0.3865   |          0.325         |         0.7109        |       0.875       |
| **SVM**           |       0.5807       |      0.4745     |          0.3062    |         0.2583         |        0.6141        |       0.812       |
| **Decision Tree** |       0.5266      |      0.3807     |          0.301   |          0.2984         |        0.5073        |       0.6302       |

## Sample Output ##

**Paper 25393036:** Phacoemulsification Versus Combined Phacoemulsification and Viscogonioplasty in Primary Angle-Closure Glaucoma: A Randomized Clinical Trial.

|      |                    |
|:------------------------|:-------------------------------------------------:|
| **Intervention Arm**       |           phacoemulsification          |
| **Comparison Arm**         | combined phacoemulsification and viscogonioplasty |
| **Patient Group**          |               82 patients with Primary Angle-Closure Glaucoma|
| **Outcome Measure**        |                    the mean Intraocular Pressure|
| **Result of Intervention** |           22.3+/-6.3 to 14.0+/-3.7 mmHg       |
| **Result of Comparison**   |           23.3+/-7.3 to 14.5+/-2.5 mmHg       |

The top row bold by default in Github Markdown, therefore it is left as empty as a workaround to achieve this look.

## Requirements ##
**This project has been tested on macOS Mojave and Ubuntu 18.04 LTS**

**General**
* Python 3.7 or higher
* sklearn
* numpy
* nltk (\*)
* GeniaTagger 3.0.2 or later
* pytest, unittest for testing

(\*) The stopword corpus and punkt model are needed. Instructions [here](http://www.nltk.org/data.html). 

**Data collection**
* Sign up and request your **[UMLS API key](https://uts.nlm.nih.gov/home.html)**
* requests 
* [beatifulsoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* html5lib
* joblib
* lxml 
* json
* urllib3
* certifi

**Results rendering**
* [pugjs](https://pugjs.org/api/getting-started.html)

This system uses [genia-tagger-py](https://github.com/bornabesic/genia-tagger-py), a python wrapper for GENIA tagger created by 
bornabesic, already included in
 *./ie_tools/libraries/*
 
 Anaconda should come with all of the above pre-installed.

## Setup and Run ##
**NOTE:** Due to absolute import python structure, we have to use `python -m <module>` to run from command line. Otherwise running from PyCharm (2018) works by default.

**Step 1:** Fork this repository, install all requirements.

**Step 2:** Install the [GENIA Tagger](http://www.nactem.ac.uk/GENIA/tagger/) in the project root directory.

**Step 3:** Get your UMLS API key and change it in *./data/umls.api_key* 

**Step 4:** Run testing code to make sure everything is working as intended.

```python -m ie_tools.scripts.testing```

The project structure should look something like this:
```bash
.
├── README.md
├── data
│   ├── abstracts_structured
│   │   └── ...
│   ├── abstracts_unstructured
│   │   └── ...
│   ├── new_data
│   │   └── ...
│   ├── preprocessed
│   │   └── ...
│   ├── testing
│   │   └── ...
│   └── ...
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
│   │   ├── semantic_class_collection.py
│   │   ├── demo.py
│   │   ├── hold_out.py
│   │   └── testing.py
│   └── src
│       ├── classifier.py
│       ├── demo.pug
│       ├── feature.py
│       ├── results.html
│       ├── results.pug
│       ├── take_abstract.py
│       ├── token_util.py
│       └── util.py
├── new_data
├── pretrained
│   └── ...
└── results
    └── ...
```

***take_abstract.py***: This file is used to retrieve new abstracts, should you want to collect new data, please modify the `main()` function of this file, the new retrieved data will be in *./new_data/*
* `take_abstract.take(pmid)` retrieves the abstract of a RCT report (identified by `pmid`) from the PubMed website and saves it in *./new_data/* folder.
* `take_title(pmid)` retrieves the title of a RCT report from its `pmid` and write it to the corresponding abstract file.

Newly retrieved abstracts should be annotated and then moved to *./data/new_data/*

***feature.py***: 
* `feature.get_feature_classes(word)` retrieves the corresponding semantic classes of a given `word`
* Defines the features which the classifier uses in class `Feature`

***util.py***:
* Defines all paths to necessary directories and files, **if you would like to change the structure please change this file accordingly**
* Defines utilities functions such as rendering results into HTML, query UMLS Metathesaurus, building knowledge base, loading and parsing abstracts, etc...

***token_util.py***: 
* Define functions to tokenise the abstract, do preprocessing tasks (tokenizing, chunking, shallow parsing), generate feature matrix for classifiers

***classifier.py***: 
* Define classifier functions, training and test, calculating precision and loss, pattern matching and selecting best tokens/phrases.
* Saves results in `.json` format
* `umls_cache=False` set this to True to rebuild the umls_cache.json (aka 
knowledge base of this system)



**Step 5:** Run *semantic_class_collection.py* to build the knowledge base for our model, output file is `umls_cache.json`:

```python -m ie_tools.scripts.data_collection```

**Step 6:** Run the code.

Specify the classifier type with: `classifier_type = Classifier.TypeRF`. Choose between `Classifier.TypeRF`, `Classifier.TypeSVM` or `Classifier.TypeDT`.

### Hold out:
***hold_out.py***:
Train the classifier on 80% of the abstract and test the classifier on the remaining 20%, if no pretrained classifier exist in *./pretrained/* it will train a new classifier.
* `random` this boolean defines whether the data is split randomly or 
not
* `persist` defines whether the system will save the classifier after 
training. Saves into *./pretrained/*
* `pretrain` defines whether the system will use pretrained classifiers or 
not (if True and no pretrained found, the system will train a new classifier)
. It looks for pretrained classifiers in *./pretrained/*
* `unstructured` defines whether the system will use the unstructured
 abstracts
* `new_test_only` defines whether the system will use the new abstracts as 
testing set only.
  

```python -m ie_tools.scripts.hold_out```

### Demo:
***demo.py***:
Runs hold_out script but without True and Predicted tokens rendered. The output is the same as Sample Output above.

```python -m ie_tools.scripts.demo```

### Cross validation:
***cross_validation.py***:
Train a new classifier at every run. There are 20 runs of 10-fold cross validation, these values can be modified in the file.
* `run_count` defines number of runs
* `fold_count` defines number of folds in data
* `new_data` defines whether you want to use new PMIDs in the run

```python -m ie_tools.scripts.cross_validation```

Results from running the code are saved in *./results/* as `.json` and 
rendered as `.html` files using `.pug` templates in *./ie_tools/src/*

