"""
Created on Dec 2018

THIS FILE IS FOR TESTING PREPROCESSING FUNCTIONS

@author: Zac Luong
"""
from nltk.corpus import stopwords
import util
import nltk
import json
import os
import re

sw = stopwords.words('english') + ['non']
script_dir = os.path.dirname(os.path.abspath(__file__))
stopword_trie = util.Trie(strings=(stopwords.words('english') + ['non']))

abbrev_path = os.path.join(script_dir, os.pardir, "data", "abbreviations.json")
with open(abbrev_path, 'r') as file:
    # TODO: This is supposed to be in abbrev_path for some reason but it
    # messes up AM times, figure it out the add it back
    #     "AM": [
    #     "Alginate",
    #     "Once",
    #     "Daily"
    #     ],
    abbrev_dict = json.load(file)

# lemmatiser instance
# lem = nltk.WordNetLemmatizer()

# genia tagger instance
# executable_path = os.path.join(".", "geniatagger-3.0.2", "geniatagger")
# tagger = GENIATagger(executable_path)

# - list of patterns
patterns = {
    # population
    '_POP_': r'''(?x) (?<=\W)[Nn]\s*\=\s*\d+
                    | (?<=_POP_\,\s)_NUM_
                    | (?<=_POP_\sand\s)_NUM_
                    | (?<=_POP_\,\s_NUM_\sand\s)_NUM_
                    | (?<=_POP_\,\s_NUM_\,\sand\s)_NUM_''',
    # confidence intervals
    '_CONFINT_': r'''(?x)\-?\d+(?:\.\d+)?\%?\s*
                        (?:\+\/\-|\Â\±)\s*
                        \d+(?:\.\d+)?\%?
                      | CI\s*(?:[\>\<]|(?:\&lt\;|\&gt\;))\s*\d+(?:\.\d+)?
                      | \(?\[?([Cc][Ii]|[Cc]onfidence\s[Ii]nterval)
                          \,?\)?\]?\s*\=?\s*_RANGE_
                      | (_NUM_|_PERC_)\s*\(?(\+\/|\Â\±|\±)\s*
                          (_NUM_|_PERC_)\)?
                      | _CONFINT_\s\(_NUM_\)
                      | _PERC_\s*_CONFINT_''',
    # confidence intervals with a measure indicator
    '_CONFINTM_': r'''(?x) _CONFINT_\s*mmHg
                         | _NUM_\s?[\(\[]_NUM_[\)\]]\s?mmHg
                         | _MEAS_\s*\(?(\+\/|\Â\±|\±)\s*_NUM_\)?
                         | _NUM_\s*\(?(\+\/|\Â\±|\±)\s*_MEAS_\)?''',
    # ranges
    '_RANGE_': r'''(?x)[\+\-]?\d+\.?\d*\s*(?:\-|to|\/)\s*[\+\-]?
                         \d+\.?\d*(?:\s*\%)?''',
    # ranges with a measure indicator
    '_RANGEM_': r'''(?x) _RANGE_\s*mmHg''',
    # p-values
    '_PVAL_': r'''(?x)[Pp]\s*
                      (?:[\>\<\=]|(?:\&lt\;|\&gt\;)){,2}\s*[01]?(\.\d+|\d+\%)
                    | [Pp]\s*
                      (?:\<|\>|\&gt\;|\&lt\;)\s*(?:or)\s*\=\s*(?:to)?\s*
                      [01]?(\.\d+|\d+\%)''',
    # percentages
    '_PERC_': r'''(?x)(?:[\-\+]\s*)?\d+\.?\d*\s*\%
                     | _NUM_\s[Pp]erc(\.|ent)? 
                     | _NUM_\s_PERC_''',
    # time indications
    '_TIME_': r'''(?x)\d+\W?\d*\s*([AaPp]\.?\s?[Mm]\.?)
                | \d+\:\d{2}
                | hrs|[Hh]ours|hh
                | _TIME_\s[Nn]oon''',
    # measurements
    '_MEAS_': r'''(?x)  (\>|\<|\&lt\;|\&gt\;|\=|\≤)(\s*or\s*\=)?
                        \s*\-?\s*\d+\.?\d*\%?
                      | (_NUM_|_MEAS_)\s*
                         \/?(mm\s?[Hh][Gg]|mm|mg\/m[Ll]|mg|m[Ll]
                         |dB(\/(y|year|month))?|DB(\/(y|year|month))?)
                      | _NUM_(?=\sand\s_MEAS_)
                      | _NUM_(?=(\,\s_NUM_)*\,?\sand\s_MEAS_)''',
    # years
    '_YEAR_': r'(?<=[^\d])([12][019]\d{2})(?=[^\d])',
    # numbers (real, integers and in words)
    '_NUM_': r'''(?x) (?:[\-\+]\s*)?\d+(?:\.\d+)?(?=[^\>])
                    | _NUM_\d
                    | (?<=\b)([Oo]ne|[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive
                         |[Ss]ix|[Ss]even|[Ee]ight|[Nn]ine
                         |[Tt]en|[Ee]leven|[Tt]welve|[A-Za-z]+teen
                         |[Tt]wenty|[Tt]hirty|[Ff]orty|[Ff]ifty
                         |[Ss]ixty|[Ss]eventy|[Ee]ighty|[Nn]inety)(?=\W)
                    | _NUM_\s?([Hh]undred|[Tt]housand)(\s(and\s)?_NUM_)?
                    | ([Tt]wenty|[Tt]hirty|[Ff]orty|[Ff]ifty
                         |[Ss]ixty|[Ss]eventy|[Ee]igthty|[Nn]inety)
                      [\s\-]?_NUM_
                    | _NUM_\-_NUM_''',
    # urls
    '_URL_': r'(?:http\:\/\/)?www\..+\.(?:com|org|gov|net|edu)',
    # dates
    '_DATE_': r'''(?x)
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
                      (?:19|20)?\d{2}           #ie: 12/jan/2001''',
    # periods of time
    '_POFT_': r'''(?x) \d{1,3}\-(?:[Mm]inutes?|[Hh]ours?|[Dd]ays?|[Ww]eeks?
                                 |[Mm]onths?|[Yy]ears?)(?=[\s\W])
                      | (?<=\W)_NUM_\-?(?=\,?\s(_NUM_\-?\,?\s)?(to|or)\s_POFT_)''',
    # arm one references
    '_GONE_': r'''(?x) (?:[Aa]rm|[Gg]roup)\s*([1IiAa]|[Oo]ne)(?=[\s\W])
                    | (?<=\W)(?:1st|[Ff]irst|[Ii]ntervention|[Oo]ne|[Ss]tudy)\s+
                        (?:[Aa]rm|[Gg]roup)(?=[\s\W])''',
    # arm two references
    '_GTWO_': r'''(?x) (?:[Aa]rm|[Gg]roup)\s*(?:[2Bb]|II|ii|[Tt]wo)(?=[\s\W])
                    | (?:2nd|[Ss]econd|[Cc]ontrol|[Pp]lacebo)\s+
						(?:[Aa]rm|[Gg]roup)(?=[\s\W])
                    | (?<=\_GONE\_\sand\s)([2Bb]|II|ii)(?=[\s\W])
                    | (?<=\_GONE\_\,\s)([2Bb]|II|ii)(?=[\s\W])''',
    # ratios
    '_RATIO_': r'''(?x) (\_NUM\_|\_RATIO\_)[\:\/]\_NUM\_
                   | _NUM_\sof\s_NUM_''',
    # other
    'mmHg': r'mm[\s\/]*[Hh][Gg]',
    ' ': r'(\-|\s+)'
}

pat_ordered = ['mmHg', '_CONFINT_', '_CONFINTM_', '_GONE_', '_GTWO_',
               '_DATE_', '_POFT_', '_URL_', '_POP_', '_RATIO_', '_RANGE_',
               '_RANGEM_', '_PVAL_', '_TIME_', '_MEAS_', '_PERC_',
               '_YEAR_', '_NUM_', ' ']


# Normalisation

def normalise_sentence(sentence):
    '''
    given string 'sentence' expands abbreviations and substitutes patterns
    with normalisation tags
    '''
    sent_expanded = expand_abbreviations(sentence)
    for key in pat_ordered * 3:
        sent_expanded = re.sub(patterns[key], key, sent_expanded)

    return sent_expanded


# Abbreviation expansion

def abbreviations(sent):
    """
    finds all the abbreviations in text and returns them as a dictionary
    with the abbreviations as keys and expansions as values
    abbreviations of the form (.) is removed from the text as the words
    preceeding them are their expanded form.

    :returns:
    new_abbrevs: a dictionary of new abbreviations we picked up from the sent.
    sent_new: new sentence with bracketed abbreviations removed.
    """
    tokens = nltk.word_tokenize(sent)
    sent_new = str(sent)
    new_abbrevs = {}
    l_bracket_i = re.compile(r'[\(\[]')
    r_bracket_i = re.compile(r's?[\)\]]')

    for i, t in enumerate(tokens[:-2]):
        tests = (l_bracket_i.search(t) and tokens[i + 1].isupper()
                 and r_bracket_i.search(tokens[i + 2])
                 and not '=' in tokens[i + 1])
        if tests:
            new_abbrevs[tokens[i + 1]] = [w.title() for w in tokens[:i]]
            sent_new = re.sub(r'[\(\[]' + tokens[i + 1] + '[\)\]]' + ' ', '',
                              sent_new)

    for a in new_abbrevs.keys():
        candidates = []
        for i, w in enumerate(reversed(new_abbrevs[a])):
            if i > len(a) + 1: break
            condition = (i > (len(a) - 3)
                         if len([1 for l in a if l == a[0]]) > 1
                         else True)
            if condition and w.lower().startswith(a[0].lower()) \
                    and stopword_trie.check(w) is None:
                candidates.append(new_abbrevs[a][-(i + 1):])
        # sort the keys in a by ascending order of length
        candidates.sort(key=lambda x: len(x))
        new_abbrevs[a] = (candidates[0] if candidates else [])

    return [new_abbrevs, sent_new]


def expand_abbreviations(sent):
    """

    :param sent: the sentence (text) we would like
    :return:
    """
    [abbrev_new, sent_new] = abbreviations(sent)
    if sent not in abbrev_dict:
        abbrev_dict.update({k: v for (k, v) in abbrev_new.items() if v})
    # sort the keys in abbrev_dict in descending order of length
    keys = sorted(abbrev_dict.keys(), key=len, reverse=True)

    for k in keys:
        replace_word = (' '.join(abbrev_dict[k]) if type(abbrev_dict[k]) is list
                        else abbrev_dict[k])

        if re.search(k, sent_new):
            next_i = re.search(k, sent_new).span()[1]
            try:
                if sent[next_i] is not ' ':
                    replace_word += ' '
            except IndexError:
                print("This is single token")

            sent_new = (re.sub(k, replace_word, sent_new)
                        if replace_word else sent_new)

    return sent_new


# Normalisation

if __name__ == "__main__":
    text = "Diurnal curves of intraocular pressure (IOP) were performed on the " \
           "baseline day and after 0.5, 3, 6, and 12 months of treatment. " \
           "The IOP measurements were performed at 8:00 AM, 12:00 noon, " \
           "4:00 PM, and 8:00 PM."

    text_2 = "Latanoprost is a PGF2 alpha analogue which reduces the intraocular pressure (IOP) by increasing the uveoscleral outflow."

    sent_new = normalise_sentence(text)

    print(sent_new)

    pass
    # data_path = os.path.join(cur_dir, 'data/annotation I/')
    # for fn in os.listdir(data_path):
    #     with open(os.path.join(data_path, fn)) as f:
    #         raw = f.read()
    #         soup = bs4.BeautifulSoup(raw, "html5lib")
    #
    #     title = take_title(soup.pmid.text)
    #     for word, base_form, pos_tag, chunk, named_entity in tagger.tag(
    #             title.text):
    #         print("{:20}{:20}{:10}{:10}{:10}".format(word, base_form, pos_tag,
    #                                                  chunk, named_entity))
    #
    #     break
