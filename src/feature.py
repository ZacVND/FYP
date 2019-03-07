from enum import Enum
import util


# Whenever you add more related to UMLS, you should probably delete the
# `umls_cache.json`
class Feature(Enum):
    # TODO: Add more features
    TOK_IS_DRUG = 0
    TOK_IS_IN_TITLE = 1
    TOK_IS_BNP = 2
    TOK_IS_PROCEDURE = 3
    TOK_IS_IN_PDICT = 4  # is in patient dictionary, P can appear anywhere
    TOK_IS_PLACEBO = 5  # sometimes A2 is a placebo
    TOK_IS_ABBREV = 6
    CHUNK_TYPE_NP = 7
    CHUNK_TYPE_VP = 8
    CHUNK_TYPE_PP = 9
    CHUNK_TYPE_ADVP = 10
    CHUNK_TYPE_ADJP = 11
    CHUNK_BOW = 12
    TOK_IS_NUMBER = 13
    TOK_IS_CD = 14
    PARA_CAT_OBJECTIVE = 15
    PARA_CAT_METHODS = 16
    PARA_CAT_RESULTS = 17
    PARA_CAT_CONCLUSIONS = 18
    # PARA_CAT_BACKGROUND = 19  # paragraph of category BACKGROUND is irrelevant
    TOK_IS_PATIENTS = 19
    TOK_IS_ARM = 20
    TOK_IS_IN_ODICT = 21
    SENT_POSITION = 22  # A/R 1 always appear before A/R 2 in a sentence
    ABSTRACT_POSITION = 23  # A always appear before OC always appear before R




feature_count = len(Feature.__members__.items())

patient_dict_trie = util.Trie(strings=["patient", "patients", "eyes",
                                       "adults", "subjects", "individuals"])

outcome_dict_trie = util.Trie(strings=["pressure", "pressures", "iop", "change",
                                       "reduction", "lowering", "values",
                                       "density", "control", "proportion",
                                       "decrease", "loss"])

class_to_feature_mapping = {
    Feature.TOK_IS_DRUG.value: ['Pharmacologic Substance', 'Antibiotic',
                                'Organic Chemical'],
    Feature.TOK_IS_PROCEDURE.value: ['Therapeutic or Preventive Procedure',
                                     'Medical Device'],
    Feature.TOK_IS_PATIENTS.value: ['Patient or Disabled Group',
                                    'Body Part, Organ, or Organ Component',
                                    'Age Group', 'Group']
}

class_to_feature_trie = util.Trie(mapping=class_to_feature_mapping)


def get_feature_classes(word):
    feature_class_map = {}
    classes = util.get_umls_classes(word)

    for cls in classes:
        feature_i = class_to_feature_trie.check(cls)
        if feature_i is not None:

            # When this is set to False, the code returns 1 if some feature
            # is present.
            # When this is set to True, the code returns the number of
            # classes that suggest that a particular feature is true.
            count_classes_suggesting_drug = False

            if count_classes_suggesting_drug:

                if feature_class_map.get(feature_i) is None:
                    feature_class_map[feature_i] = 1
                else:
                    feature_class_map[feature_i] += 1

            else:
                feature_class_map[feature_i] = 1

    return feature_class_map


if __name__ == "__main__":
    print(get_feature_classes('44.7'))
