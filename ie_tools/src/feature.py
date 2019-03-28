from enum import Enum

from ie_tools.src import util


# Whenever you add more related to UMLS, you should probably delete the
# `umls_cache.json` in data/
class Feature(Enum):
    TOK_IS_CD = 0
    TOK_IS_ARM = 1
    TOK_IS_BNP = 2
    TOK_IS_DRUG = 3
    TOK_IS_NUMBER = 4
    TOK_IS_IN_ODICT = 5
    TOK_IS_IN_PDICT = 6  # is in patient dictionary, P can appear anywhere
    TOK_IS_IN_TITLE = 7
    TOK_IS_PATIENTS = 8
    TOK_IS_PROCEDURE = 9
    CHUNK_TYPE_NP = 10
    CHUNK_TYPE_VP = 11
    CHUNK_TYPE_PP = 12
    CHUNK_TYPE_ADJP = 13
    SENT_POSITION = 14  # A/R 1 always appear before A/R 2 in a sentence
    PARA_CAT_OBJECTIVE = 15
    PARA_CAT_METHODS = 16
    PARA_CAT_RESULTS = 17
    ABSTRACT_POSITION = 18  # A always appear before OC always appear before R
    ABSTRACT_BOW = 19


feature_count = len(Feature.__members__.items())

# Hardcoding the features
patient_dict_trie = util.Trie(strings=["patient", "patients", "eyes", "adults",
                                       "subjects", "individuals"])

outcome_dict_trie = util.Trie(strings=["pressure", "iop", "change",
                                       "reduction", "lowering", "value",
                                       "density", "control", "proportion",
                                       "decrease", "loss"])

treatment_dict_trie = util.Trie(strings=["combination", "fixed-combination",
                                         "350-mm2", "500-mm2"])

class_to_feature_mapping = {
    Feature.TOK_IS_DRUG.value: ['Pharmacologic Substance', 'Antibiotic',
                                'Organic Chemical', "Clinical Drug",
                                "Substance"],
    Feature.TOK_IS_PROCEDURE.value: ['Therapeutic or Preventive Procedure',
                                     'Medical Device'],
    Feature.TOK_IS_PATIENTS.value: ['Patient or Disabled Group',
                                    'Body Part, Organ, or Organ Component',
                                    'Age Group', 'Group']
}

class_to_feature_trie = util.Trie(mapping=class_to_feature_mapping)

print("hello")

def get_feature_classes(word):
    feature_class_map = {}
    classes = util.get_umls_classes(word)

    for cls in classes:
        feature_i = class_to_feature_trie.check(cls)
        if feature_i is not None:

            # When this is set to True, the code returns the number of
            # classes that suggest that a particular feature is true.
            # When set to False, the code returns 1 if some feature is present.
            count_classes_suggesting_drug = False

            if count_classes_suggesting_drug:

                if feature_class_map.get(feature_i) is None:
                    feature_class_map[feature_i] = 1
                else:
                    feature_class_map[feature_i] += 1

            else:
                feature_class_map[feature_i] = 1

    return feature_class_map


def main():
    losses = [0.5815, 0.4965, 0.4535, 0.4806, 0.4752, 0.4178, 0.466, 0.6158,
              0.55, 0.4279, 0.4982, 0.3845, 0.5531, 0.4859, 0.4613, 0.635,
              0.51, 0.3807, 0.5077, 0.4357, 0.6182, 0.4156, 0.4439, 0.5126,
              0.4016, 0.4976, 0.5197, 0.4067, 0.5211, 0.4838, 0.6136, 0.4788,
              0.4281]

    print("Total loss: ", sum(losses))
    print("Average loss: ", sum(losses) / len(losses))


if __name__ == "__main__":
    main()
