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
    # TOK_IS_PLACEBO = 5  # sometimes A2 is a placebo
    CHUNK_TYPE_NP = 5
    CHUNK_TYPE_VP = 6
    CHUNK_TYPE_PP = 7
    CHUNK_TYPE_ADJP = 8
    TOK_IS_NUMBER = 9
    TOK_IS_CD = 10
    PARA_CAT_OBJECTIVE = 11
    PARA_CAT_METHODS = 12
    PARA_CAT_RESULTS = 13
    TOK_IS_PATIENTS = 14
    TOK_IS_ARM = 15
    TOK_IS_IN_ODICT = 16
    SENT_POSITION = 17  # A/R 1 always appear before A/R 2 in a sentence
    ABSTRACT_POSITION = 18  # A always appear before OC always appear before R




feature_count = len(Feature.__members__.items())

patient_dict_trie = util.Trie(strings=["patient", "patients", "eyes",
                                       "adults", "subjects", "individuals"])

outcome_dict_trie = util.Trie(strings=["pressure", "pressures", "iop", "change",
                                       "reduction", "lowering", "values",
                                       "density", "control", "proportion",
                                       "decrease", "loss"])

treatment_dict_trie = util.Trie(strings=["combination", "fixed-combination",
                                         "350-mm2", "500-mm2"])

class_to_feature_mapping = {
    Feature.TOK_IS_DRUG.value: ['Pharmacologic Substance', 'Antibiotic',
                                'Organic Chemical', "Clinical Drug"],
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
    losses = [0.5815, 0.4965, 0.4535, 0.4806, 0.4752, 0.4178, 0.466, 0.6158,
              0.55, 0.4279, 0.4982, 0.3845, 0.5531, 0.4859, 0.4613, 0.635,
              0.51, 0.3807, 0.5077, 0.4357, 0.6182, 0.4156, 0.4439, 0.5126,
        0.4016, 0.4976, 0.5197, 0.4067, 0.5211, 0.4838, 0.6136, 0.4788, 0.4281]

    print("Total loss: ", sum(losses))
    print("Average loss: ", sum(losses)/len(losses))
