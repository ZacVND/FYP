from enum import Enum
import util


# Whenever you add more related to UMLS, you should probably delete the
# `umls_cache.json`
class Feature(Enum):
    # TODO: Add more features
    IS_DRUG = 0
    IS_IN_TITLE = 1
    IS_BNP = 2


class_to_feature_mapping = {
    Feature.IS_DRUG.value: ['Pharmacologic Substance', 'Antibiotic',
                            'Organic Chemical', 'Biomedical or Dental Material']
}

class_to_feature_trie = util.make_trie(class_to_feature_mapping)


def get_feature_classes(word, cache=None):
    if cache is not None:
        cached_map = cache.get(word)
        if cached_map is not None:
            return cached_map

    feature_class_map = {}
    classes = util.get_umls_classes(word)

    for cls in classes:
        feature_i = util.check_trie(class_to_feature_trie, cls)
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

    if cache is not None:
        cache[word] = feature_class_map

    return feature_class_map


if __name__ == "__main__":
    print(get_feature_classes('timolol'))
