from joblib import Parallel, delayed
from os import path, listdir
import token_utils as tu
import feature as ft
import util
import bs4

script_dir = path.dirname(path.abspath(__file__))

if __name__ == "__main__":

    umls_cache_path = path.join(script_dir, 'umls_cache.json')
    umls_cache = util.load_dict(umls_cache_path)

    # Prepare papers' XML
    # paper_files = sorted(listdir(data_path))[:4]
    paper_paths = util.get_paper_paths()
    paper_count = len(paper_paths)
    paper_soups = [None] * paper_count


    def process_word(word_i, word):
        # print('Processing word {}'.format(word_i + 1))
        cached_map = umls_cache.get(word)
        if cached_map is not None:
            return word, cached_map

        class_map = ft.get_feature_classes(word, cache=None)
        return word, class_map


    for i in range(len(paper_paths)):
        soup = util.parse_paper(paper_paths[i])

        print('---- Paper #{}/{} [{}]'.format(i + 1, paper_count,
                                              soup.pmid.text))

        col = tu.TokenCollection(soup)
        col.normalize()

        print('Total tokens: {}'.format(len(col.tokens)))

        result = Parallel(n_jobs=4)(delayed(process_word)(i, token.word)
                                    for i, token in enumerate(col.tokens))

        for word, class_map in result:
            umls_cache[word] = class_map

        util.save_dict(umls_cache_path, umls_cache)
        print('\n\n')

    print('Done!')
