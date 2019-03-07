from joblib import Parallel, delayed
from os import path
import token_utils as tu
import util

script_dir = path.dirname(path.abspath(__file__))

if __name__ == "__main__":

    umls_cache = util.umls_cache

    # Prepare papers' XML
    # paper_files = sorted(listdir(data_path))[:4]
    paper_paths = util.get_paper_paths()
    paper_count = len(paper_paths)
    paper_soups = [None] * paper_count

    import util as our_util


    def process_word(word_i, word):
        # print('Processing word {}'.format(word_i + 1))
        cached_classes = umls_cache.get(word)
        if cached_classes is not None:
            return word, cached_classes

        classes = our_util.get_umls_classes(word)
        return word, classes


    for i in range(len(paper_paths)):
        soup = util.parse_paper(paper_paths[i])

        print('---- Paper #{}/{} [{}]'.format(i + 1, paper_count,
                                              soup.pmid.text))

        col = tu.TokenCollection(soup)
        col.build_tokens(umls_cache=True)

        print('Total tokens: {}'.format(len(col.tokens)))

        result = Parallel(n_jobs=4)(delayed(process_word)(i, token.word)
                                    for i, token in enumerate(col.tokens))

        for word, classes in result:
            umls_cache.set(word, classes)

        umls_cache.save()
        print('\n\n')

    print('Done!')
