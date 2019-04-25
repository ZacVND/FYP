import os
import unittest
from unittest.mock import Mock, patch

import ie_tools.src.util as util


class AuthenticationMock:

    def __init__(self, apikey):
        self.apikey = apikey

    def gettgt(self):
        mock_tgt = {"test": "tgt"}
        return {}

    def getst(self, mock_tgt):
        text = "st"
        return text


# This method will be used by the mock to replace requests.get
def mocked_requests_get(*args, **kwargs):
    class MockResponse:
        def __init__(self, text, status_code):
            self.text = text
            self.status_code = status_code

        def text(self):
            return self.text

    uri = "https://uts-ws.nlm.nih.gov"
    version = "2018AB"
    content_endpoint = "/rest/search/" + version
    text_1 = '{"pageSize":1,"pageNumber":1,"result":{"classType":"searchResults","results":[{"ui":"C0018592","rootSource":"MTH","uri":"https://uts-ws.nlm.nih.gov/rest/content/2018AB/CUI/C0018592","name":"Happiness"}]}}'

    uri_2 = "https://uts-ws.nlm.nih.gov/rest"
    content_endpoint_2 = "/content/" + version + "/CUI/"
    text_2 = '{"pageSize":25,"pageNumber":1,"pageCount":1,"result":{"classType":"Concept","ui":"C0018592","suppressible":false,"dateAdded":"09-30-1990","majorRevisionDate":"09-05-2017","status":"R","semanticTypes":[{"name":"Mental Process","uri":"https://uts-ws.nlm.nih.gov/rest/semantic-network/2018AB/TUI/T041"}],"atomCount":69,"attributeCount":0,"cvMemberCount":0,"atoms":"https://uts-ws.nlm.nih.gov/rest/content/2018AB/CUI/C0018592/atoms","definitions":"https://uts-ws.nlm.nih.gov/rest/content/2018AB/CUI/C0018592/definitions","relations":"https://uts-ws.nlm.nih.gov/rest/content/2018AB/CUI/C0018592/relations","defaultPreferredAtom":"https://uts-ws.nlm.nih.gov/rest/content/2018AB/CUI/C0018592/atoms/preferred","relationCount":10,"name":"Happiness"}}'
    uid = "C0018592"

    if args[0] == uri + content_endpoint:
        return MockResponse(text_1, 200)
    elif args[0] == uri_2 + content_endpoint_2 + uid:
        return MockResponse(text_2, 200)

    return MockResponse(None, 404)


class TestUtil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Setting up tests for util.py")
        testing_dir = util.get_testing_dir()
        cls.test_xml = os.path.join(testing_dir, "xml")
        cls.test_json = os.path.join(testing_dir, "json")
        cls.paper_paths = util.get_paper_paths(cls.test_xml)

    def test_get_paper_paths(self):
        paths = util.get_paper_paths(self.test_xml)
        self.assertTrue(os.path.exists(paths[0]))

        with self.assertRaises(FileNotFoundError):
            util.get_paper_paths("banana")

    def test_parse_paper(self):
        # paper_paths[0] is good xml, paper_paths[1] is bad.xml
        good = util.parse_paper(self.paper_paths[0])
        bad = util.parse_paper(self.paper_paths[1])

        self.assertEqual(good.pmid.text, "25393036")
        self.assertEqual(bad.text, "Lorem Ipsum")

        with self.assertRaises(AttributeError):
            foo = bad.pmid.text

    def test_load_paper_xmls(self):
        soups = util.load_paper_xmls(self.paper_paths)
        self.assertTrue(isinstance(soups[0].text, str))

        with self.assertRaises(IsADirectoryError):
            util.load_paper_xmls(self.test_xml)

    def test_load_dict(self):
        abbrev_dict = os.path.join(self.test_json, "abbreviations.json")
        self.assertEqual(util.load_dict("banana.json"), {})
        self.assertEqual(util.load_dict(abbrev_dict)["FP"], ["F"])

    def test_normalise_sentence(self):
        s = "Phaco alone reduced the mean IOP from a preoperative level of 22.3±6.3 to 14.0±3.7 mm Hg"
        s_norm = "Phaco alone reduced the mean Intraocular Pressure from a " \
                 "preoperative level of 22.3 +/- 6.3 to 14.0 +/- 3.7 mmHg"

        self.assertEqual(util.normalise_sentence(s), s_norm)
        self.assertEqual(util.normalise_sentence("hello world"), "hello world")

    def test_heap_max_n_indices(self):
        values = [20, -4, 55, 10, 100, 25, 0, 17]

        self.assertEqual(util.get_n_max_indices(values, 3), [4, 2, 5])
        with self.assertRaises(IndexError):
            util.get_n_max_indices(values, len(values) + 1)

    def test_trie_creation(self):
        mapping = {"test": ["hello"]}
        mapping_trie = util.Trie(mapping=mapping)
        string_trie = util.Trie(strings=["hello"])

        self.assertEqual(mapping_trie.check("hello"), "test")
        self.assertEqual(mapping_trie.check("bad"), None)

        self.assertTrue(string_trie.check("hello"))
        self.assertFalse(string_trie.check("bad"))

    def test_cache(self):
        self.assertEqual(util.Cache(file_path="bad").cache, {})

    def test_pug_render(self):
        template_path = util.get_demo_template_path()
        sample_json = os.path.join(self.test_json,
                                   "random_forest-results-2019-04-12_20-07.json")
        results = os.path.join(self.test_json, "results.html")

        util.render_pug(template_path, out_file=results, json_path=sample_json)
        self.assertTrue(os.path.exists(results))

    def mocked_gettgt(self):
        return "tgt"

    # Patch requests, ticket and authentication
    @patch('requests.get', new=mocked_requests_get)
    def test_get_umls_classes(self):
        get_ticket = Mock()

        def mocked_get_ticket():
            return "ticket"

        get_ticket.side_effect = mocked_get_ticket

        util.auth_client = AuthenticationMock("abc")

        classes = util.get_umls_classes("happiness")
        self.assertEqual(classes, ["Mental Process"])
        # checking that the mock object is called
        self.assertEqual(util.get_umls_classes("Vu Luong"), ["Mental Process"])


if __name__ == '__main__':
    unittest.main()
