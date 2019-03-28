import unittest

import ie_tools.src.util as util


class TestUtil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Setting up tests for util.py")
        cls.mapping = {"test": ["Pharmacy", "Patient"]}
        cls.testing_dir = util.get_testing_dir()

    def setUp(self):
        print("setUp() function")
        pass

    def get_paper_paths(self):
        good = util.load_paper_xmls(self.testing_dir)
        bad = util.load_paper_xmls("ba")
        pass


if __name__ == '__main__':
    unittest.main()
