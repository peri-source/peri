import unittest

class TemplateTestCase(unittest.TestCase):
    def setUp(self):
        self.value = 'blah'

    def tearDown(self):
        pass

    def test_setup_worked(self):
        self.assertEqual(self.value, 'blah')

    def test_string_index(self):
        self.assertEqual(self.value.index('l'), 1)

