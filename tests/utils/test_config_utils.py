import unittest

from dvml.utils.config_utils import parse_config


class TestParseConfig(unittest.TestCase):
    def test_empty_default(self):
        conf = {
            "opt1": "this",
            "opt2": "that",
            "opt3": 123,
        }

        parsed_conf = parse_config(conf)

        self.assertDictEqual(parsed_conf, conf)

    def test_empty_conf(self):
        conf = None

        default_conf = {
            "opt1": "this",
            "opt2": "that",
            "opt3": 123,
        }

        parsed_conf = parse_config(conf, default_conf)

        self.assertDictEqual(parsed_conf, default_conf)

    def test_with_default(self):
        conf = {
            "opt1": "this",
            "opt2": "that",
        }

        default_conf = {
            "opt1": "one",
            "opt2": "two",
            "opt3": 123,
        }

        expected_conf = {
            "opt1": "this",
            "opt2": "that",
            "opt3": 123,
        }

        parsed_conf = parse_config(conf, default_conf)

        self.assertDictEqual(parsed_conf, expected_conf)

    def test_with_extra(self):
        conf = {
            "opt1": "this",
            "opt2": "that",
            "opt4": [1, 2],
        }

        default_conf = {
            "opt1": "one",
            "opt2": "two",
            "opt3": 123,
        }

        expected_conf = {
            "opt1": "this",
            "opt2": "that",
            "opt3": 123,
            "opt4": [1, 2],
        }

        parsed_conf = parse_config(conf, default_conf)

        self.assertDictEqual(parsed_conf, expected_conf)
