from unittest import TestCase
from keras_xlnet.pretrained import get_pretrained_paths, PretrainedList


class TestPretrained(TestCase):

    def test_get_pretrained(self):
        get_pretrained_paths(PretrainedList.__test__)
