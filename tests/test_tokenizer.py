import os
from unittest import TestCase

from keras_xlnet import Tokenizer


class TestTokenizer(TestCase):

    def test_tokenizer(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        spm_path = os.path.join(current_path, 'spiece.model')
        tokenizer = Tokenizer(
            spm_path,
            remove_spaces=True,
            remove_accents=True,
            cased=True,
            sample=True,
        )
        text = 'build XLNet'
        for _ in range(10):
            ids = tokenizer.encode(text)
            self.assertEqual(text, tokenizer.decode(ids))
        tokenizer = Tokenizer(
            spm_path,
            remove_spaces=False,
            remove_accents=False,
            cased=False,
            sample=False,
        )
        ids = tokenizer.encode(text)
        self.assertEqual([1266, 3512, 368, 1942], ids)
        self.assertEqual(text.lower(), tokenizer.decode(ids))
