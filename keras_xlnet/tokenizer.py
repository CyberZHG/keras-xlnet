import unicodedata
import sentencepiece as spm

__all__ = ['Tokenizer']


class Tokenizer(object):

    def __init__(self,
                 spm_path,
                 remove_spaces=True,
                 remove_accents=False,
                 cased=True,
                 sample=False):
        """Initialized the tokenizer.

        :param spm_path: The path to the sentence piece model.
        :param remove_spaces: Whether to remove space characters.
        :param remove_accents: Whether to remove accent characters.
        :param cased: Whether it is cased.
        :param sample: A word will be segmented differently on each call if it is True.
        """
        self.remove_spaces = remove_spaces
        self.remove_accents = remove_accents
        self.cased = cased
        self.sample = sample
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_path)

    def tokenize(self, text):
        if self.remove_spaces:
            text = ' '.join(text.strip().split())
        if self.remove_accents:
            text = unicodedata.normalize('NFKD', text)
            text = ''.join([ch for ch in text if not unicodedata.combining(ch)])
        if not self.cased:
            text = text.lower()

        if self.sample:
            pieces = self.sp.SampleEncodeAsPieces(text, 64, 0.1)
        else:
            pieces = self.sp.EncodeAsPieces(text)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
                cur_pieces = self.sp.EncodeAsPieces(piece[:-1].replace('▁', ''))
                if piece[0] != '▁' and cur_pieces[0][0] == '▁':
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)
        return new_pieces

    def encode(self, text):
        """Encode the text.

        :param text: The text.
        :return: A list of ints represents the IDs of tokens.
        """
        pieces = self.tokenize(text)
        return [self.sp.PieceToId(piece) for piece in pieces]

    def decode(self, ids):
        return self.sp.DecodeIds(ids)
