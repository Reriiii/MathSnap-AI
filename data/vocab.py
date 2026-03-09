import json
from tqdm.auto import tqdm

_SPECIAL = ['<pad>', '<sos>', '<eos>', '<unk>', '∅']


class Vocabulary:
    def __init__(self):
        self.t2i: dict = {}
        self.i2t: dict = {}
        for t in _SPECIAL:
            self._add(t)

    def _add(self, tok: str):
        if tok not in self.t2i:
            i = len(self.t2i)
            self.t2i[tok] = i
            self.i2t[i]   = tok

    def __len__(self):           return len(self.t2i)
    @property
    def pad_idx(self):           return self.t2i['<pad>']
    @property
    def sos_idx(self):           return self.t2i['<sos>']
    @property
    def eos_idx(self):           return self.t2i['<eos>']
    @property
    def none_idx(self):          return self.t2i['∅']

    def encode(self, tokens):
        unk = self.t2i['<unk>']
        return [self.t2i.get(t, unk) for t in tokens]

    def decode(self, ids):
        return [self.i2t.get(int(i), '<unk>') for i in ids]

    @classmethod
    def build(cls, token_lists):
        v = cls()
        for tok in sorted({t for toks in token_lists for t in toks}):
            v._add(tok)
        return v

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.t2i, f, ensure_ascii=False, indent=2)
        tqdm.write(f"Vocab saved → {path}  ({len(self)} tokens)")

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            t2i = json.load(f)
        v = cls()
        v.t2i = t2i
        v.i2t = {int(idx): tok for tok, idx in t2i.items()}
        tqdm.write(f"Vocab loaded ← {path}  ({len(v)} tokens)")
        return v
