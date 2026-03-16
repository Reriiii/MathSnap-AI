"""
Vocabulary for HMER LaTeX token sequences.

Handles encoding/decoding between LaTeX token strings and integer indices.
Supports special tokens: <PAD>, <SOS>, <EOS>, <UNK>.
"""

import json
import csv
from typing import List, Optional
from pathlib import Path


class Vocab:
    """LaTeX token vocabulary for HMER."""

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Initialize with special tokens."""
        special = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        for i, tok in enumerate(special):
            self.token2idx[tok] = i
            self.idx2token[i] = tok

    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.token2idx[self.SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[self.EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.token2idx)

    def build_from_csv(self, csv_path: str, min_freq: int = 1):
        """
        Build vocabulary from a processed CSV file.

        Args:
            csv_path: path to CSV with 'latex' column
            min_freq: minimum frequency for a token to be included
        """
        token_freq = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for tok in row['latex'].split():
                    token_freq[tok] = token_freq.get(tok, 0) + 1

        # Reset and rebuild
        self.token2idx = {}
        self.idx2token = {}
        self._init_special_tokens()

        # Add tokens sorted alphabetically (for reproducibility)
        for tok in sorted(token_freq.keys()):
            if token_freq[tok] >= min_freq and tok not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[tok] = idx
                self.idx2token[idx] = tok

        print(f"Vocabulary built: {len(self)} tokens "
              f"({len(self) - 4} unique LaTeX tokens + 4 special tokens)")

    def build_from_token_list(self, tokens: List[str]):
        """Build vocabulary from a list of unique tokens."""
        self.token2idx = {}
        self.idx2token = {}
        self._init_special_tokens()

        for tok in sorted(set(tokens)):
            if tok not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[tok] = idx
                self.idx2token[idx] = tok

    def encode(self, latex_str: str, add_sos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Convert a space-separated LaTeX string to a list of token indices.

        Args:
            latex_str: space-separated LaTeX tokens
            add_sos: prepend <SOS> token
            add_eos: append <EOS> token

        Returns:
            List of integer indices
        """
        tokens = latex_str.split()
        indices = []

        if add_sos:
            indices.append(self.sos_idx)

        for tok in tokens:
            indices.append(self.token2idx.get(tok, self.unk_idx))

        if add_eos:
            indices.append(self.eos_idx)

        return indices

    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """
        Convert a list of token indices back to a LaTeX string.

        Args:
            indices: list of integer indices
            remove_special: if True, remove <PAD>, <SOS>, <EOS> tokens

        Returns:
            Space-separated LaTeX string
        """
        tokens = []
        for idx in indices:
            tok = self.idx2token.get(idx, self.UNK_TOKEN)
            if remove_special and tok in (self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN):
                if tok == self.EOS_TOKEN:
                    break  # Stop at EOS
                continue
            tokens.append(tok)
        return " ".join(tokens)

    def save(self, filepath: str):
        """Save vocabulary to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        data = {
            'token2idx': self.token2idx,
            'idx2token': {str(k): v for k, v in self.idx2token.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to {filepath} ({len(self)} tokens)")

    def load(self, filepath: str):
        """Load vocabulary from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.token2idx = data['token2idx']
        self.idx2token = {int(k): v for k, v in data['idx2token'].items()}
        print(f"Vocabulary loaded from {filepath} ({len(self)} tokens)")

    @classmethod
    def from_file(cls, filepath: str) -> 'Vocab':
        """Create a Vocab instance from a saved JSON file."""
        vocab = cls()
        vocab.load(filepath)
        return vocab


if __name__ == "__main__":
    # Build vocab from processed training data
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/processed/train.csv"
    save_path = sys.argv[2] if len(sys.argv) > 2 else "dataset/processed/vocab.json"

    vocab = Vocab()
    vocab.build_from_csv(csv_path)
    vocab.save(save_path)

    # Test encode/decode
    print("\n--- Test ---")
    test_str = "x + y = z"
    encoded = vocab.encode(test_str)
    decoded = vocab.decode(encoded)
    print(f"Original: {test_str}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
