# Leitura/mapeamento de tokens
"""Vocabulary loader and token mapping utilities without AI frameworks."""

import json
from pathlib import Path
from typing import Dict, List, Set, Union


class VocabManager:
    """Manages token mappings loaded directly from the tokenizer files.

    Attributes
    ----------
    token_to_id: Dict[str, int]
        Mapping from raw token string to token ID.
    id_to_token: Dict[int, str]
        Mapping from token ID to decoded clean string.
    string_token_ids: List[int]
        Precomputed token IDs valid inside JSON string values.
    number_token_ids: List[int]
        Precomputed token IDs consisting purely of digits.
    """

    def __init__(self, tokenizer_or_vocab_path: Union[str, Path]) -> None:
        self.path = Path(tokenizer_or_vocab_path)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.string_token_ids: List[int] = []
        self.number_token_ids: List[int] = []
        self._load_vocab()
        self._precompute_token_sets()

    @staticmethod
    def _clean_bpe_token(token_str: str) -> str:
        """Converts BPE byte-level space (Ġ) and newline (Ċ) markers."""
        return token_str.replace("Ġ", " ").replace("Ċ", "\n")

    def _load_vocab(self) -> None:
        """Parses tokenizer.json or vocab.json strictly with standard json."""
        if not self.path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_vocab: Dict[str, int] = {}

        if isinstance(data, dict) and isinstance(data.get("model"), dict):
            raw_vocab = data["model"].get("vocab", {})
        elif isinstance(data, dict):
            raw_vocab = data
        else:
            raise ValueError(f"Unrecognized vocabulary format in {self.path}")

        self.token_to_id = {str(k): int(v) for k, v in raw_vocab.items()}
        self.id_to_token = {
            int(v): self._clean_bpe_token(str(k))
            for k, v in raw_vocab.items()
        }

    def _precompute_token_sets(self) -> None:
        """Precomputes valid token IDs for numbers and string contents."""
        allowed_str_chars = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789 _-./:?=!@#$%^&*()[]+\\'"
        )
        digits = set("0123456789")
        closing_suffixes = ('"', '",', '", ', '"}', '"}\n', '", \n')

        for tid, clean in self.id_to_token.items():
            if not clean:
                continue
            if all(c in digits for c in clean):
                self.number_token_ids.append(tid)

            if all(c in allowed_str_chars for c in clean):
                self.string_token_ids.append(tid)
            else:
                for suffix in closing_suffixes:
                    if clean.endswith(suffix):
                        prefix = clean[: -len(suffix)]
                        if all(c in allowed_str_chars for c in prefix):
                            self.string_token_ids.append(tid)
                            break

    @property
    def vocab_size(self) -> int:
        """Returns the total number of unique tokens in vocabulary."""
        return len(self.token_to_id)

    def get_token_id(self, token_str: str) -> Union[int, None]:
        """Returns token ID for an exact string match, or None if absent."""
        return self.token_to_id.get(token_str)

    def find_token_ids_matching(self, allowed_prefixes: Set[str]) -> List[int]:
        """Filters token IDs matching any allowed prefix."""
        matching_ids: List[int] = []
        for token_id, token_str in self.id_to_token.items():
            if any(token_str.startswith(pref) for pref in allowed_prefixes):
                matching_ids.append(token_id)
        return matching_ids
