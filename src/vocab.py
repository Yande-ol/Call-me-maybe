# Leitura/mapeamento de tokens
"""Vocabulary loader and token mapping utilities without AI frameworks."""

import json
from pathlib import Path
from typing import Dict, List, Set


class VocabManager:
    """Manages token mappings loaded directly from the tokenizer files.

    Attributes
    ----------
    token_to_id: Dict[str, int]
        Mapping from token string/byte representation to token ID.
    id_to_token: Dict[int, str]
        Mapping from token ID to token string.
    """

    def __init__(self, tokenizer_or_vocab_path: str | Path) -> None:
        self.path = Path(tokenizer_or_vocab_path)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._load_vocab()

    def _load_vocab(self) -> None:
        """Parses tokenizer.json or vocab.json strictly with standard json."""
        if not self.path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # tokenizer.json (formato Hugging Face BPE moderno usado pelo Qwen)
        if "model" in data and "vocab" in data["model"]:
            self.token_to_id = data["model"]["vocab"]
        # vocab.json simples (mapeamento direto {"token": id})
        elif isinstance(data, dict):
            self.token_to_id = data
        else:
            raise ValueError(f"Unrecognized vocabulary format in {self.path}")

        # Inverte o mapeamento para lookup reverso eficiente
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        """Returns the total number of unique tokens in vocabulary."""
        return len(self.token_to_id)

    def get_token_id(self, token_str: str) -> int | None:
        """Returns token ID for an exact string match, or None if absent."""
        return self.token_to_id.get(token_str)

    def find_token_ids_matching(self, allowed_prefixes: Set[str]) -> List[int]:
        """Filters token IDs whose string
        representation starts with any prefix."""
        matching_ids: List[int] = []
        for token_id, token_str in self.id_to_token.items():
            if any(token_str.startswith(pref) for pref in allowed_prefixes):
                matching_ids.append(token_id)
        return matching_ids
