# Lógica de Constrained Decoding
"""Inference engine with constrained decoding for deterministic tool calls."""

import json
from typing import Any, List, Set
import numpy as np

from src.grammar import GrammarState, JSONGrammar
from src.models import FunctionDefinition, ToolCall
from src.vocab import VocabManager


class ConstrainedEngine:
    """Executes constrained decoding inference using a grammar and vocab mask.

    Parameters
    ----------
    model : Any
        Instance of Small_LLM_Model from the provided SDK.
    vocab_manager : VocabManager
        Manager for token-to-ID mappings.
    """

    def __init__(self, model: Any, vocab_manager: VocabManager) -> None:
        self.model = model
        self.vocab = vocab_manager

    def _build_system_prompt(
        self, function_definitions: List[FunctionDefinition], user_prompt: str
    ) -> str:
        """Formats the input prompt presenting available tools to the LLM."""
        tools_dict = [fn.model_dump() for fn in function_definitions]
        tools_json = json.dumps(tools_dict, indent=2)

        prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant that calls functions to solve tasks. "
            "Respond ONLY with a single JSON object containing the function "
            "name and arguments to call.\n"
            f"Available tools:\n{tools_json}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return prompt

    def _get_allowed_ids(self, allowed_strings: Set[str]) -> List[int]:
        """Maps allowed textual continuations to their token IDs."""
        allowed_ids: List[int] = []
        for text in allowed_strings:
            # Se for token exato no dicionário do vocab
            if text in self.vocab.token_to_id:
                allowed_ids.append(self.vocab.token_to_id[text])
            else:
                # Caso seja uma string maior, obtém via tokenize da SDK
                tokenized = self.model.tokenize(text)
                if tokenized:
                    allowed_ids.append(tokenized[0])
        return list(set(allowed_ids))

    def generate(
        self,
        user_prompt: str,
        function_definitions: List[FunctionDefinition],
        max_tokens: int = 128,
    ) -> ToolCall:
        """Generates a valid ToolCall under grammar constraints.

        Parameters
        ----------
        user_prompt : str
            The query or task provided by the user.
        function_definitions : List[FunctionDefinition]
            Available tools the model can invoke.
        max_tokens : int
            Maximum number of tokens to generate before aborting.

        Returns
        -------
        ToolCall
            Pydantic model validated against the generated JSON.
        """
        grammar = JSONGrammar(function_definitions)
        full_text = self._build_system_prompt(
            function_definitions, user_prompt
        )
        generated_json = ""

        input_ids: List[int] = self.model.tokenize(full_text)

        for _ in range(max_tokens):
            if grammar.state == GrammarState.DONE:
                break

            logits = self.model.get_next_token_logits(input_ids)
            allowed_literals = grammar.get_allowed_literals_or_chars()

            # Força literais fixos estruturais para acelerar inferência
            if len(allowed_literals) == 1:
                candidate = next(iter(allowed_literals))
                if len(candidate) > 1 and grammar.state in (
                    GrammarState.START,
                    GrammarState.AFTER_NAME,
                    GrammarState.COLON,
                ):
                    generated_json += candidate
                    grammar.feed(candidate)
                    token_ids: List[int] = self.model.tokenize(candidate)
                    input_ids.extend(token_ids)
                    continue

            allowed_ids = self._get_allowed_ids(allowed_literals)

            if not allowed_ids:
                allowed_ids = list(range(len(logits)))

            # Máscara vetorial com -np.inf
            mask = np.full(len(logits), -np.inf)
            mask[allowed_ids] = logits[allowed_ids]

            next_token_id = int(np.argmax(mask))

            # Acesso direto ao dicionário de IDs para Tokens
            token_str = self.vocab.id_to_token[next_token_id]
            generated_json += token_str
            grammar.feed(token_str)
            input_ids.append(next_token_id)

        return ToolCall.model_validate_json(generated_json.strip())
