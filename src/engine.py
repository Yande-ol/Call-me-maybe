"""Inference engine with constrained decoding for deterministic tool calls."""

import json
from typing import Any, List, Set
import numpy as np

from src.grammar import GrammarState, JSONGrammar
from src.models import FunctionDefinition, ToolCall, validate_tool_call
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
            '"name" and "parameters" to call. Preserve every extracted value '
            "exactly. A negative number must start with '-' immediately "
            "before its digits; never omit that sign. Preserve decimal "
            "points and string characters too. For example, if the user asks "
            "to add -2 and 3.5, the numeric values must be -2 and 3.5, not "
            "2 and 3.5. For regex requests, use simple patterns without "
            "capturing groups () and preserve escapes such as \\d+ and \\s+ "
            "exactly. When a prompt names a replacement character such as "
            "asterisk, hash, or dash, output that single character itself "
            "without spaces or repetition (e.g. '*' or '-'). When the "
            "request says all vowels, include both lowercase and uppercase "
            "vowels in the regular expression ([aeiouAEIOU]). Empty string "
            'values must be represented as "".\n'
            f"Available tools:\n{tools_json}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return prompt

    def _get_allowed_ids(
        self, grammar: JSONGrammar, allowed_strings: Set[str]
    ) -> List[int]:
        """Maps allowed textual continuations to their token IDs."""
        if grammar.state == GrammarState.ARG_VALUE:
            ptype = grammar.current_param_type()
            if ptype == "string":
                if not grammar.in_string_value:
                    return list(
                        set(
                            self.vocab.open_quote_token_ids
                            + self.vocab.empty_quote_token_ids
                        )
                    )
                return self.vocab.string_token_ids
            if ptype in ("number", "integer"):
                partial = grammar.buffer[grammar.state_start_idx:]
                ids: List[int] = []
                for token_id in self.vocab.number_token_ids:
                    tok = self.vocab.id_to_token[token_id]
                    if grammar.is_valid_number_prefix(partial + tok):
                        ids.append(token_id)
                if not partial:
                    for raw_tok in (
                        "Ġ-",
                        "-",
                        "Ġ0",
                        "Ġ1",
                        "Ġ2",
                        "Ġ3",
                        "Ġ4",
                        "Ġ5",
                        "Ġ6",
                        "Ġ7",
                        "Ġ8",
                        "Ġ9",
                    ):
                        if raw_tok in self.vocab.token_to_id:
                            ids.append(self.vocab.token_to_id[raw_tok])
                for ch in allowed_strings:
                    if ch in ("-", ".", "e", "E", "+", ",", "}"):
                        if ch in self.vocab.token_to_id:
                            ids.append(self.vocab.token_to_id[ch])
                return list(set(ids))

        allowed_ids: List[int] = []
        for text in allowed_strings:
            if text in self.vocab.token_to_id:
                allowed_ids.append(self.vocab.token_to_id[text])
            else:
                encoded = self.model.encode(text)
                tokenized: List[int] = encoded.tolist()[0]
                if tokenized:
                    allowed_ids.append(tokenized[0])
        return list(set(allowed_ids))

    def generate(
        self,
        user_prompt: str,
        function_definitions: List[FunctionDefinition],
        max_tokens: int = 128,
    ) -> ToolCall:
        """Generates a valid ToolCall under grammar constraints."""
        grammar = JSONGrammar(function_definitions)
        full_text = self._build_system_prompt(
            function_definitions, user_prompt
        )

        encoded_full = self.model.encode(full_text)
        input_ids: List[int] = encoded_full.tolist()[0]

        deterministic_states = {
            GrammarState.START,
            GrammarState.FUNCTION_NAME,
            GrammarState.AFTER_NAME,
            GrammarState.ARG_KEY,
            GrammarState.COLON,
            GrammarState.NEXT_OR_CLOSE_ARGS,
            GrammarState.CLOSE_JSON,
        }

        for _ in range(max_tokens):
            if grammar.state == GrammarState.DONE:
                break

            allowed_literals = grammar.get_allowed_literals_or_chars()
            if not allowed_literals:
                break

            if (
                len(allowed_literals) == 1
                and grammar.state in deterministic_states
            ):
                candidate = next(iter(allowed_literals))
                grammar.feed(candidate)
                token_ids: List[int] = (
                    self.model.encode(candidate).tolist()[0]
                )
                input_ids.extend(token_ids)
                continue

            # Fronteira BPE limpa (sem espaço solto após ':') no 1º token de
            # strings (para decidir entre ' "' e ' ""') e no 1º token de
            # números (para decidir entre ' -' e dígitos).
            partial = grammar.buffer[grammar.state_start_idx:]
            if grammar.state == GrammarState.ARG_VALUE and (
                (
                    grammar.current_param_type() == "string"
                    and not grammar.in_string_value
                )
                or (
                    grammar.current_param_type() in ("number", "integer")
                    and not partial
                )
            ):
                bpe_prefix = full_text + grammar.buffer.rstrip(" ")
                query_ids: List[int] = (
                    self.model.encode(bpe_prefix).tolist()[0]
                )
            else:
                query_ids = input_ids

            raw_logits = self.model.get_logits_from_input_ids(query_ids)
            allowed_ids = self._get_allowed_ids(grammar, allowed_literals)

            if not allowed_ids:
                raise RuntimeError(
                    "No valid tokens available for grammar state "
                    f"{grammar.state.name}."
                )

            logits_arr = np.array(raw_logits, dtype=np.float32)
            mask = np.full(len(logits_arr), -np.inf, dtype=np.float32)
            mask[allowed_ids] = logits_arr[allowed_ids]

            next_token_id = int(np.argmax(mask))

            if (
                grammar.state == GrammarState.ARG_VALUE
                and grammar.current_param_type() == "string"
                and not grammar.in_string_value
            ):
                if next_token_id in self.vocab.empty_quote_token_ids:
                    grammar.feed('""')
                    token_ids = self.model.encode('""').tolist()[0]
                    input_ids.extend(token_ids)
                else:
                    grammar.feed('"')
                    token_ids = self.model.encode('"').tolist()[0]
                    input_ids.extend(token_ids)
                continue

            token_str = self.vocab.id_to_token.get(
                next_token_id, self.model.decode([next_token_id])
            )
            if (
                grammar.state == GrammarState.ARG_VALUE
                and grammar.current_param_type() in ("number", "integer")
                and not partial
            ):
                token_str = token_str.lstrip(" ")

            prev_len = len(grammar.buffer)
            grammar.feed(token_str)
            added_text = grammar.buffer[prev_len:]

            if added_text != token_str:
                added_ids: List[int] = (
                    self.model.encode(added_text).tolist()[0]
                )
                input_ids.extend(added_ids)
            else:
                input_ids.append(next_token_id)

        tool_call = ToolCall.model_validate_json(grammar.buffer.strip())
        function_map = {fn.name: fn for fn in function_definitions}
        return validate_tool_call(tool_call, function_map)
